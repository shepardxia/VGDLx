"""
GVGAI validation backend: run GVGAI via subprocess and parse trace output.

Uses the TraceAgent (Java) to replay an action sequence and emit per-step
state as JSONL. The Python side parses the trace, maps GVGAI itypes to
VGDLx type_idx via sprite names, and returns normalized state dicts.
"""
import json
import os
import shutil
import subprocess
import tempfile

from .discovery import GameEntry

_GVGAI_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'GVGAI'))
_GVGAI_BIN = os.path.join(_GVGAI_DIR, 'bin')
_GVGAI_LIB = os.path.join(_GVGAI_DIR, 'lib')

_TRACE_AGENT = 'tracks.singlePlayer.simple.traceAgent.Agent'

# Cached TraceRunner: compile once, reuse across invocations
_RUNNER_DIR = os.path.join(_GVGAI_BIN, '_trace_runner')


def _ensure_compiled():
    """Compile GVGAI if bin/ directory doesn't exist or is empty."""
    if os.path.isdir(_GVGAI_BIN) and os.listdir(_GVGAI_BIN):
        return
    compile_sh = os.path.join(_GVGAI_DIR, 'compile.sh')
    if not os.path.isfile(compile_sh):
        raise RuntimeError(f"GVGAI compile.sh not found at {compile_sh}")
    result = subprocess.run(['bash', compile_sh], capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        raise RuntimeError(f"GVGAI compilation failed:\n{result.stderr}")


def _ensure_runner(classpath):
    """Compile TraceRunner.java once, cache in _RUNNER_DIR."""
    runner_class = os.path.join(_RUNNER_DIR, 'TraceRunner.class')
    if os.path.isfile(runner_class):
        return
    os.makedirs(_RUNNER_DIR, exist_ok=True)
    runner_code = f"""
import tools.ReplayRNG;
import tracks.ArcadeMachine;
public class TraceRunner {{
    public static void main(String[] args) {{
        String game = args[0];
        String level = args[1];
        int seed = Integer.parseInt(args[2]);

        // Load RNG replay if provided
        String rngFile = System.getProperty("rng.replay.file", "");
        if (!rngFile.isEmpty()) {{
            ReplayRNG.loadFromFile(rngFile);
        }}

        String agent = "{_TRACE_AGENT}";
        ArcadeMachine.runOneGame(game, level, false, agent, null, seed, 0);
    }}
}}
"""
    runner_java = os.path.join(_RUNNER_DIR, 'TraceRunner.java')
    with open(runner_java, 'w') as f:
        f.write(runner_code)
    compile_result = subprocess.run(
        ['javac', '-cp', classpath, '-d', _RUNNER_DIR, runner_java],
        capture_output=True, text=True, timeout=30)
    if compile_result.returncode != 0:
        raise RuntimeError(f"TraceRunner compilation failed:\n{compile_result.stderr}")


def _build_classpath():
    """Build Java classpath from GVGAI bin/ and lib/ directories."""
    lib_jars = []
    if os.path.isdir(_GVGAI_LIB):
        lib_jars = [os.path.join(_GVGAI_LIB, f) for f in os.listdir(_GVGAI_LIB)
                    if f.endswith('.jar')]
    return os.pathsep.join([_GVGAI_BIN] + lib_jars)


def run_gvgai_trajectory(entry: GameEntry, actions: list, seed: int = 42,
                          level_idx: int = 0,
                          action_names: tuple = (),
                          rng_file: str = None) -> list:
    """Run GVGAI via subprocess with TraceAgent, return list of state dicts.

    Args:
        entry: GameEntry with game_file and level_files
        actions: list of action indices (GVGAI ordering — direct index into action_names)
        seed: random seed
        level_idx: which level to use
        action_names: tuple of GVGAI action name strings in order
        rng_file: optional path to RNG replay JSON file for deterministic injection

    Returns:
        list of dicts, each with:
            'step': int
            'score': float
            'done': bool
            'sprites': list of {'itype': int, 'position': [row, col], ...}
            'sprite_registry': dict (name -> itype, from init record)
    """
    _ensure_compiled()
    classpath = _build_classpath()
    _ensure_runner(classpath)

    # Write action sequence to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        actions_file = f.name
        for a in actions:
            gvgai_action = action_names[a]
            f.write(gvgai_action + '\n')

    with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
        trace_file = f.name

    try:
        run_cp = os.pathsep.join([_RUNNER_DIR, classpath])
        java_args = [
            'java', '-Djava.awt.headless=true',
            f'-Dactions.file={actions_file}',
            f'-Dtrace.file={trace_file}',
        ]
        if rng_file:
            java_args.append(f'-Drng.replay.file={rng_file}')
        java_args += [
            '-cp', run_cp,
            'TraceRunner',
            entry.game_file, entry.level_files[level_idx], str(seed),
        ]
        run_result = subprocess.run(
            java_args,
            capture_output=True, text=True, timeout=60,
            cwd=_GVGAI_DIR)

        if run_result.returncode != 0:
            raise RuntimeError(
                f"GVGAI run failed (exit {run_result.returncode}):\n"
                f"stdout: {run_result.stdout[-500:]}\n"
                f"stderr: {run_result.stderr[-500:]}")

        return _parse_trace(trace_file)

    finally:
        if os.path.exists(actions_file):
            os.unlink(actions_file)
        if os.path.exists(trace_file):
            os.unlink(trace_file)


def _parse_trace(trace_file: str) -> list:
    """Parse JSONL trace file into list of state dicts."""
    if not os.path.exists(trace_file):
        raise RuntimeError(f"Trace file not found: {trace_file}")

    states = []
    init_record = None

    with open(trace_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if record.get('init'):
                init_record = record
                continue

            state = {
                'step': record['step'],
                'score': record['score'],
                'done': record['gameOver'],
                'winner': record.get('winner', 'NO_WINNER'),
                'sprites': record.get('sprites', []),
            }
            if init_record:
                state['sprite_registry'] = init_record.get('spriteRegistry', {})
                state['block_size'] = init_record.get('blockSize', 10)
                state['world_dim'] = init_record.get('worldDim', [0, 0])
            states.append(state)

    return states


def normalize_gvgai_state(gvgai_state: dict, game_def) -> dict:
    """Convert GVGAI state dict to the same format as extract_jax_state.

    Maps GVGAI itypes to VGDLx type_idx via sprite name matching.

    Args:
        gvgai_state: dict from _parse_trace
        game_def: GameDef from VGDLx parser (for sprite name -> type_idx mapping)

    Returns:
        Normalized state dict matching extract_jax_state format.
    """
    # Build name -> type_idx mapping from VGDLx game_def
    jax_name_to_idx = {sd.key: sd.type_idx for sd in game_def.sprites}

    # Build itype -> name mapping from GVGAI sprite registry
    gvgai_registry = gvgai_state.get('sprite_registry', {})
    itype_to_name = {v: k for k, v in gvgai_registry.items()}

    # Group sprites by VGDLx type_idx
    types = {}
    for sd in game_def.sprites:
        types[sd.type_idx] = {
            'key': sd.key,
            'alive_count': 0,
            'positions': [],
            'orientations': [],
        }

    for sprite in gvgai_state.get('sprites', []):
        itype = sprite['itype']
        name = itype_to_name.get(itype)
        if name is None:
            continue
        type_idx = jax_name_to_idx.get(name)
        if type_idx is None:
            continue

        row, col = sprite['position']
        types[type_idx]['positions'].append((row, col))
        types[type_idx]['alive_count'] += 1
        types[type_idx]['orientations'].append((0.0, 1.0))  # default orientation

    # Sort positions for order-independent comparison
    for info in types.values():
        info['positions'].sort()
        info['orientations'].sort()

    return {
        'types': types,
        'score': gvgai_state.get('score', 0),
        'done': gvgai_state.get('done', False),
        'step': gvgai_state.get('step', 0),
    }
