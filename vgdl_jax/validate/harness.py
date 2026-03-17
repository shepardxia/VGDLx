"""
Validation harness: orchestrates cross-engine comparison between GVGAI and VGDLx.

All functions accept GameEntry objects for game identification (no string-based path construction).

Validation levels (inspired by PuzzleJAX validate_sols.py):
  0: Game loads
  1: Initial state extracted correctly
  2: Single NOOP step runs
  3: N-step trajectory (10, 50 steps)
  4: Terminal state (score, done, win)
"""
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from .discovery import GameEntry

# Ensure py-vgdl is importable (needed by setup_pyvgdl_game)
_PYVGDL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'py-vgdl')
if _PYVGDL_DIR not in sys.path:
    sys.path.insert(0, _PYVGDL_DIR)


@dataclass
class StepComparison:
    step: int
    action: int
    state_a: dict           # normalized state from engine A (reference: GVGAI or py-vgdl)
    state_b: Optional[dict] # normalized state from engine B (vgdl-jax), None until ready
    matches: bool
    diffs: List[str]


@dataclass
class TrajectoryResult:
    game_name: str
    n_steps: int
    actions: List[int]
    steps: List[StepComparison]
    level: int  # highest passing validation level (0-4)


_POS_TOL = 1e-4  # tolerance for float32 vs float64 position comparison


def _positions_close(pos_a, pos_b, tol=_POS_TOL):
    """Check if two position lists match within tolerance (order-independent).

    Uses greedy nearest-neighbor matching to avoid false positives when
    float32 rounding changes sort order of nearly-identical positions.
    """
    if len(pos_a) != len(pos_b):
        return False
    if not pos_a:
        return True
    remaining = list(pos_b)
    for a in pos_a:
        best_idx = None
        best_dist = float('inf')
        for j, b in enumerate(remaining):
            d = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
            if d < best_dist:
                best_dist = d
                best_idx = j
        if best_dist >= tol:
            return False
        remaining.pop(best_idx)
    return True


def compare_states(state_a, state_b):
    """Field-by-field comparison of two normalized state dicts.

    Args:
        state_a: from extract_pyvgdl_state or extract_jax_state
        state_b: from extract_pyvgdl_state or extract_jax_state

    Returns:
        (matches: bool, diffs: list[str])
    """
    diffs = []

    # Score
    if state_a['score'] != state_b['score']:
        diffs.append(f"score: {state_a['score']} vs {state_b['score']}")

    # Done
    if state_a['done'] != state_b['done']:
        diffs.append(f"done: {state_a['done']} vs {state_b['done']}")

    # Per-type comparison
    all_types = set(state_a['types'].keys()) | set(state_b['types'].keys())
    for tidx in sorted(all_types):
        ta = state_a['types'].get(tidx)
        tb = state_b['types'].get(tidx)
        if ta is None:
            diffs.append(f"type {tidx}: missing from state_a")
            continue
        if tb is None:
            diffs.append(f"type {tidx}: missing from state_b")
            continue

        key = ta.get('key', tb.get('key', f'type_{tidx}'))

        if ta['alive_count'] != tb['alive_count']:
            diffs.append(
                f"{key}(t{tidx}): alive_count {ta['alive_count']} vs {tb['alive_count']}")

        pos_a = ta['positions']
        pos_b = tb['positions']
        if not _positions_close(pos_a, pos_b):
            diffs.append(
                f"{key}(t{tidx}): positions differ "
                f"(a={pos_a[:3]}{'...' if len(pos_a) > 3 else ''} "
                f"vs b={pos_b[:3]}{'...' if len(pos_b) > 3 else ''})")

    return (len(diffs) == 0, diffs)


# ── py-vgdl trajectory runner ────────────────────────────────────────


def setup_pyvgdl_game(entry: GameEntry, level_idx=0):
    """Set up py-vgdl game without renderer (state extraction only).

    Args:
        entry: GameEntry with game_file and level_files
        level_idx: which level to use (default 0)

    Returns:
        (game, action_keys, sprite_key_order)
        - game: BasicGameLevel instance
        - action_keys: list of Action objects (index = action int)
        - sprite_key_order: list of sprite keys matching type_idx order
    """
    import vgdl as pyvgdl

    with open(entry.game_file) as f:
        game_desc = f.read()
    with open(entry.level_files[level_idx]) as f:
        level_desc = f.read()

    domain = pyvgdl.VGDLParser().parse_game(game_desc)
    game = domain.build_level(level_desc)

    # Get action keys (same ordering as VGDLEnv)
    from collections import OrderedDict
    action_dict = OrderedDict(game.get_possible_actions())
    action_keys = list(action_dict.values())

    # Sprite key order from registry (matches parser registration order)
    sprite_key_order = list(game.sprite_registry.sprite_keys)

    return game, action_keys, sprite_key_order


def run_pyvgdl_trajectory(entry: GameEntry, actions, seed=42, rng_replay=None, level_idx=0):
    """Run py-vgdl for an action sequence, return state at each step.

    Args:
        entry: GameEntry
        actions: list of int action indices
        seed: random seed for py-vgdl
        rng_replay: optional ReplayRandomGenerator to inject
        level_idx: which level to use

    Returns:
        list of normalized state dicts (one per step, including initial state)
    """
    from .state_extractor import extract_pyvgdl_state

    game, action_keys, sprite_key_order = setup_pyvgdl_game(entry, level_idx)
    game.set_seed(seed)

    if rng_replay is not None:
        game.set_random_generator(rng_replay)

    # Use actual block_size from the game (default=1 when no renderer)
    block_size = game.block_size

    # Initial state
    states = [extract_pyvgdl_state(game, sprite_key_order, block_size)]

    for action_idx in actions:
        if game.ended:
            break
        action = action_keys[action_idx]
        game.tick(action)
        states.append(extract_pyvgdl_state(game, sprite_key_order, block_size))

    return states


def validate_pyvgdl_loads(entry: GameEntry, level_idx=0):
    """Validation level 0: game loads without error.

    Returns:
        (success: bool, error_msg: str or None)
    """
    try:
        setup_pyvgdl_game(entry, level_idx)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_pyvgdl_state_extraction(entry: GameEntry, level_idx=0):
    """Validation level 1: initial state is extractable and well-formed.

    Returns:
        (success: bool, state_or_error)
    """
    from .state_extractor import extract_pyvgdl_state

    try:
        game, action_keys, sprite_key_order = setup_pyvgdl_game(entry, level_idx)
        state = extract_pyvgdl_state(game, sprite_key_order, game.block_size)

        # Basic sanity checks
        assert 'types' in state
        assert 'score' in state
        assert 'done' in state
        assert len(state['types']) == len(sprite_key_order)

        for tidx, info in state['types'].items():
            assert 'alive_count' in info
            assert 'positions' in info
            assert len(info['positions']) == info['alive_count']

        return True, state
    except Exception as e:
        return False, str(e)


def validate_pyvgdl_trajectory(entry: GameEntry, n_steps=50, seed=42, level_idx=0):
    """Validation levels 2+3: run trajectory, verify states extracted at each step.

    Uses NOOP action to minimize avatar-driven complexity.

    Returns:
        (success: bool, states_or_error)
    """
    try:
        game, action_keys, _ = setup_pyvgdl_game(entry, level_idx)
        # Find NOOP action index (Action with empty keys tuple)
        noop_idx = next(i for i, a in enumerate(action_keys) if a.keys == ())
        actions = [noop_idx] * n_steps

        states = run_pyvgdl_trajectory(entry, actions, seed=seed, level_idx=level_idx)

        assert len(states) >= 2, f"Expected at least 2 states, got {len(states)}"

        # Verify all states are well-formed
        for i, s in enumerate(states):
            assert 'types' in s, f"Step {i}: missing 'types'"
            assert 'score' in s, f"Step {i}: missing 'score'"

        return True, states
    except Exception as e:
        return False, str(e)


# ── vgdl-jax trajectory runner ─────────────────────────────────────


def setup_jax_game(entry: GameEntry, level_idx=0):
    """Set up vgdl-jax compiled game from a GameEntry.

    Uses max_sprites_per_type large enough for ALL sprites (including inert
    background tiles like 'floor'/'grass') so counts match py-vgdl exactly.

    Args:
        entry: GameEntry with game_file and level_files
        level_idx: which level to use (default 0)

    Returns:
        (compiled, game_def)
    """
    from collections import Counter
    from vgdl_jax.parser import parse_vgdl
    from vgdl_jax.compiler import compile_game

    game_def = parse_vgdl(entry.game_file, entry.level_files[level_idx])
    # Compute max sprites needed across all types (including inert background)
    counts = Counter(t for t, r, c in game_def.level.initial_sprites)
    max_n = max(counts.values(), default=1) + 10  # headroom for spawns
    compiled = compile_game(game_def, max_sprites_per_type=max_n)
    return compiled, game_def


def run_jax_trajectory(entry: GameEntry, actions, seed=42, level_idx=0):
    """Run vgdl-jax for an action sequence, return state at each step.

    Returns list of normalized state dicts (same format as run_pyvgdl_trajectory).
    """
    import jax
    from .state_extractor import extract_jax_state
    from vgdl_jax.data_model import get_block_size

    compiled, game_def = setup_jax_game(entry, level_idx)
    sgm = compiled.static_grid_map
    bs = get_block_size(game_def)
    state = compiled.init_state.replace(rng=jax.random.PRNGKey(seed))

    # Initial state
    states = [extract_jax_state(state, game_def, static_grid_map=sgm, block_size=bs)]

    for action_idx in actions:
        if bool(state.done):
            break
        state = compiled.step_fn(state, action_idx)
        states.append(extract_jax_state(state, game_def, static_grid_map=sgm, block_size=bs))

    return states


def run_comparison(entry: GameEntry, actions, seed=42, use_rng_replay=False, level_idx=0):
    """Run both engines on same actions, compare state at every step.

    Args:
        entry: GameEntry
        actions: list of int action indices
        seed: random seed
        use_rng_replay: if True, record JAX RNG sequence and inject into py-vgdl
        level_idx: which level to use

    Returns:
        TrajectoryResult with per-step StepComparison.
    """
    import jax
    from .state_extractor import extract_pyvgdl_state, extract_jax_state
    from vgdl_jax.data_model import gvgai_block_size

    # ── Set up JAX side ──
    compiled, game_def = setup_jax_game(entry, level_idx)
    sgm = compiled.static_grid_map
    bs = (game_def.square_size if game_def.square_size > 0
          else gvgai_block_size(game_def.level.height, game_def.level.width))
    jax_state = compiled.init_state.replace(rng=jax.random.PRNGKey(seed))

    # ── Set up py-vgdl side ──
    game, action_keys, sprite_key_order = setup_pyvgdl_game(entry, level_idx)
    game.set_seed(seed)

    # ── Optional RNG replay ──
    rng_replay = None
    recorder = None
    if use_rng_replay:
        from .rng_replay import RNGRecorder, ReplayRandomGenerator

        sprite_configs = get_sprite_configs(compiled)
        effects = get_effects(compiled)

        recorder = RNGRecorder(sprite_configs, effects, game_def)
        rng_replay = ReplayRandomGenerator(game_def)
        game.set_random_generator(rng_replay)

    max_n = compiled.init_state.alive.shape[1]
    rng_key = jax.random.PRNGKey(seed)
    block_size = game.block_size

    # ── Compare initial state ──
    pv_state = extract_pyvgdl_state(game, sprite_key_order, block_size)
    jx_state = extract_jax_state(jax_state, game_def, static_grid_map=sgm, block_size=bs)
    matches, diffs = compare_states(pv_state, jx_state)

    step_comparisons = [StepComparison(
        step=0, action=-1,
        state_a=pv_state, state_b=jx_state,
        matches=matches, diffs=diffs,
    )]

    # ── Step through actions ──
    sprite_configs = get_sprite_configs(compiled) if recorder else None

    for i, action_idx in enumerate(actions):
        pv_done = game.ended
        jx_done = bool(jax_state.done)

        if pv_done and jx_done:
            break

        # RNG replay: record this step's draws before stepping
        if recorder is not None:
            record, rng_key = recorder.record_step(rng_key, max_n=max_n)

        # Step JAX first (so we can extract actual chaser directions)
        prev_jax_state = jax_state
        if not jx_done:
            jax_state = compiled.step_fn(jax_state, action_idx)

        # Patch chaser directions using distance field, then step py-vgdl
        if recorder is not None:
            from .rng_replay import patch_chaser_directions
            patch_chaser_directions(record, prev_jax_state, sprite_configs,
                                    game_def.level.height, game_def.level.width,
                                    block_size=bs)
            rng_replay.set_step_record(record)

        if not pv_done:
            game.tick(action_keys[action_idx])

        pv_state = extract_pyvgdl_state(game, sprite_key_order, block_size)
        jx_state = extract_jax_state(jax_state, game_def, static_grid_map=sgm, block_size=bs)
        matches, diffs = compare_states(pv_state, jx_state)

        step_comparisons.append(StepComparison(
            step=i + 1, action=action_idx,
            state_a=pv_state, state_b=jx_state,
            matches=matches, diffs=diffs,
        ))

    # ── Build result ──
    return _build_trajectory_result(entry, actions, step_comparisons)


# ── GVGAI comparison ─────────────────────────────────────────────────


def run_gvgai_comparison(entry: GameEntry, actions, seed=42, level_idx=0,
                         use_rng_replay=False):
    """Run GVGAI and VGDLx on same actions, compare state at every step.

    Args:
        entry: GameEntry (must have source='gvgai' or point to GVGAI game files)
        actions: list of int action indices (VGDLx convention)
        seed: random seed
        level_idx: which level to use
        use_rng_replay: if True, run VGDLx first to record RNG choices,
            then inject them into GVGAI so both engines make identical
            random decisions (directions, spawn rolls).

    Returns:
        TrajectoryResult with per-step StepComparison.
    """
    from .backend_gvgai import run_gvgai_trajectory, normalize_gvgai_state
    from .state_extractor import extract_jax_state
    from vgdl_jax.data_model import get_block_size

    compiled, game_def = setup_jax_game(entry, level_idx)
    sgm = compiled.static_grid_map
    bs = get_block_size(game_def)

    # ── Run JAX side + optionally build RNG replay ──
    rng_file_path = None
    if use_rng_replay:
        from .rng_replay import build_gvgai_rng_records, write_gvgai_rng_file
        import tempfile

        # Single trajectory run: produces both RNG records and raw states
        records, raw_states = build_gvgai_rng_records(
            compiled, game_def, actions, seed)
        jax_states = [extract_jax_state(s, game_def, static_grid_map=sgm, block_size=bs)
                      for s in raw_states]

        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        rng_file_path = tmp.name
        tmp.close()
        write_gvgai_rng_file(records, rng_file_path)
    else:
        jax_states = run_jax_trajectory(entry, actions, seed, level_idx)

    try:
        # ── Run GVGAI side ──
        gvgai_raw = run_gvgai_trajectory(
            entry, actions, seed=seed, level_idx=level_idx,
            action_names=compiled.action_names,
            rng_file=rng_file_path,
        )
        gvgai_states = [normalize_gvgai_state(s, game_def) for s in gvgai_raw]

        # ── Compare step by step ──
        step_comparisons = []
        n_steps = min(len(jax_states), len(gvgai_states))
        for i in range(n_steps):
            action = actions[i - 1] if i > 0 and i - 1 < len(actions) else -1
            matches, diffs = compare_states(gvgai_states[i], jax_states[i])
            step_comparisons.append(StepComparison(
                step=i, action=action,
                state_a=gvgai_states[i], state_b=jax_states[i],
                matches=matches, diffs=diffs,
            ))

        return _build_trajectory_result(entry, actions, step_comparisons)

    finally:
        if rng_file_path and os.path.exists(rng_file_path):
            os.unlink(rng_file_path)


def _build_trajectory_result(entry, actions, step_comparisons):
    """Compute validation level and build TrajectoryResult."""
    all_match = all(sc.matches for sc in step_comparisons)
    init_match = step_comparisons[0].matches if step_comparisons else False
    level = 0
    if init_match:
        level = 1
    if len(step_comparisons) > 1:
        level = 2
    if len(step_comparisons) > 10:
        level = 3
    if all_match:
        level = 4

    return TrajectoryResult(
        game_name=entry.name,
        n_steps=len(actions),
        actions=actions,
        steps=step_comparisons,
        level=level,
    )


# ── Helpers for extracting configs from compiled game ─────────────────


def get_sprite_configs(compiled):
    """Extract sprite_configs list from a CompiledGame.

    Delegates to compiler._build_sprite_configs for the actual resolution logic
    (SpawnPoint target orientation/speed, spawn_orientation override, etc.) and
    converts to dicts for backward compatibility with RNGRecorder.
    """
    from dataclasses import asdict
    from vgdl_jax.compiler import _build_sprite_configs
    from vgdl_jax.data_model import get_block_size

    gd = compiled.game_def
    block_size = get_block_size(gd)
    configs = _build_sprite_configs(gd, block_size)
    return [asdict(cfg) for cfg in configs]


def get_effects(compiled):
    """Extract compiled effects list from a CompiledGame."""
    gd = compiled.game_def
    effects = []
    for ed in gd.effects:
        is_eos = (ed.actee_stype == 'EOS')
        actor_indices = gd.resolve_stype(ed.actor_stype)

        if is_eos:
            for ta_idx in actor_indices:
                effects.append(dict(
                    type_a=ta_idx,
                    type_b=-1,
                    is_eos=True,
                    effect_type=ed.effect_type,
                    score_change=ed.score_change,
                    kwargs=dict(ed.kwargs),
                ))
        else:
            actee_indices = gd.resolve_stype(ed.actee_stype)
            for ta_idx in actor_indices:
                for tb_idx in actee_indices:
                    eff = dict(
                        type_a=ta_idx,
                        type_b=tb_idx,
                        is_eos=False,
                        effect_type=ed.effect_type,
                        score_change=ed.score_change,
                        kwargs=dict(ed.kwargs),
                    )
                    if ed.effect_type == 'teleport_to_exit':
                        # Resolve exit type from the actee portal sprite's
                        # portal_exit_stype (not from effect kwargs).
                        portal_sd = gd.sprites[tb_idx]
                        exit_stype = portal_sd.portal_exit_stype
                        if exit_stype:
                            exit_indices = gd.resolve_stype(exit_stype)
                            eff['kwargs']['exit_type_idx'] = (
                                exit_indices[0] if exit_indices else -1)
                    effects.append(eff)
    return effects
