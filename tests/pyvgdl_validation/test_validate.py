"""
Pytest tests for the validation harness.

Tests validation levels 0-3 (py-vgdl) and cross-engine comparison (py-vgdl vs vgdl-jax).
Uses discovery-based game parametrization from PYVGDL_GAMES.
"""
import os
import pytest
import numpy as np

from vgdl_jax.validate.constants import PYVGDL_GAMES, PYVGDL_GAMES_DIR, BLOCK_SIZE
from vgdl_jax.validate.discovery import GameEntry
from vgdl_jax.validate.harness import (
    validate_pyvgdl_loads,
    validate_pyvgdl_state_extraction,
    validate_pyvgdl_trajectory,
    run_pyvgdl_trajectory,
    run_jax_trajectory,
    run_comparison,
    compare_states,
    setup_pyvgdl_game,
    setup_jax_game,
    get_sprite_configs,
    get_effects,
)
from vgdl_jax.validate.state_extractor import extract_pyvgdl_state

# ── Discovered game entries ─────────────────────────────────────────

ENTRIES = list(PYVGDL_GAMES.values())
DETERMINISTIC_ENTRIES = [e for e in ENTRIES if e.name == 'sokoban']
STOCHASTIC_ENTRIES = [e for e in ENTRIES if e.name != 'sokoban']


# ── Level 0: Game loads ──────────────────────────────────────────────


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_pyvgdl_loads(entry):
    """Level 0: py-vgdl loads the game without error."""
    success, error = validate_pyvgdl_loads(entry)
    assert success, f"Failed to load {entry.name}: {error}"


# ── Level 1: State extraction ────────────────────────────────────────


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_pyvgdl_extracts_state(entry):
    """Level 1: initial state is extractable and well-formed."""
    success, result = validate_pyvgdl_state_extraction(entry)
    assert success, f"State extraction failed for {entry.name}: {result}"

    state = result
    # Every type should have non-negative alive count
    for tidx, info in state['types'].items():
        assert info['alive_count'] >= 0, f"{info['key']}: negative alive_count"
        # Positions should be non-negative grid coords
        for r, c in info['positions']:
            assert r >= 0 and c >= 0, (
                f"{info['key']}: negative grid coord ({r}, {c})")

    # Initial score should be 0
    assert state['score'] == 0, f"Initial score should be 0, got {state['score']}"
    # Game should not be ended
    assert state['done'] is False


# ── Level 2+3: Trajectory runs ───────────────────────────────────────


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_pyvgdl_trajectory_runs(entry):
    """Level 2+3: py-vgdl runs a 50-step NOOP trajectory, states extracted."""
    success, result = validate_pyvgdl_trajectory(entry, n_steps=50, seed=42)
    assert success, f"Trajectory failed for {entry.name}: {result}"

    states = result
    # Should have initial + at least 1 step
    assert len(states) >= 2

    # Step counter should increment
    assert states[0]['step'] == 0
    if not states[1]['done']:
        assert states[1]['step'] == 1


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_pyvgdl_trajectory_with_actions(entry):
    """Run a trajectory with non-NOOP actions."""
    game_obj, action_keys, _ = setup_pyvgdl_game(entry)
    n_actions = len(action_keys)

    # Cycle through all actions
    actions = [i % n_actions for i in range(30)]
    states = run_pyvgdl_trajectory(entry, actions, seed=42)

    assert len(states) >= 2, f"Expected at least 2 states for {entry.name}"

    # All states should be well-formed
    for i, s in enumerate(states):
        assert 'types' in s, f"Step {i}: missing 'types'"
        assert isinstance(s['score'], (int, float)), f"Step {i}: bad score type"


# ── Deterministic reproducibility ────────────────────────────────────


@pytest.mark.parametrize("entry", DETERMINISTIC_ENTRIES, ids=lambda e: e.name)
def test_pyvgdl_deterministic_reproducibility(entry):
    """Deterministic games produce identical trajectories with same seed."""
    game_obj, action_keys, _ = setup_pyvgdl_game(entry)
    n_actions = len(action_keys)
    actions = [i % n_actions for i in range(20)]

    states_a = run_pyvgdl_trajectory(entry, actions, seed=42)
    states_b = run_pyvgdl_trajectory(entry, actions, seed=42)

    assert len(states_a) == len(states_b), "Different trajectory lengths"
    for i, (sa, sb) in enumerate(zip(states_a, states_b)):
        assert sa['score'] == sb['score'], f"Step {i}: score differs"
        assert sa['done'] == sb['done'], f"Step {i}: done differs"
        for tidx in sa['types']:
            assert sa['types'][tidx]['alive_count'] == sb['types'][tidx]['alive_count'], (
                f"Step {i}, {sa['types'][tidx]['key']}: alive_count differs")
            assert sa['types'][tidx]['positions'] == sb['types'][tidx]['positions'], (
                f"Step {i}, {sa['types'][tidx]['key']}: positions differ")


# ── ReplayRandomGenerator integration ────────────────────────────────


@pytest.mark.parametrize("entry", STOCHASTIC_ENTRIES, ids=lambda e: e.name)
def test_pyvgdl_with_replay_rng(entry):
    """Verify ReplayRandomGenerator can be injected and game still runs."""
    import jax
    from vgdl_jax.validate.rng_replay import RNGRecorder, ReplayRandomGenerator
    from vgdl_jax.validate.state_extractor import extract_pyvgdl_state
    from vgdl_jax.parser import parse_vgdl
    from vgdl_jax.compiler import compile_game

    # Set up JAX side for RNG recording
    gd = parse_vgdl(entry.game_file, entry.level_files[0])
    compiled = compile_game(gd)
    max_n = compiled.init_state.alive.shape[1]

    recorder = RNGRecorder(
        sprite_configs=get_sprite_configs(compiled),
        effects=get_effects(compiled),
        game_def=gd,
    )
    replay_rng = ReplayRandomGenerator(gd)

    # Set up py-vgdl
    game_obj, action_keys, sprite_key_order = setup_pyvgdl_game(entry)
    game_obj.set_seed(42)
    game_obj.set_random_generator(replay_rng)

    # Run 10 NOOP steps with RNG replay
    noop_idx = len(action_keys) - 1
    rng_key = jax.random.PRNGKey(42)

    for step_i in range(10):
        if game_obj.ended:
            break
        record, rng_key = recorder.record_step(rng_key, max_n=max_n)
        replay_rng.set_step_record(record)
        game_obj.tick(action_keys[noop_idx])

    # If we got here without error, the replay RNG works
    state = extract_pyvgdl_state(game_obj, sprite_key_order, game_obj.block_size)
    assert 'types' in state
    assert state['step'] <= 10


# ── RNGRecorder unit test ────────────────────────────────────────────


def test_rng_recorder_produces_valid_records():
    """Verify RNGRecorder output has correct structure and value ranges."""
    import jax
    from vgdl_jax.validate.rng_replay import RNGRecorder
    from vgdl_jax.parser import parse_vgdl
    from vgdl_jax.compiler import compile_game
    from vgdl_jax.data_model import SpriteClass

    # Use chase — has Chaser and Fleeing NPCs
    chase = PYVGDL_GAMES['chase']
    gd = parse_vgdl(chase.game_file, chase.level_files[0])
    compiled = compile_game(gd)
    max_n = compiled.init_state.alive.shape[1]

    sprite_configs = get_sprite_configs(compiled)
    effects = get_effects(compiled)

    recorder = RNGRecorder(sprite_configs, effects, gd)
    rng_key = jax.random.PRNGKey(0)

    record, next_rng = recorder.record_step(rng_key, max_n=max_n)

    # Should have records for stochastic types
    assert len(record) > 0, "No RNG records produced for chase"

    for tidx, rec in record.items():
        if isinstance(tidx, tuple):
            continue  # teleport effect record
        sc = rec['class']
        if sc in (SpriteClass.RANDOM_NPC, SpriteClass.CHASER, SpriteClass.FLEEING):
            assert 'dir_indices' in rec
            assert rec['dir_indices'].shape == (max_n,)
            assert (rec['dir_indices'] >= 0).all()
            assert (rec['dir_indices'] < 4).all()
        elif sc in (SpriteClass.SPAWN_POINT, SpriteClass.BOMBER):
            assert 'spawn_rolls' in rec
            assert rec['spawn_rolls'].shape == (max_n,)
            assert (rec['spawn_rolls'] >= 0).all()
            assert (rec['spawn_rolls'] < 1).all()

    # next_rng should be different from input
    assert not np.array_equal(np.array(rng_key), np.array(next_rng))


# ══════════════════════════════════════════════════════════════════════
# Cross-engine comparison tests (py-vgdl vs vgdl-jax)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_cross_engine_initial_state(entry):
    """Initial state must match exactly for all games."""
    result = run_comparison(entry, actions=[], seed=42)
    init_step = result.steps[0]
    assert init_step.matches, (
        f"{entry.name}: initial state differs:\n" +
        "\n".join(f"  - {d}" for d in init_step.diffs)
    )


@pytest.mark.parametrize("entry", DETERMINISTIC_ENTRIES, ids=lambda e: e.name)
def test_cross_engine_deterministic(entry):
    """Deterministic games must match exactly for 30 steps."""
    compiled, _ = setup_jax_game(entry)
    noop_idx = compiled.noop_action

    # Cycle through all actions for broader coverage
    actions = [i % compiled.n_actions for i in range(30)]
    result = run_comparison(entry, actions, seed=42)

    failing_steps = [s for s in result.steps if not s.matches]
    assert len(failing_steps) == 0, (
        f"{entry.name}: {len(failing_steps)} steps diverged:\n" +
        "\n".join(
            f"  step {s.step}: {s.diffs}" for s in failing_steps[:5]
        )
    )


@pytest.mark.parametrize("entry", STOCHASTIC_ENTRIES, ids=lambda e: e.name)
def test_cross_engine_with_rng_replay(entry):
    """Stochastic games with RNG replay — strict comparison, no relaxation."""
    compiled, _ = setup_jax_game(entry)

    # Use NOOP to isolate NPC behavior
    noop_idx = compiled.noop_action
    actions = [noop_idx] * 20

    result = run_comparison(entry, actions, seed=42, use_rng_replay=True)

    failing_steps = [s for s in result.steps if not s.matches]
    assert len(failing_steps) == 0, (
        f"{entry.name}: {len(failing_steps)}/{len(result.steps)} steps diverged "
        f"with RNG replay:\n" +
        "\n".join(
            f"  step {s.step}: {s.diffs}" for s in failing_steps[:5]
        )
    )


# ── JAX-only trajectory tests ────────────────────────────────────────


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda e: e.name)
def test_jax_trajectory_runs(entry):
    """vgdl-jax runs a 30-step trajectory without error."""
    compiled, _ = setup_jax_game(entry)
    noop_idx = compiled.noop_action
    actions = [noop_idx] * 30

    states = run_jax_trajectory(entry, actions, seed=42)
    assert len(states) >= 2, f"Expected at least 2 states for {entry.name}"

    # All states should be well-formed
    for i, s in enumerate(states):
        assert 'types' in s, f"Step {i}: missing 'types'"
        assert isinstance(s['score'], (int, float)), f"Step {i}: bad score type"


@pytest.mark.parametrize("entry", DETERMINISTIC_ENTRIES, ids=lambda e: e.name)
def test_jax_deterministic_reproducibility(entry):
    """Deterministic games produce identical JAX trajectories with same seed."""
    compiled, _ = setup_jax_game(entry)
    actions = [i % compiled.n_actions for i in range(20)]

    states_a = run_jax_trajectory(entry, actions, seed=42)
    states_b = run_jax_trajectory(entry, actions, seed=42)

    assert len(states_a) == len(states_b), "Different trajectory lengths"
    for i, (sa, sb) in enumerate(zip(states_a, states_b)):
        matches, diffs = compare_states(sa, sb)
        assert matches, f"Step {i} differs: {diffs}"
