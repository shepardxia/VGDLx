import jax
from conftest import make_env


def test_aliens_flak_shoots_missile():
    """FlakAvatar's shoot action should create a missile."""
    env = make_env('aliens')
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    sam_idx = gd.type_idx('sam')

    # Initially no sam missile
    assert state.alive[sam_idx].sum() == 0

    # FlakAvatar GVGAI ordering: USE=0, LEFT=1, RIGHT=2, NIL=3
    shoot_action = 0
    obs, state, _, _, _ = env.step(state, shoot_action)

    # Sam missile should be spawned
    assert state.alive[sam_idx].sum() == 1


def test_aliens_missile_moves_up():
    """Sam missile should move upward (orientation=UP)."""
    env = make_env('aliens')
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    sam_idx = gd.type_idx('sam')

    # Shoot to create missile
    shoot_action = 0  # FlakAvatar GVGAI ordering: USE=0, LEFT=1, RIGHT=2, NIL=3
    obs, state, _, _, _ = env.step(state, shoot_action)
    assert state.alive[sam_idx].sum() == 1
    sam_pos_0 = state.positions[sam_idx, 0].copy()

    # NOOP once — tick-spawned missile has is_first_tick=False, moves after 1 cooldown tick
    obs, state, _, _, _ = env.step(state, env.noop_action)

    # Missile should move up (row decreases) or be dead (hit EOS or something)
    if state.alive[sam_idx, 0]:
        sam_pos_1 = state.positions[sam_idx, 0]
        assert sam_pos_1[0] < sam_pos_0[0]  # moved up


def test_aliens_flak_moves_horizontal():
    """FlakAvatar actions 0,1 should move LEFT,RIGHT (not UP,DOWN)."""
    env = make_env('aliens')
    rng = jax.random.PRNGKey(0)
    obs, state = env.reset(rng)

    gd = env.compiled.game_def
    avatar_idx = gd.type_idx('avatar')
    initial_pos = state.positions[avatar_idx, 0].copy()

    # Action 1 = LEFT for FlakAvatar (GVGAI: USE=0, LEFT=1, RIGHT=2, NIL=3)
    obs, state, _, _, _ = env.step(state, 1)
    pos_after_left = state.positions[avatar_idx, 0]
    # Row should stay same, column should decrease (or stay if at wall)
    assert pos_after_left[0] == initial_pos[0], "LEFT should not change row"

    # Action 2 = RIGHT for FlakAvatar
    obs, state, _, _, _ = env.step(state, 2)
    pos_after_right = state.positions[avatar_idx, 0]
    assert pos_after_right[0] == initial_pos[0], "RIGHT should not change row"
