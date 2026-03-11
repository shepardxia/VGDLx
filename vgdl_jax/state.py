import jax
import jax.numpy as jnp
import flax.struct


@flax.struct.dataclass
class GameState:
    positions: jnp.ndarray       # [n_types, max_n, 2] float32
    alive: jnp.ndarray           # [n_types, max_n] bool
    orientations: jnp.ndarray    # [n_types, max_n, 2] float32
    speeds: jnp.ndarray          # [n_types, max_n] float32
    cooldown_timers: jnp.ndarray # [n_types, max_n] int32
    ages: jnp.ndarray            # [n_types, max_n] int32
    spawn_counts: jnp.ndarray    # [n_types, max_n] int32
    direction_ticks: jnp.ndarray # [n_types, max_n] int32 — remaining ticks to keep current direction
    resources: jnp.ndarray       # [n_types, max_n, n_resource_types] int32
    velocities: jnp.ndarray      # [n_types, max_n, 2] float32
    passive_forces: jnp.ndarray  # [n_types, max_n, 2] float32
    is_first_tick: jnp.ndarray  # [n_types, max_n] bool — blocks passiveMovement for 1 tick
    static_grids: jnp.ndarray   # [n_static_types, height, width] bool
    score: jnp.int32
    step_count: jnp.int32
    done: jnp.bool_
    win: jnp.bool_
    rng: jnp.ndarray             # PRNGKey


def create_initial_state(n_types, max_n, height, width,
                         n_resource_types=0, n_static_types=0,
                         rng_key=None):
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
    n_res = max(n_resource_types, 1)  # at least 1 to keep shape valid
    n_static = max(n_static_types, 1)  # at least 1 to keep shape valid
    return GameState(
        positions=jnp.zeros((n_types, max_n, 2), dtype=jnp.float32),
        alive=jnp.zeros((n_types, max_n), dtype=jnp.bool_),
        orientations=jnp.zeros((n_types, max_n, 2), dtype=jnp.float32),
        speeds=jnp.ones((n_types, max_n), dtype=jnp.float32),  # speed=1.0 is the py-vgdl default
        cooldown_timers=jnp.zeros((n_types, max_n), dtype=jnp.int32),
        ages=jnp.zeros((n_types, max_n), dtype=jnp.int32),
        spawn_counts=jnp.zeros((n_types, max_n), dtype=jnp.int32),
        direction_ticks=jnp.zeros((n_types, max_n), dtype=jnp.int32),
        resources=jnp.zeros((n_types, max_n, n_res), dtype=jnp.int32),
        velocities=jnp.zeros((n_types, max_n, 2), dtype=jnp.float32),
        passive_forces=jnp.zeros((n_types, max_n, 2), dtype=jnp.float32),
        is_first_tick=jnp.zeros((n_types, max_n), dtype=jnp.bool_),
        static_grids=jnp.zeros((n_static, height, width), dtype=jnp.bool_),
        score=jnp.int32(0),
        step_count=jnp.int32(0),
        done=jnp.bool_(False),
        win=jnp.bool_(False),
        rng=rng_key,
    )
