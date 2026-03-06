"""
Profile all games: effect counts, compilation time, per-step time, stability.
"""
import time
import os
import jax
from vgdl_jax.env import VGDLJaxEnv
from vgdl_jax.parser import parse_vgdl
from vgdl_jax.data_model import STATIC_CLASSES, AVATAR_CLASSES
from vgdl_jax.effects import PARTNER_IDX_EFFECTS

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')

GAMES = [
    ("Chase", "chase.txt", "chase_lvl0.txt"),
    ("Zelda", "zelda.txt", "zelda_lvl0.txt"),
    ("Aliens", "aliens.txt", "aliens_lvl0.txt"),
    ("MissileCommand", "missilecommand.txt", "missilecommand_lvl0.txt"),
    ("Sokoban", "sokoban.txt", "sokoban_lvl0.txt"),
    ("Portals", "portals.txt", "portals_lvl0.txt"),
    ("BoulderDash", "boulderdash.txt", "boulderdash_lvl0.txt"),
    ("SurviveZombies", "survivezombies.txt", "survivezombies_lvl0.txt"),
    ("Frogs", "frogs.txt", "frogs_lvl0.txt"),
]


def analyze_game(name, game_file, level_file):
    """Print structural analysis for a game."""
    game_def = parse_vgdl(game_file, level_file)
    n_types = len(game_def.sprites)

    # Count effect rules (before Cartesian expansion)
    n_rules = len(game_def.effects)
    effect_types = {}
    for ed in game_def.effects:
        effect_types[ed.effect_type] = effect_types.get(ed.effect_type, 0) + 1

    # Count moving NPCs
    n_static = sum(1 for sd in game_def.sprites if sd.sprite_class in STATIC_CLASSES)
    n_npc_moving = sum(1 for sd in game_def.sprites
                       if sd.sprite_class not in STATIC_CLASSES
                       and sd.sprite_class not in AVATAR_CLASSES
                       and sd.speed > 0)
    n_frac = sum(1 for sd in game_def.sprites
                 if sd.speed > 0 and sd.speed != 1.0)

    print(f"\n  {name}: {n_types} types ({n_static} static, {n_frac} fractional-speed), "
          f"{n_rules} rules, {n_npc_moving} moving NPCs")
    print(f"    effects: {dict(sorted(effect_types.items()))}")


def benchmark_game(name, game_file, level_file, n_envs=256, n_steps=200, n_trials=5):
    """Run benchmark with compilation timing and stability."""
    env = VGDLJaxEnv(game_file, level_file)
    max_n = env.compiled.init_state.alive.shape[1]
    n_types = len(env.compiled.game_def.sprites)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, n_envs)
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    jax.block_until_ready(obs_batch)

    step_vmap = jax.jit(jax.vmap(env.step))
    actions = jax.random.randint(jax.random.PRNGKey(1), (n_envs,), 0, env.n_actions)

    # Measure compilation
    t0 = time.time()
    obs_batch, state_batch, _, _, _ = step_vmap(state_batch, actions)
    jax.block_until_ready(obs_batch)
    compile_time = time.time() - t0

    # Run multiple trials
    results = []
    for trial in range(n_trials):
        obs_batch, state_batch = jax.vmap(env.reset)(rngs)
        jax.block_until_ready(obs_batch)

        rng_t = jax.random.PRNGKey(trial + 10)
        t0 = time.time()
        for i in range(n_steps):
            rng_t, key = jax.random.split(rng_t)
            actions = jax.random.randint(key, (n_envs,), 0, env.n_actions)
            obs_batch, state_batch, _, _, _ = step_vmap(state_batch, actions)
        jax.block_until_ready(obs_batch)
        elapsed = time.time() - t0
        results.append((n_envs * n_steps) / elapsed)

    avg = sum(results) / len(results)
    mn, mx = min(results), max(results)
    us_per_step = 1e6 / avg * n_envs  # microseconds per batched step
    print(f"  {name:18s}  types={n_types:2d}  max_n={max_n:2d}  "
          f"compile={compile_time:.2f}s  "
          f"avg={avg:>7,.0f} sps  [{mn:>7,.0f} - {mx:>7,.0f}]  "
          f"{us_per_step:>6,.0f} us/batch_step")
    return avg


if __name__ == '__main__':
    print("=" * 60)
    print("  GAME STRUCTURE ANALYSIS")
    print("=" * 60)
    for name, gf, lf in GAMES:
        analyze_game(name,
                    os.path.join(GAMES_DIR, gf),
                    os.path.join(GAMES_DIR, lf))

    print("\n" + "=" * 100)
    print("  BENCHMARK (256 envs, 200 steps, 5 trials)")
    print("=" * 100)
    for name, gf, lf in GAMES:
        try:
            benchmark_game(name,
                          os.path.join(GAMES_DIR, gf),
                          os.path.join(GAMES_DIR, lf))
        except Exception as e:
            print(f"  {name}: ERROR: {e}")

    print()
