"""
Benchmark: measure throughput of batched VGDL-JAX environments.
Tests vmap over N parallel environments, measures steps/second.
"""
import time
import os
import jax
from vgdl_jax.env import VGDLxEnv

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')


def benchmark_game(game_name, game_file, level_file, n_envs=256, n_steps=200):
    print(f"\n{'='*60}")
    print(f"  {game_name}  |  {n_envs} envs  |  {n_steps} steps")
    print(f"{'='*60}")

    env = VGDLxEnv(game_file, level_file)
    print(f"  n_types={len(env.compiled.game_def.sprites)}, "
          f"max_n={env.compiled.init_state.alive.shape[1]}, "
          f"n_actions={env.n_actions}")

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, n_envs)

    # Reset
    t0 = time.time()
    obs_batch, state_batch = jax.vmap(env.reset)(rngs)
    jax.block_until_ready(obs_batch)
    t_reset = time.time() - t0
    print(f"  Reset (compile + exec): {t_reset:.2f}s")

    step_vmap = jax.jit(jax.vmap(env.step))

    # Warmup step (triggers compilation)
    actions = jax.random.randint(jax.random.PRNGKey(1), (n_envs,), 0, env.n_actions)
    obs_batch, state_batch, _, _, _ = step_vmap(state_batch, actions)
    jax.block_until_ready(obs_batch)

    # Benchmark
    t0 = time.time()
    for i in range(n_steps):
        rng, key = jax.random.split(rng)
        actions = jax.random.randint(key, (n_envs,), 0, env.n_actions)
        obs_batch, state_batch, rewards, dones, infos = step_vmap(state_batch, actions)
    jax.block_until_ready(obs_batch)
    elapsed = time.time() - t0

    total_steps = n_envs * n_steps
    sps = total_steps / elapsed
    print(f"  {total_steps:,} total steps in {elapsed:.2f}s")
    print(f"  Throughput: {sps:,.0f} steps/sec")
    return sps


if __name__ == '__main__':
    games = [
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

    for name, gf, lf in games:
        try:
            benchmark_game(
                name,
                os.path.join(GAMES_DIR, gf),
                os.path.join(GAMES_DIR, lf),
            )
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print("  Done.")
