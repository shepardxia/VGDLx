"""
Profile vgdl-jax environment speed with random actions across batch sizes.
Mirrors PuzzleJAX's profile_rand_jax.py methodology:
  - Sweep batch sizes from 1 to 8000
  - 3 warmup steps (JIT compilation)
  - 3 measurement loops with jax.lax.scan
  - Save JSON results per game
"""
import argparse
import json
import os
import traceback
from timeit import default_timer as timer

import jax

from vgdl_jax.env import VGDLxEnv

GAMES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'py-vgdl', 'vgdl', 'games')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'jax')

BATCH_SIZES = [1, 10, 50, 100, 400, 1_500, 2_000, 5_000, 8_000]

GAMES = [
    ("Chase",           "chase.txt",           "chase_lvl0.txt"),
    ("Zelda",           "zelda.txt",           "zelda_lvl0.txt"),
    ("Aliens",          "aliens.txt",          "aliens_lvl0.txt"),
    ("MissileCommand",  "missilecommand.txt",  "missilecommand_lvl0.txt"),
    ("Sokoban",         "sokoban.txt",         "sokoban_lvl0.txt"),
    ("Portals",         "portals.txt",         "portals_lvl0.txt"),
    ("BoulderDash",     "boulderdash.txt",     "boulderdash_lvl0.txt"),
    ("SurviveZombies",  "survivezombies.txt",  "survivezombies_lvl0.txt"),
    ("Frogs",           "frogs.txt",           "frogs_lvl0.txt"),
]

N_STEPS = 200       # steps per measurement loop
N_MEASURE_LOOPS = 3  # number of timed loops (take last as stable)


def profile_game(game_name, game_file, level_file, overwrite=False):
    """Profile a single game across all batch sizes."""
    device = jax.devices()[0].device_kind.replace(' ', '_')
    results_path = os.path.join(RESULTS_DIR, device, f'{game_name}.json')

    if os.path.exists(results_path) and not overwrite:
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    env = VGDLxEnv(game_file, level_file)
    print(f"\n{'='*60}")
    print(f"  {game_name}  |  n_types={len(env.compiled.game_def.sprites)}  "
          f"|  max_n={env.compiled.init_state.alive.shape[1]}  "
          f"|  n_actions={env.n_actions}")
    print(f"{'='*60}")

    for n_envs in BATCH_SIZES:
        if str(n_envs) in results and not overwrite:
            print(f"  batch={n_envs}: skipping (already profiled)")
            continue

        print(f"\n  batch={n_envs}:", end=" ")
        rng = jax.random.PRNGKey(42)

        # Build vmapped step inside scan
        def _env_step(carry, unused):
            state, rng = carry
            rng, k1, k2 = jax.random.split(rng, 3)
            actions = jax.random.randint(k1, (n_envs,), 0, env.n_actions)
            _, state, _, _, _ = jax.jit(jax.vmap(env.step))(state, actions)
            return (state, rng), None

        _env_step_jitted = jax.jit(_env_step)

        try:
            # Reset
            rngs = jax.random.split(rng, n_envs)
            _, state = jax.vmap(env.reset)(rngs)

            carry = (state, rng)

            # 3 warmup single-steps (JIT compilation)
            for w in range(3):
                t0 = timer()
                carry, _ = _env_step_jitted(carry, None)
                jax.block_until_ready(carry[0].alive)
                print(f"warmup{w}={timer()-t0:.2f}s", end=" ")

            # 3 measurement loops using lax.scan
            n_total = N_STEPS * n_envs
            times = []
            for i in range(N_MEASURE_LOOPS):
                t0 = timer()
                carry, _ = jax.lax.scan(_env_step_jitted, carry, None, N_STEPS)
                jax.block_until_ready(carry[0].alive)
                elapsed = timer() - t0
                times.append(elapsed)
                fps = n_total / elapsed
                print(f"loop{i}={fps:,.0f}fps", end=" ")

            fpss = [n_total / t for t in times]
            results[str(n_envs)] = fpss
            print()

        except Exception as e:
            print(f"\n    ERROR: {e}")
            results[str(n_envs)] = {"error": str(e),
                                     "traceback": traceback.format_exc()}

        # Save after each batch size (incremental)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def main():
    global N_STEPS

    parser = argparse.ArgumentParser(description="Profile vgdl-jax FPS across batch sizes")
    parser.add_argument('--game', type=str, default=None,
                        help="Profile a single game (e.g. 'Chase')")
    parser.add_argument('--overwrite', action='store_true',
                        help="Re-run even if results exist")
    parser.add_argument('--n-steps', type=int, default=N_STEPS,
                        help=f"Steps per measurement loop (default: {N_STEPS})")
    args = parser.parse_args()

    N_STEPS = args.n_steps

    device = jax.devices()[0]
    print(f"JAX device: {device.device_kind}")
    print(f"Batch sizes: {BATCH_SIZES}")
    print(f"Steps per loop: {N_STEPS}")

    games = GAMES
    if args.game:
        games = [(n, g, l) for n, g, l in GAMES if n == args.game]
        if not games:
            print(f"Unknown game '{args.game}'. Available: {[n for n,_,_ in GAMES]}")
            return

    all_results = {}
    for name, gf, lf in games:
        try:
            r = profile_game(
                name,
                os.path.join(GAMES_DIR, gf),
                os.path.join(GAMES_DIR, lf),
                overwrite=args.overwrite,
            )
            all_results[name] = r
        except Exception as e:
            print(f"  FATAL ERROR on {name}: {e}")

    print(f"\n{'='*60}")
    print(f"  Results saved to {os.path.join(RESULTS_DIR, jax.devices()[0].device_kind.replace(' ', '_'))}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
