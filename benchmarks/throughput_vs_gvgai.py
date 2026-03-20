#!/usr/bin/env python
"""
Benchmark: VGDLx vs GVGAI throughput comparison.

Discovers all GVGAI-supported games automatically (no hardcoded list).
Measures steps/sec for both engines and writes results to a JSON file
that accumulates across runs.

Usage:
    .venv/bin/python benchmarks/throughput_vs_gvgai.py
    .venv/bin/python benchmarks/throughput_vs_gvgai.py --game chase
    .venv/bin/python benchmarks/throughput_vs_gvgai.py --n-steps 500 --n-envs 256
    .venv/bin/python benchmarks/throughput_vs_gvgai.py --skip-gvgai   # JAX only
"""
import argparse
import json
import os
import time

import jax
import numpy as np

from vgdl_jax.env import VGDLxEnv
from vgdl_jax.validate.constants import GVGAI_GAMES
from vgdl_jax.validate.harness import setup_jax_game
from vgdl_jax.validate.backend_gvgai import run_gvgai_trajectory

RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'results', 'throughput_vs_gvgai.json')


# ── GVGAI benchmark ─────────────────────────────────────────────────────────


def bench_gvgai(entry, action_names, n_actions, n_steps, seed=42):
    """Measure GVGAI steps/sec via subprocess.

    Returns steps/sec (excluding JVM startup on first call — we time the
    trajectory execution only).
    """
    rng = np.random.RandomState(seed)
    actions = rng.randint(0, n_actions, size=n_steps).tolist()

    t0 = time.perf_counter()
    run_gvgai_trajectory(entry, actions, seed=seed, action_names=action_names)
    elapsed = time.perf_counter() - t0

    return n_steps / elapsed, elapsed


# ── VGDLx benchmark ─────────────────────────────────────────────────────────


def bench_jax_single(compiled, n_steps, seed=42):
    """Measure VGDLx single-env steps/sec (no vmap)."""
    n_actions = compiled.n_actions
    state = compiled.init_state.replace(rng=jax.random.PRNGKey(seed))
    step_fn = jax.jit(compiled.step_fn)

    # Warmup (triggers JIT compilation)
    rng = jax.random.PRNGKey(seed + 1)
    action = jax.random.randint(rng, (), 0, n_actions)
    state = step_fn(state, action)
    jax.block_until_ready(state.done)

    # Benchmark
    rng = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        rng, key = jax.random.split(rng)
        action = jax.random.randint(key, (), 0, n_actions)
        state = step_fn(state, action)
    jax.block_until_ready(state.done)
    elapsed = time.perf_counter() - t0

    return n_steps / elapsed, elapsed


def bench_jax_batched(entry, n_envs, n_steps, seed=42):
    """Measure VGDLx batched steps/sec (vmap)."""
    env = VGDLxEnv(entry.game_file, entry.level_files[0])
    n_actions = env.n_actions

    rngs = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    _, state_batch = jax.vmap(env.reset)(rngs)

    step_vmap = jax.jit(jax.vmap(env.step))

    # Warmup
    actions = jax.random.randint(jax.random.PRNGKey(seed + 1), (n_envs,), 0, n_actions)
    _, state_batch, _, _, _ = step_vmap(state_batch, actions)
    jax.block_until_ready(state_batch.done)

    # Benchmark
    rng = jax.random.PRNGKey(seed)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        rng, key = jax.random.split(rng)
        actions = jax.random.randint(key, (n_envs,), 0, n_actions)
        _, state_batch, _, _, _ = step_vmap(state_batch, actions)
    jax.block_until_ready(state_batch.done)
    elapsed = time.perf_counter() - t0

    total_steps = n_envs * n_steps
    return total_steps / elapsed, elapsed


# ── Discovery ────────────────────────────────────────────────────────────────


def discover_supported_games(game_filter=None):
    """Return dict of game_name -> GameEntry for GVGAI-supported games."""
    supported = {}
    for name, entry in sorted(GVGAI_GAMES.items()):
        if game_filter and name != game_filter:
            continue
        if not entry.level_files:
            continue
        try:
            setup_jax_game(entry)
            supported[name] = entry
        except Exception:
            continue
    return supported


# ── Main ─────────────────────────────────────────────────────────────────────


def load_results():
    """Load existing results file, or return empty dict."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    """Write results to JSON file."""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='VGDLx vs GVGAI throughput benchmark')
    parser.add_argument('--game', type=str, default=None, help='Benchmark a single game')
    parser.add_argument('--n-steps', type=int, default=200, help='Steps per trajectory (default: 200)')
    parser.add_argument('--n-envs', type=int, default=256, help='Batch size for VGDLx batched (default: 256)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--skip-gvgai', action='store_true', help='Skip GVGAI benchmark (JAX only)')
    args = parser.parse_args()

    print(f"Discovering GVGAI-supported games...")
    games = discover_supported_games(args.game)
    if not games:
        print("No supported games found.")
        return

    print(f"Found {len(games)} game(s): {', '.join(sorted(games))}\n")

    results = load_results()

    for name, entry in sorted(games.items()):
        print(f"{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

        game_results = results.get(name, {})

        # ── GVGAI ──
        if not args.skip_gvgai:
            try:
                compiled, _ = setup_jax_game(entry)
                action_names = compiled.action_names
                n_actions = compiled.n_actions

                # Warmup run (JVM startup, class loading)
                print(f"  GVGAI warmup...", end='', flush=True)
                bench_gvgai(entry, action_names, n_actions, n_steps=10, seed=args.seed)
                print(" done")

                # Timed run
                sps, elapsed = bench_gvgai(entry, action_names, n_actions,
                                           n_steps=args.n_steps, seed=args.seed)
                print(f"  GVGAI:         {sps:>10,.0f} steps/s  ({elapsed:.2f}s)")
                game_results['gvgai_sps'] = round(sps, 1)
                game_results['gvgai_n_steps'] = args.n_steps
            except Exception as e:
                print(f"  GVGAI:         ERROR — {e}")
                game_results['gvgai_error'] = str(e)

        # ── VGDLx single-env ──
        try:
            compiled, _ = setup_jax_game(entry)
            sps, elapsed = bench_jax_single(compiled, n_steps=args.n_steps, seed=args.seed)
            print(f"  VGDLx (1 env): {sps:>10,.0f} steps/s  ({elapsed:.2f}s)")
            game_results['jax_single_sps'] = round(sps, 1)
        except Exception as e:
            print(f"  VGDLx (1 env): ERROR — {e}")
            game_results['jax_single_error'] = str(e)

        # ── VGDLx batched ──
        try:
            sps, elapsed = bench_jax_batched(entry, n_envs=args.n_envs,
                                              n_steps=args.n_steps, seed=args.seed)
            print(f"  VGDLx ({args.n_envs:>3d}):  {sps:>10,.0f} steps/s  ({elapsed:.2f}s)")
            game_results['jax_batched_sps'] = round(sps, 1)
            game_results['jax_batched_n_envs'] = args.n_envs
        except Exception as e:
            print(f"  VGDLx ({args.n_envs:>3d}):  ERROR — {e}")
            game_results['jax_batched_error'] = str(e)

        # ── Speedup ratios ──
        if 'gvgai_sps' in game_results and 'jax_single_sps' in game_results:
            ratio = game_results['jax_single_sps'] / game_results['gvgai_sps']
            print(f"  Speedup (1 env vs GVGAI):   {ratio:>6.1f}x")
            game_results['speedup_single'] = round(ratio, 1)
        if 'gvgai_sps' in game_results and 'jax_batched_sps' in game_results:
            ratio = game_results['jax_batched_sps'] / game_results['gvgai_sps']
            print(f"  Speedup ({args.n_envs} env vs GVGAI): {ratio:>6.1f}x")
            game_results['speedup_batched'] = round(ratio, 1)

        game_results['n_steps'] = args.n_steps
        results[name] = game_results
        print()

    # ── Save ──
    save_results(results)
    print(f"Results saved to {RESULTS_FILE}")

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print(f"  {'Game':<20s} {'GVGAI':>10s} {'JAX (1)':>10s} {'JAX (batch)':>12s} {'1x':>6s} {'Nx':>6s}")
    print(f"  {'':<20s} {'sps':>10s} {'sps':>10s} {'sps':>12s} {'':>6s} {'':>6s}")
    print(f"  {'-' * 64}")
    for name in sorted(results):
        r = results[name]
        gvgai = f"{r['gvgai_sps']:,.0f}" if 'gvgai_sps' in r else '—'
        jax1 = f"{r['jax_single_sps']:,.0f}" if 'jax_single_sps' in r else '—'
        jaxn = f"{r['jax_batched_sps']:,.0f}" if 'jax_batched_sps' in r else '—'
        s1 = f"{r['speedup_single']:.1f}x" if 'speedup_single' in r else '—'
        sn = f"{r['speedup_batched']:.1f}x" if 'speedup_batched' in r else '—'
        print(f"  {name:<20s} {gvgai:>10s} {jax1:>10s} {jaxn:>12s} {s1:>6s} {sn:>6s}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
