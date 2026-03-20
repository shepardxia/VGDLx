#!/usr/bin/env python
"""
Plot VGDLx vs GVGAI throughput comparison.

Reads results from benchmarks/results/throughput_vs_gvgai.json and produces:
  1. Grouped bar chart: GVGAI vs JAX(1) vs JAX(256) for all games
  2. Speedup chart: single-env and batched speedup factors

Usage:
    .venv/bin/python benchmarks/plot_vs_gvgai.py
    .venv/bin/python benchmarks/plot_vs_gvgai.py --log-scale
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

BENCH_DIR = os.path.dirname(__file__)
RESULTS_FILE = os.path.join(BENCH_DIR, 'results', 'throughput_vs_gvgai.json')
PLOTS_DIR = os.path.join(BENCH_DIR, 'results', 'plots')


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def plot_throughput(results, log_scale=False):
    """Grouped bar chart: 3 bars per game (GVGAI, JAX×1, JAX×256)."""
    games = sorted(results.keys())
    n = len(games)

    gvgai = [results[g]['gvgai_sps'] for g in games]
    jax1 = [results[g]['jax_single_sps'] for g in games]
    jaxn = [results[g]['jax_batched_sps'] for g in games]
    n_envs = results[games[0]].get('jax_batched_n_envs', 256)

    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(14, n * 0.8), 6))

    bars_gvgai = ax.bar(x - w, gvgai, w, label='GVGAI (Java)', color='#e74c3c', zorder=3)
    bars_jax1 = ax.bar(x, jax1, w, label='VGDLx (1 env)', color='#3498db', zorder=3)
    bars_jaxn = ax.bar(x + w, jaxn, w, label=f'VGDLx ({n_envs} envs)', color='#2ecc71', zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(games, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Steps / second')
    ax.set_title('VGDLx vs GVGAI: Throughput Comparison (20 Matching Games)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)

    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    scale = 'log' if log_scale else 'linear'
    path = os.path.join(PLOTS_DIR, f'throughput_vs_gvgai_{scale}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close(fig)
    return path


def plot_speedup(results):
    """Horizontal bar chart: speedup factors (single + batched)."""
    # Sort by batched speedup
    games = sorted(results.keys(), key=lambda g: results[g].get('speedup_batched', 0))
    n = len(games)

    s1 = [results[g].get('speedup_single', 0) for g in games]
    sn = [results[g].get('speedup_batched', 0) for g in games]
    n_envs = results[games[0]].get('jax_batched_n_envs', 256)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(6, n * 0.35)))

    y = np.arange(n)

    # Single-env speedup
    bars1 = ax1.barh(y, s1, color='#3498db', zorder=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels(games, fontsize=9)
    ax1.set_xlabel('Speedup vs GVGAI')
    ax1.set_title('Single-Env Speedup')
    ax1.axvline(x=1, color='#e74c3c', linestyle='--', linewidth=1, zorder=2)
    ax1.grid(axis='x', alpha=0.3, zorder=0)
    for i, v in enumerate(s1):
        ax1.text(v + 0.3, i, f'{v:.0f}x', va='center', fontsize=8)

    # Batched speedup
    bars2 = ax2.barh(y, sn, color='#2ecc71', zorder=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels(games, fontsize=9)
    ax2.set_xlabel('Speedup vs GVGAI')
    ax2.set_title(f'Batched ({n_envs} envs) Speedup')
    ax2.axvline(x=1, color='#e74c3c', linestyle='--', linewidth=1, zorder=2)
    ax2.grid(axis='x', alpha=0.3, zorder=0)
    for i, v in enumerate(sn):
        ax2.text(v + 5, i, f'{v:.0f}x', va='center', fontsize=8)

    fig.suptitle('VGDLx Speedup over GVGAI (Java)', fontsize=13, fontweight='bold')
    fig.tight_layout()

    path = os.path.join(PLOTS_DIR, 'speedup_vs_gvgai.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f'Saved {path}')
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description='Plot VGDLx vs GVGAI throughput')
    parser.add_argument('--log-scale', action='store_true', help='Log scale for throughput chart')
    args = parser.parse_args()

    if not os.path.exists(RESULTS_FILE):
        print(f'No results file at {RESULTS_FILE}. Run throughput_vs_gvgai.py first.')
        return

    results = load_results()
    print(f'Loaded {len(results)} games from {RESULTS_FILE}\n')

    plot_throughput(results, log_scale=True)
    plot_throughput(results, log_scale=False)
    plot_speedup(results)


if __name__ == '__main__':
    main()
