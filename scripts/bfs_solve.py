#!/usr/bin/env python
"""
BFS solver for VGDL-JAX games.

Performs breadth-first search over the vgdl-jax step_fn to find
winning action sequences. After finding a solution, replays it to
verify correctness. Outputs JSON results and a LaTeX summary table.

Usage:
    python scripts/bfs_solve.py                     # all games
    python scripts/bfs_solve.py --game sokoban       # single game
    python scripts/bfs_solve.py --max-iter 500000    # increase budget
"""
import os
import json
import time
import argparse
from collections import Counter, deque

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')

from vgdl_jax.validate.constants import PYVGDL_GAMES, PYVGDL_GAMES_DIR

ALL_GAMES = sorted(PYVGDL_GAMES.keys())
STOCHASTIC_GAMES = [g for g in ALL_GAMES if g != 'sokoban']
GAMES_DIR = PYVGDL_GAMES_DIR

RESULTS_DIR = os.path.join(PROJECT_DIR, 'validation_results')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')


# ── State hashing ─────────────────────────────────────────────────────

def hash_state(state):
    """Hash observable state (positions + alive), ignoring rng."""
    pos = np.asarray(state.positions).flatten()
    alive = np.asarray(state.alive).flatten()
    return hash(pos.round().astype(np.int8).tobytes()
                + alive.astype(np.int8).tobytes())


# ── Game setup ────────────────────────────────────────────────────────

def load_game(game_name):
    """Parse and compile a VGDL game, returning the CompiledGame."""
    from vgdl_jax.parser import parse_vgdl
    from vgdl_jax.compiler import compile_game

    game_file = os.path.join(GAMES_DIR, f'{game_name}.txt')
    level_file = os.path.join(GAMES_DIR, f'{game_name}_lvl0.txt')

    game_def = parse_vgdl(game_file, level_file)

    # Compute max sprites needed across all types
    counts = Counter(t for t, r, c in game_def.level.initial_sprites)
    max_n = max(counts.values(), default=1) + 10
    compiled = compile_game(game_def, max_sprites_per_type=max_n)
    return compiled


# ── BFS solver ────────────────────────────────────────────────────────

def bfs_solve(compiled, max_iterations=100000, verbose=True):
    """
    BFS over step_fn to find a winning action sequence.

    Args:
        compiled: CompiledGame with init_state, step_fn, n_actions
        max_iterations: maximum number of states to expand
        verbose: print progress updates

    Returns:
        dict with keys: solved, actions (if solved), iterations,
        states_visited, solution_length (if solved), time_s
    """
    import jax

    init_state = compiled.init_state.replace(rng=jax.random.PRNGKey(42))
    step_fn = compiled.step_fn
    n_actions = compiled.n_actions

    visited = set()
    init_h = hash_state(init_state)
    visited.add(init_h)

    # Queue entries: (state, action_history)
    queue = deque()
    queue.append((init_state, []))

    iterations = 0
    t0 = time.time()

    while queue and iterations < max_iterations:
        state, actions = queue.popleft()
        iterations += 1

        if verbose and iterations % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  iter {iterations:,}  |  visited {len(visited):,}  "
                  f"|  queue {len(queue):,}  |  {elapsed:.1f}s")

        for a in range(n_actions):
            new_state = step_fn(state, a)
            new_actions = actions + [a]

            # Check win condition
            if bool(new_state.win):
                elapsed = time.time() - t0
                return {
                    'solved': True,
                    'actions': new_actions,
                    'iterations': iterations,
                    'states_visited': len(visited),
                    'solution_length': len(new_actions),
                    'time_s': round(elapsed, 2),
                }

            # Skip terminal (lost) states
            if bool(new_state.done):
                continue

            # Dedup by observable state hash
            h = hash_state(new_state)
            if h not in visited:
                visited.add(h)
                queue.append((new_state, new_actions))

    elapsed = time.time() - t0
    return {
        'solved': False,
        'iterations': iterations,
        'states_visited': len(visited),
        'time_s': round(elapsed, 2),
    }


# ── Solution replay & verification ───────────────────────────────────

def verify_solution(compiled, actions):
    """Replay an action sequence and verify it ends in a win."""
    import jax

    state = compiled.init_state.replace(rng=jax.random.PRNGKey(42))
    for a in actions:
        if bool(state.done):
            break
        state = compiled.step_fn(state, a)

    return bool(state.win)


# ── Output generation ─────────────────────────────────────────────────

def write_json_results(results, path):
    """Write results dict to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results written to {path}")


def write_latex_table(results, path):
    """Generate a LaTeX table summarizing BFS solvability."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lines = []
    lines.append(r'\begin{table}[ht]')
    lines.append(r'\centering')
    lines.append(r'\caption{BFS solvability of VGDL-JAX games (level 0)}')
    lines.append(r'\label{tab:bfs-solvability}')
    lines.append(r'\begin{tabular}{lcccc}')
    lines.append(r'\toprule')
    lines.append(r'Game & Solved & \# Actions & Iterations & States Visited \\')
    lines.append(r'\midrule')

    for game_name in ALL_GAMES:
        r = results.get(game_name, {})
        solved = r.get('solved', False)
        solved_str = r'\cmark' if solved else r'\xmark'
        n_act = str(r.get('solution_length', '--'))
        iters = f"{r.get('iterations', 0):,}"
        visited = f"{r.get('states_visited', 0):,}"

        if not solved:
            n_act = '--'
            note = r.get('note', '')
            if note:
                solved_str += r'\textsuperscript{*}'

        lines.append(f'{game_name} & {solved_str} & {n_act} & {iters} & {visited} \\\\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')

    # Add footnote for stochastic games
    has_stochastic_note = any(
        results.get(g, {}).get('note') for g in ALL_GAMES
    )
    if has_stochastic_note:
        lines.append(r'\vspace{2pt}')
        lines.append(r'{\small \textsuperscript{*}Stochastic game -- '
                     r'BFS explores with fixed RNG seed.}')

    lines.append(r'\end{table}')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"LaTeX table written to {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='BFS solver for VGDL-JAX games')
    parser.add_argument('--game', type=str, default=None,
                        help='Single game to solve (default: all games)')
    parser.add_argument('--max-iter', type=int, default=100000,
                        help='Max BFS iterations per game (default: 100000)')
    parser.add_argument('--output', type=str,
                        default=os.path.join(RESULTS_DIR, 'bfs_results.json'),
                        help='Output JSON path')
    parser.add_argument('--latex', type=str,
                        default=os.path.join(TABLES_DIR, 'bfs_solvability.tex'),
                        help='Output LaTeX table path')
    args = parser.parse_args()

    games = [args.game] if args.game else ALL_GAMES

    results = {}

    for game_name in games:
        is_stochastic = game_name in STOCHASTIC_GAMES
        tag = ' (stochastic)' if is_stochastic else ' (deterministic)'
        print(f"\n{'='*60}")
        print(f"BFS: {game_name}{tag}  |  max_iter={args.max_iter:,}")
        print(f"{'='*60}")

        # Load and compile game
        print(f"  Loading {game_name}...")
        compiled = load_game(game_name)
        print(f"  n_actions={compiled.n_actions}, "
              f"n_types={len(compiled.game_def.sprites)}")

        # Run BFS
        result = bfs_solve(compiled, max_iterations=args.max_iter)

        # Add stochastic note
        if is_stochastic and not result['solved']:
            result['note'] = 'stochastic - BFS not applicable'

        # Verify solution if found
        if result['solved']:
            verified = verify_solution(compiled, result['actions'])
            result['verified'] = verified
            status = 'VERIFIED' if verified else 'FAILED VERIFICATION'
            print(f"\n  SOLVED in {result['solution_length']} actions "
                  f"({result['iterations']:,} iterations, "
                  f"{result['states_visited']:,} states)  [{status}]")
        else:
            note = result.get('note', '')
            print(f"\n  NOT SOLVED after {result['iterations']:,} iterations "
                  f"({result['states_visited']:,} states)")
            if note:
                print(f"  Note: {note}")

        print(f"  Time: {result['time_s']}s")
        results[game_name] = result

    # Write outputs
    write_json_results(results, args.output)
    write_latex_table(results, args.latex)

    # Summary
    solved_count = sum(1 for r in results.values() if r.get('solved'))
    print(f"\n{'='*60}")
    print(f"Summary: {solved_count}/{len(results)} games solved")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
