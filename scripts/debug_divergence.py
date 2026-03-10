#!/usr/bin/env python
"""Debug GVGAI validation divergences by tracing VGDLx step-by-step."""
import argparse
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

from vgdl_jax.validate import discover_games
from vgdl_jax.validate.harness import setup_jax_game, run_jax_trajectory


def load_gvgai_trace(game_name, traj_type='random'):
    """Load GVGAI trace from validation results."""
    path = f'validation_results/per_game/{game_name}/trajectory_{traj_type}.json'
    with open(path) as f:
        data = json.load(f)
    return data


def trace_vgdlx(game_name, actions, seed=42, steps_range=None):
    """Run VGDLx step by step, returning per-step raw state."""
    gvgai_dir = '../GVGAI/examples/gridphysics'
    games = discover_games(gvgai_dir)
    entry = [g for g in games if g.name == game_name][0]
    compiled, game_def = setup_jax_game(entry)
    state = compiled.init_state.replace(rng=jax.random.PRNGKey(seed))

    type_names = [s.key for s in game_def.sprites]
    print(f'Game: {game_name}')
    print(f'Grid: {game_def.level.height}x{game_def.level.width}')
    print(f'Types: {type_names}')
    print(f'n_actions={compiled.n_actions}, noop={compiled.noop_action}')
    print(f'Actions: {actions}')
    print()

    if steps_range is None:
        steps_range = range(len(actions))

    for step in range(max(steps_range) + 1):
        if step < len(actions):
            action = actions[step]
            state = compiled.step_fn(state, action)

        if step in steps_range:
            print(f'--- Step {step + 1} (action={actions[step] if step < len(actions) else "N/A"}) ---')
            for ti, tname in enumerate(type_names):
                alive = state.alive[ti]
                n_alive = int(alive.sum())
                if n_alive > 0 and tname not in ('floor', 'ground', 'wall', 'water', 'land'):
                    pos = state.positions[ti]
                    ori = state.orientations[ti]
                    positions = [(float(pos[j, 0]), float(pos[j, 1])) for j in range(alive.shape[0]) if alive[j]]
                    orientations = [(float(ori[j, 0]), float(ori[j, 1])) for j in range(alive.shape[0]) if alive[j]]
                    print(f'  {tname}(t{ti}): n={n_alive} pos={positions} ori={orientations}')
            print(f'  score={float(state.score)}, done={bool(state.done)}')
            print()

    return state, compiled, game_def


def compare_with_gvgai(game_name, traj_type='random', context=3):
    """Show VGDLx vs GVGAI state at divergence points."""
    # Load GVGAI results
    result = load_gvgai_trace(game_name, traj_type)
    steps = result.get('steps', [])

    # Find first divergence
    first_fail = None
    for s in steps:
        if not s.get('matches', True):
            first_fail = s['step']
            break

    if first_fail is None:
        print(f'{game_name}/{traj_type}: all steps match!')
        return

    print(f'{game_name}/{traj_type}: first divergence at step {first_fail}')
    print(f'Diffs: {steps[first_fail].get("diffs", [])}')
    print()

    # Generate actions
    gvgai_dir = '../GVGAI/examples/gridphysics'
    games = discover_games(gvgai_dir)
    entry = [g for g in games if g.name == game_name][0]
    compiled, game_def = setup_jax_game(entry)

    rng = np.random.RandomState(42)
    actions = rng.randint(0, compiled.n_actions, size=30).tolist()

    # Show steps around divergence
    start = max(0, first_fail - context)
    end = min(len(actions), first_fail + context)
    trace_vgdlx(game_name, actions, steps_range=range(start, end))

    # Show GVGAI states at same steps
    print('--- GVGAI states ---')
    for si in range(start, end):
        if si < len(steps):
            s = steps[si]
            print(f'  step {s["step"]} (action={s.get("action", "?")}): match={s.get("matches", "?")}')
            if s.get('diffs'):
                for d in s['diffs']:
                    print(f'    {d}')
            # Show GVGAI positions from state_a
            state_a = s.get('state_a', {})
            if state_a:
                for tidx, tdata in sorted(state_a.get('types', {}).items()):
                    key = tdata.get('key', f't{tidx}')
                    alive = tdata.get('alive_count', 0)
                    if alive > 0 and key not in ('floor', 'ground', 'wall', 'water', 'land'):
                        print(f'    GVGAI {key}(t{tidx}): n={alive} pos={tdata.get("positions", [])}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game', help='Game name')
    parser.add_argument('--traj', default='random', help='Trajectory type')
    parser.add_argument('--context', type=int, default=3, help='Steps of context around divergence')
    args = parser.parse_args()

    compare_with_gvgai(args.game, args.traj, args.context)
