#!/usr/bin/env python
"""
Debug GVGAI validation divergences from saved trajectory JSON.

Pure JSON consumer — no engine imports, no re-running GVGAI or VGDLx.
Reads trajectory files written by validate_all.py and shows side-by-side
state comparisons at divergent steps.

Usage:
    python scripts/debug_divergence.py aliens
    python scripts/debug_divergence.py aliens --traj noop --context 3 --all-types
    python scripts/debug_divergence.py chase   # should say "all steps match"
"""
import argparse
import json
import os
import sys

DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "validation_results"
)

TRAJECTORY_TYPES = ["noop", "cycling", "random"]


def load_trajectory(game_name, traj_type, results_dir):
    """Load a trajectory JSON file. Returns parsed dict or None."""
    path = os.path.join(results_dir, "per_game", game_name,
                        f"trajectory_{traj_type}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_first_failing_traj(game_name, results_dir):
    """Return the first trajectory type that has a divergence, or None."""
    for ttype in TRAJECTORY_TYPES:
        data = load_trajectory(game_name, ttype, results_dir)
        if data is None:
            continue
        for s in data.get("steps", []):
            if not s.get("matches", True):
                return ttype
    return None


def format_positions(pos_list):
    """Format position list compactly."""
    if not pos_list:
        return "[]"
    return "[" + ", ".join(f"({p[0]}, {p[1]})" for p in pos_list) + "]"


def print_step(step, show_all_types=False, show_bg=False):
    """Print a single step's side-by-side comparison."""
    state_a = step.get("state_a")
    state_b = step.get("state_b")

    if state_a is None or state_b is None:
        # Old format without state data
        match_str = "MATCH" if step.get("matches") else "DIVERGENT"
        print(f"Step {step['step']} (action={step.get('action', '?')}) -- {match_str}")
        if step.get("diffs"):
            for d in step["diffs"]:
                print(f"  {d}")
        else:
            print("  (no state data saved — re-run validate_all.py to get full states)")
        return

    match_str = "MATCH" if step.get("matches") else "DIVERGENT"
    print(f"Step {step['step']} (action={step.get('action', '?')}) -- {match_str}")

    # Score and done
    score_a = state_a.get("score", 0)
    score_b = state_b.get("score", 0)
    done_a = state_a.get("done", False)
    done_b = state_b.get("done", False)
    score_mark = "  " if score_a == score_b else "**"
    done_mark = "  " if done_a == done_b else "**"
    print(f"  {score_mark}score: GVGAI={score_a}  VGDLx={score_b}")
    print(f"  {done_mark}done:  GVGAI={done_a}  VGDLx={done_b}")

    # Collect all type names from both states
    types_a = state_a.get("types", {})
    types_b = state_b.get("types", {})
    all_type_keys = sorted(set(types_a.keys()) | set(types_b.keys()))

    hidden_matching = 0
    hidden_bg = 0

    for tkey in all_type_keys:
        ta = types_a.get(tkey, {})
        tb = types_b.get(tkey, {})
        name = ta.get("key") or tb.get("key") or tkey

        alive_a = ta.get("alive_count", 0)
        alive_b = tb.get("alive_count", 0)
        pos_a = ta.get("positions", [])
        pos_b = tb.get("positions", [])

        # Check if this is a high-count background type
        is_bg = alive_a > 20 or alive_b > 20

        # Check if states match for this type
        type_matches = (alive_a == alive_b and pos_a == pos_b)

        if type_matches and not show_all_types:
            hidden_matching += 1
            continue

        if is_bg and not show_bg:
            hidden_bg += 1
            continue

        print()
        alive_mark = " " if alive_a == alive_b else "*"
        print(f"  {alive_mark} {name} (t{tkey}): alive {alive_a} vs {alive_b}")

        if pos_a != pos_b or show_all_types:
            print(f"    GVGAI: {format_positions(pos_a)}")
            print(f"    VGDLx: {format_positions(pos_b)}")

    suffixes = []
    if hidden_matching > 0:
        suffixes.append(f"{hidden_matching} matching types hidden")
    if hidden_bg > 0:
        suffixes.append(f"{hidden_bg} background types hidden")
    if suffixes:
        hint = ", use --all-types/--show-bg to show"
        print(f"\n  [{', '.join(suffixes)}{hint}]")


def main():
    parser = argparse.ArgumentParser(
        description="Debug GVGAI validation divergences from saved JSON"
    )
    parser.add_argument("game", help="Game name (e.g. aliens)")
    parser.add_argument("--traj", default=None,
                        help="Trajectory type: noop/cycling/random "
                             "(default: first failing)")
    parser.add_argument("--context", type=int, default=2,
                        help="Steps of context around divergence (default: 2)")
    parser.add_argument("--step", type=int, default=None,
                        help="Inspect a specific step instead of auto-finding")
    parser.add_argument("--all-types", action="store_true",
                        help="Show matching types too")
    parser.add_argument("--show-bg", action="store_true",
                        help="Show high-count background types (alive > 20)")
    parser.add_argument("--results-dir", default=None,
                        help=f"Results directory (default: {DEFAULT_RESULTS_DIR})")
    args = parser.parse_args()

    results_dir = args.results_dir or DEFAULT_RESULTS_DIR
    game_dir = os.path.join(results_dir, "per_game", args.game)

    if not os.path.isdir(game_dir):
        print(f"ERROR: No results for '{args.game}' in {results_dir}/per_game/")
        available = []
        per_game = os.path.join(results_dir, "per_game")
        if os.path.isdir(per_game):
            available = sorted(os.listdir(per_game))
        if available:
            print(f"Available: {', '.join(available[:20])}"
                  f"{'...' if len(available) > 20 else ''}")
        sys.exit(1)

    # Determine trajectory type
    traj_type = args.traj
    if traj_type is None:
        traj_type = find_first_failing_traj(args.game, results_dir)
        if traj_type is None:
            # Check if any trajectories exist at all
            for tt in TRAJECTORY_TYPES:
                if load_trajectory(args.game, tt, results_dir) is not None:
                    print(f"{args.game}: all steps match across all trajectories!")
                    return
            print(f"ERROR: No trajectory files found for '{args.game}'")
            sys.exit(1)

    data = load_trajectory(args.game, traj_type, results_dir)
    if data is None:
        print(f"ERROR: No trajectory file for {args.game}/{traj_type}")
        sys.exit(1)

    steps = data.get("steps", [])
    actions = data.get("actions", [])

    print(f"Game: {args.game}")
    print(f"Trajectory: {traj_type}")
    print(f"Steps: {len(steps)}, Actions: {len(actions)}")
    if actions:
        print(f"Action sequence: {actions}")
    print()

    if args.step is not None:
        # Show a specific step
        matching = [s for s in steps if s["step"] == args.step]
        if not matching:
            print(f"ERROR: Step {args.step} not found (range: "
                  f"{steps[0]['step']}-{steps[-1]['step']})")
            sys.exit(1)
        print_step(matching[0], show_all_types=args.all_types,
                   show_bg=args.show_bg)
    else:
        # Find first divergence and show context window
        first_fail_idx = None
        for i, s in enumerate(steps):
            if not s.get("matches", True):
                first_fail_idx = i
                break

        if first_fail_idx is None:
            print(f"All {len(steps)} steps match!")
            return

        fail_step = steps[first_fail_idx]["step"]
        total_failing = sum(1 for s in steps if not s.get("matches", True))
        print(f"First divergence at step {fail_step} "
              f"({total_failing}/{len(steps)} steps failing)")
        print()

        # Context window
        start = max(0, first_fail_idx - args.context)
        end = min(len(steps), first_fail_idx + args.context + 1)

        for i in range(start, end):
            print_step(steps[i], show_all_types=args.all_types,
                       show_bg=args.show_bg)
            print()


if __name__ == "__main__":
    main()
