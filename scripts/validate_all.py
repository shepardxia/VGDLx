#!/usr/bin/env python
"""
Standalone CLI validation script for vgdl-jax cross-engine comparison.

Supports both py-vgdl and GVGAI as reference engines, with auto-discovery
and compatibility filtering.

Usage:
    python scripts/validate_all.py                                    # py-vgdl, all games
    python scripts/validate_all.py --source gvgai --compat-only       # GVGAI compat report
    python scripts/validate_all.py --source gvgai --compat-filter supported --game aliens
    python scripts/validate_all.py --game chase --steps 50
    python scripts/validate_all.py --latex-only                       # regenerate tables
    python scripts/validate_all.py --render-diffs                     # generate PNGs/GIFs
"""

import argparse
import json
import os
import sys
import time

from vgdl_jax.validate.discovery import discover_games, GameEntry
from vgdl_jax.validate.constants import PYVGDL_GAMES, GVGAI_GAMES, PYVGDL_GAMES_DIR, GVGAI_GAMES_DIR
from vgdl_jax.validate.harness import (
    run_comparison,
    run_gvgai_comparison,
    setup_jax_game,
)

# ── Constants ────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")

TRAJECTORY_TYPES = ["noop", "cycling", "random"]

OUTPUT_DIR = os.path.join(PROJECT_DIR, "validation_results")


# ── Action sequence generators ───────────────────────────────────────────────


def _make_actions(traj_type, n_actions, n_steps, seed=42, noop_idx=None):
    """Generate an action sequence of the given type."""
    if noop_idx is None:
        noop_idx = n_actions - 1
    if traj_type == "noop":
        return [noop_idx] * n_steps
    elif traj_type == "cycling":
        return [i % n_actions for i in range(n_steps)]
    elif traj_type == "random":
        import numpy as np
        rng = np.random.RandomState(seed)
        return rng.randint(0, n_actions, size=n_steps).tolist()
    else:
        raise ValueError(f"Unknown trajectory type: {traj_type}")


# ── Classification ───────────────────────────────────────────────────────────


def _classify_result(result):
    """Classify a TrajectoryResult into a status string."""
    if all(sc.matches for sc in result.steps):
        return "match"
    return "state_error"


# ── Per-game validation ─────────────────────────────────────────────────────


def validate_game(entry: GameEntry, n_steps=30, seed=42, render_diffs=False,
                   output_dir=None, backend='pyvgdl'):
    """Run all 3 trajectory types for a single game.

    Args:
        entry: GameEntry
        n_steps: trajectory length
        seed: random seed
        render_diffs: generate visual diff artifacts
        output_dir: output directory for artifacts
        backend: 'pyvgdl' for py-vgdl comparison, 'gvgai' for GVGAI comparison

    Returns:
        dict with keys: status, trajectories, errors, timing
    """
    game_result = {
        "status": "match",
        "trajectories": {},
        "errors": [],
        "timing_s": 0.0,
    }

    t0 = time.time()

    # Get n_actions from JAX compiled game
    try:
        compiled, game_def = setup_jax_game(entry)
        n_actions = compiled.n_actions
        noop_idx = compiled.noop_action
    except Exception as e:
        game_result["status"] = "compile_error"
        game_result["errors"].append(f"JAX compile failed: {e}")
        game_result["timing_s"] = time.time() - t0
        return game_result

    worst_status = "match"
    status_rank = {"match": 0, "state_error": 1, "compile_error": 2}

    for traj_type in TRAJECTORY_TYPES:
        try:
            actions = _make_actions(traj_type, n_actions, n_steps, seed=seed,
                                    noop_idx=noop_idx)

            if backend == 'gvgai':
                result = run_gvgai_comparison(entry, actions, seed=seed)
            else:
                # Stochastic games use RNG replay; sokoban is deterministic
                use_rng = entry.name != "sokoban"
                result = run_comparison(
                    entry, actions, seed=seed,
                    use_rng_replay=use_rng,
                )

            status = _classify_result(result)

            # Collect per-step data
            step_data = []
            for sc in result.steps:
                step_data.append({
                    "step": sc.step,
                    "action": sc.action,
                    "matches": sc.matches,
                    "diffs": sc.diffs,
                })

            game_result["trajectories"][traj_type] = {
                "status": status,
                "n_steps": result.n_steps,
                "actual_steps": len(result.steps),
                "level": result.level,
                "steps": step_data,
            }

            # Render artifacts if requested
            if render_diffs and output_dir:
                try:
                    render_diff_artifacts(
                        entry, traj_type, result, actions,
                        seed=seed, output_dir=output_dir)
                except Exception as render_err:
                    game_result["errors"].append(
                        f"render {traj_type}: {render_err}")

            if status_rank.get(status, 99) > status_rank.get(worst_status, 0):
                worst_status = status

        except Exception as e:
            game_result["trajectories"][traj_type] = {
                "status": "compile_error",
                "error": str(e),
            }
            game_result["errors"].append(f"{traj_type}: {e}")
            worst_status = "compile_error"

    game_result["status"] = worst_status
    game_result["timing_s"] = round(time.time() - t0, 2)
    return game_result


# ── Output writers ───────────────────────────────────────────────────────────


def _serialize_step_data(step_data):
    """Make step data JSON-serializable."""
    if isinstance(step_data, dict):
        return {str(k): _serialize_step_data(v) for k, v in step_data.items()}
    elif isinstance(step_data, (list, tuple)):
        return [_serialize_step_data(item) for item in step_data]
    elif isinstance(step_data, set):
        return sorted(step_data)
    else:
        return step_data


def write_results(all_results, output_dir):
    """Write structured results to output_dir/."""
    os.makedirs(output_dir, exist_ok=True)

    total = len(all_results)
    matching = sum(1 for r in all_results.values() if r["status"] == "match")
    state_errors = sum(1 for r in all_results.values() if r["status"] == "state_error")
    compile_errors = sum(1 for r in all_results.values() if r["status"] == "compile_error")

    summary = {
        "total_games": total,
        "matching": matching,
        "state_errors": state_errors,
        "compile_errors": compile_errors,
    }

    per_game_summary = {}
    for game_name, result in all_results.items():
        per_game_summary[game_name] = {
            "status": result["status"],
            "timing_s": result["timing_s"],
            "trajectories": {
                ttype: {
                    "status": tdata.get("status", "unknown"),
                    "n_steps": tdata.get("n_steps", 0),
                    "actual_steps": tdata.get("actual_steps", 0),
                    "level": tdata.get("level", 0),
                    "failing_steps": sum(
                        1 for s in tdata.get("steps", []) if not s.get("matches", True)
                    ),
                }
                for ttype, tdata in result["trajectories"].items()
            },
        }

    output = {"summary": summary, "per_game": per_game_summary}
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Wrote {results_path}")

    for game_name, result in all_results.items():
        game_dir = os.path.join(output_dir, "per_game", game_name)
        os.makedirs(game_dir, exist_ok=True)

        for ttype, tdata in result["trajectories"].items():
            traj_path = os.path.join(game_dir, f"trajectory_{ttype}.json")
            with open(traj_path, "w") as f:
                json.dump(_serialize_step_data(tdata), f, indent=2, default=str)

        if result["errors"]:
            error_path = os.path.join(game_dir, "errors.txt")
            with open(error_path, "w") as f:
                f.write(f"Errors for {game_name}\n")
                f.write("=" * 60 + "\n\n")
                for err in result["errors"]:
                    f.write(f"- {err}\n")
            print(f"  Wrote {error_path}")

    print(f"  Wrote per-game results to {os.path.join(output_dir, 'per_game')}/")


# ── LaTeX table generation ───────────────────────────────────────────────────


_STATUS_SYMBOL = {
    "match": r"\checkmark",
    "state_error": r"$\times$",
    "compile_error": r"\textbf{ERR}",
    "unknown": "?",
}


def _traj_cell(traj_data):
    """Format a trajectory result for the LaTeX table."""
    status = traj_data.get("status", "unknown")
    symbol = _STATUS_SYMBOL.get(status, "?")
    if status in ("match", "compile_error", "unknown"):
        return symbol

    failing = traj_data.get("failing_steps", 0)
    actual = traj_data.get("actual_steps", 0)
    if failing > 0:
        return f"{symbol} ({failing}/{actual})"
    return symbol


def generate_latex(results_json_path, output_dir):
    """Generate LaTeX validation summary table from results.json."""
    with open(results_json_path) as f:
        data = json.load(f)

    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    per_game = data["per_game"]
    summary = data["summary"]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-engine validation: py-vgdl vs vgdl-jax. "
                 r"\checkmark\ = match, "
                 r"$\times$ = state error. "
                 r"Parenthetical shows (failing steps / total steps).}")
    lines.append(r"\label{tab:validation-summary}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Game & Init State & NOOP & Cycling & Random & Overall \\")
    lines.append(r"\midrule")

    for game_name in sorted(per_game.keys()):
        gdata = per_game[game_name]

        noop_traj = gdata["trajectories"].get("noop", {})
        init_level = noop_traj.get("level", 0)
        init_symbol = r"\checkmark" if init_level >= 1 else r"$\times$"

        noop_cell = _traj_cell(gdata["trajectories"].get("noop", {}))
        cycling_cell = _traj_cell(gdata["trajectories"].get("cycling", {}))
        random_cell = _traj_cell(gdata["trajectories"].get("random", {}))

        overall_symbol = _STATUS_SYMBOL.get(gdata["status"], "?")

        escaped_name = game_name.replace("_", r"\_")
        lines.append(
            f"{escaped_name} & {init_symbol} & {noop_cell} "
            f"& {cycling_cell} & {random_cell} & {overall_symbol} \\\\"
        )

    lines.append(r"\midrule")

    total = summary["total_games"]
    matching = summary["matching"]
    errors = summary["state_errors"]
    lines.append(
        f"\\textbf{{Total}} & & & & & "
        f"{matching}/{total} match, {errors}/{total} diverge \\\\"
    )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_path = os.path.join(tables_dir, "validation_summary.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Wrote {tex_path}")
    return tex_path


# ── Console output ───────────────────────────────────────────────────────────


def print_summary(all_results):
    """Print a human-readable summary to stdout."""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    status_icons = {
        "match": "[MATCH]  ",
        "state_error": "[FAIL]   ",
        "compile_error": "[ERROR]  ",
    }

    for game_name, result in all_results.items():
        icon = status_icons.get(result["status"], "[?]      ")
        timing = f"({result['timing_s']:.1f}s)"
        print(f"  {icon} {game_name:<20s} {timing}")

        for ttype, tdata in result["trajectories"].items():
            tstatus = tdata.get("status", "unknown")
            actual = tdata.get("actual_steps", "?")
            failing = sum(
                1 for s in tdata.get("steps", []) if not s.get("matches", True)
            )
            detail = f"  steps={actual}"
            if failing > 0:
                detail += f", failing={failing}"
            print(f"           {ttype:<10s} {tstatus:<16s} {detail}")

        if result["errors"]:
            for err in result["errors"]:
                print(f"           ERROR: {err}")

    total = len(all_results)
    matching = sum(1 for r in all_results.values() if r["status"] == "match")
    errors = sum(1 for r in all_results.values()
                 if r["status"] in ("state_error", "compile_error"))
    print()
    print(f"  Total: {total} games | Match: {matching} | Errors: {errors}")
    print("=" * 70)


# ── Compatibility report ─────────────────────────────────────────────────────


def print_compat_report(entries: list[GameEntry]):
    """Print compatibility report by attempting to compile each game."""
    supported = []
    unsupported = []

    for entry in entries:
        if not entry.level_files:
            unsupported.append((entry.name, "no level files"))
            continue
        try:
            setup_jax_game(entry)
            supported.append(entry.name)
        except Exception as e:
            unsupported.append((entry.name, str(e)))

    print(f"\n{'=' * 70}")
    print(f"COMPATIBILITY REPORT ({len(entries)} games)")
    print(f"{'=' * 70}")

    print(f"\n  Supported: {len(supported)}")
    for name in sorted(supported):
        print(f"    [OK] {name}")

    print(f"\n  Unsupported: {len(unsupported)}")
    for name, reason in sorted(unsupported):
        print(f"    [--] {name:<30s} {reason}")

    print(f"\n  Summary: {len(supported)}/{len(entries)} supported")
    print(f"{'=' * 70}")


# ── Visual diff rendering ───────────────────────────────────────────────────


def _render_jax_frames(entry: GameEntry, actions, seed=42, block_size=24):
    """Re-run JAX trajectory and render each step to RGB."""
    import jax
    from vgdl_jax.render import render_pygame

    compiled, game_def = setup_jax_game(entry)
    sgm = compiled.static_grid_map
    state = compiled.init_state.replace(rng=jax.random.PRNGKey(seed))
    step_fn = compiled.step_fn

    frames = []
    frames.append((0, render_pygame(state, game_def, block_size,
                                     render_sprites=False, static_grid_map=sgm)))

    for i, a in enumerate(actions):
        if bool(state.done):
            break
        state = step_fn(state, a)
        frames.append((i + 1, render_pygame(state, game_def, block_size,
                                             render_sprites=False, static_grid_map=sgm)))

    return frames


def _annotate_frame(frame, step_idx, diffs, is_divergent):
    """Add a colored border to a frame: green=match, red=divergent."""
    h, w = frame.shape[:2]
    border = 3
    annotated = frame.copy()
    color = [255, 60, 60] if is_divergent else [60, 200, 60]
    annotated[:border, :] = color
    annotated[-border:, :] = color
    annotated[:, :border] = color
    annotated[:, -border:] = color
    return annotated


def render_diff_artifacts(entry: GameEntry, traj_type, comparison_result, actions,
                          seed, output_dir, block_size=24):
    """Render and save visual diff artifacts for a trajectory."""
    import imageio.v3 as iio

    game_dir = os.path.join(output_dir, "per_game", entry.name)
    os.makedirs(game_dir, exist_ok=True)

    divergent_steps = {}
    for sc in comparison_result.steps:
        if not sc.matches:
            divergent_steps[sc.step] = sc.diffs

    frames = _render_jax_frames(entry, actions, seed=seed, block_size=block_size)

    gif_frames = []
    for step_idx, frame in frames:
        is_div = step_idx in divergent_steps
        diffs = divergent_steps.get(step_idx, [])
        annotated = _annotate_frame(frame, step_idx, diffs, is_div)
        gif_frames.append(annotated)

        if is_div:
            png_path = os.path.join(
                game_dir, f"step_{step_idx:03d}_{traj_type}_jax.png")
            iio.imwrite(png_path, frame)

    if gif_frames:
        gif_path = os.path.join(game_dir, f"trajectory_{traj_type}.gif")
        iio.imwrite(gif_path, gif_frames, duration=200, loop=0)


# ── Main ─────────────────────────────────────────────────────────────────────


def _get_games(args) -> dict[str, GameEntry]:
    """Get games dict based on --source and --game flags."""
    if args.source == 'pyvgdl':
        games = dict(PYVGDL_GAMES)
    elif args.source == 'gvgai':
        games = dict(GVGAI_GAMES)
    else:
        # Custom directory
        entries = discover_games(args.source, source='custom')
        games = {e.name: e for e in entries}

    if args.game:
        if args.game not in games:
            available = sorted(games.keys())
            print(f"ERROR: Unknown game '{args.game}'. Available ({len(available)}): "
                  f"{', '.join(available[:20])}{'...' if len(available) > 20 else ''}")
            sys.exit(1)
        games = {args.game: games[args.game]}

    return games


def main():
    parser = argparse.ArgumentParser(
        description="Cross-engine validation for vgdl-jax"
    )
    parser.add_argument(
        "--source", type=str, default="pyvgdl",
        help="Game source: 'pyvgdl', 'gvgai', or path to game directory (default: pyvgdl)",
    )
    parser.add_argument(
        "--game", type=str, default=None,
        help="Run validation for a single game by name",
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Number of steps per trajectory (default: 30)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--compat-only", action="store_true",
        help="Print compatibility report only, no validation",
    )
    parser.add_argument(
        "--compat-filter", type=str, default="supported",
        choices=["supported", "all"],
        help="Filter for validation: 'supported' (default) or 'all'",
    )
    parser.add_argument(
        "--latex-only", action="store_true",
        help="Regenerate LaTeX table from existing results.json",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--render-diffs", action="store_true",
        help="Generate PNG/GIF artifacts at divergence points",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or OUTPUT_DIR

    # ── LaTeX-only mode ──
    if args.latex_only:
        results_path = os.path.join(output_dir, "results.json")
        if not os.path.exists(results_path):
            print(f"ERROR: {results_path} not found. Run validation first.")
            sys.exit(1)
        generate_latex(results_path, output_dir)
        print("Done (LaTeX only).")
        return

    # ── Get games ──
    games = _get_games(args)

    if not games:
        print("ERROR: No games found.")
        sys.exit(1)

    # ── Compat-only mode ──
    if args.compat_only:
        print_compat_report(list(games.values()))
        return

    # ── Filter by compatibility ──
    if args.compat_filter == "supported":
        supported = {}
        n_skipped = 0
        for name, entry in games.items():
            if not entry.level_files:
                n_skipped += 1
                continue
            try:
                setup_jax_game(entry)
                supported[name] = entry
            except Exception:
                n_skipped += 1
        if n_skipped > 0:
            print(f"Skipping {n_skipped} unsupported game(s)")
        games = supported

    if not games:
        print("No supported games to validate.")
        return

    # ── Run validation ──
    print(f"Running validation for {len(games)} game(s), "
          f"{args.steps} steps, seed={args.seed}, source={args.source}")
    print(f"Trajectory types: {TRAJECTORY_TYPES}")
    print()

    # Determine backend: gvgai source uses GVGAI backend, pyvgdl uses py-vgdl backend
    backend = 'gvgai' if args.source == 'gvgai' else 'pyvgdl'

    all_results = {}
    for game_name, entry in sorted(games.items()):
        print(f"  Validating {game_name}...", end="", flush=True)
        result = validate_game(entry, n_steps=args.steps, seed=args.seed,
                               render_diffs=args.render_diffs,
                               output_dir=output_dir, backend=backend)
        all_results[game_name] = result
        icon = {"match": "ok", "state_error": "FAIL", "compile_error": "ERROR"}
        print(f" {icon.get(result['status'], '?')} ({result['timing_s']:.1f}s)")

    # ── Write outputs ──
    print()
    print("Writing results...")
    write_results(all_results, output_dir)
    generate_latex(os.path.join(output_dir, "results.json"), output_dir)

    # ── Console summary ──
    print_summary(all_results)


if __name__ == "__main__":
    main()
