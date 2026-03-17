#!/usr/bin/env python3
"""
Export Golden Trajectories for Synthetic Pipeline
==================================================
Scans baseline run directories, selects the best trajectory per task
(by checkpoint score, then model priority), and exports to a local directory
as JSON files ready for the synthetic pipeline.

The synthetic pipeline reads trajectories via pd.read_json() for segment
grounding (is_used_in_trajectory, first_use_pct, etc.).

Output: experiments/agentcompany/golden_trajectories/{task_name}.json

Usage:
    # Export from all runs in experiments/agentcompany/runs/
    python scripts/export_tac_golden_trajectories.py

    # Export from a specific run
    python scripts/export_tac_golden_trajectories.py --run-dirs "run_opus_4_5_pass3_*"

    # Export only specific tasks
    python scripts/export_tac_golden_trajectories.py --tasks finance_budget_variance hr_new_grad_job_description

    # Dry run (show what would be exported)
    python scripts/export_tac_golden_trajectories.py --dry-run
"""

import argparse
import json
import re
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import PATHS  # noqa: E402

RUNS_DIR = REPO_ROOT / PATHS["runs_dir"]
DEFAULT_OUTPUT_DIR = REPO_ROOT / PATHS["tac_golden_trajectories"]

# Baseline run dir pattern: run_{model_key}_pass{k}_{YYYYMMDD}_{HHMMSS}[_{uuid8}]
# Captures model_key (group 1). Excludes underspecified/user-sim runs.
BASELINE_RUN_RE = re.compile(r"^run_(.+?)_pass\d+_\d{8}_\d{6}(?:_[0-9a-f]{8})?$")

# Priority order for model tiebreaking (lower index = better).
# When multiple models solve the same task, we pick the trajectory from the
# highest-priority model. Ordering: frontier → fast → open-weight.
MODEL_PRIORITY = [
    "opus_4_5",
    "sonnet_4_6",
    "sonnet_4_5",
    "gpt_5_2",
    "gpt_5_1",
    "gpt_5",
    "o3_pro",
    "gemini_3_pro",
    "gemini_3_flash",
    "gemini_3_1_flash_lite",
    "haiku_4_5",
    "kimi_k2",
    "qwen3_235b",
]


def get_model_priority(model_str: str) -> int:
    """Get priority for a model string (lower = better).

    Uses longest-match to avoid substring collisions
    (e.g. 'gemini_3_flash' matching inside 'gemini_3_1_flash_lite').
    """
    best_priority = 99
    best_len = 0
    for i, key in enumerate(MODEL_PRIORITY):
        if key in model_str and len(key) > best_len:
            best_priority = i
            best_len = len(key)
    return best_priority


def load_trajectory_events(state_dir: Path) -> Optional[List[Dict]]:
    """Load trajectory events from OpenHands state directory.

    Structure: state/sessions/{session_id}/events/{0..N}.json
    Returns list of event dicts (the format expected by pd.read_json).
    """
    sessions_dir = state_dir / "sessions"
    if not sessions_dir.exists():
        return None

    session_dirs = sorted(d for d in sessions_dir.iterdir() if d.is_dir())
    if not session_dirs:
        return None

    events_dir = session_dirs[0] / "events"
    if not events_dir.exists():
        return None

    events = []
    for event_file in sorted(
        [f for f in events_dir.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    ):
        try:
            with open(event_file, encoding="utf-8") as f:
                events.append(json.load(f))
        except Exception:
            continue

    return events if events else None


def _find_best_trial(local_results_task: Path, run_dir: Path, task: str) -> Optional[Path]:
    """Find the highest-scoring trial directory with a valid state/sessions dir.

    Reads per-trial evaluation JSONs from final_results/deterministic/{task}/{model}/
    to pick the trial with the best checkpoint score. Falls back to the first valid
    trial if no evaluation files exist.
    """
    # Collect all valid trial dirs (sorted alphabetically = same order as evaluation files)
    valid_trials: List[Path] = []
    for model_dir in sorted(local_results_task.iterdir()):
        if not model_dir.is_dir():
            continue
        for trial_dir in sorted(model_dir.iterdir()):
            if not trial_dir.is_dir():
                continue
            state_dir = trial_dir / "state"
            if state_dir.exists() and (state_dir / "sessions").exists():
                valid_trials.append(trial_dir)

    if not valid_trials:
        return None

    # Try to score each trial using evaluation JSONs (aligned by sort order)
    det_task_dir = run_dir / "final_results" / "deterministic" / task
    eval_files = []
    if det_task_dir.exists():
        for model_dir in sorted(det_task_dir.iterdir()):
            if model_dir.is_dir():
                eval_files = sorted(
                    [
                        f
                        for f in model_dir.glob("evaluation_*.json")
                        if f.stem.split("_")[-1].isdigit()
                    ],
                    key=lambda f: int(f.stem.split("_")[-1]),
                )
                break

    if eval_files:
        best_score = -1.0
        best_trial = None
        for eval_file in eval_files:
            try:
                idx = int(eval_file.stem.split("_")[-1])
                if idx >= len(valid_trials):
                    continue
                with open(eval_file, encoding="utf-8") as f:
                    result = json.load(f)
                score = result.get("final_score", {})
                total = score.get("total", 0)
                achieved = score.get("result", 0)
                pct = (achieved / total * 100) if total > 0 else 0.0
                if pct > best_score:
                    best_score = pct
                    best_trial = valid_trials[idx]
            except Exception:
                continue
        if best_trial is not None:
            return best_trial

    return valid_trials[0]


def scan_runs(
    runs_dir: Path,
    run_patterns: Optional[List[str]] = None,
    task_filter: Optional[List[str]] = None,
) -> Dict[str, List[dict]]:
    """Scan baseline run directories and collect per-task candidates.

    Returns: {task_name: [{"model": ..., "avg_percent": ..., "trial_dir": ...}, ...]}
    """
    task_candidates: Dict[str, List[dict]] = {}

    if not runs_dir.exists():
        return task_candidates

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        m = BASELINE_RUN_RE.match(run_dir.name)
        if not m:
            continue

        if run_patterns and not any(fnmatch(run_dir.name, p) for p in run_patterns):
            continue

        summary_files = sorted(
            (run_dir / "final_results" / "summaries").glob("pass_k_summary_*.csv")
        )
        if not summary_files:
            continue

        model_name = m.group(1)
        model_pri = get_model_priority(model_name)
        df = pd.read_csv(summary_files[0])

        for _, row in df[df["task"] != "AGGREGATE"].iterrows():
            task = row["task"]
            if task_filter and task not in task_filter:
                continue

            local_results = run_dir / "local_results" / task
            if not local_results.exists():
                continue

            trial_dir = _find_best_trial(local_results, run_dir, task)
            if trial_dir is None:
                continue

            if task not in task_candidates:
                task_candidates[task] = []

            task_candidates[task].append(
                {
                    "model": model_name,
                    "model_priority": model_pri,
                    "avg_percent": float(row.get("avg_percent", 0)),
                    "trial_dir": trial_dir,
                }
            )

    return task_candidates


def select_best_per_task(task_candidates: Dict[str, List[dict]]) -> Dict[str, dict]:
    """Select the single best trajectory per task.

    Priority: highest avg_percent, then model priority (lower = better).
    """
    return {
        task: min(candidates, key=lambda c: (-c["avg_percent"], c["model_priority"]))
        for task, candidates in task_candidates.items()
    }


def export_trajectories(
    best_per_task: Dict[str, dict],
    output_dir: Path,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Export trajectories to output directory. Returns (success, skipped)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    skipped = 0

    for task in sorted(best_per_task):
        info = best_per_task[task]
        state_dir = info["trial_dir"] / "state"

        if dry_run:
            print(f"  [DRY RUN] {task}: {info['model']} (avg={info['avg_percent']:.1f}%)")
            success += 1
            continue

        events = load_trajectory_events(state_dir)
        if not events:
            print(f"  SKIP {task}: no events found in {state_dir}")
            skipped += 1
            continue

        out_path = output_dir / f"{task}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, default=str)

        print(
            f"  {task}: {len(events)} events from {info['model']} "
            f"(avg={info['avg_percent']:.1f}%) -> {out_path.name}"
        )
        success += 1

    return success, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Export golden trajectories for synthetic pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help=f"Directory containing run outputs (default: {RUNS_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        help="Glob patterns to filter run directories (e.g., 'run_opus_*')",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Only export these tasks (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be exported without writing files",
    )
    args = parser.parse_args()

    # Handle zsh not word-splitting: --tasks "a b c" -> ["a", "b", "c"]
    if args.tasks:
        args.tasks = [t for arg in args.tasks for t in arg.split()]

    print(f"\n{'='*60}")
    print("Golden Trajectory Export")
    print(f"{'='*60}")
    print(f"  Runs dir:   {args.runs_dir}")
    print(f"  Output dir: {args.output_dir}")

    task_candidates = scan_runs(args.runs_dir, args.run_dirs, args.tasks)
    if not task_candidates:
        print("\nNo matching runs found.")
        sys.exit(1)

    total = sum(len(v) for v in task_candidates.values())
    print(f"\n  Found {total} candidates across {len(task_candidates)} tasks\n")

    best = select_best_per_task(task_candidates)

    print(f"{'='*60}")
    print("Exporting best trajectory per task:")
    print(f"{'='*60}")

    success, skipped = export_trajectories(best, args.output_dir, args.dry_run)

    print(f"\n{'='*60}")
    if args.dry_run:
        print(f"DRY RUN: Would export {success} trajectories")
    else:
        print(f"Exported {success} trajectories ({skipped} skipped)")
        print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    if success > 0 and not args.dry_run:
        rel_out = args.output_dir.relative_to(REPO_ROOT)
        print("Use with the synthetic pipeline:")
        print(
            f"  python scripts/run_tac_underspec.py --generate " f"--trajectory-dir {rel_out} ..."
        )


if __name__ == "__main__":
    main()
