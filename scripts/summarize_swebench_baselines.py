#!/usr/bin/env python3
"""
Summarize SWE-Bench Pro baseline evaluation results as pass_k_per_task.csv.

Reads baseline_N/eval_results/ from a SWE-Bench experiment directory,
determines per-task pass/fail (all F2P tests must pass), and writes
pass_k_per_task.csv in the same format used by MCP-Atlas. This enables
filter_passed_tasks.py to work uniformly across benchmarks.

Prerequisite: baselines must be evaluated first:
    python scripts/process_swebench_underspec.py --exp-dir <dir> --run-eval --eval-only \\
        --dockerhub-username <user>

Usage:
    python scripts/summarize_swebench_baselines.py --exp-dir experiments/swebench/runs/run_my_exp_*

    # Then use with filter_passed_tasks.py for multi-model intersection:
    python scripts/filter_passed_tasks.py \\
        experiments/swebench/runs/run_my_exp_gpt_5_2_* \\
        experiments/swebench/runs/run_my_exp_sonnet_4_5_* \\
        -o passed_task_ids.json
"""

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import DEFAULTS  # noqa: E402
from evaluation.pass_k_metrics import pass_at_k, pass_hat_k  # noqa: E402
from scripts.compute_phase_b_results import compute_baseline_pass_rate  # noqa: E402
from scripts.process_swebench_underspec import ensure_swebench_csv  # noqa: E402

SWEBENCH_CSV = REPO_ROOT / "swebenchpro" / "SWE-bench_Pro-os" / "swe_bench_pro_full.csv"


def evaluate_baselines(exp_dir: Path, swebench_df, num_trials: int) -> dict:
    """Evaluate baseline results using the canonical compute_baseline_pass_rate.

    Returns: {task_id: [0/1, 0/1, ...]} for each trial
    """
    results = compute_baseline_pass_rate(exp_dir, swebench_df, num_trials)
    if not results:
        return {}

    per_task = {}
    for r in results:
        for task_id, resolved in r.get("per_instance", {}).items():
            per_task.setdefault(task_id, []).append(resolved)
    return per_task


def write_pass_k_csv(per_task: dict, output_path: Path, max_k: int):
    """Write pass_k_per_task.csv in the standard format."""
    headers = ["task_id", "n_trials", "n_success", "pass_rate"]
    for k in range(1, max_k + 1):
        headers.append(f"pass^{k}")
    for k in range(1, max_k + 1):
        headers.append(f"pass@{k}")

    rows = []
    total_trials = 0
    total_successes = 0

    for task_id in sorted(per_task):
        trials = per_task[task_id]
        n = len(trials)
        c = sum(trials)
        total_trials += n
        total_successes += c

        row = {
            "task_id": task_id,
            "n_trials": n,
            "n_success": c,
            "pass_rate": f"{c / n:.4f}" if n > 0 else "0.0000",
        }
        for k in range(1, max_k + 1):
            row[f"pass^{k}"] = f"{pass_hat_k(n, c, k):.4f}"
            row[f"pass@{k}"] = f"{pass_at_k(n, c, k):.4f}"
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

        n_tasks = len(rows)
        agg = {
            "task_id": "AGGREGATE",
            "n_trials": total_trials,
            "n_success": total_successes,
            "pass_rate": f"{total_successes / total_trials:.4f}" if total_trials else "0.0000",
        }
        for k in range(1, max_k + 1):
            if n_tasks > 0:
                avg_pass_k = sum(float(r[f"pass@{k}"]) for r in rows) / n_tasks
                avg_hat_k = sum(float(r[f"pass^{k}"]) for r in rows) / n_tasks
            else:
                avg_pass_k = avg_hat_k = 0.0
            agg[f"pass^{k}"] = f"{avg_hat_k:.4f}"
            agg[f"pass@{k}"] = f"{avg_pass_k:.4f}"
        writer.writerow(agg)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize SWE-Bench Pro baseline eval results as pass_k_per_task.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Experiment directory containing baseline_N/ subdirectories",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=DEFAULTS["num_trials"],
        help=f"Number of baseline trials (default: {DEFAULTS['num_trials']})",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_dir():
        print(f"ERROR: {exp_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    baseline_dirs = sorted(exp_dir.glob("baseline_[0-9]*"))
    if not baseline_dirs:
        print(f"ERROR: No baseline_N/ directories found in {exp_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Experiment dir: {exp_dir}")
    print(f"Baseline dirs: {len(baseline_dirs)}")

    csv_path = ensure_swebench_csv(SWEBENCH_CSV)
    swebench_df = pd.read_csv(csv_path, index_col="instance_id")
    per_task = evaluate_baselines(exp_dir, swebench_df, args.num_trials)

    if not per_task:
        print("ERROR: No baseline results found. Run --eval-only first.", file=sys.stderr)
        sys.exit(1)

    output_path = exp_dir / "pass_k_per_task.csv"
    write_pass_k_csv(per_task, output_path, max_k=args.num_trials)

    n_passed = sum(1 for trials in per_task.values() if sum(trials) > 0)
    print(f"Tasks: {len(per_task)} total, {n_passed} passed (pass@1 > 0)")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
