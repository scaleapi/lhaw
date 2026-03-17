#!/usr/bin/env python3
"""
Select tasks that passed across all baseline runs (intersection).

Reads pass_k_per_task.csv from each run directory, filters tasks where
pass@k > 0 (i.e., at least one trial succeeded), and outputs the intersection
across all runs. This implements the paper's task selection criterion:
"pass@3 > 0 for all reference models" (default: --metric pass@3).

Works with any benchmark that produces pass_k_per_task.csv (MCP-Atlas, SWE-Bench Pro).

Usage:
    # Single model baseline
    python scripts/filter_passed_tasks.py \\
        experiments/mcpatlas/runs/run_baseline_gpt_5_2_20260301_120000 \\
        -o passed_task_ids.json

    # Multi-model intersection (paper methodology)
    python scripts/filter_passed_tasks.py \\
        experiments/mcpatlas/runs/run_baseline_gpt_5_2_* \\
        experiments/mcpatlas/runs/run_baseline_sonnet_4_5_* \\
        experiments/mcpatlas/runs/run_baseline_gemini_3_pro_* \\
        -o passed_task_ids.json

    # With glob expansion from shell
    python scripts/filter_passed_tasks.py \\
        experiments/mcpatlas/runs/run_baseline_* \\
        -o passed_task_ids.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def get_passed_tasks(run_dir: Path, metric: str = "pass@3", threshold: float = 0.0) -> set:
    """Return task IDs where the given metric exceeds the threshold."""
    csv_path = run_dir / "pass_k_per_task.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found", file=sys.stderr)
        sys.exit(1)

    passed = set()
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["task_id"] == "AGGREGATE":
                continue
            value = float(row.get(metric, "0"))
            if value > threshold:
                passed.add(row["task_id"])
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Select tasks that passed across all baseline runs (intersection)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="Baseline run directories containing pass_k_per_task.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON file path for selected task IDs",
    )
    parser.add_argument(
        "--metric",
        default="pass@3",
        help="Metric column to check (default: pass@3, matching paper criterion)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Minimum metric value (exclusive) to count as passed (default: 0.0)",
    )
    args = parser.parse_args()

    per_run_passed = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str)
        if not run_dir.is_dir():
            print(f"WARNING: Skipping {run_dir} (not a directory)", file=sys.stderr)
            continue
        passed = get_passed_tasks(run_dir, args.metric, args.threshold)
        print(f"  {run_dir.name}: {len(passed)} passed tasks")
        per_run_passed.append(passed)

    if not per_run_passed:
        print("ERROR: No valid run directories found", file=sys.stderr)
        sys.exit(1)

    intersection = sorted(set.intersection(*per_run_passed))

    if not intersection:
        print("ERROR: No tasks passed ALL baseline runs", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(intersection, f, indent=2)

    print(f"Selected {len(intersection)} tasks (intersection of {len(per_run_passed)} run(s))")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
