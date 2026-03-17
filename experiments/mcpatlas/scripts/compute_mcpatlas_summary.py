#!/usr/bin/env python3
"""
Compute Table 3-style summary for MCP-Atlas experiments.

Produces a consolidated table with baseline, underspec, +user pass@3,
user behavior stats (Ask%, Avg/Traj), and Gain/Q per model.

Usage:
    python experiments/mcpatlas/scripts/compute_mcpatlas_summary.py \
        --exp-name mcpatlas_med_exp \
        --models gemini_3_flash sonnet_4_6 gpt_5_2 \
        --output experiments/mcpatlas/reports/mcpatlas_med_exp/summary.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = str(Path(__file__).parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from utils import _scores_dir_sort_key, count_ask_user_calls  # noqa: E402


def find_latest_run(runs_dir: Path, prefix: str, exclude: str = None) -> Path | None:
    candidates = sorted(runs_dir.glob(f"run_{prefix}_*"), reverse=True)
    for c in candidates:
        if exclude and exclude in c.name:
            continue
        return c
    return None


def load_aggregate_pass3(run_dir: Path) -> float | None:
    csv_path = run_dir / "pass_k_per_task.csv"
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("task_id") == "AGGREGATE":
                return float(row["pass@3"])
    return None


def load_aggregate_ckpt(run_dir: Path) -> float | None:
    """Load average checkpoint score from evaluation JSON.

    Averages per-task means (not flat across all trials) to give equal
    weight to each task regardless of trial count.
    """
    eval_files = list(run_dir.glob("evaluation_*.json"))
    if not eval_files:
        return None
    with open(eval_files[0]) as f:
        data = json.load(f)
    task_means = []
    for task_data in data.get("tasks", {}).values():
        scores = task_data.get("coverage_scores", [])
        if scores:
            task_means.append(sum(scores) / len(scores))
    return sum(task_means) / len(task_means) if task_means else None


def compute_user_stats(run_dir: Path) -> dict:
    """Compute Ask%, Avg/Traj, and total questions from a with-user run."""
    scores_dirs = sorted(run_dir.glob("scores_*"), key=_scores_dir_sort_key)
    total_trials = 0
    trials_with_ask = 0
    total_calls = 0

    for scores_dir in scores_dirs:
        for csv_path in scores_dir.glob("scored_*.csv"):
            df = pd.read_csv(csv_path)
            if "raw_conversation_history" not in df.columns or "TASK" not in df.columns:
                continue

            for _, row in df.iterrows():
                total_trials += 1
                hist = row.get("raw_conversation_history")
                if not isinstance(hist, str):
                    continue
                n = count_ask_user_calls(hist)
                if n > 0:
                    trials_with_ask += 1
                    total_calls += n

    ask_pct = trials_with_ask / total_trials if total_trials > 0 else 0
    avg_per_traj = total_calls / trials_with_ask if trials_with_ask > 0 else 0

    return {
        "total_trials": total_trials,
        "trials_with_ask": trials_with_ask,
        "ask_pct": ask_pct,
        "total_questions": total_calls,
        "avg_per_traj": avg_per_traj,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute MCP-Atlas summary table (Paper Table 3)")
    parser.add_argument("--exp-name", required=True, help="Experiment name prefix")
    parser.add_argument("--models", nargs="+", required=True, help="Model short names")
    parser.add_argument("--runs-dir", default="experiments/mcpatlas/runs", help="Runs directory")
    parser.add_argument("--output", help="Output CSV path (prints to stdout if not set)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    rows = []

    for model in args.models:
        baseline_dir = find_latest_run(runs_dir, f"{args.exp_name}_baseline_{model}")
        no_ask_dir = find_latest_run(
            runs_dir, f"{args.exp_name}_underspec_{model}", exclude="_ask_"
        )
        with_ask_dir = find_latest_run(runs_dir, f"{args.exp_name}_underspec_{model}_ask")

        baseline_p3 = load_aggregate_pass3(baseline_dir) if baseline_dir else None
        no_ask_p3 = load_aggregate_pass3(no_ask_dir) if no_ask_dir else None
        with_ask_p3 = load_aggregate_pass3(with_ask_dir) if with_ask_dir else None

        baseline_ckpt = load_aggregate_ckpt(baseline_dir) if baseline_dir else None
        no_ask_ckpt = load_aggregate_ckpt(no_ask_dir) if no_ask_dir else None
        with_ask_ckpt = load_aggregate_ckpt(with_ask_dir) if with_ask_dir else None

        user_stats = compute_user_stats(with_ask_dir) if with_ask_dir else {}

        gain_p3 = (
            (with_ask_p3 - no_ask_p3)
            if (with_ask_p3 is not None and no_ask_p3 is not None)
            else None
        )
        gain_ckpt = (
            (with_ask_ckpt - no_ask_ckpt)
            if (with_ask_ckpt is not None and no_ask_ckpt is not None)
            else None
        )
        total_qs = user_stats.get("total_questions", 0)
        gain_q_p3 = (gain_p3 / total_qs) if (gain_p3 is not None and total_qs > 0) else None
        gain_q_ckpt = (gain_ckpt / total_qs) if (gain_ckpt is not None and total_qs > 0) else None

        row = {
            "Model": model,
            "Baseline Pass@3": f"{baseline_p3:.1%}" if baseline_p3 is not None else "-",
            "Baseline Ckpt%": f"{baseline_ckpt:.1%}" if baseline_ckpt is not None else "-",
            "Underspec Pass@3": f"{no_ask_p3:.1%}" if no_ask_p3 is not None else "-",
            "+User Pass@3": f"{with_ask_p3:.1%}" if with_ask_p3 is not None else "-",
            "Gain": f"{gain_p3:+.1%}" if gain_p3 is not None else "-",
            "Ask%": f"{user_stats.get('ask_pct', 0):.0%}",
            "Avg/Traj": f"{user_stats.get('avg_per_traj', 0):.1f}",
            "Total Qs": str(total_qs),
            "Gain/Q (Pass@3)": f"{gain_q_p3:.4f}" if gain_q_p3 is not None else "-",
            "Ckpt%": f"{no_ask_ckpt:.1%}" if no_ask_ckpt is not None else "-",
            "+User Ckpt%": f"{with_ask_ckpt:.1%}" if with_ask_ckpt is not None else "-",
            "Gain/Q (Ckpt%)": f"{gain_q_ckpt:.4f}" if gain_q_ckpt is not None else "-",
        }
        rows.append(row)

    if rows:
        headers = list(rows[0].keys())
        widths = [max(len(h), max(len(r[h]) for r in rows)) for h in headers]

        header_line = "  ".join(h.rjust(w) for h, w in zip(headers, widths))
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print("  ".join(row[h].rjust(w) for h, w in zip(headers, widths)))
        print()

    if args.output and rows:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
