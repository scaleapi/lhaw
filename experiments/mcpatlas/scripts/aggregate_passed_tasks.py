"""Aggregate passed task IDs across model runs."""

import csv
import glob
import json
import os
import sys
from typing import Dict, List, Set

# Handle large CSV fields
csv.field_size_limit(10 * 1024 * 1024)

PASS_THRESHOLD = 0.75
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_passed_tasks(scored_csv_paths: List[str]) -> Set[str]:
    """
    Get task IDs that passed (coverage_score >= 0.75 or pass@3 >= 0.75) in any of the given CSVs.

    Args:
        scored_csv_paths: List of paths to scored CSV files or pass_k_per_task.csv files

    Returns:
        Set of task IDs that passed at least once
    """
    tasks: Dict[str, List[float]] = {}

    for csv_path in scored_csv_paths:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Check if this is a pass_k_per_task.csv file
            is_pass_k_file = "pass@3" in reader.fieldnames

            for row in reader:
                if is_pass_k_file:
                    # pass_k_per_task.csv format
                    task_id = row["task_id"]
                    try:
                        score = float(row["pass@3"])
                    except (ValueError, TypeError):
                        score = 0.0
                else:
                    # scored_*.csv format
                    task_id = row["TASK"]
                    try:
                        score = float(row["coverage_score"])
                    except (ValueError, TypeError):
                        score = 0.0

                if task_id not in tasks:
                    tasks[task_id] = []
                tasks[task_id].append(score)

    # Task passes if ANY run passes
    passed = {tid for tid, scores in tasks.items() if any(s >= PASS_THRESHOLD for s in scores)}
    return passed


def find_scored_csvs(run_dir: str) -> List[str]:
    """Find all scored_*.csv files in a run directory."""
    # Check if this is a full_inference_data directory with pass_k_per_task.csv
    pass_k_csv = os.path.join(run_dir, "combined", "pass_k_per_task.csv")
    if os.path.exists(pass_k_csv):
        return [pass_k_csv]
    # Otherwise look for individual scored files (inference_data structure)
    return glob.glob(os.path.join(run_dir, "scores_*/scored_*.csv"))


def main():
    # Check for command line argument to specify which data to aggregate
    use_full_inference = len(sys.argv) > 1 and sys.argv[1] == "--full"

    if use_full_inference:
        # Full inference data directories
        base_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "full_inference_data")
        runs = {
            "claude": os.path.join(
                base_dir,
                "FIRST_7_OPUS_batched_full_baseline_claude-opus-4-5-20251101_20260119_164531",
            ),
            "gemini": os.path.join(
                base_dir, "FULL_GEMINI_batched_full_baseline_gemini-3-pro-preview_20260118_233612"
            ),
            "gpt": os.path.join(base_dir, "FULL_OAI_batched_full_baseline_gpt-5.2_20260119_002656"),
        }
        output_base = base_dir
    else:
        # Original inference data directories
        runs = {
            "claude": os.path.join(SCRIPT_DIR, "1_RUN_75_claude-opus-4-5-20251101_20260116_154852"),
            "gpt": os.path.join(SCRIPT_DIR, "1_RUN_75_gpt-5.2_20260116_154930"),
            "gemini": os.path.join(SCRIPT_DIR, "2_RUNS_75_gemini-3-pro-preview_20260116_154805"),
        }
        output_base = SCRIPT_DIR

    # Get passed tasks for each model
    results = {}
    for model, run_dir in runs.items():
        csvs = find_scored_csvs(run_dir)
        if not csvs:
            print(f"Warning: No scored CSVs found for {model} in {run_dir}")
            continue
        passed = get_passed_tasks(csvs)
        results[model] = passed
        print(f"{model}: {len(passed)} tasks passed (from {len(csvs)} CSV files)")

    # Compute union and intersection
    all_passed_sets = list(results.values())
    union_passed = set.union(*all_passed_sets) if all_passed_sets else set()
    intersection_passed = set.intersection(*all_passed_sets) if all_passed_sets else set()

    print(f"\nUnion (ANY model): {len(union_passed)} tasks")
    print(f"Intersection (ALL models): {len(intersection_passed)} tasks")

    # Create output directory
    output_dir = os.path.join(output_base, "aggregated")
    os.makedirs(output_dir, exist_ok=True)

    # Count total tasks from the union of all models' task sets
    # (since different models may have tested different subsets)
    all_task_ids = set()
    for run_dir in runs.values():
        csvs = find_scored_csvs(run_dir)
        for csv_path in csvs:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                # Check if this is a pass_k_per_task.csv file
                is_pass_k_file = "pass@3" in reader.fieldnames
                task_id_field = "task_id" if is_pass_k_file else "TASK"

                for row in reader:
                    all_task_ids.add(row[task_id_field])
    total_tasks = len(all_task_ids)

    # Write individual model results
    model_full_names = {
        "claude": "claude-opus-4-5-20251101",
        "gpt": "gpt-5.2",
        "gemini": "gemini-3-pro-preview",
    }

    for model, passed in results.items():
        output_path = os.path.join(output_dir, f"{model}_passed.json")
        data = {
            "model": model_full_names.get(model, model),
            "task_ids": sorted(list(passed)),
            "count": len(passed),
            "total_tasks": total_tasks,
            "pass_rate": len(passed) / total_tasks if total_tasks > 0 else 0,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {output_path}")

    # Write union
    union_path = os.path.join(output_dir, "union_passed.json")
    with open(union_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Tasks passed by ANY model",
                "task_ids": sorted(list(union_passed)),
                "count": len(union_passed),
                "total_tasks": total_tasks,
                "pass_rate": len(union_passed) / total_tasks if total_tasks > 0 else 0,
            },
            f,
            indent=2,
        )
    print(f"Wrote {union_path}")

    # Write intersection
    intersection_path = os.path.join(output_dir, "intersection_passed.json")
    with open(intersection_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Tasks passed by ALL models",
                "task_ids": sorted(list(intersection_passed)),
                "count": len(intersection_passed),
                "total_tasks": total_tasks,
                "pass_rate": len(intersection_passed) / total_tasks if total_tasks > 0 else 0,
            },
            f,
            indent=2,
        )
    print(f"Wrote {intersection_path}")

    # Write summary
    summary_path = os.path.join(output_dir, "summary.json")
    summary = {
        "total_tasks": total_tasks,
        "models": {
            model: {
                "passed": len(passed),
                "pass_rate": len(passed) / total_tasks if total_tasks > 0 else 0,
            }
            for model, passed in results.items()
        },
        "union": {
            "passed": len(union_passed),
            "pass_rate": len(union_passed) / total_tasks if total_tasks > 0 else 0,
        },
        "intersection": {
            "passed": len(intersection_passed),
            "pass_rate": len(intersection_passed) / total_tasks if total_tasks > 0 else 0,
        },
        "venn_diagram": {
            "claude_only": len(
                results.get("claude", set())
                - results.get("gpt", set())
                - results.get("gemini", set())
            ),
            "gpt_only": len(
                results.get("gpt", set())
                - results.get("claude", set())
                - results.get("gemini", set())
            ),
            "gemini_only": len(
                results.get("gemini", set())
                - results.get("claude", set())
                - results.get("gpt", set())
            ),
            "claude_and_gpt": len(
                (results.get("claude", set()) & results.get("gpt", set()))
                - results.get("gemini", set())
            ),
            "claude_and_gemini": len(
                (results.get("claude", set()) & results.get("gemini", set()))
                - results.get("gpt", set())
            ),
            "gpt_and_gemini": len(
                (results.get("gpt", set()) & results.get("gemini", set()))
                - results.get("claude", set())
            ),
            "all_three": len(intersection_passed),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
