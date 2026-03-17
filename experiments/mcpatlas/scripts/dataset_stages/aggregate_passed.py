"""Aggregate passed tasks from baseline runs."""

import csv
import json
from pathlib import Path
from typing import Dict, Set

# Handle large CSV fields
csv.field_size_limit(10 * 1024 * 1024)


def get_passed_tasks_from_csv(csv_path: Path, pass_threshold: float) -> Set[str]:
    """
    Get task IDs that passed from a pass_k_per_task.csv file.

    Args:
        csv_path: Path to pass_k_per_task.csv file
        pass_threshold: Threshold for considering a task as passed

    Returns:
        Set of task IDs that passed
    """
    tasks = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            task_id = row["task_id"]
            try:
                score = float(row["pass@3"])
            except (ValueError, TypeError):
                score = 0.0

            if task_id not in tasks:
                tasks[task_id] = []
            tasks[task_id].append(score)

    # Task passes if ANY run passes
    passed = {tid for tid, scores in tasks.items() if any(s >= pass_threshold for s in scores)}

    return passed


def aggregate_passed_tasks(
    run_dirs: Dict[str, Path], pass_threshold: float, strategy: str, output_dir: Path
) -> Dict:
    """
    Aggregate passed tasks from baseline runs.

    Args:
        run_dirs: Dict mapping model names to run directory paths
        pass_threshold: Threshold for considering a task as passed
        strategy: Aggregation strategy ("union" or "intersection")
        output_dir: Directory to write output files

    Returns:
        Dict containing:
            - per_model: Dict[model_name, Set[task_ids]]
            - union: Set[task_ids]
            - intersection: Set[task_ids]
            - all_task_ids: Set of all task IDs seen across all models
            - summary: Summary statistics

    Outputs:
        - {output_dir}/per_model/{model}_passed.json
        - {output_dir}/union_passed.json
        - {output_dir}/intersection_passed.json
        - {output_dir}/summary.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir = output_dir / "per_model"
    per_model_dir.mkdir(exist_ok=True)

    # Model full names for output
    model_full_names = {
        "claude": "claude-opus-4-5-20251101",
        "gpt": "gpt-5.2",
        "gemini": "gemini-3-pro-preview",
    }

    # Get passed tasks for each model
    results = {}
    all_task_ids = set()

    for model, run_dir in run_dirs.items():
        pass_k_file = run_dir / "combined" / "pass_k_per_task.csv"

        if not pass_k_file.exists():
            raise FileNotFoundError(f"Pass k file not found for {model}: {pass_k_file}")

        # Get passed tasks
        passed = get_passed_tasks_from_csv(pass_k_file, pass_threshold)
        results[model] = passed

        # Collect all task IDs
        with open(pass_k_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_task_ids.add(row["task_id"])

    # Compute union and intersection
    all_passed_sets = list(results.values())
    union_passed = set.union(*all_passed_sets) if all_passed_sets else set()
    intersection_passed = set.intersection(*all_passed_sets) if all_passed_sets else set()

    total_tasks = len(all_task_ids)

    # Write individual model results
    for model, passed in results.items():
        output_path = per_model_dir / f"{model}_passed.json"
        data = {
            "model": model_full_names.get(model, model),
            "task_ids": sorted(list(passed)),
            "count": len(passed),
            "total_tasks": total_tasks,
            "pass_rate": len(passed) / total_tasks if total_tasks > 0 else 0,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # Write union
    union_path = output_dir / "union_passed.json"
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

    # Write intersection
    intersection_path = output_dir / "intersection_passed.json"
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

    # Compute Venn diagram stats
    venn_diagram = {}
    if len(results) == 3 and "claude" in results and "gpt" in results and "gemini" in results:
        venn_diagram = {
            "claude_only": len(results["claude"] - results["gpt"] - results["gemini"]),
            "gpt_only": len(results["gpt"] - results["claude"] - results["gemini"]),
            "gemini_only": len(results["gemini"] - results["claude"] - results["gpt"]),
            "claude_and_gpt": len((results["claude"] & results["gpt"]) - results["gemini"]),
            "claude_and_gemini": len((results["claude"] & results["gemini"]) - results["gpt"]),
            "gpt_and_gemini": len((results["gpt"] & results["gemini"]) - results["claude"]),
            "all_three": len(intersection_passed),
        }

    # Write summary
    summary = {
        "pass_threshold": pass_threshold,
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
    }

    if venn_diagram:
        summary["venn_diagram"] = venn_diagram

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Return structured results
    return {
        "per_model": results,
        "union": union_passed,
        "intersection": intersection_passed,
        "all_task_ids": all_task_ids,
        "summary": summary,
    }
