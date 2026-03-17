"""Extract baseline results for specific tasks."""

import csv
import json
from pathlib import Path
from typing import Dict, Set

# Handle large CSV fields
csv.field_size_limit(10 * 1024 * 1024)


def extract_pass_k_results(pass_k_csv: Path, task_ids: Set[str], output_csv: Path) -> int:
    """
    Extract pass@k results for specific tasks.

    Args:
        pass_k_csv: Path to pass_k_per_task.csv file
        task_ids: Set of task IDs to extract
        output_csv: Path to write filtered results

    Returns:
        Number of rows extracted
    """
    matching_rows = []

    with open(pass_k_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            if row["task_id"] in task_ids:
                matching_rows.append(row)

    # Write filtered results
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matching_rows)

    return len(matching_rows)


def extract_scored_results(scored_csv: Path, task_ids: Set[str], output_csv: Path) -> int:
    """
    Extract scored results for specific tasks.

    Args:
        scored_csv: Path to combined_scored.csv file
        task_ids: Set of task IDs to extract
        output_csv: Path to write filtered results

    Returns:
        Number of rows extracted
    """
    matching_rows = []

    with open(scored_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            if row["TASK"] in task_ids:
                matching_rows.append(row)

    # Write filtered results
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matching_rows)

    return len(matching_rows)


def extract_baseline_results(
    run_dirs: Dict[str, Path], task_ids: Set[str], output_dir: Path
) -> dict:
    """
    Extract baseline results for specific tasks from model runs.

    Args:
        run_dirs: Dict mapping model names to run directory paths
        task_ids: Set of task IDs to extract results for
        output_dir: Directory to write extracted results

    Returns:
        Dict containing:
            - models: List of models processed
            - tasks_extracted: Total unique tasks extracted
            - per_model_stats: Dict of per-model statistics

    Outputs:
        - {output_dir}/per_model/{model}_pass_k_per_task.csv
        - {output_dir}/per_model/{model}_combined_scored.csv
        - {output_dir}/extraction_summary.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir = output_dir / "per_model"
    per_model_dir.mkdir(exist_ok=True)

    results_summary = {}

    for model, run_dir in run_dirs.items():
        model_dir = run_dir / "combined"
        model_stats = {}

        # Extract pass_k_per_task.csv
        pass_k_input = model_dir / "pass_k_per_task.csv"
        pass_k_output = per_model_dir / f"{model}_pass_k_per_task.csv"

        if pass_k_input.exists():
            count = extract_pass_k_results(pass_k_input, task_ids, pass_k_output)
            model_stats["pass_k_tasks"] = count
        else:
            model_stats["pass_k_error"] = "pass_k file not found"

        # Extract combined_scored.csv
        scored_input = model_dir / "combined_scored.csv"
        scored_output = per_model_dir / f"{model}_combined_scored.csv"

        if scored_input.exists():
            row_count = extract_scored_results(scored_input, task_ids, scored_output)
            model_stats["scored_rows"] = row_count
        else:
            model_stats["scored_error"] = "scored file not found"

        results_summary[model] = model_stats

    # Write summary
    summary = {
        "num_tasks": len(task_ids),
        "task_ids": sorted(list(task_ids)),
        "models": results_summary,
    }

    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "models": list(run_dirs.keys()),
        "tasks_extracted": len(task_ids),
        "per_model_stats": results_summary,
    }
