"""Convert MCP Atlas CSV results to pass@k compatible format."""

import csv
import json
import os
import sys
from typing import Any, Dict, List

# Increase CSV field size limit for large trajectory fields
csv.field_size_limit(10 * 1024 * 1024)

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from evaluation.pass_k_metrics import pass_at_k, pass_hat_k  # noqa: E402
from evaluation.scoring import Checkpoint, Result  # noqa: E402

# MCP Atlas pass threshold (coverage_score >= 0.75 is a pass)
PASS_THRESHOLD = 0.75

# New comments


def load_scores_csv(scores_csv: str) -> List[Dict[str, Any]]:
    """
    Load scored results from MCP Atlas mcp_evals_scores.py output.

    Expected columns:
    - TASK: task identifier
    - PROMPT: original prompt
    - coverage_score: float 0-1 from LLM-as-judge
    - (other columns from completion)

    Args:
        scores_csv: Path to scores CSV file or directory containing scored_*.csv

    Returns:
        List of row dictionaries
    """
    # Handle case where scores_csv is a directory (mcp_evals_scores.py creates a directory)
    if os.path.isdir(scores_csv):
        # Look for scored_*.csv file inside
        import glob

        scored_files = glob.glob(os.path.join(scores_csv, "scored_*.csv"))
        if not scored_files:
            raise FileNotFoundError(f"No scored_*.csv file found in {scores_csv}")
        scores_csv = scored_files[0]
        print(f"Found scores file: {scores_csv}")

    results = []
    with open(scores_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def convert_to_checkpoint_result(
    coverage_score: float,
    pass_threshold: float = PASS_THRESHOLD,
) -> Result:
    """
    Convert MCP Atlas coverage score to our Checkpoint/Result format.

    MCP Atlas uses:
    - coverage_score: float 0-1 (average of claim scores)
    - pass: coverage_score >= 0.75

    We convert to:
    - Checkpoint(total=1, result=1 if pass else 0) for binary pass/fail
    - Or Checkpoint(total=100, result=int(coverage_score*100)) for granular scoring

    Args:
        coverage_score: Float 0-1 from MCP Atlas
        pass_threshold: Threshold for pass (default 0.75)

    Returns:
        Result object compatible with evaluation/scoring.py
    """
    # Binary pass/fail checkpoint
    passed = coverage_score >= pass_threshold
    checkpoints = [Checkpoint(total=1, result=1 if passed else 0)]

    return Result(checkpoints)


def convert_to_granular_result(coverage_score: float) -> Result:
    """
    Convert coverage score to granular checkpoint (0-100 scale).

    This preserves more information than binary pass/fail.

    Args:
        coverage_score: Float 0-1 from MCP Atlas

    Returns:
        Result with checkpoint on 0-100 scale
    """
    # Scale to 0-100 for granular scoring
    score_100 = int(round(coverage_score * 100))
    checkpoints = [Checkpoint(total=100, result=score_100)]

    return Result(checkpoints)


def convert_mcp_atlas_results(
    scores_csv: str,
    output_dir: str,
    granular: bool = False,
    experiment_idx: int = 0,
) -> List[Dict[str, Any]]:
    """
    Convert MCP Atlas scored CSV to per-task result.json files.

    Args:
        scores_csv: Path to scores CSV from mcp_evals_scores.py
        output_dir: Directory to write result.json files
        granular: If True, use 0-100 scale instead of binary pass/fail
        experiment_idx: Index of this experiment run (for pass^k with multiple runs)

    Returns:
        List of converted results for further processing
    """
    rows = load_scores_csv(scores_csv)
    converted_results = []

    for row in rows:
        task_id = row.get("TASK", "unknown")
        coverage_score_str = row.get("coverage_score", "0")

        # Parse coverage score
        try:
            coverage_score = float(coverage_score_str)
        except (ValueError, TypeError):
            coverage_score = 0.0

        # Convert to our format
        if granular:
            result = convert_to_granular_result(coverage_score)
        else:
            result = convert_to_checkpoint_result(coverage_score)

        # Create task output directory
        task_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)

        # Write result with experiment index for pass^k support
        # result.json is kept for backwards compatibility (last experiment)
        # result_exp{N}.json stores each experiment separately
        result_path = os.path.join(task_dir, f"result_exp{experiment_idx}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        # Also write result.json for backwards compatibility
        compat_path = os.path.join(task_dir, "result.json")
        with open(compat_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        converted_results.append(
            {
                "task_id": task_id,
                "coverage_score": coverage_score,
                "passed": coverage_score >= PASS_THRESHOLD,
                "result": result.to_dict(),
                "result_path": result_path,
                "experiment_idx": experiment_idx,
            }
        )

    return converted_results


def aggregate_results_for_pass_k(
    converted_results: List[Dict[str, Any]],
) -> Dict[str, List[Dict]]:
    """
    Aggregate results by task_id for pass@k calculation.

    Groups multiple runs of the same task together.

    Args:
        converted_results: List of converted results from convert_mcp_atlas_results

    Returns:
        Dict mapping task_id to list of results for that task
    """
    by_task: Dict[str, List[Dict]] = {}

    for result in converted_results:
        task_id = result["task_id"]
        if task_id not in by_task:
            by_task[task_id] = []
        by_task[task_id].append(result)

    return by_task


def write_evaluation_results_json(
    converted_results: List[Dict[str, Any]],
    output_dir: str,
    model_name: str,
    format_name: str,
) -> str:
    """
    Write aggregated evaluation results in format expected by pass_k_metrics.

    Args:
        converted_results: List of converted results
        output_dir: Directory for output
        model_name: Model identifier
        format_name: Experiment format name

    Returns:
        Path to written JSON file
    """
    # Group by task
    by_task = aggregate_results_for_pass_k(converted_results)

    # Format for pass_k_metrics
    evaluation_data = {
        "model": model_name,
        "format": format_name,
        "tasks": {},
    }

    for task_id, task_results in by_task.items():
        evaluation_data["tasks"][task_id] = {
            "n_trials": len(task_results),
            "n_success": sum(1 for r in task_results if r["passed"]),
            "results": [r["result"] for r in task_results],
            "coverage_scores": [r["coverage_score"] for r in task_results],
        }

    # Write to file
    output_path = os.path.join(output_dir, f"evaluation_{format_name}_{model_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_data, f, indent=2)

    return output_path


def compute_pass_k_metrics(
    converted_results: List[Dict[str, Any]],
    max_k: int = 3,
) -> Dict[str, Any]:
    """
    Compute pass^k and pass@k metrics from converted results.

    Uses unbiased combinatorial estimators (same as TAC/SWE-Bench pipelines):
        pass^k = C(c, k) / C(n, k)     — P(all k succeed)
        pass@k = 1 - C(n-c, k) / C(n, k) — P(at least 1 of k succeeds)

    where n = total trials, c = successful trials.

    Args:
        converted_results: List of converted results from convert_mcp_atlas_results
        max_k: Maximum k to compute (will compute pass^1 through pass^max_k)

    Returns:
        Dictionary with pass metrics
    """
    # Group by task
    by_task = aggregate_results_for_pass_k(converted_results)

    # Compute metrics for each task
    task_metrics = []
    for task_id, task_results in by_task.items():
        n = len(task_results)
        c = sum(1 for r in task_results if r["passed"])

        metrics = {
            "task_id": task_id,
            "n_trials": n,
            "n_success": c,
            "pass_rate": c / n if n > 0 else 0.0,
        }

        for k in range(1, max_k + 1):
            if k <= n:
                metrics[f"pass^{k}"] = pass_hat_k(n, c, k)
                metrics[f"pass@{k}"] = pass_at_k(n, c, k)

        task_metrics.append(metrics)

    # Compute aggregate metrics
    n_tasks = len(task_metrics)
    if n_tasks == 0:
        return {"n_tasks": 0}

    aggregate = {
        "n_tasks": n_tasks,
        "total_trials": sum(m["n_trials"] for m in task_metrics),
        "total_successes": sum(m["n_success"] for m in task_metrics),
        "avg_pass_rate": sum(m["pass_rate"] for m in task_metrics) / n_tasks,
    }

    # Average pass^k and pass@k only over tasks with enough trials (n >= k)
    for k in range(1, max_k + 1):
        eligible_hat = [m[f"pass^{k}"] for m in task_metrics if f"pass^{k}" in m]
        aggregate[f"pass^{k}"] = sum(eligible_hat) / len(eligible_hat) if eligible_hat else None
        eligible_at = [m[f"pass@{k}"] for m in task_metrics if f"pass@{k}" in m]
        aggregate[f"pass@{k}"] = sum(eligible_at) / len(eligible_at) if eligible_at else None

    aggregate["task_metrics"] = task_metrics

    return aggregate


def display_pass_k_summary(metrics: Dict[str, Any], max_k: int = 3) -> None:
    """
    Display pass^k metrics in τ-bench style.

    Args:
        metrics: Output from compute_pass_k_metrics
        max_k: Maximum k to display
    """
    if metrics.get("n_tasks", 0) == 0:
        print("No results to display.")
        return

    print("\n" + "=" * 60)
    print("📊 PASS METRICS (τ-bench style)")
    print("=" * 60)
    print(
        f"Tasks: {metrics['n_tasks']} | "
        f"Total trials: {metrics['total_trials']} | "
        f"Successes: {metrics['total_successes']}"
    )
    print("-" * 60)

    # Header
    header = f"| {'Metric':^15} |"
    for k in range(1, max_k + 1):
        header += f" k={k:^5} |"
    print(header)
    print("|" + "-" * 17 + "|" + ("---------+" * max_k))

    # pass^k row (reliability - all k must succeed)
    row = f"| {'pass^k':^15} |"
    for k in range(1, max_k + 1):
        val = metrics.get(f"pass^{k}")
        row += f" {'--':^7} |" if val is None else f" {val:^7.3f} |"
    print(row)

    # pass@k row (capability - at least 1 of k succeeds)
    row = f"| {'pass@k':^15} |"
    for k in range(1, max_k + 1):
        val = metrics.get(f"pass@{k}")
        row += f" {'--':^7} |" if val is None else f" {val:^7.3f} |"
    print(row)

    print("=" * 60)
    print("  pass^k = 1 if ALL trials pass | pass@k = 1 if ANY trial passes")
    print()


def write_pass_k_csv(
    metrics: Dict[str, Any],
    output_path: str,
    max_k: int = 3,
) -> str:
    """
    Write per-task pass^k and pass@k metrics to CSV.

    Args:
        metrics: Output from compute_pass_k_metrics
        output_path: Path to write CSV file
        max_k: Maximum k to include in columns

    Returns:
        Path to written CSV file
    """
    task_metrics = metrics.get("task_metrics", [])
    if not task_metrics:
        print("Warning: No task metrics to write")
        return output_path

    # Build headers
    headers = ["task_id", "n_trials", "n_success", "pass_rate"]
    for k in range(1, max_k + 1):
        headers.append(f"pass^{k}")
    for k in range(1, max_k + 1):
        headers.append(f"pass@{k}")

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        for task in task_metrics:
            row = {
                "task_id": task["task_id"],
                "n_trials": task["n_trials"],
                "n_success": task["n_success"],
                "pass_rate": f"{task['pass_rate']:.4f}",
            }
            for k in range(1, max_k + 1):
                row[f"pass^{k}"] = f"{task.get(f'pass^{k}', 0):.4f}"
                row[f"pass@{k}"] = f"{task.get(f'pass@{k}', 0):.4f}"
            writer.writerow(row)

        # Write aggregate row
        agg_row = {
            "task_id": "AGGREGATE",
            "n_trials": metrics.get("total_trials", 0),
            "n_success": metrics.get("total_successes", 0),
            "pass_rate": f"{metrics.get('avg_pass_rate', 0):.4f}",
        }
        for k in range(1, max_k + 1):
            agg_row[f"pass^{k}"] = f"{metrics.get(f'pass^{k}', 0):.4f}"
            agg_row[f"pass@{k}"] = f"{metrics.get(f'pass@{k}', 0):.4f}"
        writer.writerow(agg_row)

    print(f"Wrote per-task pass^k/pass@k CSV: {output_path}")
    return output_path


def load_completion_csv(completion_csv: str) -> List[Dict[str, Any]]:
    """
    Load completion results from MCP Atlas mcp_completion_script.py output.

    Useful for extracting trajectory and tool call information.

    Args:
        completion_csv: Path to completions CSV file

    Returns:
        List of row dictionaries
    """
    results = []
    with open(completion_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def merge_completion_and_scores(
    completion_csv: str,
    scores_csv: str,
) -> List[Dict[str, Any]]:
    """
    Merge completion and score CSVs by task ID.

    Expects single-trial CSVs (one row per TASK). Called once per experiment
    by convert_mcp_atlas_results; multi-trial aggregation happens upstream.

    Args:
        completion_csv: Path to completions CSV (single trial)
        scores_csv: Path to scores CSV (single trial)

    Returns:
        List of merged dictionaries
    """
    completion_rows = list(load_completion_csv(completion_csv))
    score_rows = list(load_scores_csv(scores_csv))

    if len(completion_rows) != len({r["TASK"] for r in completion_rows}):
        print("WARNING: Duplicate TASK entries in completions CSV — only last per task is kept")
    if len(score_rows) != len({r["TASK"] for r in score_rows}):
        print("WARNING: Duplicate TASK entries in scores CSV — only last per task is kept")

    completions = {row["TASK"]: row for row in completion_rows}
    scores = {row["TASK"]: row for row in score_rows}

    merged = []
    for task_id in scores:
        row = {**completions.get(task_id, {}), **scores[task_id]}
        merged.append(row)

    return merged
