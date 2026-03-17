"""
pass^k and pass@k metrics for agent evaluation.

pass^k = P(all k trials succeed)     - from τ-bench (Yao et al., 2024)
pass@k = P(at least 1 of k succeeds) - from HumanEval (Chen et al., 2021)

For empirical estimation, we use unbiased estimators:
- pass^k = C(c, k) / C(n, k)  # Matches official τ-bench implementation
- pass@k = 1 - C(n-c, k) / C(n, k)

where n = total trials, c = successful trials, k = sample size.

NOTE: τ-bench only reports pass^k (reliability metric). pass@k is included here
for comparison with code generation benchmarks but is NOT used by τ-bench.

References:
- τ-bench: https://arxiv.org/abs/2406.12045
- HumanEval: https://arxiv.org/abs/2107.03374 (pass@k origin)
"""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrialResult:
    """Result from a single trial."""

    success: bool
    score: float = 0.0
    total: float = 1.0
    result_path: str = ""


@dataclass
class TaskTrialResults:
    """Aggregated results for a task across multiple trials."""

    task_name: str
    trials: List[TrialResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        """Total number of trials."""
        return len(self.trials)

    @property
    def c(self) -> int:
        """Number of successful trials."""
        return sum(1 for t in self.trials if t.success)

    @property
    def success_rate(self) -> float:
        """Empirical success rate (pass^1)."""
        return self.c / self.n if self.n > 0 else 0.0


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k.

    From "Evaluating Large Language Models Trained on Code" (Chen et al., 2021):
    The probability that at least one of k samples succeeds.

    Args:
        n: total number of trials
        c: number of successful trials
        k: target number of samples

    Returns:
        Probability that at least one of k samples succeeds

    Examples:
        >>> pass_at_k(10, 5, 3)  # 50% success rate, k=3
        0.9167
        >>> pass_at_k(10, 10, 3)  # 100% success rate
        1.0
        >>> pass_at_k(10, 0, 3)  # 0% success rate
        0.0
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n < k:
        # Not enough trials to estimate pass@k
        # Fall back to empirical estimate: 1 - (1 - c/n)^k
        p = c / n if n > 0 else 0.0
        return 1.0 - (1.0 - p) ** k
    if n - c < k:
        # Enough successes that any k samples must include at least one success
        return 1.0
    if c == 0:
        return 0.0

    # Unbiased estimator: 1 - C(n-c, k) / C(n, k)
    # Which equals: 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def pass_hat_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass^k (pass-hat-k).

    From τ-bench (Yao et al., 2024): The probability that ALL k samples succeed.

    This uses the combinatorial estimator:
        pass^k = C(c, k) / C(n, k)

    Which is the probability of picking k successes when sampling k from n trials
    without replacement.

    Verified against official τ-bench implementation.

    Args:
        n: total number of trials
        c: number of successful trials
        k: target number of samples

    Returns:
        Probability that all k samples succeed

    Examples:
        >>> pass_hat_k(10, 5, 3)  # 50% success rate, k=3
        0.0833...  # C(5,3) / C(10,3) = 10/120
        >>> pass_hat_k(10, 10, 3)  # 100% success rate
        1.0
        >>> pass_hat_k(10, 0, 3)  # 0% success rate
        0.0
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n < k:
        # Not enough trials to estimate pass^k
        # Fall back to empirical estimate: (c/n)^k
        p = c / n if n > 0 else 0.0
        return p**k
    if c < k:
        # Not enough successes for all k to succeed
        return 0.0
    if c == n:
        # All trials succeeded
        return 1.0

    # Unbiased estimator: C(c, k) / C(n, k)
    return math.comb(c, k) / math.comb(n, k)


def compute_task_metrics(results: TaskTrialResults, k: int = 3) -> Dict[str, Any]:
    """
    Compute all pass metrics for a single task.

    Args:
        results: Aggregated trial results for the task
        k: Target number of samples for pass^k and pass@k

    Returns:
        Dictionary containing:
        - task: task name
        - n_trials: number of trials
        - n_success: number of successful trials
        - pass_1: success rate (same as pass^1)
        - pass_k: pass^k value
        - pass_at_k: pass@k value
        - avg_score: average score across trials
        - max_score: maximum score across trials
        - result_paths: semicolon-separated list of result paths
    """
    n, c = results.n, results.c
    trials = results.trials

    # Compute scores
    scores = [t.score for t in trials]
    totals = [t.total for t in trials]

    # Compute percentages (score/total * 100)
    percents = [(s / t * 100) if t > 0 else 0.0 for s, t in zip(scores, totals)]

    return {
        "task": results.task_name,
        "n_trials": n,
        "n_success": c,
        "pass_1": results.success_rate,
        f"pass_{k}": pass_hat_k(n, c, k),
        f"pass_at_{k}": pass_at_k(n, c, k),
        "avg_score": sum(scores) / n if n > 0 else 0.0,
        "avg_total": sum(totals) / n if n > 0 else 0.0,
        "avg_percent": sum(percents) / n if n > 0 else 0.0,
        "max_score": max(scores) if scores else 0.0,
        "max_percent": max(percents) if percents else 0.0,
        "result_paths": ";".join(t.result_path for t in trials),
    }


def aggregate_pass_metrics(task_results: List[TaskTrialResults], k: int = 3) -> Dict[str, Any]:
    """
    Aggregate pass metrics across all tasks.

    Args:
        task_results: List of per-task trial results
        k: Target number of samples for pass^k and pass@k

    Returns:
        Dictionary containing:
        - n_tasks: number of tasks
        - avg_pass_1: average pass^1 across tasks
        - avg_pass_k: average pass^k across tasks
        - avg_pass_at_k: average pass@k across tasks
        - task_details: list of per-task metrics
    """
    if not task_results:
        return {"n_tasks": 0, "avg_pass_1": 0.0, f"avg_pass_{k}": 0.0, f"avg_pass_at_{k}": 0.0}

    # Compute per-task metrics
    task_metrics = [compute_task_metrics(r, k) for r in task_results]
    n_tasks = len(task_metrics)

    return {
        "n_tasks": n_tasks,
        "avg_pass_1": sum(m["pass_1"] for m in task_metrics) / n_tasks,
        f"avg_pass_{k}": sum(m[f"pass_{k}"] for m in task_metrics) / n_tasks,
        f"avg_pass_at_{k}": sum(m[f"pass_at_{k}"] for m in task_metrics) / n_tasks,
        "avg_score": sum(m["avg_score"] for m in task_metrics) / n_tasks,
        "avg_percent": sum(m["avg_percent"] for m in task_metrics) / n_tasks,
        "task_details": task_metrics,
    }


def load_evaluation_results(
    final_results_dir: str, task_names: List[str], model_short: str
) -> List[TaskTrialResults]:
    """
    Load evaluation results from deterministic evaluation JSON files.

    Args:
        final_results_dir: Path to final_results directory
        task_names: List of task names to load
        model_short: Model short name (e.g., "claude-opus-4-5-20251101")

    Returns:
        List of TaskTrialResults, one per task
    """
    results = []

    for task_name in task_names:
        deterministic_dir = os.path.join(final_results_dir, "deterministic", task_name, model_short)

        if not os.path.exists(deterministic_dir):
            continue

        trials = []
        for fname in sorted(os.listdir(deterministic_dir)):
            if not fname.endswith(".json"):
                continue

            fpath = os.path.join(deterministic_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                score = data.get("final_score", {}).get("result", 0)
                total = data.get("final_score", {}).get("total", 1)

                # Consider success if score == total (all checkpoints passed)
                success = score == total and total > 0

                trials.append(
                    TrialResult(success=success, score=score, total=total, result_path=fpath)
                )
            except Exception as e:
                print(f"Warning: Failed to load {fpath}: {e}")
                continue

        if trials:
            results.append(TaskTrialResults(task_name=task_name, trials=trials))

    return results


def display_pass_k_metrics(
    task_results: List[TaskTrialResults],
    max_k: int = 3,
) -> None:
    """
    Display pass^k and pass@k metrics to console in τ-bench style.

    Args:
        task_results: List of per-task trial results
        max_k: Maximum k to display (will show pass^1 through pass^max_k)
    """
    if not task_results:
        print("No results to display.")
        return

    # Compute pass^k for each k from 1 to max_k
    n_tasks = len(task_results)
    total_trials = sum(r.n for r in task_results)
    total_successes = sum(r.c for r in task_results)

    # Compute average scores for partial credit view
    avg_scores = [sum(t.score for t in r.trials) / r.n if r.n > 0 else 0.0 for r in task_results]
    avg_totals = [sum(t.total for t in r.trials) / r.n if r.n > 0 else 0.0 for r in task_results]
    overall_avg_score = sum(avg_scores) / n_tasks if n_tasks > 0 else 0.0
    overall_avg_total = sum(avg_totals) / n_tasks if n_tasks > 0 else 0.0
    overall_avg_percent = (
        (overall_avg_score / overall_avg_total * 100) if overall_avg_total > 0 else 0.0
    )

    print("\n" + "=" * 60)
    print("📊 PASS METRICS (τ-bench style)")
    print("=" * 60)
    print(f"Tasks: {n_tasks} | Total trials: {total_trials} | Binary successes: {total_successes}")
    print(
        f"Avg score: {overall_avg_score:.2f}/{overall_avg_total:.2f} ({overall_avg_percent:.1f}%)"
    )
    print("-" * 60)

    # Header row
    header = "| {:^20} |".format("Metric")
    for k in range(1, max_k + 1):
        header += " {:^8} |".format(f"k={k}")
    print(header)
    print("|" + "-" * 22 + "|" + ("----------+" * max_k))

    # Average pass^k across tasks (τ-bench style - reliability metric)
    row = "| {:^20} |".format("Avg pass^k")
    for k in range(1, max_k + 1):
        task_pass_k_values = []
        for r in task_results:
            if r.n >= k:
                task_pass_k_values.append(pass_hat_k(r.n, r.c, k))
            else:
                p = r.c / r.n if r.n > 0 else 0.0
                task_pass_k_values.append(p**k)
        avg_pass_k = (
            sum(task_pass_k_values) / len(task_pass_k_values) if task_pass_k_values else 0.0
        )
        row += " {:^8.3f} |".format(avg_pass_k)
    print(row)

    # Average pass@k across tasks (capability metric - at least 1 success in k tries)
    row = "| {:^20} |".format("Avg pass@k")
    for k in range(1, max_k + 1):
        task_pass_at_k_values = []
        for r in task_results:
            if r.n >= k:
                task_pass_at_k_values.append(pass_at_k(r.n, r.c, k))
            else:
                p = r.c / r.n if r.n > 0 else 0.0
                task_pass_at_k_values.append(1.0 - (1.0 - p) ** k)
        avg_pass_at_k = (
            sum(task_pass_at_k_values) / len(task_pass_at_k_values)
            if task_pass_at_k_values
            else 0.0
        )
        row += " {:^8.3f} |".format(avg_pass_at_k)
    print(row)

    print("=" * 60)
    print("  pass^k = P(all k succeed)    | pass@k = P(≥1 of k succeeds)")
    print()


def write_pass_k_summary_csv(
    final_results_dir: str,
    task_names: List[str],
    format_name: str,
    model_short: str,
    k: int = 3,
) -> Optional[str]:
    """
    Write summary CSV with pass^k metrics and display to console.

    Args:
        final_results_dir: Path to final_results directory
        task_names: List of task names
        format_name: Experiment format name
        model_short: Model short name
        k: Target k for pass^k and pass@k (also max_k for display)

    Returns:
        Path to written CSV file, or None if failed
    """
    import csv

    # Load results
    task_results = load_evaluation_results(final_results_dir, task_names, model_short)

    if not task_results:
        print(f"Warning: No results found for {model_short}")
        return None

    # Display pass^k metrics to console (τ-bench style)
    display_pass_k_metrics(task_results, max_k=k)

    # Compute metrics
    aggregated = aggregate_pass_metrics(task_results, k=k)

    # Write CSV with pass^1 through pass^k and pass@1 through pass@k columns
    summary_dir = os.path.join(final_results_dir, "summaries")
    os.makedirs(summary_dir, exist_ok=True)

    csv_path = os.path.join(summary_dir, f"pass_k_summary_{format_name}_{model_short}.csv")

    # Build headers with all pass^k and pass@k values from 1 to k
    headers = ["task", "n_trials", "n_success"]
    for ki in range(1, k + 1):
        headers.append(f"pass^{ki}")  # pass^k (reliability)
    for ki in range(1, k + 1):
        headers.append(f"pass@{ki}")  # pass@k (capability)
    headers.extend(
        ["avg_score", "avg_total", "avg_percent", "max_score", "max_percent", "result_paths"]
    )

    try:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            # Per-task rows with all pass^k and pass@k values
            for r in task_results:
                row = [r.task_name, r.n, r.c]
                # pass^k columns
                for ki in range(1, k + 1):
                    if r.n >= ki:
                        row.append(f"{pass_hat_k(r.n, r.c, ki):.4f}")
                    else:
                        p = r.c / r.n if r.n > 0 else 0.0
                        row.append(f"{p ** ki:.4f}")
                # pass@k columns
                for ki in range(1, k + 1):
                    if r.n >= ki:
                        row.append(f"{pass_at_k(r.n, r.c, ki):.4f}")
                    else:
                        p = r.c / r.n if r.n > 0 else 0.0
                        row.append(f"{1.0 - (1.0 - p) ** ki:.4f}")

                # Compute scores for this task
                scores = [t.score for t in r.trials]
                totals = [t.total for t in r.trials]
                percents = [(s / t * 100) if t > 0 else 0.0 for s, t in zip(scores, totals)]

                row.extend(
                    [
                        f"{sum(scores) / r.n:.2f}" if r.n > 0 else "0.00",
                        f"{sum(totals) / r.n:.2f}" if r.n > 0 else "0.00",
                        f"{sum(percents) / r.n:.2f}" if r.n > 0 else "0.00",
                        f"{max(scores):.2f}" if scores else "0.00",
                        f"{max(percents):.2f}" if percents else "0.00",
                        ";".join(t.result_path for t in r.trials),
                    ]
                )
                writer.writerow(row)

            # Aggregate row
            total_n = sum(r.n for r in task_results)
            total_c = sum(r.c for r in task_results)
            agg_row = ["AGGREGATE", total_n, total_c]
            # Aggregate pass^k
            for ki in range(1, k + 1):
                task_pass_k_values = []
                for r in task_results:
                    if r.n >= ki:
                        task_pass_k_values.append(pass_hat_k(r.n, r.c, ki))
                    else:
                        p = r.c / r.n if r.n > 0 else 0.0
                        task_pass_k_values.append(p**ki)
                avg = (
                    sum(task_pass_k_values) / len(task_pass_k_values) if task_pass_k_values else 0.0
                )
                agg_row.append(f"{avg:.4f}")
            # Aggregate pass@k
            for ki in range(1, k + 1):
                task_pass_at_k_values = []
                for r in task_results:
                    if r.n >= ki:
                        task_pass_at_k_values.append(pass_at_k(r.n, r.c, ki))
                    else:
                        p = r.c / r.n if r.n > 0 else 0.0
                        task_pass_at_k_values.append(1.0 - (1.0 - p) ** ki)
                avg = (
                    sum(task_pass_at_k_values) / len(task_pass_at_k_values)
                    if task_pass_at_k_values
                    else 0.0
                )
                agg_row.append(f"{avg:.4f}")
            agg_row.extend(
                [
                    f"{aggregated.get('avg_score', 0):.2f}",
                    "",
                    f"{aggregated.get('avg_percent', 0):.2f}",
                    "",
                    "",
                    "",
                ]
            )
            writer.writerow(agg_row)

        print(f"Wrote pass^k/pass@k summary CSV: {csv_path}")
        return csv_path

    except Exception as e:
        print(f"Warning: Failed to write pass^k summary CSV: {e}")
        return None


if __name__ == "__main__":
    # Quick test
    print("Testing pass@k and pass^k estimators...")

    # Test cases from τ-bench paper
    test_cases = [
        (10, 5, 3),  # 50% success rate
        (10, 10, 3),  # 100% success rate
        (10, 0, 3),  # 0% success rate
        (3, 2, 3),  # 66.7% success rate, n=k
        (5, 3, 3),  # 60% success rate
    ]

    for n, c, k in test_cases:
        p = c / n if n > 0 else 0.0
        print(f"\nn={n}, c={c}, k={k} (empirical p={p:.2f})")
        print(f"  pass@{k} = {pass_at_k(n, c, k):.4f} (expected: {1 - (1-p)**k:.4f})")
        print(f"  pass^{k} = {pass_hat_k(n, c, k):.4f} (expected: {p**k:.4f})")
