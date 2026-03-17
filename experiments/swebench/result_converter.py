"""
Convert SWE-agent results to pass@k evaluation format.

This module provides utilities to:
1. Parse SWE-agent output directories
2. Compute pass@k and pass^k metrics
3. Generate summary CSV files compatible with TAC evaluation
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.pass_k_metrics import pass_at_k, pass_hat_k  # noqa: E402


def parse_sweagent_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Parse SWE-agent output directory for instance results.

    Args:
        results_dir: Path to SWE-agent output directory

    Returns:
        Dict mapping instance_id -> result info
    """
    results = {}

    # SWE-agent stores results in various formats
    # Check for results.json or individual instance files
    results_json = results_dir / "results.json"
    if results_json.exists():
        with open(results_json) as f:
            data = json.load(f)
            for instance_id, result in data.items():
                results[instance_id] = {
                    "resolved": result.get("resolved", False),
                    "exit_status": result.get("exit_status"),
                    "patch": result.get("patch"),
                }
        return results

    # Check for individual trajectory files
    for traj_file in results_dir.glob("*.traj"):
        instance_id = traj_file.stem
        try:
            with open(traj_file) as f:
                traj = json.load(f)
                results[instance_id] = {
                    "resolved": traj.get("info", {}).get("resolved", False),
                    "exit_status": traj.get("info", {}).get("exit_status"),
                    "patch": traj.get("info", {}).get("submission"),
                }
        except (json.JSONDecodeError, KeyError):
            results[instance_id] = {"resolved": False, "exit_status": "parse_error"}

    return results


def convert_sweagent_results(
    run_dir: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Convert SWE-agent run directory to unified results format.

    Args:
        run_dir: Path to run directory (contains exp_1/, exp_2/, etc.)
        output_path: Optional path to save results JSON

    Returns:
        Dict with instance results across all experiments
    """
    # Load config
    config_path = run_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    num_trials = config.get("num_trials", 3)

    # Aggregate results across experiments
    instance_results: Dict[str, List[bool]] = {}

    for exp_num in range(1, num_trials + 1):
        exp_dir = run_dir / f"exp_{exp_num}" / "results"
        if not exp_dir.exists():
            # Try alternative structure
            exp_dir = run_dir / f"exp_{exp_num}"

        if exp_dir.exists():
            exp_results = parse_sweagent_results(exp_dir)
            for instance_id, result in exp_results.items():
                if instance_id not in instance_results:
                    instance_results[instance_id] = []
                instance_results[instance_id].append(result.get("resolved", False))

    # Build output structure
    output = {
        "config": config,
        "num_trials": num_trials,
        "instances": {},
    }

    for instance_id, successes in instance_results.items():
        n = len(successes)
        c = sum(successes)
        output["instances"][instance_id] = {
            "trials": n,
            "successes": c,
            "success_rate": c / n if n > 0 else 0.0,
            "pass_at_1": pass_at_k(n, c, 1),
            "pass_at_3": pass_at_k(n, c, 3) if n >= 3 else None,
            "pass_hat_3": pass_hat_k(n, c, 3) if n >= 3 else None,
            "results": successes,
        }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    return output


def compute_pass_at_k(
    results: Dict[str, Any],
    k_values: List[int] = [1, 3, 5],
) -> Dict[str, Dict[str, float]]:
    """
    Compute pass@k metrics for all instances.

    Args:
        results: Output from convert_sweagent_results
        k_values: List of k values to compute

    Returns:
        Dict with per-instance and aggregate metrics
    """
    metrics = {
        "per_instance": {},
        "aggregate": {},
    }

    all_pass_at_k = {k: [] for k in k_values}
    all_pass_hat_k = {k: [] for k in k_values}

    for instance_id, data in results.get("instances", {}).items():
        n = data["trials"]
        c = data["successes"]

        instance_metrics = {}
        for k in k_values:
            if n >= k:
                pak = pass_at_k(n, c, k)
                phk = pass_hat_k(n, c, k)
                instance_metrics[f"pass@{k}"] = pak
                instance_metrics[f"pass^{k}"] = phk
                all_pass_at_k[k].append(pak)
                all_pass_hat_k[k].append(phk)

        metrics["per_instance"][instance_id] = instance_metrics

    # Aggregate metrics
    for k in k_values:
        if all_pass_at_k[k]:
            metrics["aggregate"][f"pass@{k}"] = sum(all_pass_at_k[k]) / len(all_pass_at_k[k])
            metrics["aggregate"][f"pass^{k}"] = sum(all_pass_hat_k[k]) / len(all_pass_hat_k[k])

    return metrics


def generate_summary_csv(
    run_dir: Path,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Generate pass@k summary CSV compatible with TAC evaluation format.

    Args:
        run_dir: Path to run directory
        output_path: Optional path to save CSV

    Returns:
        DataFrame with summary metrics
    """
    results = convert_sweagent_results(run_dir)
    metrics = compute_pass_at_k(results)

    # Build DataFrame
    rows = []
    for instance_id, data in results.get("instances", {}).items():
        row = {
            "instance_id": instance_id,
            "trials": data["trials"],
            "successes": data["successes"],
            "success_rate": data["success_rate"],
        }

        if instance_id in metrics["per_instance"]:
            row.update(metrics["per_instance"][instance_id])

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add aggregate row
    if not df.empty:
        agg_row = {
            "instance_id": "AGGREGATE",
            "trials": df["trials"].sum(),
            "successes": df["successes"].sum(),
            "success_rate": (
                df["successes"].sum() / df["trials"].sum() if df["trials"].sum() > 0 else 0.0
            ),
        }
        for col in df.columns:
            if col.startswith("pass"):
                agg_row[col] = metrics["aggregate"].get(col)

        df = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
