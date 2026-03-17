"""
Simplified orchestrator for MCP Atlas tasks.
Assumes Docker services (agent-environment + completion service) are already running.

Usage:
  # First, in separate terminals:
  cd experiments/mcpatlas/mcp-atlas && make run-docker
  cd experiments/mcpatlas/mcp-atlas && make run-mcp-completion

  # Then run this script:
  python task_completion_mcpatlas.py \
      --format_name test_run \
      --backend_model opus_4_5 \
      --num_trials 3 \
      --max_k 3
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Optional

from datasets import load_dataset

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
MCPATLAS_DIR = os.path.join(REPO_ROOT, "experiments", "mcpatlas")
DEFAULT_MCPATLAS_REPO = os.path.join(MCPATLAS_DIR, "mcp-atlas")
LOCAL_DATASET_PATH = os.path.join(MCPATLAS_DIR, "MCP-Atlas.csv")

sys.path.insert(0, REPO_ROOT)

from constants import DEFAULTS, resolve_model  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from experiments.mcpatlas.result_converter import (  # noqa: E402
    compute_pass_k_metrics,
    convert_mcp_atlas_results,
    display_pass_k_summary,
    write_evaluation_results_json,
    write_pass_k_csv,
)
from experiments.mcpatlas.visualize_results import generate_html  # noqa: E402

csv.field_size_limit(10 * 1024 * 1024)


def generate_results_html(scores_dir: str, output_path: str) -> Optional[str]:
    """
    Generate HTML visualization from scored results.

    Args:
        scores_dir: Directory containing scored_*.csv file
        output_path: Path for output HTML file

    Returns:
        Path to generated HTML file, or None if failed
    """
    import glob

    # Find the scored CSV file
    scored_files = glob.glob(os.path.join(scores_dir, "scored_*.csv"))
    if not scored_files:
        print(f"Warning: No scored_*.csv found in {scores_dir}")
        return None

    scored_csv = scored_files[0]

    # Load the CSV
    rows = []
    with open(scored_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Warning: No rows in scored CSV")
        return None

    # Generate HTML
    generate_html(rows, output_path, min_score=0.0, max_score=1.0)
    return output_path


def run_completions(
    mcp_atlas_repo: str,
    model: str,
    input_csv: str,
    output_file: str,
    concurrency: int = 10,
    num_tasks: Optional[int] = None,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
) -> bool:
    """
    Call mcp_completion_script.py via uv run.
    Returns True on success, False on failure.
    """
    mcp_eval_dir = os.path.join(mcp_atlas_repo, "services", "mcp_eval")

    cmd = [
        "uv",
        "run",
        "--frozen",
        "mcp_completion_script.py",
        "--model",
        model,
        "--input",
        input_csv,
        "--output",
        output_file,
        "--concurrency",
        str(concurrency),
    ]

    if start_index is not None:
        cmd.extend(["--start-index", str(start_index)])
    if end_index is not None:
        cmd.extend(["--end-index", str(end_index)])
    if num_tasks is not None and start_index is None and end_index is None:
        cmd.extend(["--num-tasks", str(num_tasks)])

    print(f"Running: {' '.join(cmd)}")
    print(f"  cwd: {mcp_eval_dir}")

    result = subprocess.run(cmd, cwd=mcp_eval_dir)
    return result.returncode == 0


def run_scoring(
    mcp_atlas_repo: str,
    completion_csv: str,
    output_dir: str,
    model_label: str,
    concurrency: int = 10,
    evaluator_model: Optional[str] = None,
) -> bool:
    """
    Call mcp_evals_scores.py via uv run.
    Returns True on success, False on failure.
    """
    mcp_eval_dir = os.path.join(mcp_atlas_repo, "services", "mcp_eval")

    cmd = [
        "uv",
        "run",
        "--frozen",
        "mcp_evals_scores.py",
        "--input-file",
        completion_csv,
        "--output-dir",
        output_dir,
        "--model-label",
        model_label,
        "--concurrency",
        str(concurrency),
    ]
    if evaluator_model:
        cmd.extend(["--evaluator-model", evaluator_model])

    print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=mcp_eval_dir)
    return result.returncode == 0


def filter_csv_by_task_ids(input_csv: str, task_ids: List[str], output_dir: str) -> str:
    """Filter CSV to only include specified task IDs."""
    task_id_set = set(task_ids)
    output_path = os.path.join(output_dir, "filtered_tasks.csv")

    with open(input_csv, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        if not fieldnames:
            print(f"Error: Input CSV is empty or has no header: {input_csv}")
            return output_path

        with open(output_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            matched = 0
            for row in reader:
                if row.get("TASK") in task_id_set:
                    writer.writerow(row)
                    matched += 1

    print(f"Filtered to {matched}/{len(task_ids)} tasks: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCP Atlas tasks (assumes services already running)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites - run these in separate terminals first:
  cd experiments/mcpatlas/mcp-atlas && make run-docker
  cd experiments/mcpatlas/mcp-atlas && make run-mcp-completion

Examples:
  # Basic run
  python task_completion_mcpatlas.py \\
      --format_name test \\
      --backend_model opus_4_5

  # Multiple experiments for pass@k
  python task_completion_mcpatlas.py \\
      --format_name full_baseline \\
      --backend_model opus_4_5 \\
      --num_trials 3 \\
      --max_k 3

  # Limit tasks
  python task_completion_mcpatlas.py \\
      --format_name debug \\
      --backend_model gpt_5_2 \\
      --limit 10
        """,
    )

    parser.add_argument("--format_name", required=True, help="Experiment name")
    parser.add_argument(
        "--backend_model",
        required=True,
        help="Model — short name (e.g. gpt_5_2, sonnet_4_6) or full LiteLLM identifier",
    )
    parser.add_argument(
        "--input_csv", default=LOCAL_DATASET_PATH, help=f"Input CSV (default: {LOCAL_DATASET_PATH})"
    )
    parser.add_argument("--task_ids", default=None, help="Comma-separated task IDs to run")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--start_index", type=int, default=None, help="Start index (0-based)")
    parser.add_argument("--end_index", type=int, default=None, help="End index (exclusive)")
    parser.add_argument(
        "--num_trials", type=int, default=DEFAULTS["num_trials"], help="Number of experiment runs"
    )
    parser.add_argument("--max_k", type=int, default=None, help="Max k for pass@k metrics")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULTS.get("concurrency", 10),
        help="Parallel completions",
    )
    parser.add_argument(
        "--mcp_atlas_repo", default=DEFAULT_MCPATLAS_REPO, help="Path to mcp-atlas repo"
    )
    parser.add_argument("--runs_dir", default=None, help="Output directory")
    parser.add_argument("--skip_scoring", action="store_true", help="Skip the scoring step")
    parser.add_argument(
        "--evaluator_model",
        default=None,
        help="LLM model for scoring (resolved via constants.py). Default: EVAL_LLM_MODEL env or gemini/gemini-2.5-pro",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve short model name → full LiteLLM identifier (with proxy prefix if needed)
    resolved_model = resolve_model(args.backend_model)
    if resolved_model != args.backend_model:
        print(f"Model resolved: {args.backend_model} -> {resolved_model}")

    # Resolve evaluator model: explicit CLI arg > EVAL_LLM_MODEL from mcp-atlas .env
    eval_model_raw = args.evaluator_model
    if not eval_model_raw:
        mcp_env = dotenv_values(os.path.join(args.mcp_atlas_repo, ".env"))
        eval_model_raw = mcp_env.get("EVAL_LLM_MODEL", "gemini/gemini-2.5-pro")
    resolved_evaluator = resolve_model(eval_model_raw)
    if resolved_evaluator != eval_model_raw:
        print(f"Evaluator resolved: {eval_model_raw} -> {resolved_evaluator}")

    # Validate inputs
    if not os.path.exists(args.mcp_atlas_repo):
        print(f"Error: mcp-atlas repo not found at {args.mcp_atlas_repo}")
        return 1

    input_csv = args.input_csv
    if not os.path.exists(input_csv):
        # Auto-download from HuggingFace if it's the default path
        if input_csv == LOCAL_DATASET_PATH:
            print("Dataset not found locally, downloading from HuggingFace (ScaleAI/MCP-Atlas)...")
            os.makedirs(os.path.dirname(LOCAL_DATASET_PATH), exist_ok=True)
            try:
                ds = load_dataset("ScaleAI/MCP-Atlas", split="train")
                ds.to_csv(LOCAL_DATASET_PATH)
                print(f"Downloaded {len(ds)} tasks to: {LOCAL_DATASET_PATH}")
            except Exception as e:
                print(f"Error: Failed to download MCP-Atlas.csv from HuggingFace: {e}")
                return 1
        else:
            print(f"Error: Input CSV not found: {input_csv}")
            return 1

    # Ensure input_csv is absolute (subprocess runs in a different cwd)
    input_csv = os.path.abspath(input_csv)

    # Setup output directory
    model_short = resolved_model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = args.runs_dir or os.path.join(MCPATLAS_DIR, "runs")
    run_dir = os.path.join(runs_dir, f"run_{args.format_name}_{model_short}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Model: {resolved_model}")
    print(f"Input: {input_csv}")
    print(f"Experiments: {args.num_trials}")

    # Filter by task_ids if specified
    if args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(",") if t.strip()]
        input_csv = filter_csv_by_task_ids(input_csv, task_ids, run_dir)

    all_converted_results = []

    for exp_idx in range(args.num_trials):
        print(f"\n{'='*50}")
        print(f"Experiment {exp_idx + 1}/{args.num_trials}")
        print("=" * 50)

        exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        completion_csv = os.path.abspath(
            os.path.join(run_dir, f"completions_{args.format_name}_{exp_idx}_{exp_timestamp}.csv")
        )

        # Run completions
        success = run_completions(
            mcp_atlas_repo=args.mcp_atlas_repo,
            model=resolved_model,
            input_csv=input_csv,
            output_file=completion_csv,
            concurrency=args.concurrency,
            num_tasks=args.limit,
            start_index=args.start_index,
            end_index=args.end_index,
        )

        if not success:
            print(f"Warning: Completions failed for experiment {exp_idx + 1}")
            continue

        if not os.path.exists(completion_csv):
            print(f"Warning: Output CSV not created: {completion_csv}")
            continue

        # Run scoring
        if not args.skip_scoring:
            scores_dir = os.path.abspath(
                os.path.join(run_dir, f"scores_{args.format_name}_{exp_idx}_{exp_timestamp}")
            )

            success = run_scoring(
                mcp_atlas_repo=args.mcp_atlas_repo,
                completion_csv=completion_csv,
                output_dir=scores_dir,
                model_label=model_short,
                concurrency=args.concurrency,
                evaluator_model=resolved_evaluator,
            )

            if not success:
                print(f"Warning: Scoring failed for experiment {exp_idx + 1}")
                continue

            # Generate HTML visualization
            html_path = os.path.join(run_dir, f"results_{exp_idx}.html")
            html_result = generate_results_html(scores_dir, html_path)
            if html_result:
                print(f"Results visualization: {html_result}")

            # Convert results
            final_results_dir = os.path.join(run_dir, "final_results")
            os.makedirs(final_results_dir, exist_ok=True)

            try:
                converted = convert_mcp_atlas_results(
                    scores_csv=scores_dir,
                    output_dir=final_results_dir,
                    granular=False,
                    experiment_idx=exp_idx,
                )
                all_converted_results.extend(converted)

                n_total = len(converted)
                if n_total > 0:
                    n_passed = sum(1 for r in converted if r["passed"])
                    print(
                        f"Experiment {exp_idx + 1}: {n_passed}/{n_total} passed ({100*n_passed/n_total:.1f}%)"
                    )
                else:
                    print(f"Experiment {exp_idx + 1}: No results (empty scored CSV)")
            except Exception as e:
                print(f"Warning: Result conversion failed: {e}")

    # Final summary
    if all_converted_results:
        print("\n" + "=" * 50)
        print("FINAL SUMMARY")
        print("=" * 50)

        # Write evaluation JSON
        eval_json_path = write_evaluation_results_json(
            converted_results=all_converted_results,
            output_dir=run_dir,
            model_name=model_short,
            format_name=args.format_name,
        )
        print(f"Evaluation JSON: {eval_json_path}")

        # Compute pass@k metrics
        max_k = args.max_k if args.max_k else min(args.num_trials, 10)
        pass_k_metrics = compute_pass_k_metrics(all_converted_results, max_k=max_k)
        display_pass_k_summary(pass_k_metrics, max_k=max_k)

        # Write pass@k CSV
        pass_k_csv_path = os.path.join(run_dir, "pass_k_per_task.csv")
        write_pass_k_csv(pass_k_metrics, pass_k_csv_path, max_k=max_k)

        n_passed = sum(1 for r in all_converted_results if r["passed"])
        n_total = len(all_converted_results)
        pct = 100 * n_passed / n_total if n_total else 0
        print(f"\nTotal: {n_passed}/{n_total} passed ({pct:.1f}%)")
        print(f"Run directory: {run_dir}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
