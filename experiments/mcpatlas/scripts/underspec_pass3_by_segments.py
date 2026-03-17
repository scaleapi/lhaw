#!/usr/bin/env python3
"""
Calculate pass@3 by number of segments removed for underspecified runs.

For each run, outputs pass@3 separately for:
- Prompts with 1 segment removed
- Prompts with 2 segments removed

Usage:
    python underspec_pass3_by_segments.py <run1> <run2> ... [--output results.csv]
"""

import argparse
import csv
import json
import re
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from evaluation.pass_k_metrics import pass_at_k  # noqa: E402


def _normalize_prompt(prompt: str) -> str:
    """Normalize prompt by collapsing whitespace for matching."""
    return " ".join(prompt.split())


def load_prompt_to_segment_count(underspec_json_dir: Path) -> dict:
    """
    Build index mapping normalized prompts to segment removal count.

    Used as fallback for old-format runs that don't have variant metadata in completions.
    """
    prompt_index = {}

    for json_path in underspec_json_dir.glob("*.json"):
        with open(json_path, "r") as f:
            data = json.load(f)
        for variant in data.get("variants", []):
            underspec_prompt = variant.get("underspecified_prompt", "")
            num_removed = len(variant.get("removed_segments", []))
            normalized = _normalize_prompt(underspec_prompt)
            prompt_index[normalized] = num_removed

    return prompt_index


def load_completions(run_path: Path) -> dict:
    """
    Load completions CSVs to get variant info for each trial.

    Returns dict mapping task_id -> list of {prompt, variant_id, removal_condition}
    """
    completions = {}
    csv.field_size_limit(10 * 1024 * 1024)

    for csv_path in sorted(run_path.glob("completions_*.csv")):
        csv_text = csv_path.read_text(encoding="utf-8", errors="replace")
        reader = csv.DictReader(StringIO(csv_text))
        for row in reader:
            task_id = row.get("task", row.get("TASK", ""))
            if task_id:
                completions.setdefault(task_id, []).append(
                    {
                        "prompt": row.get("PROMPT", ""),
                        "variant_id": row.get("variant_id", ""),
                        "removal_condition": row.get("removal_condition", ""),
                    }
                )
    return completions


def count_segments_from_variant_id(variant_id: str) -> int:
    """
    Parse variant_id to count segments removed.

    Supports both formats:
      "task_id::V_S1_S2_delete" -> 2
      "V_S1_S2_delete"          -> 2
    """
    if not variant_id:
        return 0
    variant_part = variant_id.split("::")[-1] if "::" in variant_id else variant_id
    return len(re.findall(r"S\d+", variant_part))


def count_segments_from_removal_condition(removal_condition: str) -> int:
    """
    Parse removal_condition to count segments removed.

    "top1_removal" or "top2_removal" -> 1 segment
    "top1_and_top2_removal" -> 2 segments
    """
    if not removal_condition:
        return 0
    if "top1_and_top2" in removal_condition:
        return 2
    if "top1" in removal_condition or "top2" in removal_condition:
        return 1
    return 0


def load_evaluation(run_path: Path) -> dict:
    """Load evaluation JSON from run directory."""
    eval_files = list(run_path.glob("evaluation_*.json"))
    if not eval_files:
        raise FileNotFoundError(f"No evaluation JSON found in {run_path}")
    with open(eval_files[0], "r") as f:
        return json.load(f)


def extract_model_name(run_name: str) -> str:
    """Extract model name from run directory name."""
    for part in run_name.split("_"):
        if any(m in part.lower() for m in ["gpt", "gemini", "claude", "opus", "sonnet"]):
            idx = run_name.find(part)
            return run_name[idx:].rsplit("_", 2)[0]
    return "unknown"


def calc_pass3(results: list) -> tuple[float, int]:
    """
    Calculate average pass@3 and task count for a list of trial results.

    Args:
        results: List of {task_id, score} where score is 0 or 1

    Returns:
        (average_pass3, num_tasks)
    """
    if not results:
        return 0.0, 0

    # Group scores by task
    by_task = {}
    for r in results:
        by_task.setdefault(r["task_id"], []).append(r["score"])

    # Calculate pass@3 for each task, then average
    pass3_values = []
    for scores in by_task.values():
        n = len(scores)
        c = sum(scores)
        k = min(3, n)
        if n > 0:
            pass3_values.append(pass_at_k(n, c, k))

    if pass3_values:
        return sum(pass3_values) / len(pass3_values), len(pass3_values)
    return 0.0, 0


def process_run(run_path: Path, prompt_to_segments: dict) -> dict | None:
    """
    Process a single run and calculate pass@3 by segment removal count.

    For each trial:
    1. Determine how many segments were removed (from metadata or prompt matching)
    2. Record the pass/fail score
    3. Group by segment count and calculate pass@3
    """
    run_name = run_path.name
    model_name = extract_model_name(run_name)

    try:
        eval_data = load_evaluation(run_path)
    except FileNotFoundError:
        print(f"Warning: No evaluation file found for {run_path}", file=sys.stderr)
        return None

    completions = load_completions(run_path)

    # Validation: check alignment between completions and evaluation
    total_eval_trials = 0
    total_completion_trials = 0
    mismatched_tasks = []

    for task_id, task_data in eval_data.get("tasks", {}).items():
        n_eval = len(task_data.get("results", []))
        n_comp = len(completions.get(task_id, []))
        total_eval_trials += n_eval
        total_completion_trials += n_comp
        if n_eval != n_comp:
            mismatched_tasks.append((task_id, n_eval, n_comp))

    if mismatched_tasks:
        print(
            f"  Warning: {len(mismatched_tasks)} tasks have mismatched trial counts:",
            file=sys.stderr,
        )
        for task_id, n_eval, n_comp in mismatched_tasks[:3]:  # Show first 3
            print(f"    {task_id}: {n_eval} eval results vs {n_comp} completions", file=sys.stderr)
        if len(mismatched_tasks) > 3:
            print(f"    ... and {len(mismatched_tasks) - 3} more", file=sys.stderr)

    # Collect results grouped by segment removal count
    results_1seg = []
    results_2seg = []
    matched_trials = 0
    skipped_no_completion = 0
    skipped_no_segment_info = 0

    for task_id, task_data in eval_data.get("tasks", {}).items():
        task_completions = completions.get(task_id, [])
        eval_results = task_data.get("results", [])

        for i, eval_result in enumerate(eval_results):
            if i >= len(task_completions):
                skipped_no_completion += 1
                continue

            completion = task_completions[i]

            # Determine segment count using available metadata
            removal_condition = completion.get("removal_condition", "")
            variant_id = completion.get("variant_id", "")

            if removal_condition:
                num_removed = count_segments_from_removal_condition(removal_condition)
            elif variant_id:
                num_removed = count_segments_from_variant_id(variant_id)
            else:
                # Fallback: match by prompt text (for old-format runs)
                normalized = _normalize_prompt(completion.get("prompt", ""))
                num_removed = prompt_to_segments.get(normalized, 0)
                if num_removed == 0:
                    skipped_no_segment_info += 1
                    continue

            matched_trials += 1

            # Extract pass/fail score
            score = 1 if eval_result.get("final_score", {}).get("result", 0) > 0 else 0

            result_entry = {"task_id": task_id, "score": score}
            if num_removed == 1:
                results_1seg.append(result_entry)
            elif num_removed >= 2:
                results_2seg.append(result_entry)

    print(
        f"  Trials: {matched_trials} matched, {skipped_no_completion} skipped (no completion), "
        f"{skipped_no_segment_info} skipped (no segment info)",
        file=sys.stderr,
    )

    # Assert we processed a reasonable number of trials
    if matched_trials == 0:
        print("  ERROR: No trials were successfully matched!", file=sys.stderr)
        return None

    if skipped_no_completion > 0:
        print(
            f"  WARNING: {skipped_no_completion} eval results had no matching completion",
            file=sys.stderr,
        )

    # Verify results add up
    total_results = len(results_1seg) + len(results_2seg)
    assert (
        total_results == matched_trials
    ), f"Results mismatch: {total_results} results but {matched_trials} matched trials"

    pass3_1seg, n_1seg = calc_pass3(results_1seg)
    pass3_2seg, n_2seg = calc_pass3(results_2seg)

    # Sanity check: n should be number of unique tasks, not trials
    print(
        f"  Results: {len(results_1seg)} trials -> {n_1seg} tasks (1-seg), "
        f"{len(results_2seg)} trials -> {n_2seg} tasks (2-seg)",
        file=sys.stderr,
    )

    return {
        "model": model_name,
        "1seg_pass3": pass3_1seg,
        "1seg_n": n_1seg,
        "2seg_pass3": pass3_2seg,
        "2seg_n": n_2seg,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate pass@3 by segment removal count for underspecified runs."
    )
    parser.add_argument("runs", nargs="+", help="Paths to underspecified run directories")
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base path containing underspec_generation/json. Inferred if not provided.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")

    args = parser.parse_args()
    run_paths = [Path(r) for r in args.runs]

    # Find base path containing variant definitions
    if args.base_path:
        base_path = Path(args.base_path)
    else:
        base_path = run_paths[0].parent
        while base_path.name:
            if (base_path / "underspec_generation" / "json").exists():
                break
            if (base_path / "json").exists():
                break
            base_path = base_path.parent
            if base_path == base_path.parent:
                raise ValueError(
                    "Could not find underspec variant JSON directory. Use --base-path."
                )

    # Support both layouts: underspec_generation/json/ and json/
    if (base_path / "underspec_generation" / "json").exists():
        underspec_json_dir = base_path / "underspec_generation" / "json"
    elif (base_path / "json").exists():
        underspec_json_dir = base_path / "json"
    else:
        raise FileNotFoundError(
            f"No json/ or underspec_generation/json/ found under {base_path}. Use --base-path."
        )
    if not underspec_json_dir.exists():
        raise FileNotFoundError(f"Directory not found: {underspec_json_dir}")

    prompt_to_segments = load_prompt_to_segment_count(underspec_json_dir)
    print(
        f"Loaded {len(prompt_to_segments)} variant prompts from {underspec_json_dir}",
        file=sys.stderr,
    )

    # Process each run
    results = []
    for run_path in run_paths:
        if not run_path.exists():
            print(f"Warning: Run path does not exist: {run_path}", file=sys.stderr)
            continue
        print(f"Processing: {run_path.name}", file=sys.stderr)
        result = process_run(run_path, prompt_to_segments)
        if result:
            results.append(result)

    if not results:
        print("No valid results to report.", file=sys.stderr)
        return 1

    # Output as DataFrame
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "Model": r["model"],
                "1-seg pass@3": r["1seg_pass3"],
                "1-seg n": r["1seg_n"],
                "2-seg pass@3": r["2seg_pass3"],
                "2-seg n": r["2seg_n"],
            }
            for r in results
        ]
    )

    print(df.to_string(index=False))

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
