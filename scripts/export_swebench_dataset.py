#!/usr/bin/env python3
"""
Export SWE-bench underspecified task dataset to benchmark JSON.

Produces lhaw_benchmark_swebench.json matching the schema of lhaw_benchmark_tac.json.

Can operate in two modes:
1. Pre-trial export: From underspec_candidates.csv (pipeline output, before running trials)
2. Post-trial export: From underspec_results_filtered.csv (after classification + filtering)

Post-trial export includes ambiguity_class and terminal_states from trial analysis.

Output schema (one entry per variant, matches lhaw_benchmark_tac.json):
{
    "variant_id": "django__django-12345__V_S3_delete",
    "original_task": "django__django-12345",
    "dataset": "SWE-Bench Pro",
    "original_prompt": "...",
    "underspecified_prompt": "...",
    "information_dimension": "constraint",
    "ambiguity_class": "outcome-critical",
    "removed_segments": [
        {
            "id": "S3",
            "dimension": "constraint",
            "subdimension": "temporal",
            "value": "version 3.2",
            "text": "for Django version 3.2"
        }
    ],
    "expected_questions": [
        {
            "segment_id": "S3",
            "questions": ["Which Django version should this target?"]
        }
    ],
    "terminal_states": "[(1, 0, 1), (0, 0, 1), (1, 0, 1)]"
}

Pipeline position:
    task_completion_swebench.py --generate  (underspec_candidates.csv)
    task_completion_swebench.py --run       (exp_1/, exp_2/, exp_3/)
    process_swebench_underspec.py           (underspec_results.csv)
    filter_swebench_samples.py              (underspec_results_filtered.csv + .json)
    >>> THIS SCRIPT <<<                     (lhaw_benchmark_swebench.json)

Usage:
    # Post-trial export (recommended - includes classification data)
    python scripts/export_swebench_dataset.py \
        --input experiments/swebench/runs/.../underspec_results_filtered.csv

    # Pre-trial export (from generate stage output)
    python scripts/export_swebench_dataset.py \
        --input experiments/swebench/runs/.../underspec_candidates.csv \
        --pre-trial

    # Custom output path
    python scripts/export_swebench_dataset.py \
        --input results_filtered.csv \
        --output lhaw_benchmark_swebench.json
"""

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import DEFAULTS, VARIANT_DELIMITER, parse_variant_id  # noqa: E402
from scripts.process_swebench_underspec import ensure_swebench_csv  # noqa: E402


def safe_parse_json(val: Any, default: Any = None) -> Any:
    """Safely parse a JSON string, returning default on failure."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default if default is not None else []
    if isinstance(val, (list, dict)):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val or val in ("nan", "None", ""):
            return default if default is not None else []
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            pass
    return default if default is not None else []


def compute_per_trial_terminal_states(
    exp_dir: Path,
    swe_bench_csv: Path,
) -> Dict[str, str]:
    """Compute per-trial F2P terminal states from eval results.

    Returns dict mapping instance_id -> terminal_states string in TAC format:
    "[(1, 0, 1), (0, 0, 1), (1, 0, 1)]" (one tuple per trial, all trials listed).
    """
    swe_df = pd.read_csv(swe_bench_csv).set_index("instance_id")
    num_trials = DEFAULTS["num_trials"]
    config_path = exp_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        num_trials = cfg.get("num_trials", num_trials)
    trial_prefixes = {f"exp_{i}": f"trial{i}" for i in range(1, num_trials + 1)}
    result = {}

    # Collect all variant IDs from instances.yaml or CSV
    filt_csv = exp_dir / "underspec_results_filtered.csv"
    if filt_csv.exists():
        filt_df = pd.read_csv(filt_csv)
        variant_ids = filt_df["instance_id"].tolist()
    else:
        return result

    for vid in variant_ids:
        base_id, _suffix = parse_variant_id(vid)
        variant_suffix = (VARIANT_DELIMITER + _suffix) if _suffix else ""

        # Get F2P test list for original instance
        if base_id not in swe_df.index:
            continue
        raw_f2p = swe_df.loc[base_id, "fail_to_pass"]
        if pd.isna(raw_f2p) or not raw_f2p:
            continue
        try:
            f2p_tests = sorted(ast.literal_eval(raw_f2p))
        except (ValueError, SyntaxError, TypeError):
            continue
        f2p_total = len(f2p_tests)
        if f2p_total == 0:
            continue

        trial_states = []
        for exp, trial_prefix in trial_prefixes.items():
            output_file = (
                exp_dir
                / exp
                / "eval_results"
                / base_id
                / f"{trial_prefix}{variant_suffix}_output.json"
            )
            if not output_file.exists():
                # Missing eval = all F2P fail
                trial_states.append(tuple(0 for _ in range(f2p_total)))
                continue

            try:
                with open(output_file) as f:
                    eval_result = json.load(f)
            except (json.JSONDecodeError, IOError):
                trial_states.append(tuple(0 for _ in range(f2p_total)))
                continue

            tests = eval_result.get("tests", [])
            if not tests:
                trial_states.append(tuple(0 for _ in range(f2p_total)))
                continue

            passed_tests = {t.get("name", "") for t in tests if t.get("status") == "PASSED"}
            state = tuple(1 if t in passed_tests else 0 for t in f2p_tests)
            trial_states.append(state)

        result[vid] = str(trial_states)

    return result


def build_benchmark_entry(row: Dict, pre_trial: bool = False) -> Dict:
    """Build a single benchmark entry from a CSV row.

    Args:
        row: Dict from CSV/DataFrame row
        pre_trial: If True, skip fields that require trial data

    Returns:
        Dict matching lhaw_benchmark_tac.json schema
    """
    # Parse removed_segments_json
    removed_segments_raw = safe_parse_json(row.get("removed_segments_json"), [])

    # Build clean segment entries matching TAC schema
    removed_segments = []
    for seg in removed_segments_raw:
        removed_segments.append(
            {
                "id": seg.get("id", ""),
                "dimension": seg.get("dimension", ""),
                "subdimension": seg.get("subdimension", ""),
                "value": seg.get("value", ""),
                "text": seg.get("text", ""),
            }
        )

    # Get primary dimension
    if removed_segments:
        primary_dim = removed_segments[0].get("dimension", "unknown")
    else:
        primary_dim = row.get("dimension", "unknown")
        if "+" in str(primary_dim):
            primary_dim = str(primary_dim).split("+")[0]

    # Parse expected_questions
    expected_questions = safe_parse_json(row.get("expected_questions"), [])

    # Build entry
    entry = {
        "variant_id": row.get("instance_id", row.get("variant_id", "")),
        "original_task": row.get("original_instance_id", ""),
        "dataset": "SWE-Bench Pro",
        "original_prompt": row.get("original_prompt", ""),
        "underspecified_prompt": row.get("underspecified_prompt", ""),
        "information_dimension": primary_dim,
        "removed_segments": removed_segments,
        "expected_questions": expected_questions,
    }

    # Add post-trial fields if available
    if not pre_trial:
        entry["ambiguity_class"] = row.get("ambiguity_class", "")
        entry["terminal_states"] = row.get("terminal_states", "")
    else:
        # Pre-trial: use severity as placeholder
        entry["severity"] = row.get("severity", "delete")
        entry["expected_failure_mode"] = row.get("expected_failure_mode", "")

    return entry


def build_dataset(
    input_path: Path,
    pre_trial: bool = False,
    per_trial_states: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Build the benchmark dataset from CSV/JSON input.

    Args:
        input_path: Path to CSV or JSON file
        pre_trial: If True, export without classification data
        per_trial_states: Optional dict of instance_id -> per-trial terminal states string

    Returns:
        List of benchmark entries
    """
    # Load data
    if input_path.suffix == ".json":
        df = pd.read_json(input_path)
    else:
        df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} variants from {input_path}")

    # Build entries
    entries = []
    skipped = 0

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        # Skip entries without removed_segments_json
        removed_segments = safe_parse_json(row_dict.get("removed_segments_json"), [])
        if not removed_segments:
            skipped += 1
            continue

        entry = build_benchmark_entry(row_dict, pre_trial=pre_trial)

        # Override terminal_states with per-trial version if available
        if per_trial_states and not pre_trial:
            vid = row_dict.get("instance_id", row_dict.get("variant_id", ""))
            if vid in per_trial_states:
                entry["terminal_states"] = per_trial_states[vid]

        entries.append(entry)

    if skipped:
        print(f"  Skipped {skipped} entries (missing removed_segments_json)")

    return entries


def print_summary(entries: List[Dict]):
    """Print dataset summary statistics."""
    print(f"\n{'='*70}")
    print("BENCHMARK DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total entries: {len(entries)}")

    # Dimension breakdown
    dim_counts = {}
    for e in entries:
        dim = e.get("information_dimension", "unknown")
        dim_counts[dim] = dim_counts.get(dim, 0) + 1
    print("\n--- By Dimension ---")
    for dim, count in sorted(dim_counts.items(), key=lambda x: -x[1]):
        print(f"  {dim:20s}: {count:4d} ({count/len(entries)*100:5.1f}%)")

    # Ambiguity class breakdown (post-trial only)
    class_counts = {}
    for e in entries:
        cls = e.get("ambiguity_class", "")
        if cls:
            class_counts[cls] = class_counts.get(cls, 0) + 1
    if class_counts:
        print("\n--- By Ambiguity Class ---")
        for cls in ["outcome-critical", "divergent", "benign", "new_task"]:
            count = class_counts.get(cls, 0)
            if count:
                print(f"  {cls:20s}: {count:4d} ({count/len(entries)*100:5.1f}%)")

    # Original task count
    tasks = set(e.get("original_task", "") for e in entries)
    print(f"\nOriginal tasks: {len(tasks)}")

    # Segment count distribution
    seg_counts = {}
    for e in entries:
        n = len(e.get("removed_segments", []))
        seg_counts[n] = seg_counts.get(n, 0) + 1
    print("\n--- By Segment Count ---")
    for n, count in sorted(seg_counts.items()):
        print(f"  {n}-segment: {count:4d} ({count/len(entries)*100:5.1f}%)")

    # Show example
    if entries:
        print("\n--- Example Entry ---")
        example = entries[0]
        print(f"  variant_id: {example['variant_id']}")
        print(f"  original_task: {example['original_task']}")
        print(f"  dimension: {example['information_dimension']}")
        if example.get("ambiguity_class"):
            print(f"  ambiguity_class: {example['ambiguity_class']}")
        if example.get("removed_segments"):
            seg = example["removed_segments"][0]
            print(f"  removed: [{seg['dimension']}] {seg['value'][:60]}...")
        if example.get("expected_questions"):
            eq = example["expected_questions"]
            if isinstance(eq, list) and eq:
                first_q = eq[0]
                if isinstance(first_q, dict):
                    qs = first_q.get("questions", [])
                    if qs:
                        print(f"  expected_question: {qs[0][:60]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Export SWE-bench underspec dataset to benchmark JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input CSV/JSON (from filter_swebench_samples.py or underspec_candidates.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON path (default: lhaw_benchmark_swebench.json in same dir)",
    )
    parser.add_argument(
        "--pre-trial",
        action="store_true",
        help="Pre-trial export (from underspec_candidates.csv, no classification data)",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Experiment directory with eval results (to compute per-trial terminal states)",
    )
    parser.add_argument(
        "--swe-bench-csv",
        type=str,
        default=str(REPO_ROOT / "swebenchpro" / "SWE-bench_Pro-os" / "swe_bench_pro_full.csv"),
        help="Path to swe_bench_pro_full.csv (for F2P test lookup)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Compute per-trial terminal states if exp-dir provided
    per_trial_states = None
    if args.exp_dir and not args.pre_trial:
        exp_dir = Path(args.exp_dir)
        swe_bench_csv = Path(args.swe_bench_csv)
        ensure_swebench_csv(swe_bench_csv)
        print(f"Computing per-trial terminal states from {exp_dir}...")
        per_trial_states = compute_per_trial_terminal_states(exp_dir, swe_bench_csv)
        print(f"  Computed states for {len(per_trial_states)} variants")

    # Build dataset
    entries = build_dataset(input_path, pre_trial=args.pre_trial, per_trial_states=per_trial_states)

    if not entries:
        print("No entries to export!")
        sys.exit(1)

    # Print summary
    print_summary(entries)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "lhaw_benchmark_swebench.json"

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Exported benchmark dataset: {output_path}")
    print(f"  {len(entries)} entries")
    print("  Schema matches lhaw_benchmark_tac.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
