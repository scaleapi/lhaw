#!/usr/bin/env python3
"""
LHAW Benchmark Sample Filtering Script for SWE-bench

This script filters SWE-bench underspecification experiment results to create
benchmark-ready samples based on outcome divergence analysis.

================================================================================
FILTERING GUIDE (from LHAW Benchmark doc)
================================================================================

STEP 2: CLASSIFY BY AMBIGUITY CLASS (done in process_swebench_underspec.py)
    - outcome-critical: 0/N success + divergent terminal states (F2P tests)
    - divergent: Some success (1-2/N) + variable outcomes
    - benign: Consistent success (N/N)
    - new_task: 0/N success + 1 unique state (filtered out)

STEP 3: APPLY TARGET DISTRIBUTION (this script)
    Default quotas match TAC (50/30/20):
    - 50% outcome-critical: Tests if agent asks when it MUST
    - 30% divergent: Tests strategic asking under uncertainty
    - 20% benign: Tests if agent avoids unnecessary asking

    Priority: outcome-critical > divergent > benign (filled in this order)
    OC variants bypass per-task cap; div/benign respect max_per_task.

STEP 4: ENSURE DIVERSITY (this script)
    - max_per_task: Caps samples per original task (default: 8)
    - max_per_dimension: Caps samples per underspec dimension
    - Random shuffle for fair representation (seed=42)

================================================================================
PIPELINE POSITION
================================================================================

    task_completion_swebench.py --generate  (underspec_candidates.csv + instances.yaml)
    task_completion_swebench.py --run       (exp_1/, exp_2/, exp_3/)
    process_swebench_underspec.py           (underspec_results.csv)
    >>> THIS SCRIPT <<<                     (underspec_results_filtered.csv + .json)
    export_swebench_dataset.py              (lhaw_benchmark_swebench.json)

================================================================================
USAGE
================================================================================

# Analyze distribution without filtering
python scripts/filter_swebench_samples.py \
    --input outputs/.../underspec_results.csv \
    --analyze

# Standard filtering (100 samples, 50/30/20 split, matches TAC)
python scripts/filter_swebench_samples.py \
    --input outputs/.../underspec_results.csv \
    --output outputs/.../filtered.csv \
    --max-total 100

# With instance-level diversity cap
python scripts/filter_swebench_samples.py \
    --input outputs/.../underspec_results.csv \
    --output outputs/.../filtered.csv \
    --max-per-instance 8

# Percentage cap mode (keeps all under-cap classes)
python scripts/filter_swebench_samples.py \
    --input results.csv --output filtered.csv \
    --percentage-caps --quotas 40,30,30

================================================================================
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import DEFAULTS, parse_variant_id  # noqa: E402

# Default target distribution for SWE-bench (50/30/20); TAC uses 40/30/30
DEFAULT_QUOTAS = {
    "outcome-critical": 0.50,
    "divergent": 0.30,
    "benign": 0.20,
}


def load_results(input_path: Path) -> pd.DataFrame:
    """Load underspec results from CSV or JSON."""
    if input_path.suffix == ".json":
        return pd.read_json(input_path)
    else:
        return pd.read_csv(input_path)


def analyze_distribution(df: pd.DataFrame) -> Dict:
    """Analyze the current distribution of the dataset."""
    evaluated = df[df["n_trials"].notna()].copy()

    if evaluated.empty:
        return {"error": "No evaluated variants found"}

    analysis = {
        "total_variants": len(evaluated),
        "ambiguity_classes": {},
        "dimensions": {},
        "tasks": {},
        "levels": {},
    }

    # Ambiguity class distribution
    if "ambiguity_class" in evaluated.columns:
        class_counts = evaluated["ambiguity_class"].value_counts()
        for cls in ["outcome-critical", "divergent", "benign", "new_task"]:
            count = class_counts.get(cls, 0)
            pct = count / len(evaluated) * 100 if len(evaluated) > 0 else 0
            analysis["ambiguity_classes"][cls] = {"count": count, "percent": round(pct, 1)}

    # Dimension distribution
    if "dimension" in evaluated.columns:
        dim_counts = evaluated["dimension"].value_counts()
        for dim, count in dim_counts.items():
            pct = count / len(evaluated) * 100
            analysis["dimensions"][dim] = {"count": count, "percent": round(pct, 1)}

    # Task distribution
    if "original_instance_id" in evaluated.columns:
        task_counts = evaluated["original_instance_id"].value_counts()
        analysis["tasks"] = {task: count for task, count in task_counts.head(20).items()}

    # Level distribution (1-seg vs 2-seg)
    if "num_segments" in evaluated.columns:
        level_counts = evaluated["num_segments"].value_counts()
        for level, count in level_counts.items():
            pct = count / len(evaluated) * 100
            analysis["levels"][f"{int(level)}-seg"] = {"count": count, "percent": round(pct, 1)}
    elif "variant_id" in evaluated.columns:

        def get_level(vid):
            parts = str(vid).replace("V_", "").replace("_delete", "").replace("_vaguify", "")
            return len([p for p in parts.split("_") if p.startswith("S")])

        evaluated["_level"] = evaluated["variant_id"].apply(get_level)
        level_counts = evaluated["_level"].value_counts()
        for level, count in level_counts.items():
            pct = count / len(evaluated) * 100
            analysis["levels"][f"{level}-seg"] = {"count": count, "percent": round(pct, 1)}

    return analysis


def filter_samples(
    df: pd.DataFrame,
    max_per_task: int = 8,
    max_total: int = 100,
    quotas: Dict[str, float] = None,
    exclude_new_task: bool = True,
    use_percentage_caps: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Filter dataset to match target class distribution.

    Matches TAC filter_tac_samples.py approach:
    - Quotas are TARGET percentages of max_total (exact target counts)
    - Priority: outcome-critical > divergent > benign
    - OC variants bypass per-task cap (we want ALL hard cases)
    - div/benign respect per-task limit for diversity
    - Random shuffle within each class for fair task representation

    Args:
        df: DataFrame with ambiguity_class column
        max_per_task: Maximum samples per original task (default: 8)
        max_total: Maximum total samples (default: 100)
        quotas: Dict mapping class name to target percentage (default: 50/30/20)
        exclude_new_task: Whether to exclude new_task class
        use_percentage_caps: If True, quotas are MAX percentages of pool (not targets)

    Returns:
        Filtered DataFrame
    """
    if quotas is None:
        quotas = DEFAULT_QUOTAS

    evaluated = df[df["n_trials"].notna()].copy()

    if exclude_new_task:
        evaluated = evaluated[evaluated["ambiguity_class"] != "new_task"]

    if evaluated.empty:
        return pd.DataFrame()

    # ── Percentage caps mode (alternative, not default) ──
    if use_percentage_caps:
        pool_size = len(evaluated)
        print(f"  Pool size after excluding new_task: {pool_size}")

        selected = []
        for cls, quota in quotas.items():
            class_df = evaluated[evaluated["ambiguity_class"] == cls]
            available = len(class_df)
            max_allowed = int(pool_size * quota)

            if available == 0:
                print(f"  {cls}: 0 available, 0 kept")
                continue

            current_pct = available / pool_size * 100

            if available <= max_allowed:
                print(
                    f"  {cls}: {available} available ({current_pct:.1f}% <= {quota*100:.0f}% cap), keeping all"
                )
                selected.append(class_df)
            else:
                print(
                    f"  {cls}: {available} available ({current_pct:.1f}% > {quota*100:.0f}% cap), sampling {max_allowed}"
                )
                selected.append(class_df.sample(n=max_allowed, random_state=seed))

        if not selected:
            return pd.DataFrame()

        result = pd.concat(selected, ignore_index=True)
        print(f"  Total after percentage caps: {len(result)}")
        return result

    # ── Target count mode (default, matches TAC) ──
    # Compute class quotas as exact target counts
    class_quotas = {}
    for cls, quota in quotas.items():
        class_quotas[cls] = int(max_total * quota)

    # Adjust for rounding
    total_allocated = sum(class_quotas.values())
    if total_allocated < max_total:
        largest_class = max(quotas.keys(), key=lambda k: quotas[k])
        class_quotas[largest_class] += max_total - total_allocated

    print(f"  Class quotas: {class_quotas}")

    # Extract base dimension for diversity tracking
    evaluated["_base_dimension"] = evaluated["dimension"].apply(
        lambda x: str(x).split("+")[0] if pd.notna(x) else "unknown"
    )

    # Extract original task for per-task cap
    if "original_instance_id" in evaluated.columns:
        evaluated["_task"] = evaluated["original_instance_id"]
    else:
        evaluated["_task"] = evaluated["instance_id"].apply(lambda x: parse_variant_id(x)[0])

    # Compute oracle_decision if not present
    if "oracle_decision" not in evaluated.columns:
        evaluated["oracle_decision"] = evaluated["ambiguity_class"].apply(
            lambda c: "clarify" if c == "outcome-critical" else "proceed"
        )

    # Sort: class priority first, then shuffle within each class
    class_rank = {"outcome-critical": 0, "divergent": 1, "benign": 2}
    evaluated["_class_rank"] = evaluated["ambiguity_class"].map(class_rank).fillna(3)

    shuffled_dfs = []
    for cls_rank, group in evaluated.groupby("_class_rank", sort=True):
        shuffled_dfs.append(group.sample(frac=1, random_state=seed))
    evaluated = pd.concat(shuffled_dfs, ignore_index=True)

    # Greedy selection with per-task and per-class caps
    filtered_rows = []
    task_counts = defaultdict(int)
    class_counts_sel = defaultdict(int)

    for _, row in evaluated.iterrows():
        if len(filtered_rows) >= max_total:
            break

        amb_class = row.get("ambiguity_class", "")
        task = row["_task"]

        # Check class quota
        if class_counts_sel[amb_class] >= class_quotas.get(amb_class, max_total):
            continue

        # OC variants bypass per-task limit (we want ALL hard cases)
        # Divergent/Benign: respect per-task limit for diversity
        if amb_class != "outcome-critical":
            if task_counts[task] >= max_per_task:
                continue

        filtered_rows.append(row)
        task_counts[task] += 1
        class_counts_sel[amb_class] += 1

    final_df = pd.DataFrame(filtered_rows)

    # Clean up temporary columns
    cols_to_drop = [c for c in ["_class_rank", "_base_dimension", "_task"] if c in final_df.columns]
    if cols_to_drop:
        final_df = final_df.drop(columns=cols_to_drop)

    print(f"  After filtering: {len(final_df)} (target: {max_total})")

    return final_df


def print_analysis(analysis: Dict):
    """Print analysis in a readable format."""
    print("\n" + "=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)

    print(f"\nTotal evaluated variants: {analysis.get('total_variants', 0)}")

    print("\n--- Ambiguity Class Distribution ---")
    for cls in ["outcome-critical", "divergent", "benign", "new_task"]:
        info = analysis.get("ambiguity_classes", {}).get(cls, {"count": 0, "percent": 0})
        print(f"  {cls:20s}: {info['count']:4d} ({info['percent']:5.1f}%)")

    print("\n--- Dimension Distribution ---")
    for dim, info in analysis.get("dimensions", {}).items():
        print(f"  {dim or 'unknown':20s}: {info['count']:4d} ({info['percent']:5.1f}%)")

    print("\n--- Level Distribution ---")
    for level, info in analysis.get("levels", {}).items():
        print(f"  {level:20s}: {info['count']:4d} ({info['percent']:5.1f}%)")

    if analysis.get("tasks"):
        print("\n--- Top Tasks ---")
        for task, count in list(analysis["tasks"].items())[:10]:
            print(f"  {task[:50]:50s}: {count:4d}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter SWE-bench underspec results for benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input CSV/JSON from process_swebench_underspec.py",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output filtered CSV (default: input_filtered.csv)",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=DEFAULTS["max_total"],
        help=f"Maximum total samples (default: {DEFAULTS['max_total']})",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=DEFAULTS["max_per_task_swebench"],
        help=f"Max samples per original task (default: {DEFAULTS['max_per_task_swebench']}, OC bypasses this)",
    )
    parser.add_argument(
        "--quotas",
        type=str,
        default="50,30,20",
        help="Comma-separated quotas: outcome-critical,divergent,benign (default: 50,30,20)",
    )
    parser.add_argument(
        "--max-per-instance",
        type=int,
        default=None,
        help="Max samples per SWE-bench instance (applied after class filtering)",
    )
    parser.add_argument(
        "--max-per-dimension",
        type=int,
        default=None,
        help="Max samples per underspec dimension (applied after class filtering)",
    )
    parser.add_argument(
        "--include-new-task",
        action="store_true",
        help="Include new_task class (excluded by default)",
    )
    parser.add_argument(
        "--critical-only",
        action="store_true",
        help="Only keep outcome-critical variants",
    )
    parser.add_argument(
        "--percentage-caps",
        action="store_true",
        help="Use percentage caps instead of exact target counts",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze distribution without filtering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
        help=f"Random seed for reproducibility (default: {DEFAULTS['seed']})",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading: {input_path}")
    df = load_results(input_path)
    print(f"  Loaded {len(df)} variants")

    # Analyze
    analysis = analyze_distribution(df)
    print_analysis(analysis)

    if args.analyze:
        print("\n[Analyze mode - no filtering applied]")
        return

    # Parse quotas
    quota_values = [float(q) / 100 for q in args.quotas.split(",")]
    if len(quota_values) != 3:
        print("Error: quotas must have 3 values (outcome-critical,divergent,benign)")
        sys.exit(1)
    if abs(sum(quota_values) - 1.0) > 0.01:
        print(f"Error: quotas must sum to 100, got {sum(quota_values) * 100:.1f}")
        sys.exit(1)

    quotas = {
        "outcome-critical": quota_values[0],
        "divergent": quota_values[1],
        "benign": quota_values[2],
    }

    # Override for critical-only mode
    if args.critical_only:
        quotas = {"outcome-critical": 1.0, "divergent": 0.0, "benign": 0.0}

    # Filter
    print("\n" + "=" * 70)
    print("FILTERING")
    print("=" * 70)
    print(f"\nMax total: {args.max_total}")
    print(f"Max per task: {args.max_per_task}")
    print(f"Quotas: {quotas}")

    filtered = filter_samples(
        df,
        max_per_task=args.max_per_task,
        max_total=args.max_total,
        quotas=quotas,
        exclude_new_task=not args.include_new_task,
        use_percentage_caps=args.percentage_caps,
        seed=args.seed,
    )
    print(f"\nAfter class filtering: {len(filtered)} samples")

    # Apply additional diversity controls (post-filter)
    if args.max_per_instance or args.max_per_dimension:
        if args.max_per_instance and "original_instance_id" in filtered.columns:
            groups = []
            for inst in filtered["original_instance_id"].unique():
                inst_df = filtered[filtered["original_instance_id"] == inst]
                if len(inst_df) > args.max_per_instance:
                    inst_df = inst_df.sample(n=args.max_per_instance, random_state=args.seed)
                groups.append(inst_df)
            filtered = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()

        if args.max_per_dimension and "dimension" in filtered.columns:
            groups = []
            for dim in filtered["dimension"].unique():
                dim_df = filtered[filtered["dimension"] == dim]
                if len(dim_df) > args.max_per_dimension:
                    dim_df = dim_df.sample(n=args.max_per_dimension, random_state=args.seed)
                groups.append(dim_df)
            filtered = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()

        print(f"After diversity controls: {len(filtered)} samples")

    # Final analysis
    print("\n--- Final Distribution ---")
    final_analysis = analyze_distribution(filtered)
    for cls in ["outcome-critical", "divergent", "benign"]:
        info = final_analysis.get("ambiguity_classes", {}).get(cls, {"count": 0, "percent": 0})
        print(f"  {cls:20s}: {info['count']:4d} ({info['percent']:5.1f}%)")

    # Print per-class performance
    if "pass_at_3" in filtered.columns:
        print("\n--- Per-Class Performance ---")
        for cls in ["outcome-critical", "divergent", "benign"]:
            cls_df = filtered[filtered["ambiguity_class"] == cls]
            if not cls_df.empty:
                print(f"  {cls:20s}: pass@3={cls_df['pass_at_3'].mean():.3f}, " f"n={len(cls_df)}")

    # SWE-Bench output columns
    output_cols = [
        "instance_id",
        "original_instance_id",
        "variant_id",
        "dimension",
        "subdimension",
        "severity",
        "n_trials",
        "pass_at_1",
        "pass_at_3",
        "pass_at_N",
        "n_unique_states",
        "divergence_ratio",
        "state_entropy",
        "terminal_states",
        "ambiguity_class",
        "oracle_decision",
        "criticality",
        "guessability",
        "priority_score",
        "removed_value",
        "removed_segments_json",
        "expected_questions",
        "expected_failure_mode",
        "original_prompt",
        "underspecified_prompt",
    ]
    output_cols = [c for c in output_cols if c in filtered.columns]

    # Save CSV
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + "_filtered.csv")
    )
    filtered[output_cols].to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"Saved filtered results to: {output_path}")

    # Also save JSON (same data as CSV, for programmatic access)
    json_path = output_path.with_suffix(".json")
    filtered[output_cols].to_json(json_path, orient="records", indent=2)
    print(f"Saved JSON to: {json_path}")

    # Print next steps
    print("\nNext steps:")
    print("  # Export benchmark JSON (canonical exporter):")
    print(f"  python scripts/export_swebench_dataset.py --input {output_path}")


if __name__ == "__main__":
    main()
