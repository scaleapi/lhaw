#!/usr/bin/env python3
"""
LHAW Benchmark Sample Filtering Script

This script filters synthetic underspecification candidates to create 
benchmark-ready samples based on outcome divergence analysis.

================================================================================
FILTERING GUIDE: From Candidates to Benchmark-Ready Samples
================================================================================

STEP 1: GENERATE CANDIDATES
    Run underspec generation with multiple reference models (Sonnet-4.5, GPT-5.2, 
    Gemini-3-Flash) to get trial outcomes. This populates the candidates CSV.

STEP 2: CLASSIFY BY AMBIGUITY CLASS (4 classes)
    Based on trial outcomes, each variant is classified as:
    
    - outcome-critical: 0/N success + divergent terminal states
        → Agent ALWAYS fails without this info
        → Oracle decision: CLARIFY (must ask)
        
    - divergent: Some success (1-2/3) + variable outcomes  
        → Agent sometimes succeeds, but outcomes vary
        → Oracle decision: PROCEED (can succeed without asking)
        
    - benign: Consistent success (N/N)
        → Agent infers intent correctly without the info
        → Oracle decision: PROCEED (no need to ask)
        
    - new_task: 0/N success + 1 unique state (LLM judged as different task)
        → Deletion created a fundamentally different task
        → FILTERED OUT by default (not genuine ambiguity)

STEP 3: APPLY TARGET DISTRIBUTION (40/30/30 split)
    Default quotas ensure a balanced benchmark:
    - 40% outcome-critical: Tests if agent asks when it MUST
    - 30% divergent: Tests strategic asking under uncertainty
    - 30% benign: Tests if agent avoids unnecessary asking

STEP 4: ENSURE DIVERSITY
    Several mechanisms prevent over-representation:
    - max_per_task (default: 8): Caps samples per task
    - max_per_dimension: Optionally limit GOAL/INPUT/CONSTRAINT per task
    - Random shuffle if : Fair task representation


================================================================================
RECOMMENDED FILTERING COMMANDS
================================================================================

# Standard filtering (100 samples, 40/30/30 split)
python scripts/filter_tac_samples.py -e delete_exp -o outputs/filtered.csv

# With diversity controls
python scripts/filter_tac_samples.py -e delete_exp --max-per-task 8 -o outputs/filtered.csv

# Critical-only (for focused testing)
python scripts/filter_tac_samples.py -e delete_exp --critical-only

# Analyze without filtering (see distribution)
python scripts/filter_tac_samples.py -e delete_exp --analyze

================================================================================

Output Format (matches core.py UnderspecVariant):
{
  "variant_id": "finance_check_attendance_payroll_V_S3",
  "agent_prompt": "Create payroll using the attendance file...",
  "ambiguity_type": "input",
  "removed_segments": [...],
  "ambiguity_class": "outcome-critical | divergent | benign | new_task",
  "oracle_decision": "clarify | proceed",
  "terminal_states": [(0,1,0), (1,1,0), ...],
  ...
}
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import DEFAULTS, PATHS, SELECTED_TASKS  # noqa: E402
from scripts.process_tac_underspec import build_summary_df, load_variants  # noqa: E402

# Paths
SYNTH_OUTPUTS = REPO_ROOT / PATHS["synthetic_outputs"]
RUNS_DIR = REPO_ROOT / PATHS["runs_dir"]

# Sorted by length descending to match longest task name first (avoids prefix collision)
SELECTED_TASKS_SORTED = sorted(SELECTED_TASKS, key=len, reverse=True)
# Minimum number of trials to consider a variant for filtering
MIN_N_TRIALS = DEFAULTS["num_trials"]
OUTPUT_COLS = [
    "experiment",
    "variant_id",
    "task",
    "dimension",
    "subdimension",
    "n_trials",
    "pass_at_1",
    "pass_at_3",
    "pass_at_N",
    "baseline_pass@3",
    "delta_pass@3",
    "n_unique_states",
    "divergence_ratio",
    "state_entropy",
    "ambiguity_class",
    "oracle_decision",
    "criticality",
    "guessability",
    "priority_score",
    "removed_value",
    "removed_segments_json",
    "expected_questions",
]
REQUIRED_EXPORT_COLS = {
    "variant_id",
    "task",
    "dimension",
    "ambiguity_class",
    "oracle_decision",
    "underspecified_prompt",
    "removed_segments_json",
    "pass_at_1",
    "pass_at_3",
    "baseline_pass@3",
    "delta_pass@3",
    "criticality",
    "guessability",
    "priority_score",
}


def extract_task_from_variant_id(variant_id: str) -> str | None:
    """Extract task name from variant_id using longest-match-first to avoid prefix collisions."""
    return next((t for t in SELECTED_TASKS_SORTED if variant_id.startswith(t)), None)


def _load_baseline_from_pass_k_summaries() -> dict:
    """Scan per-run pass_k_summary CSVs to build baseline metrics.

    Returns: {task: {"pass@3": float, "avg_ckpt_acc": float}} averaged across
    all baseline runs (non-underspec) for selected tasks.
    """
    csvs = list(RUNS_DIR.glob("run_*/final_results/summaries/pass_k_summary_*.csv"))
    if not csvs:
        return {}

    task_set = set(SELECTED_TASKS)
    # Collect best pass@3 per (task, run) — multiple runs may cover same task
    task_scores: dict = {}  # task -> list of {"pass@3": float, "avg_ckpt_acc": float}

    for csv_path in csvs:
        run_dir_name = csv_path.parents[2].name
        if "underspec" in run_dir_name or "user_sim" in run_dir_name or "baseline_" in run_dir_name:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        for _, row in df.iterrows():
            task = row.get("task", "")
            if task == "AGGREGATE" or task not in task_set:
                continue
            p3 = float(row.get("pass@3", 0))
            avg_pct = float(row.get("avg_percent", 0))
            task_scores.setdefault(task, []).append(
                {
                    "pass@3": p3,
                    "avg_ckpt_acc": avg_pct / 100.0,
                }
            )

    # Average across all runs for each task
    result = {}
    for task, scores in task_scores.items():
        result[task] = {
            "pass@3": sum(s["pass@3"] for s in scores) / len(scores),
            "avg_ckpt_acc": sum(s["avg_ckpt_acc"] for s in scores) / len(scores),
        }
    return result


def load_baseline() -> dict:
    """Load baseline TAC results for selected tasks.

    Tries aggregated CSV first, then falls back to scanning per-run
    pass_k_summary CSVs from baseline runs.

    Returns: {task: {"pass@3": float, "avg_ckpt_acc": float}}
    """
    csv_path = RUNS_DIR / "aggregated_results_all_models.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            df = df[df["task"].isin(SELECTED_TASKS)]
            result = {}
            for task in SELECTED_TASKS:
                task_df = df[df["task"] == task]
                if not task_df.empty:
                    result[task] = {
                        "pass@3": task_df["pass@3"].mean(),
                        "avg_ckpt_acc": task_df["avg_ckpt_acc"].mean(),
                    }
            if result:
                return result
        except Exception:
            pass

    # Fallback: scan per-run pass_k_summary CSVs
    result = _load_baseline_from_pass_k_summaries()
    if result:
        print(f"  📋 Loaded baseline from per-run pass_k_summary CSVs ({len(result)} tasks)")
    else:
        print("  ⚠️  No baseline results found — baseline metrics will be 0")
    return result


def get_task_baseline(baseline: dict, task: str) -> dict:
    """Get baseline metrics for a specific task."""
    return baseline.get(task, {"pass@3": 0, "avg_ckpt_acc": 0})


def _resolve_exp_dirs(exp_key: str) -> tuple[list[str], str]:
    """Resolve experiment key to directory list and display name.

    If the exact directory exists under synthetic/outputs/, use it directly.
    Otherwise, expand to domain-specific directories (e.g. lhaw_tac -> lhaw_tac_ds,
    lhaw_tac_finance, lhaw_tac_hr, lhaw_tac_sde).
    """
    synth_outputs = Path(PATHS["synthetic_outputs"])
    if not synth_outputs.is_absolute():
        synth_outputs = Path(__file__).parent.parent / synth_outputs

    exact = synth_outputs / exp_key
    if exact.is_dir() and (exact / "underspec_candidates.csv").exists():
        return [exp_key], exp_key

    domains = ["ds", "finance", "hr", "sde"]
    expanded = []
    for domain in domains:
        domain_dir = synth_outputs / f"{exp_key}_{domain}"
        if domain_dir.is_dir() and (domain_dir / "underspec_candidates.csv").exists():
            expanded.append(f"{exp_key}_{domain}")

    if expanded:
        return expanded, exp_key

    return [exp_key], exp_key


def analyze_divergence(exp_key: str) -> pd.DataFrame:
    """
    Analyze outcome divergence across all variants in an experiment.

    Returns DataFrame with divergence metrics for each variant:
    - n_unique_states: Number of unique terminal states
    - divergence_ratio: Fraction of trials with unique states
    - state_entropy: Entropy of terminal state distribution
    - ambiguity_class: outcome-critical | divergent | benign | new_task
    """
    exp_dirs, _ = _resolve_exp_dirs(exp_key)
    exp_df = build_summary_df(exp_dirs)

    if exp_df.empty:
        return pd.DataFrame()

    # Filter to variants with trial results
    exp_df = exp_df[exp_df["n_trials"].notna()].copy()

    return exp_df


def print_divergence_analysis(exp_key: str):
    """Print detailed divergence analysis for an experiment."""
    df = analyze_divergence(exp_key)

    if df.empty:
        print(f"No data for {exp_key}")
        return

    _, exp_name = _resolve_exp_dirs(exp_key)
    print(f"\n{'='*80}")
    print(f"OUTCOME DIVERGENCE ANALYSIS: {exp_name}")
    print(f"{'='*80}")
    print(f"Total variants: {len(df)}")

    # Ambiguity class breakdown
    print("\n--- Ambiguity Classification ---")
    if "ambiguity_class" in df.columns:
        class_counts = df["ambiguity_class"].value_counts()
        for cls, count in class_counts.items():
            pct = count / len(df) * 100
            print(f"  {cls}: {count} ({pct:.1f}%)")

    # Divergence metrics
    print("\n--- Terminal State Divergence ---")
    if "n_unique_states" in df.columns:
        print(
            f"  Variants with 1 unique state (no divergence): {(df['n_unique_states'] == 1).sum()}"
        )
        print(f"  Variants with 2+ unique states (divergence): {(df['n_unique_states'] > 1).sum()}")
        print(f"  Avg unique states per variant: {df['n_unique_states'].mean():.2f}")
        print(f"  Max unique states: {df['n_unique_states'].max()}")

    if "state_entropy" in df.columns:
        print(f"  Avg state entropy: {df['state_entropy'].mean():.3f}")
        print(f"  Max state entropy: {df['state_entropy'].max():.3f}")

    # Show examples of high divergence variants
    if "divergence_ratio" in df.columns:
        high_div = df[df["divergence_ratio"] > 0.5].head(5)
        if not high_div.empty:
            print("\n--- High Divergence Examples (ratio > 0.5) ---")
            for _, row in high_div.iterrows():
                print(f"  {row['variant_id'][:50]}...")
                print(f"    States: {row.get('terminal_states', 'N/A')[:80]}")
                print(
                    f"    Divergence: {row['divergence_ratio']:.2f}, Entropy: {row.get('state_entropy', 0):.2f}"
                )

    # Per-task breakdown
    print("\n--- Per-Task Ambiguity Classification ---")
    df["task"] = df["variant_id"].apply(lambda vid: extract_task_from_variant_id(vid) or "unknown")

    if "ambiguity_class" in df.columns:
        task_class = df.groupby(["task", "ambiguity_class"]).size().unstack(fill_value=0)
        print(task_class.to_string())


def filter_samples(
    exp_key: str,
    max_per_task: int = 10,
    max_total: int = 100,
    critical_only: bool = False,
    max_per_dimension: int = 0,
    judge_model: str | None = None,
) -> pd.DataFrame:
    """
    Filter underspec candidates to create benchmark with 40/30/30 quota distribution.

    Target distribution:
        - 40% outcome-critical (agent MUST ask)
        - 30% divergent (agent SHOULD ask)
        - 30% benign (agent should NOT ask)
        - new_task variants are always filtered out (LLM judge detects these)

    Args:
        exp_key: Experiment name (matches directory under synthetic/outputs/)
        max_per_task: Maximum samples per task (default: 10)
        max_total: Maximum total samples (default: 100)
        critical_only: If True, only include outcome-critical (100% quota)
        max_per_dimension: Max variants per dimension per task (0=unlimited)
        judge_model: LLM model for new_task classification

    Returns:
        Filtered DataFrame with benchmark-ready samples
    """
    exp_dirs, _ = _resolve_exp_dirs(exp_key)

    # Load data with divergence analysis (optionally with LLM judge for new_task detection)
    exp_df = build_summary_df(exp_dirs, judge_model=judge_model)
    baseline_df = load_baseline()
    variants = {}
    for exp_dir in exp_dirs:
        variants.update(load_variants(exp_dir, include_prompts=True))

    if exp_df.empty:
        print(f"No data for experiment: {exp_key}")
        return pd.DataFrame()

    # Filter to variants with trial results
    exp_df = exp_df[exp_df["n_trials"].notna()].copy()

    # Skip incomplete samples (< 3 trials) - judge results may be unreliable
    incomplete = exp_df[exp_df["n_trials"] < MIN_N_TRIALS]
    if not incomplete.empty:
        print(f"\n⚠️  Skipping {len(incomplete)} incomplete variants (< {MIN_N_TRIALS} trials)")
        exp_df = exp_df[exp_df["n_trials"] >= MIN_N_TRIALS].copy()

    # Extract task from variant_id
    exp_df["task"] = exp_df["variant_id"].apply(extract_task_from_variant_id)
    exp_df = exp_df[exp_df["task"].notna()]

    # Add baseline metrics per task
    exp_df["baseline_pass@3"] = exp_df["task"].apply(
        lambda t: get_task_baseline(baseline_df, t)["pass@3"]
    )
    exp_df["baseline_ckpt_acc"] = exp_df["task"].apply(
        lambda t: get_task_baseline(baseline_df, t)["avg_ckpt_acc"]
    )

    # Calculate delta
    exp_df["delta_pass@3"] = exp_df["pass_at_3"] - exp_df["baseline_pass@3"]

    # Determine oracle_decision based on ambiguity_class
    # outcome-critical → clarify (agent always fails without this info)
    # divergent/benign → proceed (agent can succeed without this info)
    exp_df["oracle_decision"] = exp_df["ambiguity_class"].apply(
        lambda c: "clarify" if c == "outcome-critical" else "proceed"
    )

    # Extract base dimension for diversity tracking
    exp_df["base_dimension"] = exp_df["dimension"].apply(
        lambda x: str(x).split("+")[0] if pd.notna(x) else "unknown"
    )

    _, exp_name = _resolve_exp_dirs(exp_key)
    print(f"\n{'='*70}")
    print(f"Filtering: {exp_name}")
    print(f"{'='*70}")
    print(f"Total variants: {len(exp_df)}")

    # Show ambiguity class distribution before filtering
    if "ambiguity_class" in exp_df.columns:
        print("\nAmbiguity class distribution:")
        for cls, count in exp_df["ambiguity_class"].value_counts().items():
            print(f"  {cls}: {count} ({count/len(exp_df)*100:.1f}%)")

    # Show dimension distribution
    print("\nDimension distribution:")
    for dim, count in exp_df["base_dimension"].value_counts().items():
        print(f"  {dim}: {count} ({count/len(exp_df)*100:.1f}%)")

    # Filter 1: Ambiguity class selection (new_task always filtered out)
    if critical_only:
        class_filters = ["outcome-critical"]
    else:
        class_filters = ["outcome-critical", "divergent", "benign"]

    filter1 = exp_df[exp_df["ambiguity_class"].isin(class_filters)]
    print(f"\nAfter ambiguity_class filter: {len(filter1)} ({len(filter1)/len(exp_df)*100:.1f}%)")

    # Filter 2: Coverage - cap per task with DIVERSITY controls
    filtered_rows = []
    task_counts = defaultdict(int)
    task_dim_counts = defaultdict(lambda: defaultdict(int))  # task -> dim -> count

    # Sort by priority: outcome-critical first, then shuffle within each class
    # Shuffle ensures fair task/level representation instead of alphabetical bias
    filter1 = filter1.copy()
    filter1["n_levels"] = filter1["dimension"].apply(lambda x: len(str(x).split("+")))

    # Create priority groups by class, then shuffle within each class
    class_rank = {"outcome-critical": 0, "divergent": 1, "benign": 2}
    filter1["_class_rank"] = filter1["ambiguity_class"].map(class_rank).fillna(3)

    # Shuffle within each class to get random mix of 1-level and 2-level variants
    shuffled_dfs = []
    for cls_rank, group in filter1.groupby("_class_rank", sort=True):
        shuffled_dfs.append(group.sample(frac=1, random_state=42))  # Fixed seed for reproducibility
    filter1 = pd.concat(shuffled_dfs, ignore_index=True)

    filter1 = filter1.drop(columns=["_class_rank"])

    # === QUOTA-BASED SAMPLING ===
    class_available = filter1["ambiguity_class"].value_counts().to_dict()
    print(
        f"\n  Available: critical={class_available.get('outcome-critical', 0)}, "
        f"divergent={class_available.get('divergent', 0)}, benign={class_available.get('benign', 0)}"
    )

    # Target quotas: 40/30/30 split (or 100% critical if critical_only)
    if critical_only:
        quotas = {"outcome-critical": max_total, "divergent": 0, "benign": 0}
        print(f"  Quotas: critical={quotas['outcome-critical']} (critical-only mode)")
    else:
        q_critical = int(max_total * 0.40)
        q_divergent = int(max_total * 0.30)
        q_benign = max_total - q_critical - q_divergent  # absorbs rounding remainder
        quotas = {
            "outcome-critical": q_critical,
            "divergent": q_divergent,
            "benign": q_benign,
        }
        print(
            f"  Quotas: critical={quotas['outcome-critical']}, divergent={quotas['divergent']}, benign={quotas['benign']}"
        )

    class_counts = defaultdict(int)

    def can_add_sample(row) -> bool:
        """Check if variant passes filters (task limit, dimension limit)."""
        task = row["task"]
        base_dim = row["base_dimension"]

        # Respect per-task limit for all classes (diversity)
        if task_counts[task] >= max_per_task:
            return False

        # Check dimension limit (if enabled)
        if max_per_dimension > 0 and task_dim_counts[task][base_dim] >= max_per_dimension:
            return False

        return True

    # Process highest-priority classes first so they claim per-task slots before lower-priority
    for target_class in ["outcome-critical", "divergent", "benign"]:
        quota = quotas[target_class]
        class_rows = filter1[filter1["ambiguity_class"] == target_class]

        for _, row in class_rows.iterrows():
            if class_counts[target_class] >= quota:
                break  # Hit quota for this class
            if can_add_sample(row):
                filtered_rows.append(row)
                task_counts[row["task"]] += 1
                task_dim_counts[row["task"]][row["base_dimension"]] += 1
                class_counts[target_class] += 1

    final_df = pd.DataFrame(filtered_rows)
    print(f"After per-task cap ({max_per_task}/task, {max_total} total): {len(final_df)}")

    # Show diversity stats
    if not final_df.empty:
        print("\nDiversity in selected samples:")
        print("  Dimension distribution:")
        for dim, count in final_df["base_dimension"].value_counts().items():
            print(f"    {dim}: {count} ({count/len(final_df)*100:.1f}%)")

    # Add variant metadata for output (NO truncation - full data for JSON export)
    if not final_df.empty:
        # Add prompts from variants dict (loaded with include_prompts=True)
        final_df["expected_questions"] = final_df["variant_id"].apply(
            lambda vid: variants.get(vid, {}).get("expected_questions", "")
        )
        final_df["underspecified_prompt"] = final_df["variant_id"].apply(
            lambda vid: variants.get(vid, {}).get("underspecified_prompt", "")
        )
        # Add full segment JSON (preserves segment IDs for benchmark export)
        final_df["removed_segments_json"] = final_df["variant_id"].apply(
            lambda vid: variants.get(vid, {}).get("removed_segments_json", "")
        )

    return final_df


def print_summary(df: pd.DataFrame, exp_key: str):
    """Print summary of filtered samples."""
    if df.empty:
        print("No samples passed filters.")
        return

    _, exp_name = _resolve_exp_dirs(exp_key)
    print(f"\n--- Summary: {exp_name} ---")
    print(f"Total samples: {len(df)}")

    # Ambiguity class breakdown
    if "ambiguity_class" in df.columns:
        print("\nAmbiguity Class:")
        for cls, count in df["ambiguity_class"].value_counts().items():
            print(f"  {cls}: {count} ({count/len(df)*100:.1f}%)")

    # Oracle decision
    if "oracle_decision" in df.columns:
        print("\nOracle Decision:")
        for dec, count in df["oracle_decision"].value_counts().items():
            print(f"  {dec}: {count} ({count/len(df)*100:.1f}%)")

    # Divergence stats
    if "n_unique_states" in df.columns:
        print("\nDivergence Stats:")
        print(f"  Avg unique terminal states: {df['n_unique_states'].mean():.2f}")
        print(f"  Avg divergence ratio: {df['divergence_ratio'].mean():.2f}")

    print("\nPerformance Impact:")
    print(f"  Avg delta pass@3: {df['delta_pass@3'].mean()*100:.1f}pp")
    if "priority_score" in df.columns:
        print(f"  Avg priority_score: {df['priority_score'].mean():.2f}")

    # Per-task breakdown
    print("\n--- Per-Task Breakdown ---")
    task_summary = (
        df.groupby("task")
        .agg(
            {
                "variant_id": "count",
                "ambiguity_class": lambda x: (x == "outcome-critical").sum(),
                "delta_pass@3": "mean",
                "n_unique_states": "mean",
            }
        )
        .rename(
            columns={
                "variant_id": "n_samples",
                "ambiguity_class": "n_critical",
                "n_unique_states": "avg_states",
            }
        )
    )

    print(f"{'Task':<40} {'Samples':>8} {'Critical':>8} {'AvgStates':>10} {'Delta':>10}")
    print("-" * 80)
    for task, row in task_summary.iterrows():
        task_short = task[:39]
        print(
            f"{task_short:<40} {int(row['n_samples']):>8} {int(row['n_critical']):>8} {row['avg_states']:>10.1f} {row['delta_pass@3']*100:>9.1f}pp"
        )

    # Dimension breakdown
    print("\n--- Dimension Breakdown ---")
    df["base_dim"] = df["dimension"].apply(lambda x: str(x).split("+")[0])
    dim_counts = df["base_dim"].value_counts()
    for dim, count in dim_counts.items():
        print(f"  {dim}: {count} ({count/len(df)*100:.1f}%)")


def export_benchmark_format(df: pd.DataFrame, output_path: Path):
    """
    Export filtered samples in the benchmark-ready JSON format from core.py.

    Output format matches UnderspecVariant + additional analysis fields.

    Requires removed_segments_json column (preserves segment IDs from core.py).
    Skips samples without this data.
    """
    missing_cols = sorted(REQUIRED_EXPORT_COLS - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns for export: {', '.join(missing_cols)}")

    samples = []
    skipped = 0

    for _, row in df.iterrows():
        # Require full segment JSON (preserves original segment IDs)
        removed_segments_json = row["removed_segments_json"]
        if removed_segments_json in ("", "nan", "[]"):
            skipped += 1
            continue

        try:
            removed_segments = json.loads(removed_segments_json)
        except (json.JSONDecodeError, TypeError):
            skipped += 1
            continue

        if not removed_segments:
            skipped += 1
            continue

        # Keep full dimension string for multi-segment variants when available
        primary_dim = removed_segments[0].get("dimension", "unknown")
        full_dim = row.get("dimension") or primary_dim

        # Parse expected_questions from source JSON string -> store as list
        raw_eq = row.get("expected_questions", "")
        try:
            expected_questions = json.loads(raw_eq) if raw_eq else []
        except (json.JSONDecodeError, TypeError):
            expected_questions = []

        sample = {
            "variant_id": row["variant_id"],
            "task": row["task"],
            "agent_prompt": row["underspecified_prompt"],
            # Preserve full dimension string (e.g., "goal+input") when present
            "ambiguity_type": full_dim,
            "removed_segments": removed_segments,
            # Classification from pass@N analysis (required fields)
            "ambiguity_class": row[
                "ambiguity_class"
            ],  # Required: outcome-critical|divergent|benign|new_task
            "oracle_decision": row["oracle_decision"],  # Required: clarify|proceed
            "expected_questions": expected_questions,  # List of {segment_id, questions}
            # Divergence metrics
            "n_unique_terminal_states": int(row.get("n_unique_states", 1)),
            "divergence_ratio": row.get("divergence_ratio", 0),
            "state_entropy": row.get("state_entropy", 0),
            "terminal_states": row.get("terminal_states", ""),
            # Performance metrics
            "pass_at_1": row["pass_at_1"],
            "pass_at_3": row["pass_at_3"],
            "baseline_pass_at_3": row["baseline_pass@3"],
            "delta_pass_at_3": row["delta_pass@3"],
            "criticality": float(row["criticality"]) if pd.notna(row["criticality"]) else 0.0,
            "guessability": float(row["guessability"]) if pd.notna(row["guessability"]) else 0.0,
            "priority_score": (
                float(row["priority_score"]) if pd.notna(row["priority_score"]) else 0.0
            ),
            # VOI cost structure (placeholder for future calibration)
            "costs": {
                "clarification": 0.2,  # Default: interruption cost
                "wrong_outcome": 1.0,  # Default: failure cost
            },
        }
        samples.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"\nExported benchmark format: {output_path}")
    print(f"  {len(samples)} samples exported, {skipped} skipped (missing removed_segments_json)")


def main():
    parser = argparse.ArgumentParser(
        description="Filter LHAW benchmark samples (40/30/30 quota: critical/divergent/benign)"
    )
    parser.add_argument(
        "--experiment", "-e", help="Experiment name (matches directory under synthetic/outputs/)"
    )
    parser.add_argument(
        "--analyze", "-a", action="store_true", help="Show divergence analysis without filtering"
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=DEFAULTS["max_per_task_tac"],
        help=f"Max samples per task (default: {DEFAULTS['max_per_task_tac']})",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=DEFAULTS["max_total"],
        help=f"Max total samples (default: {DEFAULTS['max_total']})",
    )
    parser.add_argument(
        "--critical-only", action="store_true", help="Only outcome-critical variants (100%% quota)"
    )
    parser.add_argument(
        "--max-per-dimension", type=int, default=0, help="Max per dimension per task (0=unlimited)"
    )
    parser.add_argument("--output", "-o", type=Path, help="Output path (.csv and .json written)")
    parser.add_argument(
        "--judge-model",
        default="gemini/gemini-3-flash-preview",
        help="LLM model for new_task classification (default: gemini-3-flash-preview)",
    )
    args = parser.parse_args()

    if not args.experiment and not args.analyze:
        parser.print_help()
        return

    # Analyze mode
    if args.analyze:
        if not args.experiment:
            parser.error("--analyze requires --experiment")
        print_divergence_analysis(args.experiment)
        return

    # Filter mode
    print(f"🧑‍⚖️ Using LLM judge: {args.judge_model}")

    df = filter_samples(
        exp_key=args.experiment,
        max_per_task=args.max_per_task,
        max_total=args.max_total,
        critical_only=args.critical_only,
        max_per_dimension=args.max_per_dimension,
        judge_model=args.judge_model,
    )

    if df.empty:
        print("No samples passed filters.")
        return

    df["experiment"] = args.experiment
    print_summary(df, args.experiment)

    # Save output
    default_dir = SYNTH_OUTPUTS / "benchmark_filtered"
    default_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or default_dir / f"{args.experiment}_filtered.csv"
    filtered_output_cols = [c for c in OUTPUT_COLS if c in df.columns]

    df[filtered_output_cols].to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n{'='*70}")
    print(f"Total benchmark samples: {len(df)}")
    print(f"{'='*70}")
    print(f"CSV: {output_path}")

    json_path = output_path.with_suffix(".json")
    export_benchmark_format(df, json_path)


if __name__ == "__main__":
    main()
