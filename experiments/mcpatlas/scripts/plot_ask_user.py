#!/usr/bin/env python
"""
Plotting script for ask_user analysis results.

Usage:
    python plot_ask_user.py \
        --input ask_user_analysis.csv \
        --output-dir plots/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from evaluation.ask_user_types import ALL_FLAGS, FLAG_CATEGORIES  # noqa: E402


def load_analysis_csv(csv_path: Path) -> pd.DataFrame:
    """Load and prepare analysis CSV.

    Args:
        csv_path: Path to the analysis CSV

    Returns:
        DataFrame with parsed columns
    """
    df = pd.read_csv(csv_path)

    # Parse JSON columns
    if "questions_json" in df.columns:
        df["questions_json"] = df["questions_json"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )

    if "failure_categories" in df.columns:
        df["failure_categories"] = df["failure_categories"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )

    return df


def plot_flag_rates_by_ambiguity(df: pd.DataFrame, output_dir: Path, prefix: str = ""):
    """Plot flag rates grouped by ambiguity class.

    Args:
        df: Analysis DataFrame
        output_dir: Output directory for plots
        prefix: Filename prefix (e.g. "gemini_3_flash_")
    """
    if "ambiguity_class" not in df.columns:
        print("Skipping flag rates plot: no ambiguity_class column")
        return

    # Filter to rows with judgments
    flag_cols = [f for f in ALL_FLAGS if f in df.columns]
    if not flag_cols:
        print("Skipping flag rates plot: no flag columns found")
        return

    # Get unique ambiguity classes
    ambiguity_classes = df["ambiguity_class"].dropna().unique()
    if len(ambiguity_classes) == 0:
        print("Skipping flag rates plot: no ambiguity classes")
        return

    # Calculate flag rates per ambiguity class
    rates_data = []
    for ac in ambiguity_classes:
        ac_df = df[df["ambiguity_class"] == ac]
        n_rows = len(ac_df)
        if n_rows == 0:
            continue

        for flag in flag_cols:
            rate = ac_df[flag].sum() / n_rows if n_rows > 0 else 0
            rates_data.append(
                {
                    "ambiguity_class": ac,
                    "flag": flag.replace("flag_", "").replace("_", " "),
                    "rate": rate,
                }
            )

    rates_df = pd.DataFrame(rates_data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 8))

    # Pivot for grouped bars
    pivot_df = rates_df.pivot(index="flag", columns="ambiguity_class", values="rate")
    pivot_df.plot(kind="bar", ax=ax, width=0.8)

    ax.set_xlabel("Flag")
    ax.set_ylabel("Rate (fraction of trials)")
    ax.set_title("Flag Rates by Ambiguity Class")
    ax.legend(title="Ambiguity Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    fname = f"{prefix}flag_rates_by_ambiguity.png"
    plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_question_count_distribution(df: pd.DataFrame, output_dir: Path, prefix: str = ""):
    """Plot histogram of ask_user counts by ambiguity class.

    Args:
        df: Analysis DataFrame
        output_dir: Output directory for plots
        prefix: Filename prefix
    """
    if "ask_user_count" not in df.columns:
        print("Skipping question count plot: no ask_user_count column")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ambiguity_classes = []
    if "ambiguity_class" in df.columns:
        ambiguity_classes = list(df["ambiguity_class"].dropna().unique())

    if ambiguity_classes:
        for ac in ambiguity_classes:
            ac_df = df[df["ambiguity_class"] == ac]
            counts = ac_df["ask_user_count"].value_counts().sort_index()
            ax.bar(
                counts.index + 0.2 * ambiguity_classes.index(ac),
                counts.values,
                width=0.2,
                label=ac,
                alpha=0.7,
            )
        ax.legend(title="Ambiguity Class")
    else:
        counts = df["ask_user_count"].value_counts().sort_index()
        ax.bar(counts.index, counts.values, alpha=0.7)

    ax.set_xlabel("Number of ask_user Calls")
    ax.set_ylabel("Number of Trials")
    ax.set_title("Distribution of ask_user Call Counts")
    ax.set_xticks(range(int(df["ask_user_count"].max()) + 1))

    plt.tight_layout()
    fname = f"{prefix}question_count_distribution.png"
    plt.savefig(output_dir / fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_failure_category_heatmap(df: pd.DataFrame, output_dir: Path, prefix: str = ""):
    """Plot heatmap of failure categories by ambiguity class.

    Args:
        df: Analysis DataFrame
        output_dir: Output directory for plots
        prefix: Filename prefix
    """
    if "failure_categories" not in df.columns or "ambiguity_class" not in df.columns:
        print("Skipping failure category heatmap: missing required columns")
        return

    # Count failures per category per ambiguity class
    all_categories = list(FLAG_CATEGORIES.keys())
    ambiguity_classes = df["ambiguity_class"].dropna().unique()

    if len(ambiguity_classes) == 0:
        print("Skipping failure category heatmap: no ambiguity classes")
        return

    # Map old LLM-generated names to our canonical names
    category_aliases = {
        "Question Quality Deficiency": "Question Quality",
        "Question Targeting Error": "Question Targeting",
        "Question Targeting Errors": "Question Targeting",
        "Information Integration Failure": "Information Integration",
        "Over-Clarification Issues": "Over-Clarification",
        "Over-Clarification": "Over-Clarification",
        "Under-Clarification Issues": "Under-Clarification",
        "Under-Clarification": "Under-Clarification",
        "Timing & Strategy Error": "Timing & Strategy",
        "Timing & Strategy Errors": "Timing & Strategy",
        "User Response Misinterpretation": "User Response Misinterpretation",
    }

    # Track unknown categories
    unknown_categories = set()

    # Build matrix
    matrix = np.zeros((len(all_categories), len(ambiguity_classes)))

    for j, ac in enumerate(ambiguity_classes):
        ac_df = df[df["ambiguity_class"] == ac]
        n_rows = len(ac_df)
        if n_rows == 0:
            continue

        for i, cat in enumerate(all_categories):
            # Count rows where this category appears in failure_categories
            # Handle both canonical names and LLM-generated aliases
            count = 0
            for cats in ac_df["failure_categories"]:
                if not isinstance(cats, list):
                    continue
                # Normalize category names and track unknowns
                normalized = []
                for c in cats:
                    if c in category_aliases:
                        normalized.append(category_aliases[c])
                    elif c in all_categories:
                        normalized.append(c)
                    else:
                        unknown_categories.add(c)
                if cat in normalized:
                    count += 1
            matrix[i, j] = count / n_rows

    # Warn about unknown categories
    if unknown_categories:
        print(f"  Warning: Unknown failure categories ignored: {sorted(unknown_categories)}")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        matrix,
        xticklabels=ambiguity_classes,
        yticklabels=all_categories,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Ambiguity Class")
    ax.set_ylabel("Failure Category")
    ax.set_title("Failure Category Rates by Ambiguity Class")

    plt.tight_layout()
    fname = f"{prefix}failure_category_heatmap.png"
    plt.savefig(output_dir / fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def plot_flag_correlation_matrix(df: pd.DataFrame, output_dir: Path, prefix: str = ""):
    """Plot correlation matrix of flags.

    Args:
        df: Analysis DataFrame
        output_dir: Output directory for plots
        prefix: Filename prefix
    """
    flag_cols = [f for f in ALL_FLAGS if f in df.columns]
    if len(flag_cols) < 2:
        print("Skipping flag correlation plot: not enough flag columns")
        return

    # Get flag data
    flag_df = df[flag_cols].copy()

    # Convert to numeric
    for col in flag_cols:
        flag_df[col] = pd.to_numeric(flag_df[col], errors="coerce").fillna(0)

    # Compute correlation
    corr = flag_df.corr()

    # Rename columns for readability
    short_names = [f.replace("flag_", "").replace("_", " ")[:20] for f in flag_cols]
    corr.columns = short_names
    corr.index = short_names

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 12))

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        vmin=-1,
        vmax=1,
        square=True,
    )

    ax.set_title("Flag Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    fname = f"{prefix}flag_correlation_matrix.png"
    plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_coverage_vs_questions(df: pd.DataFrame, output_dir: Path, prefix: str = ""):
    """Plot coverage score vs number of questions.

    Args:
        df: Analysis DataFrame
        output_dir: Output directory for plots
        prefix: Filename prefix
    """
    if "coverage_score" not in df.columns or "ask_user_count" not in df.columns:
        print("Skipping coverage vs questions plot: missing columns")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ambiguity_classes = []
    if "ambiguity_class" in df.columns:
        ambiguity_classes = list(df["ambiguity_class"].dropna().unique())

    if ambiguity_classes:
        for ac in ambiguity_classes:
            ac_df = df[df["ambiguity_class"] == ac]
            ax.scatter(
                ac_df["ask_user_count"],
                ac_df["coverage_score"],
                label=ac,
                alpha=0.6,
            )
        ax.legend(title="Ambiguity Class")
    else:
        ax.scatter(df["ask_user_count"], df["coverage_score"], alpha=0.6)

    ax.set_xlabel("Number of ask_user Calls")
    ax.set_ylabel("Coverage Score")
    ax.set_title("Coverage Score vs Number of Questions Asked")

    plt.tight_layout()
    fname = f"{prefix}coverage_vs_questions.png"
    plt.savefig(output_dir / fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")


def main():
    # Import here to avoid circular imports
    from aggregate_passed_tasks import PASS_THRESHOLD

    parser = argparse.ArgumentParser(description="Generate plots from ask_user analysis CSV")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to analysis CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--filter-passed",
        type=str,
        choices=["true", "false"],
        default=None,
        help=f"Filter by pass/fail (coverage_score >= {PASS_THRESHOLD}). 'true' = passed only, 'false' = failed only",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for output filenames (e.g. model name). Files become {prefix}_question_count_distribution.png",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading {args.input}...")
    df = load_analysis_csv(args.input)
    print(f"Loaded {len(df)} rows")

    # Apply pass/fail filter
    if args.filter_passed is not None:
        if "coverage_score" not in df.columns:
            print("Warning: No coverage_score column, cannot filter by passed")
        else:
            if args.filter_passed == "true":
                df = df[df["coverage_score"] >= PASS_THRESHOLD]
                print(f"Filtered to {len(df)} passed rows (coverage_score >= {PASS_THRESHOLD})")
            else:
                df = df[df["coverage_score"] < PASS_THRESHOLD]
                print(f"Filtered to {len(df)} failed rows (coverage_score < {PASS_THRESHOLD})")

    # Generate plots
    prefix = f"{args.prefix}_" if args.prefix else ""
    print(f"\nGenerating plots (prefix='{prefix}')...")
    plot_flag_rates_by_ambiguity(df, args.output_dir, prefix)
    plot_question_count_distribution(df, args.output_dir, prefix)
    plot_failure_category_heatmap(df, args.output_dir, prefix)
    plot_flag_correlation_matrix(df, args.output_dir, prefix)
    plot_coverage_vs_questions(df, args.output_dir, prefix)

    print(f"\nAll plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
