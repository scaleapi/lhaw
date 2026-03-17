"""
Generate pass@3 comparison charts from explicit run directories.

Usage:
    python plot_pass3_from_runs.py \
        --baseline-run /path/to/run_sampled_baseline_o3_YYYYMMDD/ \
        --no-user-run /path/to/run_sampled_underspecified_o3_YYYYMMDD/ \
        --with-user-run /path/to/run_sampled_underspecified_with_user_o3_YYYYMMDD/ \
        --output-dir /path/to/output/ \
        --model-name "o3" \
        --mapping-csv /path/to/sampled_filtered_underspec_with_baseline.csv

Creates:
    - pass3_comparison.png - Bar chart with 3 bars (baseline, no_user, with_user)
    - pass3_by_ambiguity.png - Breakdown by ambiguity class (if --mapping-csv provided)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import load_ask_user_counts

AMBIGUITY_CLASSES = ["benign", "divergent", "outcome-critical"]
AMBIGUITY_COLORS = {
    "benign": "#2ecc71",
    "divergent": "#f39c12",
    "outcome-critical": "#e74c3c",
}


def load_run_pass3(run_dir: Path) -> pd.DataFrame:
    """Load pass@3 data from a run directory.

    Args:
        run_dir: Path to run directory

    Returns:
        DataFrame with columns: task_id, pass@3
    """
    csv_path = run_dir / "pass_k_per_task.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run pass_k file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Filter out AGGREGATE row
    df = df[df["task_id"] != "AGGREGATE"]
    return df[["task_id", "pass@3"]]


def load_task_mapping(mapping_csv: Path) -> pd.DataFrame:
    """Load variant-to-task mapping from sampled CSV.

    Args:
        mapping_csv: Path to sampled_filtered_underspec_with_baseline.csv

    Returns:
        DataFrame with variant_index, task, and ambiguity_class columns
    """
    df = pd.read_csv(mapping_csv)
    # Add variant_index based on row position
    df = df.reset_index().rename(columns={"index": "variant_index"})
    df["variant_index"] = df["variant_index"].astype(int)
    return df[["variant_index", "task", "ambiguity_class"]]


def plot_comparison(
    model_name: str,
    baseline_pass3: float,
    no_user_pass3: float,
    with_user_pass3: float,
    output_path: Path,
    ambiguity_data: dict = None,
) -> None:
    """Generate bar chart comparing baseline vs underspec conditions.

    Args:
        model_name: Display name of the model
        baseline_pass3: Mean pass@3 for baseline
        no_user_pass3: Mean pass@3 for underspec (no user tool)
        with_user_pass3: Mean pass@3 for underspec (with user tool)
        output_path: Path to save the chart
        ambiguity_data: Optional dict with per-ambiguity-class data:
            {
                "benign": {"baseline": 1.0, "no_user": 0.8, "with_user": 0.9, "count": 20},
                ...
            }
    """
    colors = ["#7f8c8d", "#3498db", "#2ecc71"]

    if ambiguity_data:
        # Create grouped bar chart with Overall + each ambiguity class
        classes = [c for c in AMBIGUITY_CLASSES if c in ambiguity_data]
        categories = ["Overall"] + classes
        n_categories = len(categories)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(n_categories)
        width = 0.25

        # Collect values for each condition
        baseline_vals = [baseline_pass3] + [ambiguity_data[c]["baseline"] for c in classes]
        no_user_vals = [no_user_pass3] + [ambiguity_data[c]["no_user"] for c in classes]
        with_user_vals = [with_user_pass3] + [ambiguity_data[c]["with_user"] for c in classes]

        # Create bars
        bars_baseline = ax.bar(
            x - width,
            baseline_vals,
            width,
            label="Baseline",
            color=colors[0],
            edgecolor="black",
            linewidth=0.5,
        )
        bars_no_user = ax.bar(
            x,
            no_user_vals,
            width,
            label="Underspec (no user tool)",
            color=colors[1],
            edgecolor="black",
            linewidth=0.5,
        )
        bars_with_user = ax.bar(
            x + width,
            with_user_vals,
            width,
            label="Underspec (with user tool)",
            color=colors[2],
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels on bars
        for bars in [bars_baseline, bars_no_user, bars_with_user]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # Format category labels with counts
        total_count = sum(ambiguity_data[c]["count"] for c in classes)
        category_labels = [f"Overall\n(n={total_count})"]
        category_labels += [f"{c}\n(n={ambiguity_data[c]['count']})" for c in classes]

        ax.set_xticks(x)
        ax.set_xticklabels(category_labels, fontsize=10)
        ax.legend(loc="upper right")
    else:
        # Simple 3-bar chart for overall comparison only
        conditions = ["Baseline", "Underspec\n(no user tool)", "Underspec\n(with user tool)"]
        values = [baseline_pass3, no_user_pass3, with_user_pass3]

        fig, ax = plt.subplots(figsize=(8, 6))

        x = np.arange(len(conditions))
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=11)

    # Common formatting
    ax.set_ylabel("Mean Pass@3", fontsize=12)
    ax.set_title(f"{model_name}: Pass@3 Comparison", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart to: {output_path}")


def plot_ambiguity_breakdown(
    model_name: str,
    data: dict,
    output_path: Path,
) -> None:
    """Generate bar chart showing pass@3 by ambiguity class.

    Args:
        model_name: Display name of the model
        data: Dict with structure:
            {
                "benign": {"baseline": 1.0, "no_user": 0.8, "with_user": 0.9, "count": 20, "ask_user_mean": 0.5},
                "divergent": {...},
                "outcome-critical": {...},
            }
        output_path: Path to save the chart
    """
    classes = [c for c in AMBIGUITY_CLASSES if c in data]
    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract values
    baseline_vals = [data[c]["baseline"] for c in classes]
    no_user_vals = [data[c]["no_user"] for c in classes]
    with_user_vals = [data[c]["with_user"] for c in classes]
    counts = [data[c]["count"] for c in classes]
    ask_user_means = [data[c].get("ask_user_mean", 0) for c in classes]

    # Get actual total ask_user calls (stored with special key)
    total_ask_user = data.get("_total_ask_user", 0)

    # Create bars
    bars_baseline = ax.bar(
        x - width,
        baseline_vals,
        width,
        label="Baseline",
        color="#7f8c8d",
        edgecolor="black",
        linewidth=0.5,
    )
    bars_no_user = ax.bar(
        x,
        no_user_vals,
        width,
        label="Underspec (no user tool)",
        color="#3498db",
        edgecolor="black",
        linewidth=0.5,
    )
    bars_with_user = ax.bar(
        x + width,
        with_user_vals,
        width,
        label="Underspec (with user tool)",
        color="#2ecc71",
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels on bars
    for bars in [bars_baseline, bars_no_user, bars_with_user]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Format class labels with counts
    class_labels = [f"{c}\n(n={counts[i]})" for i, c in enumerate(classes)]

    # Formatting
    ax.set_xlabel("Ambiguity Class", fontsize=12)
    ax.set_ylabel("Mean Pass@3", fontsize=12)
    ax.set_title(f"{model_name}: Pass@3 by Ambiguity Class", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Add secondary y-axis for ask_user counts
    ax2 = ax.twinx()
    ax2.plot(
        x + width,
        ask_user_means,
        marker="D",
        markersize=8,
        color="#e74c3c",
        linewidth=2,
        label=f"Mean ask_user calls (total: {total_ask_user:.1f})",
        alpha=0.6,
    )
    # Add value labels for ask_user
    for i, (xi, val) in enumerate(zip(x + width, ask_user_means)):
        ax2.annotate(
            f"{val:.2f}",
            xy=(xi, val),
            xytext=(5, 5),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9,
            color="#e74c3c",
            fontweight="bold",
            alpha=0.7,
        )
    ax2.set_ylabel("Mean ask_user Calls", fontsize=12, color="#e74c3c", alpha=0.8)
    ax2.tick_params(axis="y", labelcolor="#e74c3c")
    ax2.set_ylim(0, max(ask_user_means) * 1.5 if max(ask_user_means) > 0 else 1)
    ax2.legend(loc="upper left", framealpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pass@3 comparison charts from explicit run directories"
    )
    parser.add_argument(
        "--baseline-run",
        required=True,
        help="Path to baseline run directory",
    )
    parser.add_argument(
        "--no-user-run",
        required=True,
        help="Path to underspec (no user tool) run directory",
    )
    parser.add_argument(
        "--with-user-run",
        required=True,
        help="Path to underspec (with user tool) run directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for charts",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name for chart title (e.g., 'o3')",
    )
    parser.add_argument(
        "--mapping-csv",
        help="Path to sampled_filtered_underspec_with_baseline.csv for ambiguity breakdown",
    )
    args = parser.parse_args()

    baseline_run = Path(args.baseline_run)
    no_user_run = Path(args.no_user_run)
    with_user_run = Path(args.with_user_run)
    output_dir = Path(args.output_dir)

    # Validate paths
    for path, name in [
        (baseline_run, "baseline-run"),
        (no_user_run, "no-user-run"),
        (with_user_run, "with-user-run"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} directory not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load pass@3 data from each run
    print("Loading pass@3 data...")
    baseline_df = load_run_pass3(baseline_run)
    no_user_df = load_run_pass3(no_user_run)
    with_user_df = load_run_pass3(with_user_run)

    print(f"  Baseline: {len(baseline_df)} tasks")
    print(f"  No user tool: {len(no_user_df)} tasks")
    print(f"  With user tool: {len(with_user_df)} tasks")

    # Compute mean pass@3 for each condition
    baseline_mean = baseline_df["pass@3"].mean()
    no_user_mean = no_user_df["pass@3"].mean()
    with_user_mean = with_user_df["pass@3"].mean()

    print("\nMean pass@3:")
    print(f"  Baseline: {baseline_mean:.3f}")
    print(f"  No user tool: {no_user_mean:.3f}")
    print(f"  With user tool: {with_user_mean:.3f}")

    # Compute ambiguity breakdown if mapping CSV provided
    ambiguity_data = None
    if args.mapping_csv:
        mapping_csv = Path(args.mapping_csv)
        if not mapping_csv.exists():
            print(f"Warning: mapping CSV not found: {mapping_csv}")
        else:
            print("\nComputing ambiguity breakdown...")
            mapping_df = load_task_mapping(mapping_csv)

            # Load ask_user counts from with_user run
            ask_user_df, total_ask_user_calls = load_ask_user_counts(with_user_run)
            print(
                f"  Loaded ask_user counts for {len(ask_user_df)} tasks (total: {total_ask_user_calls})"
            )

            # For baseline and with_user runs, task_id should be actual task names
            # For no_user run, task_id may be numeric indices
            # Check if no_user_df has numeric task_ids
            first_no_user_id = str(no_user_df["task_id"].iloc[0])
            no_user_uses_indices = first_no_user_id.isdigit()

            if no_user_uses_indices:
                # Map variant indices to tasks
                no_user_df = no_user_df.copy()
                no_user_df["variant_index"] = no_user_df["task_id"].astype(int)
                no_user_df = no_user_df.merge(
                    mapping_df[["variant_index", "task", "ambiguity_class"]],
                    on="variant_index",
                    how="left",
                )
            else:
                # Merge with mapping to get ambiguity_class
                no_user_df = no_user_df.merge(
                    mapping_df[["task", "ambiguity_class"]],
                    left_on="task_id",
                    right_on="task",
                    how="left",
                )

            # Add ambiguity_class to baseline and with_user
            baseline_df = baseline_df.merge(
                mapping_df[["task", "ambiguity_class"]],
                left_on="task_id",
                right_on="task",
                how="left",
            )
            with_user_df = with_user_df.merge(
                mapping_df[["task", "ambiguity_class"]],
                left_on="task_id",
                right_on="task",
                how="left",
            )

            # Add ask_user counts to with_user_df
            if not ask_user_df.empty:
                with_user_df = with_user_df.merge(ask_user_df, on="task", how="left")
                with_user_df["ask_user_count"] = with_user_df["ask_user_count"].fillna(0)

            # Compute per-ambiguity-class stats
            ambiguity_data = {}
            for amb_class in AMBIGUITY_CLASSES:
                baseline_class = baseline_df[baseline_df["ambiguity_class"] == amb_class]
                no_user_class = no_user_df[no_user_df["ambiguity_class"] == amb_class]
                with_user_class = with_user_df[with_user_df["ambiguity_class"] == amb_class]

                if len(baseline_class) == 0:
                    continue

                ambiguity_data[amb_class] = {
                    "baseline": baseline_class["pass@3"].mean(),
                    "no_user": no_user_class["pass@3"].mean() if len(no_user_class) > 0 else 0,
                    "with_user": (
                        with_user_class["pass@3"].mean() if len(with_user_class) > 0 else 0
                    ),
                    "count": len(with_user_class),  # Use underspec task count, not baseline
                }

                if "ask_user_count" in with_user_class.columns:
                    ambiguity_data[amb_class]["ask_user_mean"] = with_user_class[
                        "ask_user_count"
                    ].mean()

                print(
                    f"  {amb_class}: baseline={ambiguity_data[amb_class]['baseline']:.3f}, "
                    f"no_user={ambiguity_data[amb_class]['no_user']:.3f}, "
                    f"with_user={ambiguity_data[amb_class]['with_user']:.3f} "
                    f"(n={ambiguity_data[amb_class]['count']})"
                )

            # Store total ask_user calls for legend
            ambiguity_data["_total_ask_user"] = total_ask_user_calls

    # Generate main comparison chart (includes ambiguity breakdown if available)
    safe_name = args.model_name.replace("/", "_").replace(" ", "_")
    comparison_path = output_dir / f"{safe_name}_pass3_comparison.png"
    plot_comparison(
        args.model_name,
        baseline_mean,
        no_user_mean,
        with_user_mean,
        comparison_path,
        ambiguity_data=ambiguity_data,
    )

    # Also generate separate ambiguity breakdown chart with ask_user counts
    if ambiguity_data:
        ambiguity_path = output_dir / f"{safe_name}_pass3_by_ambiguity.png"
        plot_ambiguity_breakdown(args.model_name, ambiguity_data, ambiguity_path)

    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
