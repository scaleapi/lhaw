"""
Generate pass@3 comparison bar charts for sampled underspecification runs.

Usage:
    python plot_pass3_comparison.py --dataset-dir /path/to/dataset --config runs_config.yaml

    # Or with legacy hardcoded defaults (no --config):
    python plot_pass3_comparison.py --dataset-dir /path/to/dataset

Creates sampled_runs/ subdirectory with:
    - pass3_comparison_by_model.png
    - pass3_by_ambiguity_<model>.png (one per model)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from utils import load_ask_user_counts

# Legacy hardcoded defaults — use --config to override
DEFAULT_MODELS = {
    "Claude Opus 4.5": {
        "baseline_prefix": "claude",
        "no_user": "run_sampled_underspecified_claude-opus-4-5-20251101_20260123_060406",
        "with_user": "run_sampled_underspecified_with_user_new_claude-opus-4-5-20251101_20260128_044057",
    },
    "Gemini 3 Pro": {
        "baseline_prefix": "gemini",
        "no_user": "run_sampled_underspecified_gemini-3-pro-preview_20260123_060447",
        "with_user": "run_sampled_underspecified_with_user_new_gemini-3-pro-preview_20260128_155002",
    },
    "GPT-5.2": {
        "baseline_prefix": "gpt",
        "no_user": "run_sampled_underspecified_gpt-5.2_20260124_023055",
        "with_user": "Run_sampled_underspecified_with_user_new_gpt-5.2_20260128_044134",
    },
}


def load_baseline_pass3(baseline_dir: Path, model_prefix: str) -> pd.DataFrame:
    """Load baseline pass@3 data for a model family.

    Args:
        baseline_dir: Path to baseline_results/per_model/
        model_prefix: Model prefix (claude, gemini, gpt)

    Returns:
        DataFrame with columns: task_id, pass@3
    """
    csv_path = baseline_dir / f"{model_prefix}_pass_k_per_task.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    return df[["task_id", "pass@3"]].rename(columns={"pass@3": "baseline_pass3"})


def load_run_pass3(run_dir: Path) -> pd.DataFrame:
    """Load pass@3 data from a run directory.

    Args:
        run_dir: Path to run directory (e.g., run_sampled_underspecified_claude-...)

    Returns:
        DataFrame with columns: (variant_index OR task), pass@3
        - no_user runs have numeric variant_index
        - with_user runs have original task_id
    """
    csv_path = run_dir / "pass_k_per_task.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Run pass_k file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Filter out AGGREGATE row
    df = df[df["task_id"] != "AGGREGATE"]

    # Check if task_id column contains numeric indices or actual task IDs
    first_id = str(df["task_id"].iloc[0])
    is_numeric = first_id.isdigit()

    if is_numeric:
        # no_user runs use numeric variant indices
        df = df.rename(columns={"task_id": "variant_index", "pass@3": "underspec_pass3"})
        df["variant_index"] = df["variant_index"].astype(int)
        return df[["variant_index", "underspec_pass3"]]
    else:
        # with_user runs use original task IDs
        df = df.rename(columns={"task_id": "task", "pass@3": "underspec_pass3"})
        return df[["task", "underspec_pass3"]]


def load_task_mapping(dataset_dir: Path) -> pd.DataFrame:
    """Load variant-to-task mapping from sampled CSV.

    Args:
        dataset_dir: Path to dataset directory

    Returns:
        DataFrame with variant_index, task, and ambiguity_class columns
    """
    csv_path = dataset_dir / "sampled_filtered_underspec_with_baseline.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Sampled mapping file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Add variant_index based on row position
    df = df.reset_index().rename(columns={"index": "variant_index"})
    # Ensure variant_index is int for merging
    df["variant_index"] = df["variant_index"].astype(int)
    return df[["variant_index", "task", "ambiguity_class"]]


def match_tasks(
    baseline_df: pd.DataFrame,
    underspec_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """Match underspec results to baseline by task_id.

    Args:
        baseline_df: DataFrame with task_id, baseline_pass3
        underspec_df: DataFrame with (variant_index OR task), underspec_pass3
        mapping_df: DataFrame with variant_index, task, ambiguity_class

    Returns:
        Merged DataFrame with both baseline and underspec pass@3, plus ambiguity_class
    """
    if "variant_index" in underspec_df.columns:
        # no_user runs: join with mapping to get task_id and ambiguity_class
        underspec_with_task = underspec_df.merge(mapping_df, on="variant_index", how="left")
    else:
        # with_user runs: already have task column, need to add ambiguity_class
        underspec_with_task = underspec_df.merge(
            mapping_df[["task", "ambiguity_class"]],
            on="task",
            how="left",
        )

    # Join with baseline
    merged = underspec_with_task.merge(
        baseline_df,
        left_on="task",
        right_on="task_id",
        how="inner",
    )

    return merged


AMBIGUITY_CLASSES = ["benign", "divergent", "outcome-critical"]
AMBIGUITY_COLORS = {
    "benign": "#2ecc71",
    "divergent": "#f39c12",
    "outcome-critical": "#e74c3c",
}


def plot_ambiguity_breakdown(
    model_name: str,
    data: dict,
    output_path: Path,
) -> None:
    """Generate bar chart showing pass@3 by ambiguity class for a single model.

    Args:
        model_name: Display name of the model (e.g., "Claude")
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

    # Get actual total ask_user calls (stored separately)
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


def plot_comparison(data: dict, output_path: Path) -> None:
    """Generate grouped bar chart comparing baseline vs underspec conditions.

    Args:
        data: Dict with structure:
            {
                "Claude": {"baseline": 0.8, "no_user": 0.3, "with_user": 0.5},
                ...
            }
        output_path: Path to save the chart
    """
    models = list(data.keys())
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract values
    baseline_vals = [data[m]["baseline"] for m in models]
    no_user_vals = [data[m]["no_user"] for m in models]
    with_user_vals = [data[m]["with_user"] for m in models]

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

    # Formatting
    ax.set_xlabel("Model Family", fontsize=12)
    ax.set_ylabel("Mean Pass@3", fontsize=12)
    ax.set_title("Pass@3 Comparison: Baseline vs Underspecified", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart to: {output_path}")


def load_models_config(config_path: Path) -> dict:
    """Load models configuration from YAML file.

    Expected format:
        models:
          "Claude Opus 4.5":
            baseline_prefix: claude
            no_user: run_sampled_underspecified_claude-opus-4-5-20251101_20260123_060406
            with_user: run_sampled_underspecified_with_user_new_claude-opus-4-5-20251101_20260128_044057
          ...
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config.get("models", {})


def main():
    parser = argparse.ArgumentParser(
        description="Generate pass@3 comparison charts for sampled runs"
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to dataset directory (e.g., intersection_delete_2segments_20260121_161230)",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML config file with model run directories (overrides hardcoded defaults)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    baseline_dir = dataset_dir / "baseline_results" / "per_model"
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")

    # Create output directory
    output_dir = dataset_dir / "sampled_runs"
    output_dir.mkdir(exist_ok=True)

    # Load models config
    if args.config:
        models = load_models_config(Path(args.config))
        print(f"Loaded config from: {args.config}")
    else:
        models = DEFAULT_MODELS
        print("Using hardcoded default model configs")

    # Load task mapping
    mapping_df = load_task_mapping(dataset_dir)
    print(f"Loaded mapping for {len(mapping_df)} variants")

    # Aggregate data for plotting
    plot_data = {}
    ambiguity_data = {}  # {model: {ambiguity_class: {baseline, no_user, with_user, count}}}

    for display_name, config in models.items():
        baseline_prefix = config["baseline_prefix"]
        no_user_dir = dataset_dir / config["no_user"]
        with_user_dir = dataset_dir / config["with_user"]

        # Load baseline
        try:
            baseline_df = load_baseline_pass3(baseline_dir, baseline_prefix)
            print(f"Loaded baseline for {display_name}: {len(baseline_df)} tasks")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        # Process each condition
        model_data = {}
        model_ambiguity = {}  # {ambiguity_class: {baseline, no_user, with_user, count}}

        for condition in ["no_user", "with_user"]:
            run_dir = no_user_dir if condition == "no_user" else with_user_dir
            if not run_dir.exists():
                print(f"Warning: Run directory not found for {display_name} {condition}: {run_dir}")
                continue

            try:
                underspec_df = load_run_pass3(run_dir)
                matched_df = match_tasks(baseline_df, underspec_df, mapping_df)

                mean_pass3 = matched_df["underspec_pass3"].mean()
                model_data[condition] = mean_pass3
                print(f"  {display_name} {condition}: {mean_pass3:.3f} (n={len(matched_df)})")

                # Compute baseline mean for matched tasks only
                if "baseline" not in model_data:
                    model_data["baseline"] = matched_df["baseline_pass3"].mean()
                    print(f"  {display_name} baseline: {model_data['baseline']:.3f}")

                # Load ask_user counts for with_user condition
                ask_user_df = None
                total_ask_user_calls = 0
                if condition == "with_user":
                    ask_user_df, total_ask_user_calls = load_ask_user_counts(run_dir)
                    if not ask_user_df.empty:
                        matched_df = matched_df.merge(ask_user_df, on="task", how="left")
                        matched_df["ask_user_count"] = matched_df["ask_user_count"].fillna(0)

                # Aggregate by ambiguity class
                for amb_class in AMBIGUITY_CLASSES:
                    class_df = matched_df[matched_df["ambiguity_class"] == amb_class]
                    if len(class_df) == 0:
                        continue

                    if amb_class not in model_ambiguity:
                        model_ambiguity[amb_class] = {
                            "baseline": class_df["baseline_pass3"].mean(),
                            "count": len(class_df),
                        }

                    model_ambiguity[amb_class][condition] = class_df["underspec_pass3"].mean()

                    # Add ask_user stats for with_user condition
                    if condition == "with_user" and "ask_user_count" in class_df.columns:
                        model_ambiguity[amb_class]["ask_user_mean"] = class_df[
                            "ask_user_count"
                        ].mean()
                        # Store total for this ambiguity class
                        if "ask_user_sum" in class_df.columns:
                            model_ambiguity[amb_class]["ask_user_total"] = int(
                                class_df["ask_user_sum"].sum()
                            )

                # Store grand total for this model
                if condition == "with_user" and total_ask_user_calls > 0:
                    model_ambiguity["_total_ask_user"] = total_ask_user_calls

            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue

        if model_data:
            plot_data[display_name] = model_data

        if model_ambiguity:
            ambiguity_data[display_name] = model_ambiguity

    if not plot_data:
        print("Error: No data to plot")
        return

    # Generate main comparison chart
    output_path = output_dir / "pass3_comparison_by_model.png"
    plot_comparison(plot_data, output_path)

    # Generate per-model ambiguity breakdown charts
    for model_name, amb_data in ambiguity_data.items():
        # Check that we have both conditions for all classes
        valid_data = {}
        for key, value in amb_data.items():
            if key.startswith("_"):  # Preserve special keys like _total_ask_user
                valid_data[key] = value
            elif "no_user" in value and "with_user" in value:
                valid_data[key] = value

        if valid_data:
            model_slug = model_name.lower().replace(" ", "_")
            amb_output_path = output_dir / f"pass3_by_ambiguity_{model_slug}.png"
            plot_ambiguity_breakdown(model_name, valid_data, amb_output_path)

    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
