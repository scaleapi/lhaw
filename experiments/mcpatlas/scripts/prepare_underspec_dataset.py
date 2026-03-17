#!/usr/bin/env python3
"""
Prepare underspecified dataset from baseline inference results.

This script orchestrates the complete dataset preparation pipeline:
1. Auto-discover baseline runs from full_inference_data/
2. Aggregate passed tasks (intersection/union)
3. Generate underspecified prompt variants
4. Extract baseline results for comparison

Usage:
    # Run full pipeline
    python scripts/prepare_underspec_dataset.py configs/intersection_delete_v1.yaml

    # Validate config only
    python scripts/prepare_underspec_dataset.py configs/intersection_delete_v1.yaml --validate

    # Dry run (show what would happen)
    python scripts/prepare_underspec_dataset.py configs/intersection_delete_v1.yaml --dry-run
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
REPO_ROOT = EXPERIMENTS_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")
api_key = os.getenv("LLM_API_KEY") or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["LLM_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
base_url = os.getenv("LLM_BASE_URL") or os.getenv("LITELLM_BASE_URL")
if base_url:
    os.environ["OPENAI_BASE_URL"] = base_url

import utils  # noqa: E402
from config import DatasetConfig  # noqa: E402
from dataset_stages import (  # noqa: E402
    aggregate_passed,
    discover_baselines,
    extract_baselines,
    generate_underspec,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def print_banner(message: str, char: str = "="):
    """Print a formatted banner."""
    width = 70
    print(f"\n{char * width}")
    print(f"{message:^{width}}")
    print(f"{char * width}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare underspecified dataset from baseline results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python scripts/prepare_underspec_dataset.py configs/intersection_delete_v1.yaml

  # Validate config only
  python scripts/prepare_underspec_dataset.py configs/intersection_delete_v1.yaml --validate

  # Dry run
  python scripts/prepare_underspec_dataset.py configs/intersection_delete_v1.yaml --dry-run
        """,
    )
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument("--validate", action="store_true", help="Validate config and exit")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would happen without executing"
    )

    args = parser.parse_args()

    # Load and validate config
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config}")
            return 1

        config = DatasetConfig.from_yaml(str(config_path))
        print(f"✓ Config loaded: {config.dataset['name']}")

        if args.validate:
            print("✓ Config validation passed")
            return 0

    except Exception as e:
        print(f"Error loading config: {e}")
        return 1

    # Dry run: show plan
    if args.dry_run:
        print_banner("DRY RUN - Pipeline Plan")
        print(f"Dataset name: {config.dataset['name']}")
        print(f"Description: {config.dataset.get('description', 'N/A')}")
        print(f"\nBaseline data: {config.baseline.data_dir}")
        print(f"Models to discover: {list(config.baseline.model_patterns.keys())}")
        print(f"Aggregation strategy: {config.baseline.aggregation_strategy}")
        print(f"Pass threshold: {config.baseline.pass_threshold}")
        print(f"\nUnderspec severity: {config.underspec.severity}")
        print(f"Variants per task: {config.underspec.variant_top_k}")
        print(f"Max level: {config.underspec.max_level}")
        print(f"\nOutput base: {config.output['base_dir']}")
        print("\nStages:")
        print("  1. Discover baseline runs")
        print("  2. Aggregate passed tasks")
        print("  3. Generate underspecified variants")
        print("  4. Extract baseline results")
        print("\nRun without --dry-run to execute")
        return 0

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = f"{config.dataset['name']}_{timestamp}"
    output_base = Path(config.output["base_dir"]) / dataset_name
    utils.ensure_dir(output_base)

    print_banner(f"Preparing Dataset: {dataset_name}")
    print(f"Output directory: {output_base}\n")

    try:
        # Save config snapshot
        config.to_yaml(str(output_base / "config.yaml"))
        logger.info("Saved config snapshot")

        # Stage 1: Discover baseline runs
        print_banner("[1/4] Discovering Baseline Runs", char="-")
        base_dir = EXPERIMENTS_DIR / config.baseline.data_dir
        run_dirs = discover_baselines.discover_baseline_runs(
            base_dir, config.baseline.model_patterns
        )
        for model, path in run_dirs.items():
            print(f"  ✓ {model:8s}: {path.name}")

        # Stage 2: Aggregate passed tasks
        print_banner("[2/4] Aggregating Passed Tasks", char="-")
        agg_output = output_base / "baseline_aggregation"
        utils.ensure_dir(agg_output)

        agg_results = aggregate_passed.aggregate_passed_tasks(
            run_dirs,
            config.baseline.pass_threshold,
            config.baseline.aggregation_strategy,
            agg_output,
        )

        strategy = config.baseline.aggregation_strategy
        selected_tasks = agg_results[strategy]

        print(f"  Strategy: {strategy}")
        print(f"  Total tasks: {len(agg_results['all_task_ids'])}")
        print(f"  Selected tasks: {len(selected_tasks)}")
        print("\n  Per-model pass rates:")
        for model, stats in agg_results["summary"]["models"].items():
            print(f"    {model:8s}: {stats['passed']:4d} tasks ({stats['pass_rate']:.1%})")

        # Stage 3: Generate underspecified variants
        print_banner("[3/4] Generating Underspecified Variants", char="-")
        underspec_output = output_base / "underspec_generation"
        utils.ensure_dir(underspec_output)

        gen_results = generate_underspec.generate_underspec_variants(
            task_ids=list(selected_tasks),
            config=config.underspec.model_dump(),
            output_dir=underspec_output,
        )

        print(f"  Processed: {gen_results['tasks_processed']} tasks")
        print(f"  Failed: {gen_results['tasks_failed']} tasks")
        print(f"  Total rows: {gen_results['total_rows']}")

        if gen_results["tasks_failed"] > 0:
            print("\n  Failed tasks:")
            for ft in gen_results["failed_tasks"][:5]:
                print(f"    - {ft['task_id']}: {ft['error'][:60]}...")
            if len(gen_results["failed_tasks"]) > 5:
                print(f"    ... and {len(gen_results['failed_tasks']) - 5} more")

        # Stage 4: Extract baseline results
        print_banner("[4/4] Extracting Baseline Results", char="-")
        baseline_output = output_base / "baseline_results"
        utils.ensure_dir(baseline_output)

        extract_results = extract_baselines.extract_baseline_results(
            run_dirs, selected_tasks, baseline_output
        )

        print(f"  Extracted results for {extract_results['tasks_extracted']} tasks")
        print("\n  Per-model extraction:")
        for model, stats in extract_results["per_model_stats"].items():
            if "pass_k_tasks" in stats:
                print(f"    {model:8s}: {stats['pass_k_tasks']} tasks")
            else:
                print(f"    {model:8s}: Error - {stats.get('pass_k_error', 'Unknown')}")

        # Done
        print_banner("Dataset Preparation Complete!", char="=")
        print(f"Output directory:\n  {output_base}\n")
        print("Key files:")
        print("  Config snapshot:")
        print(f"    {output_base / 'config.yaml'}")
        print("\n  Underspecified prompts (use this for experiments):")
        print(f"    {output_base / 'underspec_generation' / 'underspec_prompts.csv'}")
        print("\n  Baseline results (for comparison):")
        print(f"    {output_base / 'baseline_results' / 'per_model'}/*.csv")
        print("\n  Aggregation details:")
        print(f"    {output_base / 'baseline_aggregation' / 'summary.json'}")
        print("\n" + "=" * 70 + "\n")

        print("Next steps:")
        print("  1. Review the generated underspecified prompts")
        print("  2. Run experiments with underspec_prompts.csv")
        print("  3. Compare results with baseline metrics")
        print()

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
