#!/usr/bin/env python
"""
CLI adapter for analyzing ask_user tool calls from scored CSVs.

Usage:
    # Full analysis with LLM judging
    python analyze_ask_user.py \
        --run-dir /path/to/run_*_with_user_*/ \
        --output analysis.csv \
        --judge-model gpt-4o

    # Quick counts only (no LLM judging)
    python analyze_ask_user.py \
        --run-dir /path/to/run_*/ \
        --output counts.csv \
        --skip-judging
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env", override=True)

api_key = os.getenv("LLM_API_KEY") or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["LLM_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
base_url = os.getenv("LLM_BASE_URL") or os.getenv("LITELLM_BASE_URL")
if base_url:
    os.environ["OPENAI_BASE_URL"] = base_url

from constants import get_model  # noqa: E402
from evaluation.ask_user_analysis import analyze_scored_csv  # noqa: E402
from evaluation.ask_user_types import ALL_FLAGS, AskUserAnalysis  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_scored_csvs(run_dir: Path) -> list[tuple[Path, int]]:
    """Find all scored CSVs in a run directory with their trial indices.

    Args:
        run_dir: Path to run directory

    Returns:
        List of (csv_path, trial_index) tuples
    """
    results = []

    # Look for scores_*_N_* directories
    for scores_dir in run_dir.glob("scores_*"):
        if not scores_dir.is_dir():
            continue

        # Extract trial index from directory name
        # Pattern: scores_{name}_{trial_index}_{timestamp}
        match = re.search(r"scores_.*?_(\d+)_\d{8}_\d{6}$", scores_dir.name)
        if not match:
            # Try simpler pattern: scores_*_N
            match = re.search(r"_(\d+)$", scores_dir.name)

        if match:
            trial_index = int(match.group(1))
        else:
            # Default to 0 if we can't parse
            trial_index = 0

        # Find scored CSV in this directory
        for csv_file in scores_dir.glob("scored_*.csv"):
            results.append((csv_file, trial_index))
            logger.info(f"Found: {csv_file.name} (trial {trial_index})")

    return sorted(results, key=lambda x: (x[1], x[0].name))


def analyses_to_dataframe(analyses: list[AskUserAnalysis]) -> pd.DataFrame:
    """Convert list of analyses to DataFrame.

    Args:
        analyses: List of AskUserAnalysis objects

    Returns:
        DataFrame with one row per analysis
    """
    rows = [a.to_flat_dict() for a in analyses]

    # Convert lists to JSON strings for CSV compatibility
    for row in rows:
        if isinstance(row.get("questions_json"), list):
            row["questions_json"] = json.dumps(row["questions_json"])
        if isinstance(row.get("failure_categories"), list):
            row["failure_categories"] = json.dumps(row["failure_categories"])

    return pd.DataFrame(rows)


async def main():
    parser = argparse.ArgumentParser(description="Analyze ask_user tool calls from scored CSVs")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory containing scores_* subdirectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o",
        help="Model to use for LLM judging (default: gpt-4o)",
    )
    parser.add_argument(
        "--skip-judging",
        action="store_true",
        help="Skip LLM judging, only extract counts",
    )
    parser.add_argument(
        "--skip-no-calls",
        action="store_true",
        help="Skip rows with no ask_user calls",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent LLM judge calls (default: 5)",
    )

    args = parser.parse_args()

    if not args.skip_judging:
        args.judge_model = get_model(args.judge_model)

    if not args.run_dir.exists():
        logger.error(f"Run directory not found: {args.run_dir}")
        sys.exit(1)

    # Find all scored CSVs
    csv_files = find_scored_csvs(args.run_dir)
    if not csv_files:
        logger.error(f"No scored CSVs found in {args.run_dir}")
        sys.exit(1)

    logger.info(f"Found {len(csv_files)} scored CSV(s)")

    # Count total rows for progress bar
    total_rows = 0
    for csv_path, _ in csv_files:
        df = pd.read_csv(csv_path)
        total_rows += len(df)

    # Analyze each CSV with progress bar
    all_analyses: list[AskUserAnalysis] = []
    with tqdm(total=total_rows, desc="Analyzing rows", unit="row") as pbar:
        for csv_path, trial_index in csv_files:
            pbar.set_postfix({"file": csv_path.name, "trial": trial_index})

            analyses = await analyze_scored_csv(
                csv_path=csv_path,
                trial_index=trial_index,
                judge_model=None if args.skip_judging else args.judge_model,
                concurrency=args.concurrency,
                skip_judging=args.skip_judging,
                skip_no_calls=args.skip_no_calls,
                progress_bar=pbar,
            )

            all_analyses.extend(analyses)

    if not all_analyses:
        logger.warning("No analyses produced")
        sys.exit(0)

    # Convert to DataFrame and save
    df = analyses_to_dataframe(all_analyses)

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(df)} rows to {args.output}")

    # Print summary
    total_calls = df["ask_user_count"].sum()
    rows_with_calls = (df["ask_user_count"] > 0).sum()
    logger.info("Summary:")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Rows with ask_user calls: {rows_with_calls}")
    logger.info(f"  Total ask_user calls: {total_calls}")

    if not args.skip_judging:
        # Count flag frequencies
        logger.info("  Flag frequencies:")
        for flag in ALL_FLAGS:
            if flag in df.columns:
                count = df[flag].sum()
                if count > 0:
                    logger.info(f"    {flag}: {count}")


if __name__ == "__main__":
    asyncio.run(main())
