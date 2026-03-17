import logging
import os

# Import from LHAW evaluation system
import sys
from typing import List

sys.path.append("../../../")
import pandas as pd
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result

logging.basicConfig(level=logging.INFO)


def _paths():
    workspace_dir = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    generated_file = os.path.join(workspace_dir, "flagged_variances.xlsx")
    budget_path = os.path.join(eval_assets_root, "budget.xlsx")
    actual_path = os.path.join(eval_assets_root, "actual_spending.xlsx")
    return generated_file, budget_path, actual_path


def _compute_expected_variances(budget_path: str, actual_path: str) -> pd.Series:
    budget_df = pd.read_excel(budget_path)
    actual_df = pd.read_excel(actual_path)
    merged = pd.merge(budget_df, actual_df, on=["Department", "Category", "Month"])
    merged["Variance_Amount"] = merged["Actual_Amount"] - merged["Budgeted_Amount"]
    merged["Variance_Percentage"] = (
        merged["Variance_Amount"] / merged["Budgeted_Amount"] * 100
    ).round(2)
    significant = merged[
        (merged["Variance_Percentage"].abs() > 10) & (merged["Variance_Amount"].abs() > 5000)
    ].copy()
    significant["Variance_ID"] = significant.apply(
        lambda x: f"{x['Department']}_{x['Category']}_{pd.to_datetime(x['Month']).strftime('%Y%m')}",
        axis=1,
    )
    return significant["Variance_ID"].reset_index(drop=True)


@grader
def grade_checkpoint1(trajectory) -> bool:
    signals = ["/Documents/Financials", "dir=/Documents/Financials"]
    return any(sig in trajectory for sig in signals)


@grader
def grade_checkpoint2() -> int:
    """Validate contents and format of flagged_variances.xlsx against expected computation."""
    generated_file, budget_path, actual_path = _paths()
    if not os.path.isfile(generated_file):
        logging.warning("flagged_variances.xlsx not found")
        return 0
    # Read agent output; if unreadable, treat as missing
    try:
        df = pd.read_excel(generated_file)
    except Exception as e:
        logging.warning(f"Failed to read flagged_variances.xlsx: {e}")
        return 0
    if "Flagged_Variances" not in df.columns:
        return 1
    # Compute expected from reference inputs; failures here mean validation could not be confirmed
    try:
        expected_series = _compute_expected_variances(budget_path, actual_path)
        agent_series = df["Flagged_Variances"].reset_index(drop=True)
        return 3 if agent_series.equals(expected_series) else 2
    except Exception as e:
        logging.warning(f"Validation error while computing expected: {e}")
        return 2


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    checkpoints.append(Checkpoint(1, int(grade_checkpoint1(trajectory))))
    checkpoints.append(Checkpoint(3, int(grade_checkpoint2())))
    return result
