import logging
import os

# Import from LHAW evaluation system
import sys
from typing import List

import pandas as pd

sys.path.append("../../../")
from evaluation.common import grader  # noqa: E402
from evaluation.scoring import Checkpoint, Result  # noqa: E402

logging.basicConfig(level=logging.INFO)


def _paths():
    """Return paths for evaluation - output files should be in MRT_OUTPUT_ROOT."""
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    corrected_path = os.path.join(output_root, "expenses_corrected.xlsx")
    analysis_path = os.path.join(output_root, "expenses_analysis.csv")
    ref_corrected = os.path.join(eval_assets_root, "ref_expenses_corrected.xlsx")
    ref_analysis = os.path.join(eval_assets_root, "ref_expenses_analysis.csv")
    return corrected_path, analysis_path, ref_corrected, ref_analysis


@grader
def grade_checkpoint1(trajectory: str) -> bool:
    expected_path = "Documents/Financials/Expenses"
    return expected_path in trajectory


@grader
def grade_checkpoint2() -> bool:
    corrected_path, _, ref_corrected, _ = _paths()
    if not os.path.isfile(corrected_path):
        return False
    try:
        corrected_df = pd.read_excel(corrected_path)
        reference_df = pd.read_excel(ref_corrected)
        corrected_sorted = corrected_df.sort_values(by=corrected_df.columns.tolist()).reset_index(
            drop=True
        )
        reference_sorted = reference_df.sort_values(by=reference_df.columns.tolist()).reset_index(
            drop=True
        )
        return corrected_sorted.equals(reference_sorted)
    except Exception:
        return False


@grader
def grade_checkpoint3() -> bool:
    _, analysis_path, _, _ = _paths()
    if not os.path.isfile(analysis_path):
        return False
    try:
        analysis_df = pd.read_csv(analysis_path)
        expected_columns = {"Category", "Total_Amount", "Number_of_Employees", "Cost_Per_Employee"}
        return set(analysis_df.columns) == expected_columns
    except Exception:
        return False


@grader
def grade_checkpoint4() -> bool:
    _, analysis_path, _, ref_analysis = _paths()
    if not (os.path.isfile(analysis_path) and os.path.isfile(ref_analysis)):
        return False
    try:
        analysis_df = pd.read_csv(analysis_path)
        reference_df = pd.read_csv(ref_analysis)
        analysis_sorted = analysis_df.sort_values(by=analysis_df.columns.tolist()).reset_index(
            drop=True
        )
        reference_sorted = reference_df.sort_values(by=reference_df.columns.tolist()).reset_index(
            drop=True
        )
        return analysis_sorted.equals(reference_sorted)
    except Exception:
        return False


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    checkpoints.append(Checkpoint(1, int(grade_checkpoint1(trajectory))))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint2())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint3())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint4())))
    return result
