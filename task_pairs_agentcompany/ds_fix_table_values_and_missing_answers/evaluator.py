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
    """Return paths for evaluation - generated file should be in MRT_OUTPUT_ROOT."""
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    generated_file_path = os.path.join(output_root, "corrected_other_corp_area_totals.xlsx")
    ref_file_path = os.path.join(eval_assets_root, "ref_data.xlsx")
    return output_root, generated_file_path, ref_file_path


def _read_xlsx(path: str) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl")


@grader
def grade_checkpoint1() -> bool:
    """[2pts] Check if all missing values (previously 'x' in ref) are replaced with 0 in generated file."""
    _, generated_file_path, ref_file_path = _paths()
    try:
        if not (os.path.isfile(ref_file_path) and os.path.isfile(generated_file_path)):
            logging.warning("Reference or generated file missing in workspace")
            return False
        ref_file = _read_xlsx(ref_file_path)
        gen_file = _read_xlsx(generated_file_path)
        # Ensure same shape
        if ref_file.shape != gen_file.shape:
            logging.warning("Reference and generated files have different shapes")
            return False
        for row in range(ref_file.shape[0]):
            for col in range(ref_file.shape[1]):
                if ref_file.iloc[row, col] == "x":
                    val = gen_file.iloc[row, col]
                    if val not in [0, "0"]:
                        logging.warning(
                            f"Cell ({row},{col}) expected 0 where ref had 'x', got: {val}"
                        )
                        return False
        return True
    except Exception as e:
        logging.error(f"Error validating checkpoint1: {e}")
        return False


@grader
def grade_checkpoint2() -> bool:
    """[2pts] Check if question 1 (cell B24) is 2809 or formula '=B5'."""
    _, generated_file_path, _ = _paths()
    try:
        if not os.path.isfile(generated_file_path):
            logging.warning("Generated file missing in workspace")
            return False
        gen = _read_xlsx(generated_file_path)
        # B24 -> row index 22 (0-based), column index 1 (B)
        answer = gen.iloc[22, 1]
        return answer == 2809 or answer == "=B5"
    except Exception as e:
        logging.error(f"Error validating checkpoint2: {e}")
        return False


@grader
def grade_checkpoint3() -> bool:
    """[2pts] Check if question 2 (cell B92) is ~98.8% or formula '=H18/B18'."""
    _, generated_file_path, _ = _paths()
    try:
        if not os.path.isfile(generated_file_path):
            logging.warning("Generated file missing in workspace")
            return False
        gen = _read_xlsx(generated_file_path)
        # B92 -> row index 90, column index 1
        answer = gen.iloc[90, 1]
        numerical_ref = 0.988
        if isinstance(answer, str) and "%" in answer:
            try:
                answer = float(answer.replace("%", "").strip()) / 100.0
            except ValueError:
                pass
        if isinstance(answer, (int, float)):
            return abs(float(answer) - numerical_ref) < 0.01
        return answer == "=H18/B18"
    except Exception as e:
        logging.error(f"Error validating checkpoint3: {e}")
        return False


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []

    workspace_dir, generated_file_path, ref_file_path = _paths()
    logging.info(f"Resolved workspace_dir={workspace_dir}")
    logging.info(
        f"Resolved generated_file_path={generated_file_path} (exists={os.path.isfile(generated_file_path)})"
    )
    logging.info(f"Resolved ref_file_path={ref_file_path} (exists={os.path.isfile(ref_file_path)})")

    result = Result(checkpoints)

    passed1 = grade_checkpoint1()
    checkpoints.append(Checkpoint(2, 2 * int(passed1)))

    passed2 = grade_checkpoint2()
    checkpoints.append(Checkpoint(2, 2 * int(passed2)))

    passed3 = grade_checkpoint3()
    checkpoints.append(Checkpoint(2, 2 * int(passed3)))

    return result
