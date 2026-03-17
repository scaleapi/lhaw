import logging
import os

# Import from LHAW evaluation system
import sys
from typing import List

sys.path.append("../../../")
import openpyxl
import pandas as pd
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result

logging.basicConfig(level=logging.INFO)


def _paths():
    workspace_dir = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    # Resolve evaluator assets root (use task dir by default)
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    working_file_path = os.path.join(workspace_dir, "respondent_counts.xlsx")
    init_file_path = os.path.join(eval_assets_root, "respondent_counts.xlsx")
    return workspace_dir, working_file_path, init_file_path


def load_spreadsheet(file_path: str, sheet_name: str, return_type: str = "workbook"):
    if not os.path.exists(file_path):
        logging.warning(f"Excel file not found: {file_path}")
        return None
    try:
        if return_type == "workbook":
            spreadsheet = openpyxl.load_workbook(file_path)[sheet_name]
        elif return_type == "dataframe":
            spreadsheet = pd.read_excel(file_path, sheet_name=sheet_name)
        else:
            logging.warning(f"Unsupported return_type: {return_type}")
            return None
        return spreadsheet
    except Exception:
        logging.warning(f"Cannot read the file or sheet: {file_path} / {sheet_name}")
        return None


@grader
def grade_checkpoint1() -> bool:
    """The 'unformatted' sheet in the working file should remain identical to the initial reference."""
    _, working_file, init_file = _paths()
    df_init = load_spreadsheet(init_file, sheet_name="unformatted", return_type="dataframe")
    df_work = load_spreadsheet(working_file, sheet_name="unformatted", return_type="dataframe")
    return df_work is not None and df_init is not None and df_init.equals(df_work)


@grader
def grade_checkpoint2() -> bool:
    """The content of the 'formatted' sheet should be the same as 'unformatted' in the working file."""
    _, working_file, _ = _paths()
    df_unformatted = load_spreadsheet(
        working_file, sheet_name="unformatted", return_type="dataframe"
    )
    df_formatted = load_spreadsheet(working_file, sheet_name="formatted", return_type="dataframe")
    return (
        df_unformatted is not None
        and df_formatted is not None
        and df_unformatted.equals(df_formatted)
    )


@grader
def grade_checkpoint3() -> bool:
    """Verify skyblue (#87CEEB) background applied to the specified header cells only."""
    _, working_file, _ = _paths()
    ws_orig = load_spreadsheet(working_file, sheet_name="unformatted", return_type="workbook")
    ws = load_spreadsheet(working_file, sheet_name="formatted", return_type="workbook")
    if ws is None or ws_orig is None:
        return False

    target_cells = [
        "A3",
        "B3",
        "B5",
        "C3",
        "C4",
        "D4",
        "D5",
        "E5",
        "F3",
        "F4",
        "G4",
        "G5",
        "H5",
        "I3",
        "I4",
        "J4",
        "J5",
        "K5",
    ]
    expected_rgb = "FF87CEEB"  # ARGB format used by openpyxl

    for col in "ABCDEFGHIJK":
        for row in range(1, 22):
            cell_name = f"{col}{row}"
            cell = ws[cell_name]
            fill = cell.fill

            if cell_name in target_cells:
                if fill.fill_type == "solid":
                    color_hex = fill.fgColor.rgb
                    if color_hex != expected_rgb:
                        logging.warning(
                            f"Cell {cell_name} has wrong background color: Expected #87CEEB, obtained {color_hex}."
                        )
                        return False
                else:
                    logging.warning(
                        f"Cell {cell_name} has wrong background color: Expected #87CEEB, obtained 'None'."
                    )
                    return False
            else:
                # Non-target cells should remain unchanged compared to 'unformatted'
                cell_orig = ws_orig[cell_name]
                fill_orig = cell_orig.fill

                if fill.fill_type != fill_orig.fill_type:
                    logging.warning(f"Cell {cell_name} should not change the background fill type.")
                    return False

                if fill_orig.fill_type == "solid":
                    color_hex_orig = fill_orig.fgColor.rgb
                    color_hex = fill.fgColor.rgb
                    if color_hex != color_hex_orig:
                        logging.warning(f"Cell {cell_name} should not change the background color.")
                        return False

    logging.info("The required background color changes are correctly applied.")
    return True


@grader
def grade_checkpoint4() -> bool:
    """Verify all populated cells are horizontally centered in 'formatted' sheet."""
    _, working_file, _ = _paths()
    ws = load_spreadsheet(working_file, sheet_name="formatted", return_type="workbook")
    if ws is None:
        return False

    for col in "ABCDEFGHIJK":
        for row in range(3, 22):
            cell_name = f"{col}{row}"
            cell = ws[cell_name]
            if cell.value and getattr(cell.alignment, "horizontal", None) != "center":
                logging.warning(
                    f"Cell {cell_name} is not horizontally centered (obtained: {cell.alignment.horizontal})"
                )
                return False

    logging.info("All the cells are horizontally centered.")
    return True


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)

    # 4 checkpoints, weight 1 each
    checkpoints.append(Checkpoint(1, int(grade_checkpoint1())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint2())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint3())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint4())))

    return result
