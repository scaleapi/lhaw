import logging
import os

# LHAW imports
import sys

import numpy as np
import pandas as pd

sys.path.append("../../../")
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result

LOCAL_DST = "openhands_evaluation.xlsx"

sheet1 = [
    [["Lemur"], ["Lemur-chat-70b"], np.nan, 5.3, np.nan, np.nan],
    [["CodeActAgent", "v1.8"], ["claude-3-5-sonnet"], 26.0, 15.3, 52.0, np.nan],
]

sheet2 = [
    [["SWE-agent", "1-shot"], ["gpt-4-turbo"], 87.7, np.nan],
    [["OH", "CodeActAgent", "v1.5"], ["gpt-3.5-turbo-16k-0613"], 20.1, 0.11],
]

sheet3 = [
    [["WebArena", "Agent"], ["Llama3-chat-70b"], 7.0, np.nan],
]

sheet4 = [[["OH", "CodeActAgent", "v1.5"], ["gpt-3.5-turbo-0125"], 11.8, 0.006]]


def found_matching_row(df, sheet: list):
    if df is None:
        return False
    try:
        for entry in sheet:
            column = 0
            cond = True
            for keywords in entry:
                if column >= len(df.columns):
                    break
                if isinstance(keywords, list):
                    for key in keywords:
                        cond &= df.iloc[:, column].astype(str).str.contains(key, case=False)
                elif np.isnan(keywords):
                    cond &= df.iloc[:, column].isna()
                else:
                    # Handle numeric comparison
                    cond &= df.iloc[:, column] == keywords

                column += 1
            matching_rows = df[cond]
            if matching_rows.empty:
                return False
    except Exception as e:
        logging.error("Error finding matching row: %s", e)
        return False
    return True


def read_sheet(file_path, sheet_index: int):
    if not os.path.isfile(file_path):
        logging.error(f"File not found at {file_path}")
        return None

    try:
        df = pd.read_excel(file_path, sheet_name=sheet_index)
    except Exception as e:
        logging.error("Error reading sheet %s: %s", sheet_index, e)
        return None
    return df


@grader
def grade_checkpoint1():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    file_path = os.path.join(output_root, "openhands_evaluation.xlsx")
    return os.path.isfile(file_path)


@grader
def grade_checkpoint2():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    file_path = os.path.join(output_root, "openhands_evaluation.xlsx")
    return found_matching_row(read_sheet(file_path, 0), sheet1)


@grader
def grade_checkpoint3():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    file_path = os.path.join(output_root, "openhands_evaluation.xlsx")
    return found_matching_row(read_sheet(file_path, 1), sheet2)


@grader
def grade_checkpoint4():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    file_path = os.path.join(output_root, "openhands_evaluation.xlsx")
    return found_matching_row(read_sheet(file_path, 2), sheet3)


@grader
def grade_checkpoint5():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    file_path = os.path.join(output_root, "openhands_evaluation.xlsx")
    return found_matching_row(read_sheet(file_path, 3), sheet4)


def grade_checkpoints(trajectory="") -> Result:
    checkpoints = []
    checkpoints.append(Checkpoint(1, 1 * int(grade_checkpoint1())))
    checkpoints.append(Checkpoint(1, 1 * int(grade_checkpoint2())))
    checkpoints.append(Checkpoint(1, 1 * int(grade_checkpoint3())))
    checkpoints.append(Checkpoint(1, 1 * int(grade_checkpoint4())))
    checkpoints.append(Checkpoint(1, 1 * int(grade_checkpoint5())))
    return Result(checkpoints)
