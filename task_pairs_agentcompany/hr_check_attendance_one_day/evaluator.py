import logging
import os

# LHAW imports
import sys
from datetime import datetime
from typing import List

import pandas as pd

sys.path.append("../../../")
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result, bonus_for_completing_final


def check_excel_file_exists(file_path):
    abs_path = os.path.abspath(file_path)
    try:
        return os.path.isfile(abs_path)
    except Exception as e:
        logging.warning(f"Error in is_file_exist: {e}")
        return False


def read_excel(filename):
    try:
        df = pd.read_excel(filename)
        names = set(df["Name"].tolist())
        return names
    except Exception:
        return None


def filter_name(filename):
    try:
        df = pd.read_csv(filename)

        # Strip out the spaces if there are any
        df.columns = df.columns.str.strip()
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        df["Clock-out"] = pd.to_datetime(df["Clock-out"], format="%H:%M").dt.time

        start_time = datetime.strptime("17:30", "%H:%M").time()
        end_time = datetime.strptime("18:00", "%H:%M").time()

        filtered_df = df[(df["Clock-out"] >= start_time) & (df["Clock-out"] <= end_time)]
        return set(filtered_df["Name"].tolist())
    except Exception:
        return None


@grader
def grade_checkpoint1(trajectory):
    urls = [
        f"dir=/Documents/Human%20Resources%20Team/Attendance",
        "/Documents/Human Resources Team/Attendance",
    ]
    return any(url in trajectory for url in urls)


@grader
def grade_checkpoint2():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    path = os.path.join(output_root, "2024-03-attendance-summary.xlsx")

    if not check_excel_file_exists(path):
        return False
    try:
        df = pd.read_excel(path)
        return "Name" in df.columns
    except Exception:
        return False


@grader
def grade_checkpoint3():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))

    name = read_excel(os.path.join(output_root, "2024-03-attendance-summary.xlsx"))
    correct_answer = filter_name(os.path.join(eval_assets_root, "attendance-2024-03-01.csv"))

    if name is None or correct_answer is None:
        return False
    if name == correct_answer:
        return True
    return False


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints, bonus_for_completing_final)
    checkpoints.append(Checkpoint(1, grade_checkpoint1(trajectory)))
    checkpoints.append(Checkpoint(1, grade_checkpoint2()))
    checkpoints.append(Checkpoint(1, grade_checkpoint3()))
    return result
