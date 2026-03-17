import logging
import math
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
    """Return paths for evaluation - payroll file should be in MRT_OUTPUT_ROOT."""
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    csv_path = os.path.join(eval_assets_root, "april-attendance-data.csv")
    rates_path = os.path.join(eval_assets_root, "rates_info.txt")
    payroll_path = os.path.join(output_root, "april-payroll.xlsx")
    return payroll_path, csv_path, rates_path


def _parse_rates(path: str) -> pd.DataFrame:
    data = {"Rate": [], "Name": []}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            name, rate = line.strip().split(":", 1)
            data["Name"].append(name.strip())
            data["Rate"].append(float(rate.strip()))
    return pd.DataFrame(data)


def _analyze_attendance(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Clock-in"] = pd.to_datetime(df["Clock-in"], format="%H:%M").dt.time
    df["Clock-out"] = pd.to_datetime(df["Clock-out"], format="%H:%M").dt.time

    def work_length(row):
        clock_in = pd.Timestamp.combine(pd.Timestamp(row["Date"]), row["Clock-in"])
        clock_out = pd.Timestamp.combine(pd.Timestamp(row["Date"]), row["Clock-out"])
        return (clock_out - clock_in).seconds / 3600

    df["Work Length"] = df.apply(work_length, axis=1)
    results = df.groupby("Name").agg(total_work_length=("Work Length", "sum")).reset_index()
    results["total_work_length"] = results["total_work_length"].apply(math.ceil)
    results.columns = ["Name", "Total Work Length"]
    return results


def _expected_payroll(csv_path: str, rates_path: str) -> pd.DataFrame:
    employee = _analyze_attendance(csv_path)
    rates = _parse_rates(rates_path)
    merged = pd.merge(employee, rates, on="Name", how="left")
    merged["Total Earnings"] = (merged["Total Work Length"] * merged["Rate"]).round(2)
    return merged[["Name", "Total Earnings"]].sort_values(by="Name").reset_index(drop=True)


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    df = df.copy()
    df.columns = df.columns.str.replace(" ", "").str.lower()
    return df.sort_values(by="name").reset_index(drop=True)


@grader
def grade_checkpoint1(trajectory: str) -> bool:
    signals = [
        "/Documents/Human Resources Team/Attendance",
        "dir=/Documents/Human%20Resources%20Team/Attendance",
    ]
    return any(sig in trajectory for sig in signals)


@grader
def grade_checkpoint2() -> bool:
    payroll_path, _, _ = _paths()
    try:
        df = pd.read_excel(payroll_path)
        std = _standardize(df)
        return set(std.columns).issuperset({"name", "totalearnings"})
    except Exception:
        return False


@grader
def grade_checkpoint3() -> bool:
    payroll_path, csv_path, rates_path = _paths()
    try:
        output = _standardize(pd.read_excel(payroll_path))
        expected = _standardize(_expected_payroll(csv_path, rates_path))
        if output is None or expected is None:
            return False
        if set(output["name"].tolist()) != set(expected["name"].tolist()):
            return False
        # compare values row-by-row
        for name in expected["name"].tolist():
            v_out = float(output.loc[output["name"] == name, "totalearnings"])
            v_exp = float(expected.loc[expected["name"] == name, "totalearnings"])
            if abs(v_out - v_exp) > 1e-2:
                return False
        return True
    except Exception:
        return False


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    checkpoints.append(Checkpoint(1, int(grade_checkpoint1(trajectory))))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint2())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint3())))
    return result
