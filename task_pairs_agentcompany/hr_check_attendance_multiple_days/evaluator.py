import logging
import os

# Import from LHAW evaluation system
import sys
from datetime import datetime
from typing import List

sys.path.append("../../../")
import pandas as pd
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result

logging.basicConfig(level=logging.INFO)


def _paths():
    workspace_dir = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    report_path = os.path.join(workspace_dir, "april-attendance.xlsx")
    csv_asset = os.path.join(eval_assets_root, "april-attendance-data.csv")
    return report_path, csv_asset


def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    df.columns = df.columns.str.replace(" ", "").str.lower()
    df = df.sort_values(by="name").reset_index(drop=True)
    return df


def _analyze_csv(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)

        # Normalize spaces and strings
        df.columns = df.columns.str.strip()
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

        df["Clock-in"] = pd.to_datetime(df["Clock-in"], format="%H:%M").dt.time
        df["Clock-out"] = pd.to_datetime(df["Clock-out"], format="%H:%M").dt.time

        # Calculate work length in hours
        def work_length(row):
            clock_in = pd.Timestamp.combine(pd.Timestamp(row["Date"]), row["Clock-in"])
            clock_out = pd.Timestamp.combine(pd.Timestamp(row["Date"]), row["Clock-out"])
            return (clock_out - clock_in).seconds / 3600

        start_time = datetime.strptime("17:30", "%H:%M").time()
        end_time = datetime.strptime("18:00", "%H:%M").time()

        def is_ontime_departure(row):
            return row["Clock-out"] > start_time and row["Clock-out"] < end_time

        def is_late(row):
            return row["Clock-in"] > datetime.strptime("9:00", "%H:%M").time()

        df["Work Length"] = df.apply(work_length, axis=1)
        df["On-Time Departure"] = df.apply(is_ontime_departure, axis=1)
        df["Late Arrival"] = df.apply(is_late, axis=1)

        results = (
            df.groupby("Name")
            .agg(
                Average_Work_Length=("Work Length", "mean"),
                Ontime_Departure=("On-Time Departure", "sum"),
                Late_Arrival=("Late Arrival", "sum"),
            )
            .reset_index()
        )

        results.columns = [
            "Name",
            "Average Work Length",
            "On-time Departure Count",
            "Late Arrival Count",
        ]
        return results
    except Exception:
        return None


def _compare_row_by_row(output_df: pd.DataFrame, expected_df: pd.DataFrame) -> bool:
    expected_cols_name = ["averageworklength", "on-timedeparturecount", "latearrivalcount"]
    for name in output_df["name"].to_list():
        answer_row = output_df[output_df["name"] == name]
        expected_row = expected_df[expected_df["name"] == name]
        for col in expected_cols_name:
            if not (abs(answer_row[col] - expected_row[col]) < 1e-5).item():
                return False
    return True


@grader
def grade_checkpoint1(trajectory: str) -> bool:
    urls = [
        f"dir=/Documents/Human%20Resources%20Team/Attendance",
        "/Documents/Human Resources Team/Attendance",
    ]
    return any(url in trajectory for url in urls)


@grader
def grade_checkpoint2() -> bool:
    report_path, _ = _paths()
    if not os.path.isfile(report_path):
        return False
    try:
        df = pd.read_excel(report_path)
        expected_cols = {
            "Name",
            "Average Work Length",
            "On-time Departure Count",
            "Late Arrival Count",
        }
        return set(df.columns) >= expected_cols
    except Exception:
        return False


@grader
def grade_checkpoint3() -> bool:
    report_path, csv_asset = _paths()
    if not (os.path.isfile(report_path) and os.path.isfile(csv_asset)):
        return False
    try:
        output = _standardize_df(pd.read_excel(report_path))
        expected = _standardize_df(_analyze_csv(csv_asset))
        if output is None or expected is None:
            return False
        if set(output["name"].to_list()) != set(expected["name"].to_list()):
            return False
        return _compare_row_by_row(output, expected)
    except Exception:
        return False


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    checkpoints.append(Checkpoint(1, int(grade_checkpoint1(trajectory))))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint2())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint3())))
    return result
