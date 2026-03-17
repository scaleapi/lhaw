"""Shared utilities for underspecification dataset preparation and analysis."""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _scores_dir_sort_key(path: Path) -> int:
    """Extract numeric trial index from scores_* directory for stable sorting.

    Directory names follow scores_{name}_{index}_{date}_{time}.
    Falls back to 0 if the pattern doesn't match.
    """
    m = re.search(r"_(\d+)_\d{8}_\d{6}$", path.name)
    return int(m.group(1)) if m else 0


# Set once globally to handle large CSV fields
csv.field_size_limit(10 * 1024 * 1024)


def safe_csv_read(path: Path) -> List[Dict]:
    """
    Read CSV file safely, handling large fields.

    Args:
        path: Path to CSV file

    Returns:
        List of dictionaries, one per row
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_csv_write(path: Path, rows: List[Dict], fieldnames: List[str] = None):
    """
    Write CSV file safely.

    Args:
        path: Output path for CSV file
        rows: List of dictionaries to write
        fieldnames: Column names (if None, inferred from first row)
    """
    if not rows:
        raise ValueError("Cannot write empty CSV")

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_dir(path: Path) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create

    Returns:
        The created/existing directory path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_file_exists(path: Path, description: str = "File"):
    """
    Validate that a file exists, raising descriptive error if not.

    Args:
        path: File path to check
        description: Human-readable description for error message
    """
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{description} is not a file: {path}")


def count_ask_user_calls(conv_str: str) -> int:
    """Count number of ask_user tool calls in a conversation."""
    try:
        conv = json.loads(conv_str)
        count = 0
        for msg in conv:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    name = tc.get("function", {}).get("name", "")
                    if name == "ask_user":
                        count += 1
        return count
    except Exception:
        return 0


def load_ask_user_counts(run_dir: Path) -> tuple[pd.DataFrame, int]:
    """Load ask_user call counts from scored CSVs.

    Args:
        run_dir: Path to run directory

    Returns:
        Tuple of (DataFrame with columns: task, ask_user_count, ask_user_sum), total_calls)
        - ask_user_count is mean per task across trials
        - ask_user_sum is total calls per task across trials
        - total_calls is the grand total across all tasks and trials
    """
    scores_dirs = sorted(run_dir.glob("scores_*"), key=_scores_dir_sort_key)
    if not scores_dirs:
        return pd.DataFrame(columns=["task", "ask_user_count", "ask_user_sum"]), 0

    all_counts = []
    for scores_dir in scores_dirs:
        for csv_path in scores_dir.glob("scored_*.csv"):
            df = pd.read_csv(csv_path)
            if "raw_conversation_history" not in df.columns:
                continue
            if "TASK" not in df.columns:
                continue

            df["ask_user_count"] = df["raw_conversation_history"].apply(count_ask_user_calls)
            all_counts.append(df[["TASK", "ask_user_count"]].rename(columns={"TASK": "task"}))

    if not all_counts:
        return pd.DataFrame(columns=["task", "ask_user_count", "ask_user_sum"]), 0

    # Combine all trials
    combined = pd.concat(all_counts, ignore_index=True)

    # Compute total across all trials
    total_calls = combined["ask_user_count"].sum()

    # Compute mean and sum per task
    grouped = combined.groupby("task")["ask_user_count"].agg(["mean", "sum"]).reset_index()
    grouped.columns = ["task", "ask_user_count", "ask_user_sum"]

    return grouped, int(total_calls)
