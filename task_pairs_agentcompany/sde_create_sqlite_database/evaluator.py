import os
import sys
from typing import List

# LHAW imports
sys.path.append("../../../")
from evaluation.common import grader  # noqa: E402
from evaluation.scoring import Checkpoint, Result, bonus_for_completing_any  # noqa: E402


@grader
def grade_checkpoint1(trajectory):
    urls = ["dir=/Documents/Financials", "/Documents/Financials"]
    return any(url in trajectory for url in urls)


@grader
def grade_checkpoint2():
    # In LHAW, files are pre-staged in /inputs/THE OWNCLOUD FOLDER/.
    # This checkpoint is worth 0 points but kept for structural consistency.
    # output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    # file_path = os.path.join(output_root, "July-September 2024 Financials.ods")
    # return os.path.isfile(file_path)
    return False


@grader
def grade_checkpoint3():
    # Check if sqlite3 is installed.
    # NOTE: Python's sqlite3 is a standard library module and is ALWAYS available.
    # This checkpoint is worth 0 points as it tests nothing meaningful.
    # Kept for structural consistency with original AgentCompany evaluator.
    # try:
    #     import sqlite3  # noqa: F401
    #     return True
    # except ImportError:
    #     return False
    return False


@grader
def grade_checkpoint4():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    db_path = os.path.join(output_root, "financial_report.db")
    try:
        import sqlite3  # noqa: F401

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        conn.close()
        return len(tables) > 0
    except sqlite3.Error:
        return False
    except Exception:
        return False


@grader
def grade_checkpoint5():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    db_path = os.path.join(output_root, "financial_report.db")
    try:
        import sqlite3  # noqa: F401

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(financial_categories);")
        columns = [col[1] for col in cursor.fetchall()]
        conn.close()
        return set(columns) == {"category_id", "category_name", "category_type"}
    except sqlite3.Error:
        return False
    except Exception:
        return False


@grader
def grade_checkpoint6():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    db_path = os.path.join(output_root, "financial_report.db")
    try:
        import sqlite3  # noqa: F401

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(financial_details);")
        columns = [col[1] for col in cursor.fetchall()]
        conn.close()
        return set(columns) == {
            "detail_id",
            "category_id",
            "month",
            "actual",
            "budget",
            "variance",
            "percent_of_budget",
        }
    except sqlite3.Error:
        return False
    except Exception:
        return False


@grader
def grade_checkpoint7():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    db_path = os.path.join(output_root, "financial_report.db")
    try:
        import sqlite3  # noqa: F401

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT month, variance
            FROM financial_details
            JOIN financial_categories ON financial_details.category_id = financial_categories.category_id
            WHERE category_name = 'Total Income'
            ORDER BY variance ASC
            LIMIT 1
        """
        )
        result = cursor.fetchone()
        conn.close()
        # Relax variance type check
        return result == ("2024-08", -35000) or result == ("2024-08", -35000.0)
    except sqlite3.Error:
        return False
    except Exception:
        return False


@grader
def grade_checkpoint8():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    db_path = os.path.join(output_root, "financial_report.db")
    try:
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT SUM(actual)
            FROM financial_details
            JOIN financial_categories ON financial_details.category_id = financial_categories.category_id
            WHERE category_name IN ('Software Development Services', 'Consulting')
            AND month = '2024-07'
        """
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] == 95000 or result[0] == 95000.0
    except sqlite3.Error:
        return False
    except Exception:
        return False


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints, bonus_for_completing_any)

    # 8 checkpoints (6 pts max - CP2 & CP3 are 0pts, N/A in LHAW setup)
    checkpoints.append(Checkpoint(1, int(grade_checkpoint1(trajectory))))  # Navigate to file
    checkpoints.append(
        Checkpoint(0, int(grade_checkpoint2()))
    )  # Download file (0pts - N/A in LHAW, files pre-staged)
    checkpoints.append(
        Checkpoint(0, int(grade_checkpoint3()))
    )  # sqlite3 installed (0pts - always available in Python)
    checkpoints.append(Checkpoint(1, int(grade_checkpoint4())))  # DB exists
    checkpoints.append(Checkpoint(1, int(grade_checkpoint5())))  # financial_categories schema
    checkpoints.append(Checkpoint(1, int(grade_checkpoint6())))  # financial_details schema
    checkpoints.append(Checkpoint(1, int(grade_checkpoint7())))  # Min variance query
    checkpoints.append(Checkpoint(1, int(grade_checkpoint8())))  # Sum query

    return result
