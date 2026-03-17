import logging
import os
import sqlite3

# Import from LHAW evaluation system
import sys
from typing import List

import pandas as pd

sys.path.append("../../../")
from evaluation.common import grader  # noqa: E402
from evaluation.scoring import Checkpoint, Result  # noqa: E402

logging.basicConfig(level=logging.INFO)


def _paths():
    """Return paths for evaluation - all files should be in MRT_OUTPUT_ROOT."""
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    csv_dir = os.path.join(eval_assets_root, "eval_data")
    db_path = os.path.join(output_root, "coffee_shop.db")
    return output_root, csv_dir, db_path


def view_exists(view_name: str, conn: sqlite3.Connection) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type='view' AND name=?
        """,
            (view_name,),
        )
        return cur.fetchone()[0] > 0
    except sqlite3.Error as e:
        logging.warning(f"Error checking view existence: {e}")
        return False


def dataframes_are_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    # Column set equality (case-insensitive)
    if set(c.lower() for c in df1.columns) != set(c.lower() for c in df2.columns):
        logging.warning("Columns do not match")
        return False
    # Row count
    if len(df1) != len(df2):
        logging.warning("Number of rows do not match")
        return False
    # Content (rounded numeric)
    df1s = df1.copy()
    df2s = df2.copy()
    df1s = df1s.sort_values(by=list(df1s.columns)).reset_index(drop=True).round(2)
    df2s = df2s.sort_values(by=list(df2s.columns)).reset_index(drop=True).round(2)
    if not df1s.equals(df2s):
        logging.warning("Content does not match")
        return False
    return True


def validate_tables_with_csv(conn: sqlite3.Connection, csv_dir: str) -> bool:
    try:
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        db_tables = [t[0] for t in cur.fetchall()]
        logging.info(f"Tables found: {db_tables}")

        for csv_file in csv_files:
            table_name = csv_file[:-4]
            if table_name not in db_tables:
                logging.warning(f"Table {table_name} not present in the DB")
                return False
            csv_path = os.path.join(csv_dir, csv_file)
            df_csv = pd.read_csv(csv_path)
            df_db = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            if not dataframes_are_equal(df_csv, df_db):
                return False
        return True
    except Exception as e:
        logging.error(f"Error validating tables: {e}")
        return False


@grader
def grade_checkpoint1(output_root: str) -> bool:
    """Check CSVs downloaded into output directory."""
    filenames = ["customers.csv", "products.csv", "inventory.csv", "orders.csv", "order_items.csv"]
    for fn in filenames:
        if not os.path.exists(os.path.join(output_root, fn)):
            logging.warning(f"{fn} not found in {output_root}")
            return False
    return True


@grader
def grade_checkpoint2(conn: sqlite3.Connection, csv_dir: str) -> bool:
    print(f"Checking checkpoint 2 with conn={conn} and csv_dir={csv_dir}")
    return validate_tables_with_csv(conn, csv_dir)


@grader
def grade_checkpoint3(conn: sqlite3.Connection, csv_dir: str) -> bool:
    if not view_exists("v_short_stock", conn):
        logging.warning("View v_short_stock does not exist")
        return False
    csv_path = os.path.join(csv_dir, "short_stock.csv")
    df_csv = pd.read_csv(csv_path)
    df_db = pd.read_sql_query("SELECT * FROM v_short_stock", conn)
    return dataframes_are_equal(df_csv, df_db)


@grader
def grade_checkpoint4(conn: sqlite3.Connection, csv_dir: str) -> bool:
    if not view_exists("v_sales", conn):
        logging.warning("View v_sales does not exist")
        return False
    csv_path = os.path.join(csv_dir, "average_sales.csv")
    df_csv = pd.read_csv(csv_path)
    df_db = pd.read_sql_query("SELECT * FROM v_sales", conn)
    return dataframes_are_equal(df_csv, df_db)


@grader
def grade_checkpoint5(output_root: str) -> int:
    """Validate answers written to analysis.txt in output directory."""
    ref_answers = [
        ["p001", "p003", "p005"],
        ["p001", "p002", "p004"],
        ["sarah johnson", "emma brown"],
    ]
    score = 0
    analysis_path = os.path.join(output_root, "analysis.txt")
    if not os.path.exists(analysis_path):
        logging.warning(f"Analysis sheet does not exist at {analysis_path}")
        return 0
    with open(analysis_path, "r") as f:
        for i, line in enumerate(f):
            if i >= len(ref_answers):
                break
            qa = line.split(":", 1)
            if len(qa) < 2:
                logging.warning("Analysis sheet corrupted")
                return 0
            answers = set(
                part.strip().lower().replace("'", "").replace('"', "") for part in qa[1].split(",")
            )
            score += 1 if answers == set(ref_answers[i]) else 0
            logging.info(f"Question {qa[0]}, Expected: {ref_answers[i]}, Agent: {answers}")
    return score


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)

    output_root, csv_dir, db_path = _paths()
    logging.info(f"Output root={output_root}, CSV dir={csv_dir}, DB={db_path}")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
    except Exception as e:
        logging.error(f"Could not connect to database at {db_path}: {e}")

    checkpoints.append(Checkpoint(1, int(grade_checkpoint1(output_root))))
    checkpoints.append(Checkpoint(2, 2 * int(grade_checkpoint2(conn, csv_dir)) if conn else 0))
    checkpoints.append(Checkpoint(2, 2 * int(grade_checkpoint3(conn, csv_dir)) if conn else 0))
    checkpoints.append(Checkpoint(2, 2 * int(grade_checkpoint4(conn, csv_dir)) if conn else 0))
    checkpoints.append(Checkpoint(3, int(grade_checkpoint5(output_root))))

    return result
