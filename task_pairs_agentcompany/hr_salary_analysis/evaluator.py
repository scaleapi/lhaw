import os

# Import from LHAW evaluation system
import sys
from io import BytesIO
from typing import List

import pandas as pd
from odf.opendocument import load
from odf.text import P

sys.path.append("../../../")
from evaluation.common import grader  # noqa: E402
from evaluation.scoring import Checkpoint, Result  # noqa: E402


def _paths():
    workspace_dir = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    report_path = os.path.join(workspace_dir, "Salary_Analysis_Report.odt")
    salary_txt = os.path.join(eval_assets_root, "salary.txt")
    return report_path, salary_txt


def _extract_text_from_odt_file(odt_file_path: str) -> str:
    try:
        doc = load(odt_file_path)
        text_content = ""
        for paragraph in doc.getElementsByType(P):
            if paragraph.firstChild is not None:
                text_content += str(paragraph.firstChild) + "\n"
        return text_content
    except Exception:
        return ""


def _extract_text_from_odt_bytes(binary_content: bytes) -> str:
    try:
        odt_file = BytesIO(binary_content)
        document = load(odt_file)
        text_content = ""
        for paragraph in document.getElementsByType(P):
            if paragraph.firstChild is not None:
                text_content += str(paragraph.firstChild) + "\n"
        return text_content
    except Exception:
        return ""


def _calc_salary_stats(salary_txt_path: str):
    try:
        # TAC sample's salary.txt appears misaligned; handle both CSV-like and simple delimited formats.
        df = pd.read_csv(salary_txt_path)
        # If the file is actually malformed (e.g., 4 columns but header suggests 3), attempt to coerce
        if "Salary" not in df.columns:
            # Try to infer last numeric column as Salary
            for col in df.columns[::-1]:
                try:
                    df[col] = pd.to_numeric(df[col])
                    df = df.rename(columns={col: "Salary"})
                    break
                except (ValueError, TypeError):
                    continue
        avg_salary = int(df["Salary"].mean())
        max_salary = int(df["Salary"].max())
        min_salary = int(df["Salary"].min())
        median_salary = int(df["Salary"].median())
        return [avg_salary, max_salary, min_salary, median_salary]
    except Exception:
        return []


def _check_keywords_in_content(content: str, keywords: List[int]) -> float:
    # Normalize content: remove $ and commas for flexible number matching
    # This handles both "78300" and "$78,300" formats
    import re

    content_lower = (content or "").lower()
    # Extract all numbers from the content (strip $, commas, etc.)
    content_numbers = set(re.findall(r"\d+", content_lower.replace(",", "")))

    found = [str(k) for k in keywords if str(k) in content_numbers]
    return found, len(found) / len(keywords) if keywords else 0.0


@grader
def grade_checkpoint1() -> float:
    # Mirror TAC: compute keyword score against calculated salary statistics
    report_path, salary_txt = _paths()
    if not (os.path.isfile(report_path) and os.path.isfile(salary_txt)):
        return 0.0
    content = _extract_text_from_odt_file(report_path)
    if not content:
        return 0.0
    keywords = _calc_salary_stats(salary_txt)
    print(f"Keywords: {keywords}")
    if not keywords:
        return 0.0
    found_keywords, keyword_score = _check_keywords_in_content(content, keywords)
    if found_keywords:
        print(f"Keywords found in the document: {', '.join(found_keywords)}")
    return keyword_score


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    # Single checkpoint worth 2 points, as in TAC
    score_float = grade_checkpoint1()
    checkpoints.append(Checkpoint(2, int(score_float * 2)))
    return result
