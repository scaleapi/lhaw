import logging
import os
import sys
from typing import List

# LHAW imports
sys.path.append("../../../")
from evaluation.common import evaluate_with_llm, grader
from evaluation.scoring import Checkpoint, Result


def read_from_docx(fn):
    # use python-docx to read the content of the docx file
    import docx

    doc = docx.Document(fn)
    # Filter out empty lines and get actual answer lines
    lines = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return [l for l in lines if l != ""]


@grader
def grade_checkpoint1(answers):
    # [2pts] The agent correctly answers the question 1 (Tea (including ice tea)).
    try:
        if len(answers) < 1:
            logging.warning("Not enough lines found in answer.docx for question 1")
            return False

        content = answers[0]
        if "Tea (including ice tea)" in content or evaluate_with_llm(
            content=content,
            predicate="the content contains the correct answer `Tea (including ice tea)` or similar phrasing like 'iced tea'",
        ):
            logging.info("Correct answer found in answer.docx for question 1.")
            return True
        else:
            logging.warning(
                f"Incorrect answer found in answer.docx for question 1. Content: {content}"
            )
            return False
    except Exception as e:
        logging.error(f"Error evaluating question 1: {e}")
        return False


@grader
def grade_checkpoint2(answers):
    # [2pts] The agent correctly answers the question 2 (88.475)
    try:
        if len(answers) < 2:
            logging.warning("Not enough lines found in answer.docx for question 2")
            return False

        content = answers[1]
        if evaluate_with_llm(
            content=content,
            predicate="the content contains the correct answer `88.475` (may be rounded)",
        ):
            logging.info("Correct answer found in answer.docx for question 2.")
            return True
        else:
            logging.warning(
                f"Incorrect answer found in answer.docx for question 2. Content: {content}"
            )
            return False
    except Exception as e:
        logging.error(f"Error evaluating question 2: {e}")
        return False


@grader
def grade_checkpoint3(answers):
    # [2pts] The agent correctly answers the question 3 (74.775)
    try:
        if len(answers) < 3:
            logging.warning("Not enough lines found in answer.docx for question 3")
            return False

        content = answers[2]
        if evaluate_with_llm(
            content=content,
            predicate="the content has the correct answer `74.775` (may be rounded)",
        ):
            logging.info("Correct answer found in answer.docx for question 3.")
            return True
        else:
            logging.warning(
                f"Incorrect answer found in answer.docx for question 3. Content: {content}"
            )
            return False
    except Exception as e:
        logging.error(f"Error evaluating question 3: {e}")
        return False


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    filepath = os.path.join(output_root, "answer.docx")

    # Parse answers once and share across all checkpoints
    answers = []
    try:
        lines = read_from_docx(filepath)
        # Take the last 3 non-empty lines (the actual answers)
        answers = lines[-3:] if len(lines) >= 3 else lines
        logging.info(f"Parsed {len(answers)} answer lines from answer.docx")
    except Exception as e:
        logging.error(f"Could not read answer.docx: {e}")

    checkpoints.append(Checkpoint(2, int(grade_checkpoint1(answers)) * 2))
    checkpoints.append(Checkpoint(2, int(grade_checkpoint2(answers)) * 2))
    checkpoints.append(Checkpoint(2, int(grade_checkpoint3(answers)) * 2))

    return Result(checkpoints)
