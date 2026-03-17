import logging
import os

# Import from LHAW evaluation system
import sys
from typing import List

sys.path.append("../../../")
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result

logging.basicConfig(level=logging.INFO)


def _paths():
    workspace_dir = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    paper_path = os.path.join(workspace_dir, "noise_simulation_paper.pdf")
    analysis_path = os.path.join(workspace_dir, "noise_simulation_analysis_sheet.txt")
    reference_answers_path = os.path.join(eval_assets_root, "eval_data", "answers.txt")
    return paper_path, analysis_path, reference_answers_path


def _normalize(text: str) -> str:
    return (text or "").strip().lower()


def _evaluate_answer(answer: str, reference: str) -> bool:
    return _normalize(answer) == _normalize(reference)


@grader
def grade_checkpoint1() -> bool:
    paper_path, analysis_path, _ = _paths()
    return os.path.isfile(paper_path) and os.path.isfile(analysis_path)


@grader
def grade_checkpoint2() -> int:
    """Score up to 11 based on correctness of answers vs reference list."""
    _, analysis_path, reference_answers_path = _paths()
    # Load reference answers
    try:
        with open(reference_answers_path, "r", encoding="utf-8", errors="ignore") as f:
            ref_answers = [line.strip() for line in f]
    except Exception as e:
        logging.warning(f"Unable to read reference answers: {e}")
        return 0

    # Ensure analysis sheet exists
    if not os.path.exists(analysis_path):
        logging.warning("Analysis sheet does not exist in the workspace directory")
        return 0

    score = 0
    try:
        with open(analysis_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i > 10:
                    break
                qa_pair = line.split(":")
                if len(qa_pair) < 2:
                    logging.warning("Analysis sheet corrupted")
                    return 0
                answer = qa_pair[1].strip()
                correct = _evaluate_answer(answer, ref_answers[i])
                score += 1 if correct else 0
                logging.info(
                    f"Question {qa_pair[0]}, Expected Answer: {ref_answers[i]}, Agent Answer: {answer}"
                )
    except Exception as e:
        logging.warning(f"Unable to evaluate analysis sheet: {e}")
        return 0

    return score


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    # Checkpoint 1: files downloaded (1pt)
    checkpoints.append(Checkpoint(1, int(grade_checkpoint1())))
    # Checkpoint 2: answers correctness (11pt)
    checkpoints.append(Checkpoint(11, int(grade_checkpoint2())))
    return result
