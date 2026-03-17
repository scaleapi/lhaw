import logging
import os

# Import from LHAW evaluation system
import sys
from typing import List

from PyPDF2 import PdfReader

sys.path.append("../../../")
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _paths():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    expected_pdf_path = os.path.join(output_root, "openhands_short.pdf")
    reference_pdf_path = os.path.join(eval_assets_root, "expected_openhands_short.pdf")
    return expected_pdf_path, reference_pdf_path


EXPECTED_PAGE_COUNT = 17


@grader
def grade_checkpoint1():
    """Check if the output PDF file exists."""
    expected_pdf_path, _ = _paths()
    exists = os.path.exists(expected_pdf_path)
    if exists:
        logging.info(f"Checkpoint 1 passed: {expected_pdf_path} exists.")
    else:
        logging.warning(f"Checkpoint 1 failed: {expected_pdf_path} does not exist.")
    return exists


@grader
def grade_checkpoint2():
    """Check if the output PDF file has exactly EXPECTED_PAGE_COUNT pages."""
    expected_pdf_path, _ = _paths()
    if not os.path.exists(expected_pdf_path):
        logging.warning(f"Checkpoint 2 failed: {expected_pdf_path} does not exist.")
        return False
    try:
        pdf = PdfReader(expected_pdf_path)
        page_count = len(pdf.pages)
        if page_count == EXPECTED_PAGE_COUNT:
            logging.info(
                f"Checkpoint 2 passed: {expected_pdf_path} has {EXPECTED_PAGE_COUNT} pages."
            )
            return True
        else:
            logging.warning(
                f"Checkpoint 2 failed: {expected_pdf_path} has {page_count} pages instead of {EXPECTED_PAGE_COUNT}."
            )
            return False
    except Exception as e:
        logging.error(f"Checkpoint 2 failed: Error reading PDF for page count check - {e}")
        return False


@grader
def grade_checkpoint3():
    """Check if the content of all EXPECTED_PAGE_COUNT pages matches the expected content."""
    expected_pdf_path, reference_pdf_path = _paths()
    if not os.path.exists(expected_pdf_path):
        logging.warning(f"Checkpoint 3 failed: {expected_pdf_path} does not exist.")
        return False
    if not os.path.exists(reference_pdf_path):
        logging.warning(f"Checkpoint 3 failed: Reference file {reference_pdf_path} does not exist.")
        return False
    try:
        generated_pdf = PdfReader(expected_pdf_path)
        expected_pdf = PdfReader(reference_pdf_path)

        if (
            len(generated_pdf.pages) != EXPECTED_PAGE_COUNT
            or len(expected_pdf.pages) != EXPECTED_PAGE_COUNT
        ):
            logging.warning(
                f"Checkpoint 3 failed: Page count mismatch (generated: {len(generated_pdf.pages)}, expected: {len(expected_pdf.pages)})."
            )
            return False

        for i in range(EXPECTED_PAGE_COUNT):
            gen_text = (generated_pdf.pages[i].extract_text() or "").strip()
            exp_text = (expected_pdf.pages[i].extract_text() or "").strip()
            if gen_text != exp_text:
                logging.warning(f"Checkpoint 3 failed: Mismatch found on page {i + 1}.")
                return False

        logging.info("Checkpoint 3 passed: All pages match the expected content.")
        return True
    except Exception as e:
        logging.error(f"Checkpoint 3 failed: Error reading or comparing PDFs - {e}")
        return False


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)

    passed1 = grade_checkpoint1()
    checkpoints.append(Checkpoint(1, int(passed1)))

    passed2 = grade_checkpoint2()
    checkpoints.append(Checkpoint(1, int(passed2)))

    passed3 = grade_checkpoint3()
    checkpoints.append(Checkpoint(1, int(passed3)))

    return result
