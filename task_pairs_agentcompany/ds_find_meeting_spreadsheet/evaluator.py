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
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    # Files are saved directly in the output directory
    link_path = os.path.join(output_root, "link.txt")
    downloaded_xlsx = os.path.join(output_root, "Seed Area Estimates.xlsx")
    return output_root, link_path, downloaded_xlsx


@grader
def grade_checkpoint1() -> bool:
    """Verify link.txt exists in workspace and contains the correct folder link."""
    _, link_path, _ = _paths()
    try:
        if not os.path.exists(link_path):
            logging.warning("File link.txt not found in workspace")
            return False
        with open(link_path, "r") as f:
            link = f.read().strip()
        # Note: Changed from "dir=/Documents/Data%20Analysis" to "/Documents/Data Analysis"
        if link and "/Documents/Data Analysis" in link:
            logging.info("Correct link found in link.txt")
            return True
        logging.warning("Incorrect link found in link.txt")
        return False
    except Exception as e:
        logging.error(f"Error occurred while checking link.txt: {e}")
        return False


@grader
def grade_checkpoint2() -> bool:
    """Verify the target Excel file was downloaded into the workspace."""
    _, _, downloaded_xlsx = _paths()
    try:
        if os.path.isfile(downloaded_xlsx):
            logging.info("File 'Seed Area Estimates.xlsx' successfully downloaded")
            return True
        logging.warning("File 'Seed Area Estimates.xlsx' not found in workspace")
        return False
    except Exception as e:
        logging.error(f"Error occurred while checking the downloaded file: {e}")
        return False


def grade_checkpoints(trajectory: str = "") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)

    passed1 = grade_checkpoint1()
    checkpoints.append(Checkpoint(1, int(passed1)))

    passed2 = grade_checkpoint2()
    checkpoints.append(Checkpoint(1, int(passed2)))

    return result
