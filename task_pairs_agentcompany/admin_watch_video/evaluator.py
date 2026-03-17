import logging
import os
import re

# LHAW imports
import sys
from typing import List

sys.path.append("../../../")
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result


@grader
def grade_checkpoint1():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    try:
        with open(os.path.join(output_root, "ans.txt"), "r") as file:
            content = file.read().strip()

            # Use regex to find all numbers in the content
            numbers = re.findall(r"\b\d+\b", content)

            # Check if there's exactly one number
            if len(numbers) != 1:
                return 0
            else:
                number = int(numbers[0])
                if number == 4:
                    return 2
                else:
                    return 0
    except FileNotFoundError:
        logging.warning(f"{os.path.join(output_root, 'ans.txt')} is not found")
        return 0


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)

    total_score = 2
    checkpoints.append(Checkpoint(total_score, int(grade_checkpoint1())))

    return result
