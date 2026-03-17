import os

# LHAW imports
import sys
from typing import List

sys.path.append("../../../")
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result, bonus_for_completing_final


@grader
def grade_final_checkpoint():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    path = os.path.join(output_root, "reimburse.txt")

    if not os.path.exists(path):
        return False

    try:
        text = open(path, "r").read()
    except Exception:
        return False

    if any([word in text for word in ["85.6", "25.0", "$"]]):
        return False

    return sum(
        [
            phrase.lower() in text.lower()
            for phrase in ["not qualified", "Credit card statements not accepted as receipts"]
        ]
    )


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints, bonus_for_completing_final)

    passed = grade_final_checkpoint()
    checkpoints.append(Checkpoint(2, int(passed)))

    return result
