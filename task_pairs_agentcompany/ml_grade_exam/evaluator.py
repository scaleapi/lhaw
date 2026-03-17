import logging
import os

# LHAW imports
import sys
from typing import List

sys.path.append("../../../")
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result


@grader
def grade_checkpoint1():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    base_name = "AgentCompany_ML_Quiz_Attempt"
    # Assuming 6 files based on original evaluator
    files_found = {f"{base_name}{i}": False for i in range(1, 7)}

    for file in os.listdir(output_root):
        if file.startswith(base_name):
            # Extract the part that corresponds to AttemptX
            # file format: AgentCompany_ML_Quiz_AttemptX...
            # We check if it starts with the key
            for key in files_found:
                if file.startswith(key):
                    files_found[key] = True

    for key in files_found:
        if not files_found[key]:
            logging.warning(f"File starting with {key} not found")
            return False

    return True


@grader
def grade_checkpoint2():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    base_name = "AgentCompany_ML_Quiz_Attempt"
    # Expected scores from original evaluator
    files_graded = {
        "AgentCompany_ML_Quiz_Attempt1_1": False,
        "AgentCompany_ML_Quiz_Attempt2_15": False,
        "AgentCompany_ML_Quiz_Attempt3_9": False,
        "AgentCompany_ML_Quiz_Attempt4_12": False,
        "AgentCompany_ML_Quiz_Attempt5_11": False,
        "AgentCompany_ML_Quiz_Attempt6_10": False,
    }

    for file in os.listdir(output_root):
        # Split filename to get base + score
        # e.g. AgentCompany_ML_Quiz_Attempt1_1.pdf -> AgentCompany_ML_Quiz_Attempt1_1
        # or AgentCompany_ML_Quiz_Attempt2_15-submit.pdf -> AgentCompany_ML_Quiz_Attempt2_15
        file_prefix = file.split(".")[0].split("-")[0]
        if file_prefix in files_graded:
            files_graded[file_prefix] = True

    score = 0
    for key in files_graded:
        if not files_graded[key]:
            logging.warning(f"File starting with {key} not graded correctly")
        else:
            score += 1

    return score


@grader
def grade_checkpoint3():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    correct_submission = "AgentCompany_ML_Quiz_Attempt2_15-submit.pdf"

    # Check if any OTHER file has -submit
    for file in os.listdir(output_root):
        if "submit" in file and file != correct_submission:
            return False

    # Check if correct file is picked
    if correct_submission not in os.listdir(output_root):
        return False

    return True


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)

    checkpoints.append(Checkpoint(1, int(grade_checkpoint1())))
    checkpoints.append(Checkpoint(6, int(grade_checkpoint2())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint3())))

    return result
