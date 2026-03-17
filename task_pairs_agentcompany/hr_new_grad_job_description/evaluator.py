import logging
import os

# LHAW imports
import sys
from typing import List

sys.path.append("../../../")
from evaluation.common import evaluate_with_llm, grader
from evaluation.scoring import Checkpoint, Result


@grader
def check_job_description_content():
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))

    path = os.path.join(output_root, "job_description.md")

    if not os.path.exists(path):
        return False

    with open(path, "r") as f:
        final_content = f.read()

    if "[Insert Information]" in final_content:
        return False

    # Load template and requirements from local artifacts
    template_path = os.path.join(eval_assets_root, "template.md")
    req_path = os.path.join(eval_assets_root, "requirement.txt")

    try:
        with open(template_path, "r") as f:
            template_content = f.read()
        with open(req_path, "r") as f:
            requirement_content = f.read()
    except Exception as e:
        logging.error(f"Failed to read assets: {e}")
        return False

    # LLM-based verification matching TAC
    predicate = f'a successful combination of the markdown template ""{template_content}"" and the requirement file ""{requirement_content}""'
    additional_prompt = "Pay special consideration to all of the numerical details. "

    return evaluate_with_llm(final_content, predicate, additional_prompt)


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)

    # In LHAW, we skip the OwnCloud link verification and download steps
    # because the environment doesn't support the full OwnCloud interaction.
    # We simply check if the job description content is correct.

    # Checkpoint 1: Content verification (merged into one checkpoint or just one check)
    # Original task had 3 checkpoints. If we want to strictly follow "remove that dimension",
    # we might just return the content check.

    passed = check_job_description_content()
    checkpoints.append(Checkpoint(1, int(passed)))

    return result
