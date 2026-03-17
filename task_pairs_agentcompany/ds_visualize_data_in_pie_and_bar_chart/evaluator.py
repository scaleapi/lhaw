import logging
import os
import sys

# Image-based evaluation requires a vision-capable model
os.environ.setdefault("LLM_EVAL_MODEL", "gpt-4o")

sys.path.append("../../../")
from evaluation.common import IMAGE_PNG, evaluate_with_llm, grader  # noqa: E402
from evaluation.scoring import Checkpoint, Result  # noqa: E402


@grader
def grade_checkpoint1():
    """Validate pie chart"""
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    pie_chart_file = os.path.join(output_root, "2011_pie.png")

    if not os.path.isfile(pie_chart_file):
        logging.warning(f"Pie chart not found at {pie_chart_file}")
        return False

    return evaluate_with_llm(
        content="",
        predicate="The picture contains a pie chart, with the value of Garlic being the highest, Sweet potato being the second highest, Chinese vegetables and Kale being the least.",
        image_path=pie_chart_file,
        image_type=IMAGE_PNG,
    )


@grader
def grade_checkpoint2():
    """Validate bar chart"""
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    bar_chart_file = os.path.join(output_root, "2016_bar.png")

    if not os.path.isfile(bar_chart_file):
        logging.warning(f"Bar chart not found at {bar_chart_file}")
        return False

    return evaluate_with_llm(
        content="",
        predicate="The picture contains a bar chart, with the value of Garlic being the highest, Kale being the second highest, Chinese vegetables and Sweet potato being the least.",
        image_path=bar_chart_file,
        image_type=IMAGE_PNG,
    )


def grade_checkpoints(trajectory="") -> Result:
    checkpoint1 = Checkpoint(2, 2 * int(grade_checkpoint1()))
    checkpoint2 = Checkpoint(2, 2 * int(grade_checkpoint2()))
    return Result([checkpoint1, checkpoint2])
