import logging
import os
import sys
from typing import List

import torch
import torch.nn.functional as F

# Image-based evaluation requires a vision-capable model
os.environ.setdefault("LLM_EVAL_MODEL", "gpt-4o")

sys.path.append("../../../")
from evaluation.common import compare_images_with_llm, evaluate_with_llm, grader  # noqa: E402
from evaluation.scoring import Checkpoint, Result  # noqa: E402


@grader
def grade_checkpoint1():
    """Compare the visual similarity of generated GradCAM visualizations"""
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))

    gradcam_output = os.path.join(output_root, "gradcam_output.jpg")
    reference_gradcam = os.path.join(eval_assets_root, "eval_data", "reference_gradcam.jpg")

    # Note: eval_data/reference_gradcam.jpg might need to be adjusted depending on asset location
    # In LHAW, assets are usually in the same directory or a subdirectory relative to __file__

    try:
        return compare_images_with_llm(
            gradcam_output,
            reference_gradcam,
            "Do these two GradCAM visualizations highlight similar regions in the image?",
            "Focus on the regions highlighted in red/orange and their intensity patterns.",
        )
    except Exception as e:
        logging.error(f"Failed to compare GradCAM visualizations: {e}")
        return False


@grader
def grade_checkpoint2():
    """Check if both GradCAMs are highlighting semantically similar features"""
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    path = os.path.join(output_root, "gradcam_explanation.txt")

    if not os.path.exists(path):
        logging.warning("GradCAM explanation file not found")
        return False

    try:
        with open(path, "r") as f:
            gradcam_explanation = f.read()
        return evaluate_with_llm(
            gradcam_explanation,
            "The model made the prediction by focusing on the head area to a high degree and also the body to a lower degree.",
        )
    except Exception as e:
        logging.error(f"Failed to compare GradCAM explanations: {e}")
        return False


@grader
def grade_checkpoint3():
    """Compare the similarity of gradients"""
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))

    try:
        # Load saved gradients
        student_gradients = torch.load(os.path.join(output_root, "raw_gradients.pt"))
        reference_gradients = torch.load(
            os.path.join(eval_assets_root, "eval_data", "reference_gradients.pt")
        )

        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            student_gradients.flatten(), reference_gradients.flatten(), dim=0
        )

        # Consider them similar if cosine similarity is above 0.8
        return similarity.item() > 0.8
    except Exception as e:
        logging.error(f"Failed to compare gradients: {e}")
        return False


@grader
def grade_checkpoint4():
    """Compare the similarity of activations"""
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))

    try:
        # Load saved activations
        student_activations = torch.load(os.path.join(output_root, "raw_activations.pt"))
        reference_activations = torch.load(
            os.path.join(eval_assets_root, "eval_data", "reference_activations.pt")
        )

        # Calculate cosine similarity
        similarity = F.cosine_similarity(
            student_activations.flatten(), reference_activations.flatten(), dim=0
        )

        # Consider them similar if cosine similarity is above 0.8
        return similarity.item() > 0.8
    except Exception as e:
        logging.error(f"Failed to compare activations: {e}")
        return False


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)

    checkpoints.append(Checkpoint(1, int(grade_checkpoint1())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint2())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint3())))
    checkpoints.append(Checkpoint(1, int(grade_checkpoint4())))

    return result
