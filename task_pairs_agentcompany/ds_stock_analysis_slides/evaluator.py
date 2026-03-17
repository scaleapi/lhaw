import io
import logging
import os
import sys
from collections import defaultdict
from typing import List

import pandas as pd
from PIL import Image
from pptx import Presentation

# Image-based evaluation requires a vision-capable model
os.environ.setdefault("LLM_EVAL_MODEL", "gpt-4o")

sys.path.append("../../../")
from evaluation.common import IMAGE_PNG, evaluate_with_llm, grader  # noqa: E402
from evaluation.scoring import Checkpoint, Result  # noqa: E402


def _paths():
    """Resolve file paths for evaluation."""
    # Output directory where agent writes results
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    csv_file_path = os.path.join(output_root, "stock_presentation", "stock_cleaned.csv")
    pptx_file_path = os.path.join(output_root, "stock_presentation", "stock_analysis.pptx")
    script_file_path = os.path.join(output_root, "stock_presentation", "stock_analysis_script.md")

    # Evaluator assets root (where stock_data.csv lives)
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    golden_csv_path = os.path.join(eval_assets_root, "stock_data.csv")

    return csv_file_path, pptx_file_path, script_file_path, golden_csv_path


@grader
def grade_checkpoint1(csv_file_path: str, golden_csv_path: str):

    # Check if files exist
    if not os.path.isfile(csv_file_path):
        logging.warning(f"Generated CSV file not found at {csv_file_path}")
        return 0

    golden_df = pd.read_csv(golden_csv_path)

    try:
        # Strict check: exact equality matching gold stock dataset
        df = pd.read_csv(csv_file_path)
        return df.equals(golden_df)
    except Exception as e:
        logging.warning(f"Error reading or comparing CSV files: {e}")
        return 0


@grader
def grade_checkpoint2(pptx_file_path: str):

    if not os.path.isfile(pptx_file_path):
        logging.warning(f"PPTX file not found at {pptx_file_path}")
        return 0

    try:
        presentation = Presentation(pptx_file_path)
    except Exception as e:
        logging.warning(f"Failed to read PPTX: {e}")
        return 0

    # Create output directory for extracted images
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    output_dir = os.path.join(output_root, "figs")
    os.makedirs(output_dir, exist_ok=True)

    # Predicates matching TAC evaluator
    predicates = [
        "Closing Prices Over Time: Display the stock's closing prices over time to show the overall trend.",
        "Close Price with 50 and 200-Day Moving Averages: Overlay the 50-day and 200-day moving averages onto the closing price plot.",
        "Correlation Between Columns: Visualize the correlation matrix.",
        "Daily Return Distribution: Calculate and plot the daily returns to highlight volatility and risk.",
        "Volume Distribution: Plot the distribution of trade volumes to identify trends in stock interest over time.",
    ]

    # Key: predicate, Value: evaluation result (0 or 1)
    results = defaultdict(int)

    for idx, slide in enumerate(presentation.slides, start=1):
        if idx > len(predicates):
            break

        image_found = 0
        for shape in slide.shapes:
            if shape.shape_type == 13:  # Picture type
                image_found += 1
                try:
                    image = shape.image
                    image_bytes = image.blob
                    image_stream = io.BytesIO(image_bytes)
                    pil_image = Image.open(image_stream)
                    image_path = os.path.join(output_dir, f"slide_{idx}_{image_found}.png")
                    pil_image.save(image_path, format="PNG")
                except Exception as e:
                    logging.warning(f"Failed to extract image from slide {idx}: {e}")
                    continue

        if image_found >= 1:
            predicate = predicates[idx - 1]
            # Check each image on the slide
            for img in range(1, image_found + 1):
                image_path = os.path.join(output_dir, f"slide_{idx}_{img}.png")
                if os.path.isfile(image_path):
                    evaluation_result = evaluate_with_llm(
                        content=None,
                        predicate=predicate,
                        image_path=image_path,
                        image_type=IMAGE_PNG,
                    )
                    # If any image on the slide passes, the slide passes
                    if evaluation_result:
                        results[predicate] = 1
                        break

    # Log failures
    for idx, predicate in enumerate(predicates):
        if results[predicate] == 0:
            logging.warning(f"Predicate {idx} failed, Predicate: {predicate}")

    return sum(results.values())


@grader
def grade_checkpoint3(script_file_path: str):

    if not os.path.isfile(script_file_path):
        logging.warning(f"Script file not found at {script_file_path}")
        return 0

    try:
        with open(script_file_path, "r", encoding="utf-8") as f:
            script_content = f.read()
    except Exception as e:
        logging.warning(f"Failed to read script file: {e}")
        return 0

    # Predicates matching TAC evaluator
    predicates = [
        "The script contains clear introductions, addressing and greeting the audience, provides presentation context for the audience, analyzes the business context and impact of each graph. The script ends with closing comments expressing gratitude.",
        "The script includes correspondence for each slide below: Slide 1: Closing Prices Over Time, Slide 2: Close Price with 50 and 200-Day Moving Averages, Slide 3: Correlation Between Columns, Slide 4: Daily Return Distribution, Slide 5: Volume Distribution.",
    ]

    results = {pred: evaluate_with_llm(script_content, pred) for pred in predicates}

    for pred, result in results.items():
        logging.info(f"{pred}: {'Pass' if result else 'Fail'}")

    # Sum over all pass results as int
    return sum([int(result) for result in results.values()])


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []

    csv_file_path, pptx_file_path, script_file_path, golden_csv_path = _paths()
    logging.info(f"CSV file path: {csv_file_path} (exists={os.path.isfile(csv_file_path)})")
    logging.info(f"PPTX file path: {pptx_file_path} (exists={os.path.isfile(pptx_file_path)})")
    logging.info(
        f"Script file path: {script_file_path} (exists={os.path.isfile(script_file_path)})"
    )
    logging.info(f"Golden CSV path: {golden_csv_path} (exists={os.path.isfile(golden_csv_path)})")

    result = Result(checkpoints)
    checkpoints.append(Checkpoint(1, int(grade_checkpoint1(csv_file_path, golden_csv_path))))
    # Checkpoint 2 returns a score up to 5
    checkpoints.append(Checkpoint(5, int(grade_checkpoint2(pptx_file_path))))
    checkpoints.append(Checkpoint(2, int(grade_checkpoint3(script_file_path))))
    return result
