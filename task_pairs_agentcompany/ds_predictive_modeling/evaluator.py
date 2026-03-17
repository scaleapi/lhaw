import logging
import os

# LHAW imports
import sys
from typing import List

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append("../../../")
from evaluation.common import grader
from evaluation.scoring import Checkpoint, Result


def _paths():
    """Resolve file paths for evaluation."""
    # Output directory where agent writes results
    output_root = os.environ.get("MRT_OUTPUT_ROOT", "/workspace/output")
    generated_file_path = os.path.join(output_root, "stock_test_answer.csv")

    # Evaluator assets root (where golden_answer.csv lives)
    eval_assets_root = os.environ.get("MRT_EVAL_ASSETS_ROOT", os.path.dirname(__file__))
    golden_path = os.path.join(eval_assets_root, "golden_answer.csv")

    return generated_file_path, golden_path


@grader
def grade_checkpoint1():
    generated_file_path, golden_path = _paths()

    # Check if the generated file exists
    if not os.path.isfile(generated_file_path):
        logging.warning(f"Generated file not found at {generated_file_path}")
        return 0

    golden_answer = pd.read_csv(golden_path)

    try:
        stock_test_answer = pd.read_csv(generated_file_path)
        actual, pred = golden_answer["Close Price"], stock_test_answer["Close Price"]
        mse = mean_squared_error(actual, pred)
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)

        logging.info(f"On private test dataset, MSE: {mse}, MAE: {mae}, R^2: {r2}")

        # 3 credit granted if MSE < 500, MAE < 10, R^2 score > 0.8
        if mse < 500 and mae < 10 and r2 > 0.8:
            return 3
        # 2 credits granted if MSE < 550, MAE < 20, R^2 score > 0.7
        elif mse < 550 and mae < 20 and r2 > 0.7:
            return 2
        # No credits granted if MSE > 550, MAE > 20, R^2 score < 0.7
        else:
            return 0
    except KeyError:
        logging.warning("Column 'Close Price' not found in answer or golden file.")
        return 0
    except ValueError as e:
        logging.warning(f"Value error during metric calculation: {e}")
        return 0


def grade_checkpoints(trajectory="") -> Result:
    checkpoints: List[Checkpoint] = []
    result = Result(checkpoints)
    checkpoints.append(Checkpoint(3, int(grade_checkpoint1())))

    return result
