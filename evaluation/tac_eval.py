#!/usr/bin/env python3
"""
Evaluation script for LHAW AgentCompany tasks.
Based on TheAgentCompany/workspaces/base_image/eval.py
"""

import argparse
import importlib.util
import json
import os
import sys


def load_trajectory(file_path):
    """Load trajectory from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            return content
    except Exception as e:
        print(f"Warning: Error reading trajectory file: {e}", file=sys.stderr)
        return ""


def main():

    # Set up argument parser (same as TAC)
    parser = argparse.ArgumentParser(
        description="Grade checkpoints from trajectory and save results"
    )
    parser.add_argument(
        "--trajectory_path", required=False, default=None, help="Path to the trajectory file"
    )
    parser.add_argument(
        "--result_path",
        required=False,
        default="./result.json",
        help="Path to save the evaluation result JSON",
    )
    parser.add_argument(
        "--output_path",
        required=False,
        default=None,
        help="Alias for result path (TAC compatibility)",
    )
    parser.add_argument(
        "--evaluator_path",
        required=False,
        default="./evaluator.py",
        help="Path to evaluator.py (avoids copying)",
    )
    parser.add_argument(
        "--lhaw_root",
        required=False,
        default=None,
        help="Path to LHAW repo root (added to sys.path). Defaults to parent of directory containing this script.",
    )
    parser.add_argument(
        "--workspace_path",
        required=False,
        default=None,
        help="Path to exported session workspace to resolve /workspace/...",
    )
    parser.add_argument(
        "--eval_assets_root",
        required=False,
        default=None,
        help="Path to task-local evaluation assets (e.g., golden CSVs) used by evaluators via /utils/* paths",
    )
    parser.add_argument(
        "--output_root",
        required=False,
        default=None,
        help="Root directory where task outputs are stored (maps /workspace/output/* and OwnCloud-like paths to this directory)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Default lhaw_root to parent of directory containing this script (i.e. lhaw/ repo root)
    if args.lhaw_root is None:
        args.lhaw_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Ensure LHAW root is importable for evaluator dependencies
    if args.lhaw_root and args.lhaw_root not in sys.path:
        sys.path.append(args.lhaw_root)

    # Load the evaluator module (supports external path to avoid copying)
    evaluator_path = args.evaluator_path
    if not os.path.exists(evaluator_path):
        print(f"Error: evaluator.py not found at {evaluator_path}")
        return 1

    try:
        spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
        if spec and spec.loader:
            evaluator_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluator_module)
            grade_checkpoints = getattr(evaluator_module, "grade_checkpoints", None)

            if not callable(grade_checkpoints):
                print("Error: grade_checkpoints function not found in evaluator.py")
                return 1
        else:
            print("Error: Could not load evaluator module")
            return 1
    except Exception as e:
        print(f"Error loading evaluator: {e}")
        return 1

    trajectory = ""
    try:
        # Check if trajectory path is provided
        if args.trajectory_path is None:
            print(
                "Warning: No trajectory file provided, assuming empty trajectory", file=sys.stderr
            )
        else:
            trajectory = load_trajectory(args.trajectory_path)

        # Set environment variables for evaluators
        if args.workspace_path:
            # Default output_root to workspace_path/workspace/output if not specified
            output_root = (
                args.output_root
                if args.output_root
                else os.path.join(args.workspace_path, "workspace", "output")
            )
            os.environ["MRT_OUTPUT_ROOT"] = output_root

        if args.eval_assets_root:
            os.environ["MRT_EVAL_ASSETS_ROOT"] = args.eval_assets_root

        # Evaluators use MRT_OUTPUT_ROOT and MRT_EVAL_ASSETS_ROOT directly now; no path monkeypatching needed
        result = grade_checkpoints(trajectory)

        from evaluation.scoring import Result

        if not isinstance(result, Result):
            raise TypeError(f"grade_checkpoints must return Result type, got {type(result)}")

        if not result.checkpoints:
            raise ValueError(f"Result must have at least one checkpoint, got {result}")

        # Save result to JSON file
        result_json = result.to_dict()
        # TAC-compatible: prefer --output_path when provided
        output_path = args.output_path if args.output_path else args.result_path
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=4)

    except Exception as e:
        print(f"Error: Failed to grade the task: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
