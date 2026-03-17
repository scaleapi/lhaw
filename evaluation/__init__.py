"""
LHAW Evaluation Module
======================

Evaluation utilities and scoring system for LHAW benchmark tasks.

Key exports:
- Checkpoint, Result: Core scoring types (from TAC)
- grader: Decorator for checkpoint grading functions
- pass_at_k, pass_hat_k: Pass metrics from τ-bench
"""

from .common import (
    compare_images_with_llm,
    evaluate_with_llm,
    grader,
    llm_complete,
    verify_image_file,
)
from .pass_k_metrics import (
    TaskTrialResults,
    TrialResult,
    aggregate_pass_metrics,
    compute_task_metrics,
    pass_at_k,
    pass_hat_k,
)
from .scoring import Checkpoint, Result, bonus_for_completing_any, bonus_for_completing_final

__all__ = [
    # Scoring
    "Checkpoint",
    "Result",
    "bonus_for_completing_any",
    "bonus_for_completing_final",
    # Grading
    "grader",
    "evaluate_with_llm",
    "compare_images_with_llm",
    "verify_image_file",
    "llm_complete",
    # Pass metrics
    "pass_at_k",
    "pass_hat_k",
    "TrialResult",
    "TaskTrialResults",
    "compute_task_metrics",
    "aggregate_pass_metrics",
]
