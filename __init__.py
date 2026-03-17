"""
LHAW: Long-Horizon Autonomous Workflows Benchmark
=================================================

A benchmark for evaluating AI agents on strategic clarification
in underspecified, long-horizon enterprise workflows.

Key Components:
- synthetic: Unified pipeline for generating underspecified task variants

Usage:
    from lhaw.synthetic import SyntheticPipeline, Severity

    pipeline = SyntheticPipeline()

    # TAC tasks
    result = pipeline.process_tac_task("finance_check_attendance_payroll")

"""

from .synthetic import (
    Dimension,
    PipelineResult,
    Segment,
    Severity,
    SyntheticPipeline,
    UnderspecVariant,
    UnifiedTask,
)

__all__ = [
    "Dimension",
    "Severity",
    "Segment",
    "UnderspecVariant",
    "UnifiedTask",
    "PipelineResult",
    "SyntheticPipeline",
]

__version__ = "0.0.0"  # Prototyping phase
