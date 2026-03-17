"""
Dataset preparation stages for underspecified task generation.

This package contains modular stages for the dataset preparation pipeline:
- discover_baselines: Auto-discover baseline model runs
- aggregate_passed: Aggregate passed tasks across models
- generate_underspec: Generate underspecified variants
- extract_baselines: Extract baseline results for comparison
"""

from . import aggregate_passed, discover_baselines, extract_baselines, generate_underspec

__all__ = [
    "discover_baselines",
    "aggregate_passed",
    "generate_underspec",
    "extract_baselines",
]
