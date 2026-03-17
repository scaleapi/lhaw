"""
LHAW Synthetic Underspecification Pipeline
==========================================

Generate underspecified task variants from well-specified prompts.

## Supported Domains

| Domain | Description | Tasks |
|--------|-------------|-------|
| TAC | TheAgentCompany enterprise workflows | ~30 |

## Usage

    from synthetic import SyntheticPipeline, Severity
    from synthetic.adapters.tac import TACAdapter
    
    pipeline = SyntheticPipeline()
    tac_adapter = TACAdapter()
    
    # TAC tasks - default is prompt-only (for ablation baseline)
    result = pipeline.process_tac_task("finance_check_attendance_payroll")
    
    # With trajectory grounding (improves segment prioritization)
    result = pipeline.process_tac_task(
        "finance_check_attendance_payroll",
        trajectory_path=tac_adapter.get_trajectory_url("finance_check_attendance_payroll"),
    )
    
    # With full grounding (trajectory + checkpoints)
    result = pipeline.process_tac_task(
        "finance_check_attendance_payroll",
        trajectory_path=tac_adapter.get_trajectory_url("finance_check_attendance_payroll"),
        use_checkpoints=True,
    )
    
    # Access results
    for variant in result.variants:
        print(variant.underspecified_prompt)
        print(f"  priority_score: {variant.predicted_difficulty}")

## Output Format

Pipeline outputs contain all metadata for downstream ablation/filtering:
- severity: DELETE | VAGUIFY | GENERICIZE
- criticality, guessability, priority_score
- is_used_in_trajectory, first_use_pct, checkpoint_refs
"""

from .core import Dimension, PipelineResult, Segment, Severity, UnderspecVariant, UnifiedTask
from .pipeline import SyntheticPipeline

__all__ = [
    "Dimension",
    "Severity",
    "Segment",
    "UnderspecVariant",
    "UnifiedTask",
    "PipelineResult",
    "SyntheticPipeline",
]
