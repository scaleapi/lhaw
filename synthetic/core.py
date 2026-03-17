"""
Core Data Structures for LHAW Synthetic Pipeline
=================================================
Domain-agnostic types that work across TAC and future domains.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# =============================================================================
# TAXONOMY
# =============================================================================


class Dimension(Enum):
    """The four dimensions of underspecification."""

    GOAL = "goal"  # WHAT to produce
    CONSTRAINT = "constraint"  # HOW to do it
    INPUT = "input"  # FROM WHERE
    CONTEXT = "context"  # WHAT BACKGROUND KNOWLEDGE


class Severity(Enum):
    """Removal strategies / severity levels."""

    DELETE = "delete"  # Remove entirely (HIGH severity)
    VAGUIFY = "vaguify"  # Replace with vague language (MEDIUM)
    GENERICIZE = "genericize"  # Subtle rewording (LOW)


# =============================================================================
# CORE TYPES
# =============================================================================


@dataclass
class Segment:
    """
    An atomic piece of information that can be removed to create underspecification.

    Field ordering:
        1. Identity: id, dimension, subdimension, value, text
        2. Grounding: is_used_in_trajectory, first_use_pct, checkpoint_refs
        3. Scores: criticality, guessability
        4. Computed: priority_score
        5. Metadata: dataset-specific fields
    """

    # Identity (1-indexed: S1, S2, S3, ...)
    id: str  # e.g., "S1", "S2"
    dimension: Dimension
    subdimension: str = ""
    value: str = ""
    text: str = ""

    # Grounding (from trajectory/checkpoints analysis)
    is_used_in_trajectory: Optional[bool] = None
    first_use_pct: Optional[float] = None  # 0.0=start, 1.0=end
    checkpoint_refs: List[str] = field(default_factory=list)

    # Scores (0-1 scale)
    criticality: float = 0.5  # Task fails without this?
    guessability: float = 0.5  # Agent can infer?

    # Computed
    priority_score: float = 0.0  # criticality × (1 - guessability)

    # Dataset-specific metadata (flexible dict for domain-specific info)
    # Examples:
    #   TAC: {"trajectory_step": 5, "tool_used": "write_file"}
    #   SWE-bench: {"file_path": "src/main.py", "line_number": 42}
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "id": self.id,
            "dimension": self.dimension.value,
            "subdimension": self.subdimension,
            "value": self.value if isinstance(self.value, str) else str(self.value),
            "text": self.text,
            "is_used_in_trajectory": self.is_used_in_trajectory,
            "first_use_pct": self.first_use_pct,
            "checkpoint_refs": self.checkpoint_refs,
            "criticality": self.criticality,
            "guessability": self.guessability,
            "priority_score": self.priority_score,
        }
        # Include metadata if present
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class UnderspecVariant:
    """A modified task with controlled underspecification."""

    id: str
    original_prompt: str
    underspecified_prompt: str
    removed_segments: List[Segment]
    severity: Severity

    # For evaluation
    expected_failure_mode: str = ""
    expected_questions: List[Dict[str, Any]] = field(
        default_factory=list
    )  # [{segment_id, questions}]
    predicted_difficulty: float = 0.0  # mean(seg.priority_score); bounded [0,1]

    def to_dict(self) -> Dict[str, Any]:
        """Full dict for debugging/intermediate storage."""
        return {
            "id": self.id,
            "underspecified_prompt": self.underspecified_prompt,
            "removed_segments": [s.to_dict() for s in self.removed_segments],
            "severity": self.severity.value if self.severity else None,
            "expected_failure_mode": self.expected_failure_mode,
            "expected_questions": self.expected_questions,
            "predicted_difficulty": self.predicted_difficulty,
        }


@dataclass
class UnifiedTask:
    """Domain-agnostic task representation."""

    id: str
    domain: str  # "tac" | "swebench" | "mcpatlas" | ...
    category: str = ""
    prompt: str = ""

    # Structured info (optional)
    known_info: Optional[str] = None

    # Grounding sources
    trajectory_path: Optional[str] = None
    checkpoints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "category": self.category,
            "prompt": self.prompt,
            "known_info": self.known_info,
            "trajectory_path": self.trajectory_path,
            "checkpoints": self.checkpoints,
        }


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass
class PipelineResult:
    """Output of the synthetic pipeline for a single task."""

    task: UnifiedTask
    segments: List[Segment]
    variants: List[UnderspecVariant]
    extraction_model: str = ""
    generation_model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "segments": [s.to_dict() for s in self.segments],
            "variants": [v.to_dict() for v in self.variants],
            "extraction_model": self.extraction_model,
            "generation_model": self.generation_model,
        }
