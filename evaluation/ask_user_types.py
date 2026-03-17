"""
Data types for ask_user analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class AskUserOutcome(Enum):
    """Outcome of an ask_user call."""
    ANSWERED = "answered"      # User provided a response
    REFUSED = "refused"        # User refused to answer
    PARTIAL = "partial"        # User gave partial/unclear response
    UNKNOWN = "unknown"        # Could not determine outcome


@dataclass
class AskUserCall:
    """Single ask_user tool call extracted from conversation."""
    call_index: int              # Order among ask_user calls (0-indexed)
    tool_call_id: str
    question_text: str           # The question asked
    user_response: Optional[str] # User's response
    outcome: AskUserOutcome      # ANSWERED/REFUSED/PARTIAL/UNKNOWN
    message_index: int           # Position in message list


@dataclass
class AskUserSequence:
    """All ask_user calls from a single trial."""
    task_id: str
    variant_id: str
    trial_index: int
    model: str
    calls: List[AskUserCall] = field(default_factory=list)

    @property
    def count(self) -> int:
        """Number of ask_user calls."""
        return len(self.calls)

    @property
    def questions(self) -> List[str]:
        """List of question texts."""
        return [c.question_text for c in self.calls]


# All 21 flag names grouped by category
FLAG_CATEGORIES = {
    "Question Quality": [
        "flag_vague_questions",
        "flag_compound_questions",
        "flag_closed_questions",
    ],
    "Question Targeting": [
        "flag_missed_critical_segments",
        "flag_wrong_dimension_focus",
        "flag_inferable_information_asked",
    ],
    "Information Integration": [
        "flag_ignoring_user_response",
        "flag_incorrect_information_application",
        "flag_partial_information_utilization",
    ],
    "Over-Clarification": [
        "flag_redundant_questions",
        "flag_unnecessary_questions",
        "flag_excessive_cost",
    ],
    "Under-Clarification": [
        "flag_premature_execution",
        "flag_insufficient_question_depth",
        "flag_false_confidence",
    ],
    "Timing & Strategy": [
        "flag_late_clarification",
        "flag_upfront_question_dump",
        "flag_inefficient_question_sequence",
    ],
    "User Response Misinterpretation": [
        "flag_literal_misunderstanding",
        "flag_context_misapplication",
        "flag_implicit_information_missed",
    ],
}

ALL_FLAGS = [flag for flags in FLAG_CATEGORIES.values() for flag in flags]


@dataclass
class AskUserJudgment:
    """LLM judge evaluation with all 21 flags from ask_user_judge.j2."""
    # Question Quality (3 flags)
    flag_vague_questions: bool
    flag_compound_questions: bool
    flag_closed_questions: bool

    # Question Targeting (3 flags)
    flag_missed_critical_segments: bool
    flag_wrong_dimension_focus: bool
    flag_inferable_information_asked: bool

    # Information Integration (3 flags)
    flag_ignoring_user_response: bool
    flag_incorrect_information_application: bool
    flag_partial_information_utilization: bool

    # Over-Clarification (3 flags)
    flag_redundant_questions: bool
    flag_unnecessary_questions: bool
    flag_excessive_cost: bool

    # Under-Clarification (3 flags)
    flag_premature_execution: bool
    flag_insufficient_question_depth: bool
    flag_false_confidence: bool

    # Timing & Strategy (3 flags)
    flag_late_clarification: bool
    flag_upfront_question_dump: bool
    flag_inefficient_question_sequence: bool

    # User Response Misinterpretation (3 flags)
    flag_literal_misunderstanding: bool
    flag_context_misapplication: bool
    flag_implicit_information_missed: bool

    # Summary
    overall_assessment: str
    reasoning: str
    failure_categories: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV/JSON output."""
        return {
            "flag_vague_questions": self.flag_vague_questions,
            "flag_compound_questions": self.flag_compound_questions,
            "flag_closed_questions": self.flag_closed_questions,
            "flag_missed_critical_segments": self.flag_missed_critical_segments,
            "flag_wrong_dimension_focus": self.flag_wrong_dimension_focus,
            "flag_inferable_information_asked": self.flag_inferable_information_asked,
            "flag_ignoring_user_response": self.flag_ignoring_user_response,
            "flag_incorrect_information_application": self.flag_incorrect_information_application,
            "flag_partial_information_utilization": self.flag_partial_information_utilization,
            "flag_redundant_questions": self.flag_redundant_questions,
            "flag_unnecessary_questions": self.flag_unnecessary_questions,
            "flag_excessive_cost": self.flag_excessive_cost,
            "flag_premature_execution": self.flag_premature_execution,
            "flag_insufficient_question_depth": self.flag_insufficient_question_depth,
            "flag_false_confidence": self.flag_false_confidence,
            "flag_late_clarification": self.flag_late_clarification,
            "flag_upfront_question_dump": self.flag_upfront_question_dump,
            "flag_inefficient_question_sequence": self.flag_inefficient_question_sequence,
            "flag_literal_misunderstanding": self.flag_literal_misunderstanding,
            "flag_context_misapplication": self.flag_context_misapplication,
            "flag_implicit_information_missed": self.flag_implicit_information_missed,
            "overall_assessment": self.overall_assessment,
            "reasoning": self.reasoning,
            "failure_categories": self.failure_categories,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AskUserJudgment":
        """Create from dictionary."""
        return cls(
            flag_vague_questions=data.get("flag_vague_questions", False),
            flag_compound_questions=data.get("flag_compound_questions", False),
            flag_closed_questions=data.get("flag_closed_questions", False),
            flag_missed_critical_segments=data.get("flag_missed_critical_segments", False),
            flag_wrong_dimension_focus=data.get("flag_wrong_dimension_focus", False),
            flag_inferable_information_asked=data.get("flag_inferable_information_asked", False),
            flag_ignoring_user_response=data.get("flag_ignoring_user_response", False),
            flag_incorrect_information_application=data.get("flag_incorrect_information_application", False),
            flag_partial_information_utilization=data.get("flag_partial_information_utilization", False),
            flag_redundant_questions=data.get("flag_redundant_questions", False),
            flag_unnecessary_questions=data.get("flag_unnecessary_questions", False),
            flag_excessive_cost=data.get("flag_excessive_cost", False),
            flag_premature_execution=data.get("flag_premature_execution", False),
            flag_insufficient_question_depth=data.get("flag_insufficient_question_depth", False),
            flag_false_confidence=data.get("flag_false_confidence", False),
            flag_late_clarification=data.get("flag_late_clarification", False),
            flag_upfront_question_dump=data.get("flag_upfront_question_dump", False),
            flag_inefficient_question_sequence=data.get("flag_inefficient_question_sequence", False),
            flag_literal_misunderstanding=data.get("flag_literal_misunderstanding", False),
            flag_context_misapplication=data.get("flag_context_misapplication", False),
            flag_implicit_information_missed=data.get("flag_implicit_information_missed", False),
            overall_assessment=data.get("overall_assessment", ""),
            reasoning=data.get("reasoning", ""),
            failure_categories=data.get("failure_categories", []),
        )


@dataclass
class AskUserAnalysis:
    """Complete analysis for a single trajectory."""
    sequence: AskUserSequence
    judgment: Optional[AskUserJudgment]
    coverage_score: Optional[float]
    ambiguity_class: Optional[str]
    removed_value: Optional[str]
    dimension: Optional[str]

    def to_flat_dict(self) -> dict:
        """Flatten to a single dict for CSV output."""
        result = {
            "task_id": self.sequence.task_id,
            "variant_id": self.sequence.variant_id,
            "model": self.sequence.model,
            "trial_index": self.sequence.trial_index,
            "ambiguity_class": self.ambiguity_class,
            "coverage_score": self.coverage_score,
            "removed_value": self.removed_value,
            "dimension": self.dimension,
            "ask_user_count": self.sequence.count,
            "questions_json": self.sequence.questions,
        }

        if self.judgment:
            result.update(self.judgment.to_dict())
        else:
            # Add empty flag columns
            for flag in ALL_FLAGS:
                result[flag] = None
            result["overall_assessment"] = None
            result["reasoning"] = None
            result["failure_categories"] = None

        return result
