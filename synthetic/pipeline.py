"""
Unified Synthetic Pipeline
===========================
Domain-agnostic pipeline for generating underspecified task variants.

Architecture:
    
    PHASE 1: SEGMENT EXTRACTION (single LLM call)
    ├── LLM extracts segments and self-assesses ALL fields:
    │   • Identity: dimension, subdimension, value, text
    │   • Grounding: is_used_in_trajectory, first_use_pct, checkpoint_refs
    │   • Scores: criticality, guessability
    └── Pipeline only computes: id (S1, S2) and priority_score
    
    PHASE 2: VARIANT GENERATION
    ├── Apply severity (DELETE | VAGUIFY | GENERICIZE) with segment(s) to remove
    └── Output ranked variants with all metadata

Filtering is done DOWNSTREAM via pass@k experiments, not by this pipeline.
Output contains all metadata needed for ablation:
    - All grounding/scores from LLM in single extraction call
    - Pipeline only adds: id, priority_score = criticality × (1 - guessability)

Usage:
    from synthetic.pipeline import SyntheticPipeline, Severity
    
    pipeline = SyntheticPipeline()
    
    # Generate ranked variants (TAC)
    result = pipeline.process_tac_task("finance_check_attendance_payroll")
    
    # MCP Atlas task
    result = pipeline.process_mcpatlas_task("68a398aa2f58036e8d45eddf")
    
    # With specific severity
    result = pipeline.process_tac_task(..., severity=Severity.DELETE)
    
    # Export for downstream filtering
    output = result.to_dict()
"""

import json
import logging
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from .core import Dimension, PipelineResult, Segment, Severity, UnderspecVariant, UnifiedTask
from .llm import SEGMENT_EXTRACTION_SCHEMA, VARIANT_GENERATION_SCHEMA, llm_completion

logger = logging.getLogger(__name__)


def clean_prompt(prompt: str, original_prompt: str | None = None) -> str:
    """Clean mechanical artifacts from segment removal.

    Only applies a fix if the pattern exists in the underspecified prompt
    but NOT in the original (i.e., the artifact was introduced by removal).

    Args:
        prompt: The underspecified prompt to clean.
        original_prompt: Optional original prompt for safety checking.
    """
    cleanup_rules = [
        (r",\s*,", r",\s*,", ","),
        (r"\band,\s+", r"\band,\s+", "and "),
        (r"\n\n,\s*[a-z]", r"\n\n,\s*([a-z])", lambda m: "\n" + m.group(1).upper()),
        (r"\n-\s*\n-", r"\n-\s*\n-", "\n-"),
        (r"\n\s*[-*]\s*\n", r"\n\s*[-*]\s*\n", "\n"),
        (r"\n{3,}", r"\n{3,}", "\n\n"),
        (r" {2,}", r" {2,}", " "),
    ]

    cleaned = prompt
    for pattern_check, pattern_replace, replacement in cleanup_rules:
        if original_prompt is not None:
            if re.search(pattern_check, original_prompt) or not re.search(pattern_check, cleaned):
                continue
        cleaned = re.sub(pattern_replace, replacement, cleaned)

    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    return cleaned.replace("\x00", "")


# =============================================================================
# TAXONOMY LOADER
# =============================================================================


def load_taxonomy() -> Dict[str, Any]:
    """Load underspecification taxonomy from configs/taxonomy.yaml."""
    taxonomy_path = Path(__file__).parent / "configs" / "taxonomy.yaml"
    if taxonomy_path.exists():
        with open(taxonomy_path) as f:
            return yaml.safe_load(f)
    return {}


def format_taxonomy_for_prompt(taxonomy: Dict[str, Any]) -> str:
    """Format taxonomy dimensions into a prompt-friendly string."""
    if not taxonomy or "dimensions" not in taxonomy:
        # Fallback to minimal taxonomy
        return """Each segment belongs to one of four dimensions:
1. **GOAL** - WHAT to produce (output file, format, success criteria)
2. **CONSTRAINT** - HOW to do it (thresholds, rules, methods)
3. **INPUT** - FROM WHERE (file paths, data sources)
4. **CONTEXT** - WHAT BACKGROUND (jargon, business rules)"""

    lines = ["Each segment belongs to one of four dimensions:\n"]

    for i, (dim_name, dim_data) in enumerate(taxonomy["dimensions"].items(), 1):
        question = dim_data.get("question", "")
        definition = dim_data.get("definition", "").strip()

        lines.append(f"### {i}. {dim_name.upper()} — {question}")
        lines.append(f"{definition}\n")

        # Add indicators
        indicators = dim_data.get("indicators", [])
        if indicators:
            lines.append("**Indicators:**")
            for ind in indicators[:3]:  # Top 3 to keep prompt concise
                lines.append(f"- {ind}")
            lines.append("")

        # Add subdimensions with examples
        subdims = dim_data.get("subdimensions", {})
        if subdims:
            lines.append("**Subdimensions:**")
            for sub_name, sub_data in subdims.items():
                examples = sub_data.get("examples", [])[:2]
                examples_str = ", ".join(f'"{e}"' for e in examples)
                lines.append(
                    f"- `{sub_name}`: {sub_data.get('description', '')} (e.g., {examples_str})"
                )
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# SEGMENT EXTRACTOR (Unified)
# =============================================================================

EXTRACTION_PROMPT_TEMPLATE = """You are an expert at analyzing task prompts to identify critical information segments.

A **segment** is an atomic piece of information that, if removed, would create outcome-critical ambiguity — different interpretations lead to different task outcomes.

## Taxonomy

{taxonomy}

## Task Prompt

{prompt}
{additional_info_section}
## Instructions

Extract segments that are specific, actionable, and match the taxonomy.

**Each segment MUST:**
- Fit one of the 4 dimensions: GOAL, CONSTRAINT, INPUT, or CONTEXT
- Be removal-critical: if removed, the agent would fail or produce wrong output
- Be specific: skip generic verbs ("analyze", "process") or standard terms

**Segment fields:**
- `text`: Short verbatim span with minimal context around the key info
- `value`: The ATOMIC piece of info (filename, number, rule) — must be exact substring, keep SHORT
- `dimension` / `subdimension`: From taxonomy

**Rules:**
- Both `text` and `value` must be EXACT substrings from the prompt
- Keep `value` as SHORT as possible (prefer "report.csv" over "a report called report.csv")
- NO SUBSTRINGS: If "csv" and "report.csv" both appear, keep only "report.csv"
- NO REDUNDANT: One segment per distinct fact

## Scoring (use ONLY 0.0, 0.5, or 1.0)

criticality — What happens if agent uses WRONG value?
| Score | Outcome | Example |
|-------|---------|---------|
| 1.0 | Task FAILS | "report.xlsx" → agent creates "output.csv" (wrong file) |
| 0.5 | Task PARTIAL | "round up hours" → agent rounds down (degraded result) |
| 0.0 | Task OK | "use blue theme" → cosmetic, doesn't affect correctness |

guessability — If removed, can agent recover correct value?
| Score | Recovery | Example |
|-------|----------|---------|
| 0.0 | Impossible | Output filename "april-payroll.xlsx" — must be told |
| 0.5 | Discoverable | Input file "data.csv" — could run `ls` to find it |
| 1.0 | Obvious | "CSV delimiter" — universal default, or stated elsewhere |

Grounding (if trajectory or checkpoints are available):
- `is_used_in_trajectory`: bool/null, is the segment used in the trajectory?
- `first_use_pct`: if used, what is the percentage of the trajectory where it is used (trajectory step / total steps, 0.0=start, 1.0=end, null if not used)
- `checkpoint_refs`: list of checkpoint references (e.g., ["CP1", "CP3"]) or empty list if not used in any checkpoints.

## Output

Return JSON with numeric scores (not strings).
```json
{{
  "segments": [
    {{
      "text": "<verbatim span with context>",
      "value": "<shortest verbatim key info>",
      "dimension": "input|goal|constraint|context",
      "subdimension": "<from taxonomy>",
      "is_used_in_trajectory": true,
      "first_use_pct": 0.15,
      "checkpoint_refs": ["CP1"],
      "criticality": 1.0,
      "guessability": 0.5
    }}
  ]
}}
```"""


class SegmentExtractor:
    """Extract segments from any UnifiedTask."""

    def __init__(self, model: str = "openai/gpt-4o", taxonomy: Optional[Dict] = None):
        self.model = model
        # Load taxonomy from YAML if not provided
        self._taxonomy = taxonomy or load_taxonomy()
        self._taxonomy_str = format_taxonomy_for_prompt(self._taxonomy)

    def extract(
        self,
        task: UnifiedTask,
        trajectory_summary: Optional[str] = None,
        checkpoints: Optional[List[str]] = None,
    ) -> List[Segment]:
        """
        Extract segments from a task prompt.

        Args:
            task: The task to extract segments from
            trajectory_summary: Optional summary of golden trajectory (files/paths used)
            checkpoints: Optional list of evaluation checkpoints
        """
        # Build prompt with any known_info context
        prompt_text = task.prompt
        if task.known_info:
            prompt_text += f"\n\nAdditional context:\n{task.known_info}"

        # Build additional info section (trajectory + checkpoints)
        additional_parts = []

        if trajectory_summary:
            additional_parts.append(
                f"""### Golden Trajectory
The "golden trace" - a successful agent's execution. Analyze this to determine:
- `is_used_in_trajectory`: Was the segment's value used? (true/false)
- `first_use_pct`: When first used? (0.0 = start, 1.0 = end)

{trajectory_summary}"""
            )

        if checkpoints:
            checkpoints_text = "\n".join(f"- CP{i+1}: {cp}" for i, cp in enumerate(checkpoints))
            additional_parts.append(
                f"""### Checkpoints
Success criteria for task completion. Include in `checkpoint_refs` if removing a segment would cause failure.

{checkpoints_text}"""
            )

        additional_info_section = ""
        if additional_parts:
            additional_info_section = (
                "\n## Additional Task Information\n\n" + "\n\n".join(additional_parts) + "\n"
            )

        user_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
            taxonomy=self._taxonomy_str,
            prompt=prompt_text,
            additional_info_section=additional_info_section,
        )

        response = llm_completion(
            model=self.model,
            messages=[{"role": "user", "content": user_prompt}],
            response_format={"type": "json_schema", "json_schema": SEGMENT_EXTRACTION_SCHEMA},
        )

        # Parse response (schema ensures structure)
        try:
            data = json.loads(response)
            segments_data = data.get("segments", [])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response[:500]}...")
            return []

        segments = []
        for i, s in enumerate(segments_data, 1):
            dim_str = s.get("dimension", "input").lower()
            try:
                dimension = Dimension(dim_str)
            except ValueError:
                dimension = Dimension.INPUT

            # Auto-generate ID
            segment_id = f"S{i}"

            # Parse grounding fields from LLM
            checkpoint_refs = s.get("checkpoint_refs", [])
            if not isinstance(checkpoint_refs, list):
                checkpoint_refs = []

            # Grounding fields (optional)
            is_used_in_trajectory = s.get("is_used_in_trajectory")
            first_use_pct = s.get("first_use_pct")

            # Parse scores
            criticality = float(s.get("criticality", 0.5))
            guessability = float(s.get("guessability", 0.5))

            value = s.get("value", "")

            metadata = {}

            segments.append(
                Segment(
                    # Identity (pipeline assigns id)
                    id=segment_id,
                    dimension=dimension,
                    subdimension=s.get("subdimension", ""),
                    value=value,
                    text=s.get("text", ""),
                    # Grounding (from LLM, optional if no trajectory/checkpoints)
                    checkpoint_refs=checkpoint_refs,
                    is_used_in_trajectory=is_used_in_trajectory,
                    first_use_pct=first_use_pct,
                    # Scores (LLM self-assesses)
                    criticality=criticality,
                    guessability=guessability,
                    # Quick metric to filter data with single value, balancing outcome-critical and guessability (search space)
                    priority_score=criticality * (1 - guessability),
                    # Dataset-specific metadata
                    metadata=metadata,
                )
            )

        # Post-process: filter segments
        original_prompt = task.prompt

        # 1. Filter segments whose value is not in the original prompt
        segments = [s for s in segments if s.value and s.value in original_prompt]

        # 2. Deduplicate
        segments = self._deduplicate_segments(segments)

        # Re-assign IDs after filtering
        for i, seg in enumerate(segments, 1):
            seg.id = f"S{i}"

        return segments

    def _deduplicate_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Remove segments where one value is contained in another (keep longer).

        Only deduplicates within the same dimension to preserve semantically
        distinct segments (e.g., input path vs output path that share a substring).
        """
        if not segments:
            return segments

        # Normalize for comparison
        def norm(s: str) -> str:
            return s.lower().strip("/`' ")

        # Sort by value length descending (longer values first)
        sorted_segs = sorted(segments, key=lambda s: len(s.value), reverse=True)

        kept = []
        # Track kept values per dimension to only dedupe within same dimension
        kept_vals_by_dim: dict[str, list[str]] = {}

        for seg in sorted_segs:
            val = norm(seg.value)
            if not val:
                continue

            dim = seg.dimension.value
            kept_vals = kept_vals_by_dim.get(dim, [])

            # Skip if this value is a substring of (or contains) a kept value
            # in the SAME dimension
            if any(val in kv or kv in val for kv in kept_vals):
                continue

            kept.append(seg)
            if dim not in kept_vals_by_dim:
                kept_vals_by_dim[dim] = []
            kept_vals_by_dim[dim].append(val)

        return kept


# =============================================================================
# VARIANT GENERATOR (Unified)
# =============================================================================

VARIANT_PROMPT = """You are creating an underspecified version of a task prompt.

The goal is to create **outcome-critical ambiguity**: the agent could reasonably interpret the underspecified prompt in multiple ways that lead to different task outcomes.

## Original Prompt
```
{prompt}
```

## Segment to Remove
- Dimension: {dimension}
- Value: "{value}"
- Text: "{text}"
{grounding_context}
## Removal Strategy
{severity_instruction}

## Instructions

Create an underspecified version of the prompt:
1. Remove or obscure the listed segment - including ALL occurrences of its value
2. The removed information should be genuinely missing — an agent could plausibly assume a WRONG value resulting in task failure or a different outcome
3. Keep the result grammatically correct and natural-sounding
4. **Preserve exact formatting**: Keep all line breaks, whitespace, indentation, bullet points, and markdown exactly as in the original. Only change the minimal text needed to remove the segment value.

Output JSON:
- `underspecified_prompt`: The modified prompt (preserve original formatting exactly)
- `expected_questions`: [{{"segment_id": "S1", "questions": ["What file?", "Which version?"]}}]
- `failure_mode`: What goes wrong if agent assumes wrong value"""

# Severity-specific instructions (optional, for controlled experiments)
SEVERITY_INSTRUCTIONS = {
    Severity.DELETE: "Remove this information entirely, adjusting sentence structure to remain natural.",
    Severity.VAGUIFY: "Replace with vague language (e.g., 'the file', 'the output').",
    Severity.GENERICIZE: "Replace with generic reference (e.g., 'appropriate format', 'suitable location').",
}

# Default: natural underspecification (mimics real user prompts)
DEFAULT_SEVERITY_INSTRUCTION = """Remove or obscure this information in a way that mimics how a real user might underspecify a task. 
Users often omit details they think are obvious or don't realize are needed. Make it sound natural."""

# Multi-segment removal prompt (for 2-level variants)
MULTI_SEGMENT_VARIANT_PROMPT = """You are creating an underspecified version of a task prompt.

The goal is to create **outcome-critical ambiguity**: the agent could reasonably interpret the underspecified prompt in multiple ways that lead to different task outcomes.

## Original Prompt
```
{prompt}
```

## Segments to Remove (remove ALL of these)
{segments_list}

## Removal Strategy
{severity_instruction}

## Instructions

Create an underspecified version of the prompt:

1. Remove or obscure ALL listed segments — including ALL occurrences of each value
2. The removed information should be genuinely missing — an agent could plausibly assume WRONG values resulting in task failure or a different outcome
3. Keep the result grammatically correct and natural-sounding
4. **Preserve exact formatting**: Keep all line breaks, whitespace, indentation, bullet points, and markdown exactly as in the original. Only change the minimal text needed to remove the segment values.

Output JSON:
- `underspecified_prompt`: The modified prompt (preserve original formatting exactly)
- `expected_questions`: [{{"segment_id": "S1", "questions": ["What file?"]}}, {{"segment_id": "S2", "questions": ["What format?"]}}]
- `failure_mode`: What goes wrong if agent assumes wrong values"""


class VariantGenerator:
    """Generate underspecified variants from segments."""

    def __init__(self, model: str = "openai/gpt-4o"):
        self.model = model

    def generate(
        self,
        task: UnifiedTask,
        segments: List[Segment],
        severity: Severity = Severity.DELETE,
        max_level: int = 1,
        max_variants_per_level: Optional[int] = None,
        selection_strategy: str = "auto",
    ) -> List[UnderspecVariant]:
        """
        Generate underspecified variants by removing segment(s).

        Args:
            task: The original task
            segments: Segments to consider for removal
            severity: Removal strategy
            max_level: Max segments to remove together (1=single, 2=pairs, etc.)
            max_variants_per_level: Limit variants per level (by priority score)
            selection_strategy: Within-level combo ranking (cross-level uses
                predicted_difficulty = mean priority, bounded [0,1]):
                - "auto": Grounding-weighted — trajectory, checkpoints, then
                  mean priority as tiebreaker (RECOMMENDED)
                - "priority_mean": Mean of segment priority scores
                - "priority_max": Max segment priority score (legacy)

        Returns:
            List of UnderspecVariant (one per segment/combination)
        """

        if not segments:
            return []

        # Sort by priority score descending
        sorted_segments = sorted(segments, key=lambda s: s.priority_score, reverse=True)

        # Cap max_level at number of available segments (can't remove more than we have)
        effective_max_level = min(max_level, len(sorted_segments))

        all_variants = []

        for level in range(1, effective_max_level + 1):
            combos = list(combinations(sorted_segments, level))

            # Score each combo based on strategy
            scored_combos = self._score_combos(combos, selection_strategy)
            scored_combos.sort(key=lambda x: x[1], reverse=True)

            # Limit per level if specified
            if max_variants_per_level:
                scored_combos = scored_combos[:max_variants_per_level]

            for combo, _ in scored_combos:
                variant = self._generate_single_variant(task, list(combo), severity)
                if variant is not None:
                    all_variants.append(variant)

        return all_variants

    def _score_combos(self, combos: List[tuple], strategy: str) -> List[tuple]:
        """Rank segment combinations within a single level for generation order.

        Only compares combos of the SAME segment count (within one level of
        the generate loop).  Cross-level ranking is handled separately by
        predicted_difficulty (mean of priority scores, bounded [0, 1]).

        Strategies:
            - auto: Grounding-weighted — trajectory use, then checkpoint refs,
                    then mean priority as tiebreaker (RECOMMENDED).
            - priority_mean: Mean of segment priority scores.
            - priority_max:  Max segment priority score (legacy).
        """
        scored = []
        for combo in combos:
            if strategy == "auto":
                traj_count = sum(1 for s in combo if s.is_used_in_trajectory)
                ckpt_count = sum(1 for s in combo if s.checkpoint_refs)
                mean_priority = sum(s.priority_score for s in combo) / len(combo)
                score = traj_count * 1000 + ckpt_count * 100 + mean_priority
            elif strategy == "priority_mean":
                score = sum(s.priority_score for s in combo) / len(combo)
            else:
                # priority_max (legacy)
                score = max(s.priority_score for s in combo)
            scored.append((combo, score))
        return scored

    def _format_segment(self, seg: Segment, index: Optional[int] = None) -> str:
        """Format a segment for the removal prompt.

        Args:
            seg: The segment to format
            index: Optional 1-based index for multi-segment lists

        Returns:
            Formatted string like: "S1 [INPUT] "filename.csv" (used at 15%, affects CP1)"
        """
        grounding = []
        if seg.is_used_in_trajectory:
            grounding.append(
                f"used at {seg.first_use_pct:.0%}"
                if seg.first_use_pct is not None
                else "used in trajectory"
            )
        if seg.checkpoint_refs:
            grounding.append(f"affects {', '.join(seg.checkpoint_refs)}")
        grounding_str = f" ({', '.join(grounding)})" if grounding else ""

        # Use segment ID (e.g., "S1") as prefix
        seg_id = seg.id if seg.id else f"S{index}" if index else "S1"
        return f'{seg_id} [{seg.dimension.value.upper()}] "{seg.value}"{grounding_str}'

    def _generate_single_variant(
        self,
        task: UnifiedTask,
        segments: List[Segment],
        severity: Severity = Severity.DELETE,
    ) -> Optional[UnderspecVariant]:
        """Generate one variant removing the given segment(s). Returns None on parse failure."""

        severity_instruction = SEVERITY_INSTRUCTIONS.get(severity, DEFAULT_SEVERITY_INSTRUCTION)

        if len(segments) == 1:
            # Single segment removal
            seg = segments[0]
            grounding_context = self._format_segment(seg) + "\n"
            user_prompt = VARIANT_PROMPT.format(
                prompt=task.prompt,
                dimension=seg.dimension.value.upper(),
                value=seg.value,
                text=seg.text,
                grounding_context=grounding_context,
                severity_instruction=severity_instruction,
            )
        else:
            # Multi-segment removal
            segments_list = "\n".join(
                self._format_segment(seg, i) for i, seg in enumerate(segments, 1)
            )
            user_prompt = MULTI_SEGMENT_VARIANT_PROMPT.format(
                prompt=task.prompt,
                segments_list=segments_list,
                severity_instruction=severity_instruction,
            )

        response = llm_completion(
            model=self.model,
            messages=[{"role": "user", "content": user_prompt}],
            response_format={"type": "json_schema", "json_schema": VARIANT_GENERATION_SCHEMA},
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            seg_ids = "_".join(seg.id for seg in segments)
            logger.warning(f"Failed to parse variant JSON for {seg_ids}, skipping")
            logger.debug(f"Response was: {response[:500]}...")
            return None

        # Mean priority: segment-count-neutral so neither 1-seg nor 2-seg
        # is systematically favored; quality per removed segment decides.
        predicted_difficulty = sum(s.priority_score for s in segments) / len(segments)
        segment_ids = "_".join(seg.id for seg in segments)

        underspec_prompt = data["underspecified_prompt"]
        underspec_prompt = clean_prompt(underspec_prompt, original_prompt=task.prompt)

        output_severity = severity
        return UnderspecVariant(
            id=f"V_{segment_ids}_{output_severity.value}",
            original_prompt=task.prompt,
            underspecified_prompt=underspec_prompt,
            removed_segments=segments,
            severity=output_severity,
            expected_failure_mode=data.get("failure_mode", ""),
            expected_questions=data.get("expected_questions", []),
            predicted_difficulty=predicted_difficulty,
        )


# =============================================================================
# MAIN PIPELINE
# =============================================================================


class SyntheticPipeline:
    """
    Unified synthetic data pipeline for underspecification.

    Two-phase architecture:
        Phase 1: Segment Extraction
            - Extract segments with criticality, guessability
            - Ground to trajectory/checkpoints (optional)

        Phase 2: Variant Generation
            - Create underspecified prompts
            - Rank by priority_score

    Filtering is done DOWNSTREAM via pass@k experiments, not by this pipeline.
    Works across TAC, tau2-bench, MCP Atlas, and future domains.
    """

    def __init__(
        self,
        extraction_model: str = "gemini/gemini-3-flash-preview",
        generation_model: str = "gemini/gemini-3-flash-preview",
        tac_task_dir: str = "task_pairs_agentcompany",
        swebench_trajectory_dir: str = "experiments/swebench/golden_trajectories",
    ):
        self.extraction_model = extraction_model
        self.generation_model = generation_model
        self._tac_task_dir = tac_task_dir
        self._swebench_trajectory_dir = swebench_trajectory_dir

        # Lazy: adapters only created when domain is accessed
        self._adapters: Dict[str, Any] = {}

        # Initialize processors
        self.extractor = SegmentExtractor(model=extraction_model)
        self.generator = VariantGenerator(model=generation_model)

    def _get_adapter(self, domain: str):
        """Get or create adapter for domain (lazy instantiation)."""
        if domain not in self._adapters:
            if domain == "tac":
                from .adapters.tac import TACAdapter  # noqa: F401

                self._adapters["tac"] = TACAdapter(self._tac_task_dir)
            elif domain == "mcpatlas":
                from .adapters.mcpatlas import MCPAtlasAdapter

                self._adapters["mcpatlas"] = MCPAtlasAdapter()
            elif domain == "swebench":
                from .adapters.swebench import SWEBenchProAdapter

                self._adapters["swebench"] = SWEBenchProAdapter(
                    trajectory_dir=self._swebench_trajectory_dir,
                )
            else:
                raise ValueError(f"Unknown domain: {domain}")
        return self._adapters[domain]

    def load_task(
        self,
        domain: str,
        **kwargs,
    ) -> UnifiedTask:
        """
        Load a task from any domain.

        Args:
            domain: "tac" | "tau2" | "mcpatlas"
            **kwargs: Domain-specific arguments
                TAC: task_name, trajectory_path
                tau2: tau2_domain (airline/retail/telecom), task_id
                mcpatlas: task_id
        """
        adapter = self._get_adapter(domain)

        if domain == "tac":
            task_name = kwargs.get("task_name")
            trajectory_path = kwargs.get("trajectory_path")  # Optional override
            return adapter.load_task(task_name, trajectory_path)
        elif domain == "mcpatlas":
            task_id = kwargs.get("task_id")
            return adapter.load_task(task_id)
        elif domain == "swebench":
            instance_id = kwargs.get("instance_id")
            return adapter.load_task(instance_id)

        else:
            raise ValueError(f"Unknown domain: {domain}")

    def load_checkpoints(self, task: UnifiedTask) -> List[str]:
        """
        Load checkpoints from task directory (for PTE level).

        Returns list of checkpoint descriptions.
        """
        if task.checkpoints:
            return task.checkpoints

        # Try to load from checkpoints.md in task directory
        if task.domain == "tac":
            checkpoints_path = Path(self._tac_task_dir) / task.id / "checkpoints.md"
            if checkpoints_path.exists():
                with open(checkpoints_path) as f:
                    content = f.read()
                # Parse markdown checkpoints (lines starting with - or *)
                checkpoints = []
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.")):
                        checkpoints.append(line.lstrip("-* 0123456789.").strip())
                return checkpoints

        return []

    def get_trajectory_summary(self, task: UnifiedTask, max_chars: int = 100000) -> Optional[str]:
        """
        Load trajectory content for LLM to analyze.

        Returns the raw trajectory data (truncated if too long).
        LLM will determine is_used_in_trajectory and first_use_pct from this.
        Returns None if trajectory is not available (graceful degradation).

        Supports three formats:
        - TAC: S3 JSON with flat tabular data (pd.read_json compatible)
        - SWE-bench: .traj files with {trajectory: [{action, observation, ...}, ...]}
        - MCP-Atlas: JSON array of tool call dicts [{role, content, tool_calls}, ...]
        """
        if not task.trajectory_path:
            return None

        traj_path = task.trajectory_path

        try:
            # Try loading as .traj (SWE-bench format: nested dict with trajectory array)
            if str(traj_path).endswith(".traj"):
                with open(traj_path) as f:
                    data = json.load(f)
                steps = data.get("trajectory", [])
                # Format steps as compact summaries for LLM
                lines = []
                for i, step in enumerate(steps):
                    action = str(step.get("action", ""))[:500]
                    observation = str(step.get("observation", ""))[:500]
                    thought = str(step.get("thought", ""))[:300]
                    lines.append(f"Step {i+1}:")
                    if thought:
                        lines.append(f"  Thought: {thought}")
                    lines.append(f"  Action: {action}")
                    if observation:
                        lines.append(f"  Observation: {observation}")
                    lines.append("")
                content = "\n".join(lines)
                total_steps = len(steps)
            else:
                with open(traj_path) as f:
                    data = json.load(f)

                if (
                    isinstance(data, list)
                    and data
                    and isinstance(data[0], dict)
                    and "role" in data[0]
                ):
                    # MCP-Atlas format: list of message dicts with role/content/tool_calls
                    lines = []
                    for i, msg in enumerate(data):
                        role = msg.get("role", "unknown")
                        msg_content = msg.get("content", "")
                        tool_calls = msg.get("tool_calls", [])
                        lines.append(f"Step {i+1} ({role}):")
                        if tool_calls:
                            for tc in tool_calls:
                                fn = tc.get("function", {})
                                lines.append(f"  Tool: {fn.get('name', 'unknown')}")
                                args_str = str(fn.get("arguments", ""))[:300]
                                lines.append(f"  Args: {args_str}")
                        if msg_content:
                            lines.append(f"  Content: {str(msg_content)[:500]}")
                        lines.append("")
                    content = "\n".join(lines)
                    total_steps = len(data)
                else:
                    # TAC format: flat tabular JSON
                    df = pd.read_json(traj_path)
                    content = df.to_json(orient="records", indent=2, date_format="iso")
                    total_steps = len(df)
        except Exception as e:
            logger.warning(
                "Trajectory unavailable for %s (%s); proceeding without trajectory grounding",
                task.id,
                e,
            )
            return None

        # Truncate if too long (keep beginning and end for context)
        if len(content) > max_chars:
            half = max_chars // 2 - 50
            content = (
                content[:half]
                + f"\n\n... [{len(content) - max_chars} chars truncated] ...\n\n"
                + content[-half:]
            )

        return f"Total steps: {total_steps}\n\n{content}"

    def process(
        self,
        task: UnifiedTask,
        severity: Severity = Severity.DELETE,
        max_segments: Optional[int] = None,
        max_level: int = 1,
        max_variants_per_level: Optional[int] = None,
        selection_strategy: str = "auto",
    ) -> PipelineResult:
        """
        Run the synthetic data pipeline on a task.

        Phase 1: Segment Extraction
            - Extract segments with criticality, guessability, priority_score
            - Ground to trajectory/checkpoints if available

        Phase 2: Variant Generation
            - Generate underspecified prompts for each segment (1-level)
            - Optionally generate multi-segment variants (2-level, etc.)
            - Rank by priority score (criticality × (1 - guessability))

        Filtering: Done DOWNSTREAM via pass@k experiments, NOT by this pipeline.
        Output contains all metadata needed for ablation (severity, scores, grounding).

        Args:
            task: The task to process
            severity: Removal strategy: DELETE (high) | VAGUIFY (medium) | GENERICIZE (low)
            max_segments: Limit to top K segments (by priority score). None = all segments.
            max_level: Max segments to remove together (1=single, 2=pairs, etc.). Default 1.
            max_variants_per_level: Limit variants per level when max_level > 1.
            selection_strategy: Within-level combo ranking (cross-level uses
                predicted_difficulty = mean priority, bounded [0,1]):
                - "auto": Grounding-weighted — trajectory, checkpoints, then
                  mean priority as tiebreaker (RECOMMENDED)
                - "priority_mean": Mean of segment priority scores
                - "priority_max": Max segment priority score (legacy)

        Returns:
            PipelineResult with segments and ranked variants (all metadata for ablation)
        """
        # =====================================================================
        # PHASE 1: SEGMENT EXTRACTION
        # =====================================================================

        # 1a. Extract segments with grounding context
        # LLM sees trajectory/checkpoints and outputs grounding + scores
        trajectory_summary = self.get_trajectory_summary(task)
        checkpoints = self.load_checkpoints(task)

        segments = self.extractor.extract(
            task,
            trajectory_summary=trajectory_summary,
            checkpoints=checkpoints,
        )

        # 1b. Rank by priority (highest first)
        # priority_score already computed inline during extraction
        segments.sort(key=lambda s: s.priority_score, reverse=True)

        if max_segments and len(segments) > max_segments:
            segments = segments[:max_segments]

        # =====================================================================
        # PHASE 2: VARIANT GENERATION
        # =====================================================================

        variants = self.generator.generate(
            task,
            segments,
            severity,
            max_level=max_level,
            max_variants_per_level=max_variants_per_level,
            selection_strategy=selection_strategy,
        )

        # Sort variants by priority (highest first)
        variants.sort(key=lambda v: v.predicted_difficulty, reverse=True)

        return PipelineResult(
            task=task,
            segments=segments,
            variants=variants,
            extraction_model=self.extraction_model,
            generation_model=self.generation_model,
        )

    def process_tac_task(
        self,
        task_name: str,
        trajectory_path: Optional[str] = None,
        use_checkpoints: bool = False,
        severity: Severity = Severity.DELETE,
        max_segments: Optional[int] = None,
        max_level: int = 1,
        max_variants_per_level: Optional[int] = None,
        selection_strategy: str = "auto",
    ) -> PipelineResult:
        """
        Convenience method for TAC tasks.

        For ablation studies, control grounding information explicitly:
            - Prompt only (default): trajectory_path=None, use_checkpoints=False
            - Prompt + trajectory:   trajectory_path=<url>, use_checkpoints=False
            - Prompt + checkpoints:  trajectory_path=None, use_checkpoints=True
            - Full grounding:        trajectory_path=<url>, use_checkpoints=True

        Args:
            task_name: Name of the TAC task
            trajectory_path: S3 path to golden trajectory, or None to skip.
                           Use TACAdapter.get_trajectory_url(task_name) to get the URL.
            use_checkpoints: If True, load checkpoints from checkpoints.md. Default False.
            severity: DELETE (high) | VAGUIFY (medium) | GENERICIZE (low)
            max_segments: Limit total segments by priority score. None = all.
            max_level: Max segments to remove together (1=single, 2=pairs). Default 1.
            max_variants_per_level: Limit variants per level when max_level > 1.
            selection_strategy: Within-level combo ranking (cross-level uses
                predicted_difficulty = mean priority, bounded [0,1]):
                - "auto": Grounding-weighted — trajectory, checkpoints, then
                  mean priority as tiebreaker (RECOMMENDED)
                - "priority_mean": Mean of segment priority scores
                - "priority_max": Max segment priority score (legacy)

        Returns:
            PipelineResult with all variants + metadata for downstream ablation/filtering
        """
        task = self.load_task("tac", task_name=task_name, trajectory_path=trajectory_path)

        # Clear checkpoints if not using them (for ablation)
        if not use_checkpoints:
            task = UnifiedTask(
                id=task.id,
                domain=task.domain,
                category=task.category,
                prompt=task.prompt,
                trajectory_path=task.trajectory_path,
                checkpoints=[],  # Clear checkpoints
                known_info=task.known_info,
            )

        return self.process(
            task, severity, max_segments, max_level, max_variants_per_level, selection_strategy
        )

    def process_swebench_task(
        self,
        instance_id: str,
        severity: Severity = Severity.DELETE,
        top_k: Optional[int] = None,
        max_level: int = 1,
        top_k_per_level: Optional[int] = None,
    ) -> PipelineResult:
        """Convenience method for SWE-bench Pro tasks.

        Args:
            instance_id: SWE-bench Pro instance ID (e.g., "django__django-12345")
            severity: DELETE | VAGUIFY | GENERICIZE | None
            top_k: Limit to top K segments by priority score. None = all.
            max_level: Max segments to remove together (1=single, 2=pairs). Default 1.
            top_k_per_level: Limit combinations per level when max_level > 1.

        Returns:
            PipelineResult with all variants + metadata
        """
        task = self.load_task("swebench", instance_id=instance_id)
        return self.process(
            task,
            severity,
            top_k,
            max_level,
            top_k_per_level,
        )

    def process_mcpatlas_task(
        self,
        task_id: str,
        severity: Severity = Severity.DELETE,
        top_k: Optional[int] = None,
        max_level: int = 1,
        top_k_per_level: Optional[int] = None,
    ) -> PipelineResult:
        """
        Convenience method for MCP Atlas tasks.

        Args:
            task_id: MCP Atlas task ID (hex string like "68a398aa2f58036e8d45eddf")
            severity: DELETE (high) | VAGUIFY (medium) | GENERICIZE (low) | None
            top_k: Limit to top K segments by priority score. None = all.
            max_level: Max segments to remove together (1=single, 2=pairs). Default 1.
            top_k_per_level: Limit combinations per level when max_level > 1.

        Returns:
            PipelineResult with all variants + metadata for downstream ablation/filtering
        """
        adapter = self._get_adapter("mcpatlas")
        task = adapter.load_task(task_id)
        return self.process(task, severity, top_k, max_level, top_k_per_level)

    @staticmethod
    def uniform_sample_variants(
        results: List[PipelineResult],
        target_count: int = 200,
        min_per_task: int = 2,
    ) -> List[Tuple[str, UnderspecVariant]]:
        """Select dimension-diverse subset of variants from pipeline results.

        Selection priorities (in order):
        1. Per-task coverage: every task gets min_per_task variants
        2. Dimension diversity: spread picks across dimensions within each task
        3. Priority score: only used as a final tiebreaker

        The classification trials decide which variants stay (200→100 filtering),
        so this step maximizes coverage rather than picking "best" variants.

        Algorithm:
        - For each task, pick min_per_task variants:
          - 1st pick: 1-seg from the rarest global dimension available for this task
          - 2nd pick: 2-seg from a different dimension than the 1st pick
          - Ties broken by priority_score
        - If budget remains after per-task picks, fill with dimension-balanced sampling

        Args:
            results: List of PipelineResult from process_swebench_task()
            target_count: Target number of variants to select (default 200)
            min_per_task: Minimum variants per task (default 2)

        Returns:
            List of (task_id, UnderspecVariant) tuples
        """
        DIMENSIONS = ["context", "goal", "constraint", "input"]

        # ── Collect all variants per task, split by level ──
        # task_id -> {1: [(variant, dim_key, score)], 2: [(variant, dim_key, score)]}
        task_pool: Dict[str, Dict[int, List[Tuple[UnderspecVariant, str, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Global dimension counts (to know which dimensions are rare)
        global_dim_count: Dict[str, int] = defaultdict(int)

        for pr in results:
            task_id = pr.task.id
            for v in pr.variants:
                n_segs = len(v.removed_segments)
                if n_segs == 1:
                    dim = v.removed_segments[0].dimension.value
                    score = v.removed_segments[0].priority_score
                    task_pool[task_id][1].append((v, dim, score))
                    global_dim_count[dim] += 1
                elif n_segs == 2:
                    dims = sorted(s.dimension.value for s in v.removed_segments)
                    dim_key = f"{dims[0]}+{dims[1]}"
                    combined = sum(s.priority_score for s in v.removed_segments) / len(
                        v.removed_segments
                    )
                    task_pool[task_id][2].append((v, dim_key, combined))
                    for d in dims:
                        global_dim_count[d] += 1

        if not task_pool:
            return []

        # Rarest-first dimension ordering (for diversity: pick rare dims first)
        dim_order = sorted(DIMENSIONS, key=lambda d: global_dim_count.get(d, 0))

        # ── Per-task picks: 1 best 1-seg + 1 best 2-seg, dimension-diverse ──
        selected: List[Tuple[str, UnderspecVariant]] = []
        selected_ids: set = set()

        for task_id, pool in task_pool.items():
            task_picked: List[Tuple[str, UnderspecVariant]] = []
            used_dims: set = set()

            # Pick 1: best 1-seg, preferring the rarest available dimension
            seg1 = pool.get(1, [])
            if seg1:
                # Group by dimension, pick from rarest-first
                by_dim: Dict[str, List[Tuple[UnderspecVariant, float]]] = defaultdict(list)
                for v, dim, score in seg1:
                    by_dim[dim].append((v, score))
                for d in dim_order:
                    if d in by_dim:
                        # Tiebreak by priority_score within this dimension
                        best_v, _ = max(by_dim[d], key=lambda x: x[1])
                        task_picked.append((task_id, best_v))
                        selected_ids.add((task_id, best_v.id))
                        used_dims.add(d)
                        break

            # Pick 2: best 2-seg, preferring a pair that covers a new dimension
            if len(task_picked) < min_per_task:
                seg2 = pool.get(2, [])
                if seg2:
                    # Try pairs that introduce a dimension not yet used
                    best_new = None  # (variant, score) for pair with new dim
                    best_any = None  # (variant, score) fallback
                    for v, dim_key, score in seg2:
                        pair_dims = set(dim_key.split("+"))
                        if best_any is None or score > best_any[1]:
                            best_any = (v, score)
                        if pair_dims - used_dims:
                            if best_new is None or score > best_new[1]:
                                best_new = (v, score)
                    pick = best_new or best_any
                    if pick:
                        task_picked.append((task_id, pick[0]))
                        selected_ids.add((task_id, pick[0].id))

            # Backfill: if still under min (e.g. task only has 1-seg or only 2-seg)
            if len(task_picked) < min_per_task:
                all_variants = [(v, s) for v, _, s in seg1] + [
                    (v, s) for v, _, s in pool.get(2, [])
                ]
                all_variants.sort(key=lambda x: -x[1])
                for v, _ in all_variants:
                    if len(task_picked) >= min_per_task:
                        break
                    if (task_id, v.id) not in selected_ids:
                        task_picked.append((task_id, v))
                        selected_ids.add((task_id, v.id))

            selected.extend(task_picked)

        # ── Fill remaining budget (if target > n_tasks * min_per_task) ──
        remaining = target_count - len(selected)
        if remaining > 0:
            backfill: List[Tuple[str, UnderspecVariant, float]] = []
            for task_id, pool in task_pool.items():
                for level in [1, 2]:
                    for v, dim_key, score in pool.get(level, []):
                        if (task_id, v.id) not in selected_ids:
                            backfill.append((task_id, v, score))
            # Sort by priority_score as final tiebreaker
            backfill.sort(key=lambda x: -x[2])
            for task_id, v, _ in backfill[:remaining]:
                selected.append((task_id, v))
                selected_ids.add((task_id, v.id))

        return selected
