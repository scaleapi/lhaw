"""Generate underspecified variants for tasks."""

import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

# LLM_API_KEY is canonical; LITELLM_API_KEY accepted as legacy alias
api_key = os.getenv("LLM_API_KEY") or os.getenv("LITELLM_API_KEY") or os.getenv("OPENAI_API_KEY")
if api_key:
    os.environ["LLM_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key
base_url = os.getenv("LLM_BASE_URL") or os.getenv("LITELLM_BASE_URL")
if base_url:
    os.environ["OPENAI_BASE_URL"] = base_url

from synthetic.adapters.mcpatlas import MCPAtlasAdapter  # noqa: E402
from synthetic.pipeline import SyntheticPipeline  # noqa: E402

from synthetic.core import Severity  # noqa: E402

logger = logging.getLogger(__name__)


def generate_underspec_variants(task_ids: List[str], config: dict, output_dir: Path) -> dict:
    """
    Generate underspecified variants for the given tasks.

    Args:
        task_ids: List of task IDs to generate variants for
        config: Configuration dict with keys:
            - severity: "DELETE", "VAGUIFY", or "GENERICIZE"
            - top_k: Optional[int] - limit segments
            - variant_top_k: int - variants per task
            - max_level: int - max segments to remove together
            - extraction_model: str - model for extraction
            - generation_model: str - model for generation
        output_dir: Directory to write output files

    Returns:
        Dict containing:
            - tasks_processed: Number of successfully processed tasks
            - tasks_failed: Number of failed tasks
            - total_rows: Number of rows in output CSVs
            - csv_files: Dict of output file paths
            - failed_tasks: List of failed task info

    Outputs:
        - {output_dir}/underspec_prompts.csv
        - {output_dir}/original_prompts.csv
        - {output_dir}/json/{task_id}.json (one per task)
        - {output_dir}/summary.json
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir = output_dir / "json"
    json_output_dir.mkdir(exist_ok=True)

    # Initialize pipeline and adapter
    pipeline = SyntheticPipeline(
        extraction_model=config["extraction_model"],
        generation_model=config["generation_model"],
    )
    adapter = MCPAtlasAdapter(task_ids=task_ids)

    # Parse severity
    severity = Severity[config["severity"]]

    # Process tasks
    results = []
    failed_tasks = []

    for i, task_id in enumerate(task_ids):
        logger.info(f"[{i+1}/{len(task_ids)}] Processing task {task_id}")

        try:
            # Run pipeline
            result = pipeline.process_mcpatlas_task(
                task_id,
                severity=severity,
                top_k=config.get("top_k"),
                max_level=config.get("max_level", 1),
            )

            # Get raw row for original CSV columns
            raw_row = adapter.get_raw_row(task_id)

            # Save full JSON output
            json_path = json_output_dir / f"{task_id}.json"
            with open(json_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            # Filter variants by min_level (number of segments removed)
            min_level = config.get("min_level", 1)
            filtered_variants = [v for v in result.variants if len(v.removed_segments) >= min_level]

            # Select top variants based on variant_top_k
            top_variants = filtered_variants[: config.get("variant_top_k", 1)]

            # Skip if no variants meet min_level requirement
            if not top_variants:
                logger.warning(
                    f"  ⚠ No variants with >={min_level} segments removed "
                    f"(generated {len(result.variants)} total variants)"
                )
                continue

            # Create one row per variant
            for variant in top_variants:
                original_prompt = raw_row.get("PROMPT", "")
                underspec_prompt = variant.underspecified_prompt

                removed_vals_json = json.dumps([s.value for s in variant.removed_segments])
                row = {
                    # Original MCP Atlas columns
                    "TASK": task_id,
                    "PROMPT": original_prompt,  # Will be modified when writing CSVs
                    "original_prompt": original_prompt,
                    "underspecified_prompt": underspec_prompt,
                    "ENABLED_TOOLS": raw_row.get("ENABLED_TOOLS", ""),
                    "GTFA_CLAIMS": raw_row.get("GTFA_CLAIMS", ""),
                    "TRAJECTORY": raw_row.get("TRAJECTORY", ""),
                    # Variant metadata
                    "variant_id": variant.id,
                    "removed_segment_ids": ",".join(s.id for s in variant.removed_segments),
                    "removed_value": removed_vals_json,
                    "severity": variant.severity.value,
                    "expected_questions": json.dumps(variant.expected_questions),
                    "expected_failure_mode": variant.expected_failure_mode,
                    "predicted_difficulty": variant.predicted_difficulty,
                    # Metadata
                    "num_segments_extracted": len(result.segments),
                    "num_variants_generated": len(result.variants),
                    "category": result.task.category,
                    "json_output_path": str(json_path.relative_to(output_dir)),
                }
                results.append(row)

            logger.info(
                f"  ✓ Extracted {len(result.segments)} segments, "
                f"generated {len(result.variants)} variants, "
                f"filtered to {len(filtered_variants)} (min_level={min_level}), "
                f"kept top {len(top_variants)}"
            )

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            failed_tasks.append({"task_id": task_id, "error": str(e)})

    # Write CSVs
    csv_files = {}

    if results:
        fieldnames = list(results[0].keys())

        # CSV 1: underspec_prompts.csv (PROMPT has underspecified prompt)
        underspec_path = output_dir / "underspec_prompts.csv"
        with open(underspec_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                # Swap PROMPT to underspecified
                modified_row = row.copy()
                modified_row["PROMPT"] = row["underspecified_prompt"]
                writer.writerow(modified_row)

        csv_files["underspec_prompts"] = str(underspec_path)

        # CSV 2: original_prompts.csv (PROMPT has original prompt)
        original_path = output_dir / "original_prompts.csv"
        with open(original_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                # PROMPT already has original, just write as-is
                writer.writerow(row)

        csv_files["original_prompts"] = str(original_path)

    # Write summary
    summary = {
        "config": config,
        "results": {
            "total_tasks": len(task_ids),
            "successful": len(task_ids) - len(failed_tasks),
            "failed": len(failed_tasks),
            "total_rows": len(results),
        },
        "output_files": {
            "underspec_prompts_csv": csv_files.get("underspec_prompts"),
            "original_prompts_csv": csv_files.get("original_prompts"),
            "json_output_dir": str(json_output_dir.relative_to(output_dir)),
        },
        "failed_tasks": failed_tasks,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "tasks_processed": len(task_ids) - len(failed_tasks),
        "tasks_failed": len(failed_tasks),
        "total_rows": len(results),
        "csv_files": csv_files,
        "failed_tasks": failed_tasks,
    }
