"""
Generate underspecified variants for MCP Atlas tasks.

Takes a JSON file with task IDs and generates:
1. underspec_prompts.csv - PROMPT column has underspecified prompts, plus original_prompt and underspecified_prompt columns
2. original_prompts.csv - PROMPT column has original prompts, plus original_prompt and underspecified_prompt columns
3. Full JSON outputs in a subdirectory with all segment/variant details

Usage:
    cd lhaw
    source .venv/bin/activate && source .env

    # Generate for intersection_passed tasks
    python scripts/generate_mcpatlas_underspec.py \
        experiments/mcpatlas/inference_data/aggregated/intersection_passed.json \
        --output-dir experiments/mcpatlas/underspec_output

    # With options
    python scripts/generate_mcpatlas_underspec.py \
        experiments/mcpatlas/inference_data/aggregated/intersection_passed.json \
        --output-dir experiments/mcpatlas/underspec_output \
        --severity DELETE \
        --top-k 5 \
        --variant-top-k 1
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from constants import DEFAULTS  # noqa: E402
from synthetic.adapters.mcpatlas import MCPAtlasAdapter  # noqa: E402
from synthetic.pipeline import SyntheticPipeline  # noqa: E402

from synthetic.core import Severity  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_task_ids(json_path: str) -> List[str]:
    """Load task IDs from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle both formats: {"task_ids": [...]} or just [...]
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "task_ids" in data:
        return data["task_ids"]
    else:
        raise ValueError("Invalid JSON format. Expected list or dict with 'task_ids' key.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate underspecified variants for MCP Atlas tasks"
    )
    parser.add_argument(
        "task_ids_json",
        help="Path to JSON file with task IDs (e.g., intersection_passed.json)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="experiments/mcpatlas/underspec_output",
        help="Output directory for CSV and JSON files",
    )
    parser.add_argument(
        "--severity",
        choices=["DELETE", "VAGUIFY", "GENERICIZE"],
        default=DEFAULTS["severity"].upper(),
        help=f"Removal strategy (default: {DEFAULTS['severity'].upper()})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Limit to top K segments by priority (default: all)",
    )
    parser.add_argument(
        "--variant-top-k",
        type=int,
        default=1,
        help="Number of variants to include per task (default: 1, top priority)",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=DEFAULTS["max_level"],
        help=f"Max segments to remove together (1=single, 2=pairs, default: {DEFAULTS['max_level']})",
    )
    parser.add_argument(
        "--extraction-model",
        default="gemini/gemini-3-flash-preview",
        help="Model for segment extraction",
    )
    parser.add_argument(
        "--generation-model",
        default="gemini/gemini-3-flash-preview",
        help="Model for variant generation",
    )

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir = output_dir / "json"
    json_output_dir.mkdir(parents=True, exist_ok=True)

    # Load task IDs
    task_ids = load_task_ids(args.task_ids_json)
    logger.info(f"Loaded {len(task_ids)} task IDs from {args.task_ids_json}")

    # Initialize pipeline and adapter
    pipeline = SyntheticPipeline(
        extraction_model=args.extraction_model,
        generation_model=args.generation_model,
    )
    adapter = MCPAtlasAdapter(task_ids=task_ids)

    # Parse severity
    severity = Severity[args.severity]

    # Process tasks and collect results
    results = []
    failed_tasks = []

    for i, task_id in enumerate(task_ids):
        logger.info(f"[{i+1}/{len(task_ids)}] Processing task {task_id}")

        try:
            # Run pipeline
            result = pipeline.process_mcpatlas_task(
                task_id,
                severity=severity,
                top_k=args.top_k,
                max_level=args.max_level,
            )

            # Get raw row for original CSV columns
            raw_row = adapter.get_raw_row(task_id)

            # Save full JSON output
            json_path = json_output_dir / f"{task_id}.json"
            with open(json_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)

            # Select top variants based on variant_top_k
            top_variants = result.variants[: args.variant_top_k]

            # Create one row per variant (or one row with top variant)
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
                    "json_output_path": str(json_path),
                }
                results.append(row)

            logger.info(
                f"  ✓ Extracted {len(result.segments)} segments, "
                f"generated {len(result.variants)} variants, "
                f"kept top {len(top_variants)}"
            )

        except Exception as e:
            logger.error(f"  ✗ Failed: {e}")
            failed_tasks.append({"task_id": task_id, "error": str(e)})

    # Write two CSVs
    underspec_path = output_dir / "underspec_prompts.csv"
    original_path = output_dir / "original_prompts.csv"
    if results:
        fieldnames = list(results[0].keys())

        # CSV 1: underspec_prompts.csv (PROMPT has underspecified prompt)
        with open(underspec_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                # Swap PROMPT to underspecified
                modified_row = row.copy()
                modified_row["PROMPT"] = row["underspecified_prompt"]
                writer.writerow(modified_row)

        logger.info(f"✓ Wrote {len(results)} rows to {underspec_path}")

        # CSV 2: original_prompts.csv (PROMPT has original prompt)
        with open(original_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                # PROMPT already has original, just write as-is
                writer.writerow(row)

        logger.info(f"✓ Wrote {len(results)} rows to {original_path}")

    # Write summary
    summary = {
        "input_file": args.task_ids_json,
        "output_dir": str(output_dir),
        "output_files": {
            "underspec_prompts_csv": str(underspec_path.name) if results else None,
            "original_prompts_csv": str(original_path.name) if results else None,
            "json_output_dir": str(json_output_dir.name),
        },
        "settings": {
            "severity": args.severity,
            "top_k": args.top_k,
            "variant_top_k": args.variant_top_k,
            "max_level": args.max_level,
            "extraction_model": args.extraction_model,
            "generation_model": args.generation_model,
        },
        "results": {
            "total_tasks": len(task_ids),
            "successful": len(task_ids) - len(failed_tasks),
            "failed": len(failed_tasks),
            "total_rows": len(results),
        },
        "failed_tasks": failed_tasks,
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✓ Wrote summary to {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Tasks processed: {len(task_ids) - len(failed_tasks)}/{len(task_ids)}")
    print(f"Rows in each CSV: {len(results)}")
    print(f"Output directory: {output_dir}")
    if results:
        print(f"  - Underspec CSV: {underspec_path.name}")
        print(f"  - Original CSV: {original_path.name}")
    print(f"  - JSON files: {json_output_dir.name}/")
    print(f"  - Summary: {summary_path.name}")

    if failed_tasks:
        print(f"\nFailed tasks ({len(failed_tasks)}):")
        for ft in failed_tasks[:5]:
            print(f"  - {ft['task_id']}: {ft['error'][:50]}...")
        if len(failed_tasks) > 5:
            print(f"  ... and {len(failed_tasks) - 5} more")

    print("=" * 60)


if __name__ == "__main__":
    main()
