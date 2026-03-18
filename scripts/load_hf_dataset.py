"""
Load pre-computed LHAW variants from HuggingFace and write CSVs
compatible with the run scripts, skipping the expensive LLM generation step.

Usage:
    cd lhaw
    source .venv/bin/activate

    # MCP-Atlas (joins with source CSV for ENABLED_TOOLS / GTFA_CLAIMS / TRAJECTORY)
    python scripts/load_hf_dataset.py --dataset MCP-Atlas \
        --output-dir experiments/mcpatlas/underspec_output/hf_variants

    # TAC
    python scripts/load_hf_dataset.py --dataset TheAgentCompany \
        --output-dir experiments/agentcompany/hf_variants

    # SWE-Bench Pro
    python scripts/load_hf_dataset.py --dataset "SWE-Bench Pro" \
        --output-dir experiments/swebench/hf_variants

Then point the run scripts at the generated outputs:
    # MCP-Atlas: pass CSV directly
    --input_csv experiments/mcpatlas/underspec_output/hf_variants/underspec_prompts.csv
    # SWE-Bench: copy instances.yaml + underspec_candidates.csv into experiment directory (see README)
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset

REPO_ROOT = Path(__file__).parent.parent
HF_DATASET_ID = "ScaleAI/lhaw"

MCPATLAS_CSV = REPO_ROOT / "experiments" / "mcpatlas" / "MCP-Atlas.csv"
MCPATLAS_HF_ID = "ScaleAI/MCP-Atlas"


def load_mcpatlas_source() -> pd.DataFrame:
    """Load MCP-Atlas source CSV, downloading from HF if needed."""
    if not MCPATLAS_CSV.exists():
        print(f"MCP-Atlas source CSV not found at {MCPATLAS_CSV}, downloading...")

        MCPATLAS_CSV.parent.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(MCPATLAS_HF_ID, split="train")
        ds.to_csv(str(MCPATLAS_CSV))
        print(f"Downloaded {len(ds)} tasks to {MCPATLAS_CSV}")

    df = pd.read_csv(MCPATLAS_CSV, dtype=str).fillna("")
    return df.set_index("TASK")


def main():
    parser = argparse.ArgumentParser(description="Load LHAW variants from HuggingFace")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["MCP-Atlas", "TheAgentCompany", "SWE-Bench Pro"],
        help="Which dataset subset to export",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for CSVs")
    args = parser.parse_args()

    print(f"Loading LHAW dataset from HuggingFace ({HF_DATASET_ID})...")
    ds = load_dataset(HF_DATASET_ID, split="test")
    variants = [row for row in ds if row["dataset"] == args.dataset]
    print(f"Found {len(variants)} variants for {args.dataset}")

    if not variants:
        print("No variants found. Check --dataset value.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "MCP-Atlas":
        _write_mcpatlas(variants, output_dir)
    elif args.dataset == "TheAgentCompany":
        _write_tac(variants, output_dir)
    elif args.dataset == "SWE-Bench Pro":
        _write_swebench(variants, output_dir)


def _write_mcpatlas(variants: list, output_dir: Path):
    """Write MCP-Atlas CSVs, joining with source CSV for tool/claim/trajectory data."""
    source = load_mcpatlas_source()

    underspec_rows = []
    original_rows = []
    skipped = 0

    for v in variants:
        task_id = v["original_task"]
        if task_id not in source.index:
            skipped += 1
            continue
        src = source.loc[task_id]

        base = {
            "TASK": task_id,
            "original_prompt": v["original_prompt"],
            "underspecified_prompt": v["underspecified_prompt"],
            "ENABLED_TOOLS": src.get("ENABLED_TOOLS", ""),
            "GTFA_CLAIMS": src.get("GTFA_CLAIMS", ""),
            "TRAJECTORY": src.get("TRAJECTORY", ""),
            "variant_id": v["variant_id"],
            "ambiguity_class": v["ambiguity_class"],
            "information_dimension": json.dumps(v["information_dimension"]),
            "removed_segments": json.dumps(v["removed_segments"]),
            "expected_questions": json.dumps(v["expected_questions"]),
        }

        underspec_rows.append({**base, "PROMPT": v["underspecified_prompt"]})
        original_rows.append({**base, "PROMPT": v["original_prompt"]})

    pd.DataFrame(underspec_rows).to_csv(output_dir / "underspec_prompts.csv", index=False)
    pd.DataFrame(original_rows).to_csv(output_dir / "original_prompts.csv", index=False)
    print(f"Wrote {len(underspec_rows)} variants ({skipped} skipped — missing source data)")


def _write_tac(variants: list, output_dir: Path):
    """Write TAC variants as a benchmark JSON.

    Field names match what run_tac_underspec.py --load-samples expects:
      task (not task_name), agent_prompt (not underspecified_prompt),
      ambiguity_type (joined dimension string, not information_dimension list).
    """
    benchmark = []
    for v in variants:
        # information_dimension is a list like ["input", "goal"] → join as "input+goal"
        dims = v["information_dimension"]
        ambiguity_type = "+".join(dims) if isinstance(dims, list) else str(dims)

        benchmark.append(
            {
                "variant_id": v["variant_id"],
                "task": v["original_task"],
                "agent_prompt": v["underspecified_prompt"],
                "original_prompt": v["original_prompt"],
                "ambiguity_class": v["ambiguity_class"],
                "ambiguity_type": ambiguity_type,
                "removed_segments": v["removed_segments"],
                "expected_questions": v["expected_questions"],
            }
        )

    out_path = output_dir / "benchmark.json"
    with open(out_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    print(f"Wrote {len(benchmark)} TAC variants to {out_path}")


def _write_swebench(variants: list, output_dir: Path):
    """Write SWE-Bench Pro variants as instances.yaml + metadata CSV.

    The --run flag of task_completion_swebench.py reads instances.yaml,
    so we must produce that format (not just a CSV). Each YAML entry needs
    image_name, problem_statement, instance_id, base_commit, and repo_name.
    We fetch repo/base_commit from the SWE-bench Pro HF dataset.
    """

    sys.path.insert(0, str(REPO_ROOT))
    from task_completion_swebench import get_dockerhub_image_uri  # noqa: E402

    # Fetch SWE-bench Pro source data for repo + base_commit
    print("Loading SWE-bench Pro source dataset for image URIs...")
    swebench_ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    source = {row["instance_id"]: row for row in swebench_ds}

    # Default DockerHub username (same as run_swebench_example.sh)
    dockerhub_user = "jefzda"

    instances = []
    original_instances = []
    original_ids_seen = set()
    rows = []
    skipped = 0

    for v in variants:
        original_id = v["original_task"]
        src = source.get(original_id)
        if not src:
            skipped += 1
            continue

        repo_name = src.get("repo", "")
        variant_instance_id = v["variant_id"]

        try:
            image_name = get_dockerhub_image_uri(original_id, dockerhub_user, repo_name)
        except ValueError:
            skipped += 1
            continue

        instances.append(
            {
                "image_name": image_name,
                "problem_statement": v["underspecified_prompt"],
                "instance_id": variant_instance_id,
                "base_commit": src.get("base_commit", ""),
                "repo_name": "app",
            }
        )

        # Collect original instances (deduplicated)
        if original_id not in original_ids_seen:
            original_ids_seen.add(original_id)
            original_instances.append(
                {
                    "image_name": image_name,
                    "problem_statement": v["original_prompt"],
                    "instance_id": original_id,
                    "base_commit": src.get("base_commit", ""),
                    "repo_name": "app",
                }
            )

        segments = v["removed_segments"] if isinstance(v["removed_segments"], list) else []
        dims = [s.get("dimension", "") for s in segments if isinstance(s, dict)]
        removed_vals = [s.get("value", "") for s in segments if isinstance(s, dict)]

        rows.append(
            {
                "instance_id": variant_instance_id,
                "original_instance_id": original_id,
                "variant_id": (
                    variant_instance_id.split("__V_", 1)[-1]
                    if "__V_" in variant_instance_id
                    else variant_instance_id
                ),
                "dimension": "+".join(dims),
                "subdimension": "",
                "removed_value": " | ".join(removed_vals),
                "removed_segments_json": json.dumps(segments),
                "num_segments": len(segments),
                "original_prompt": v["original_prompt"],
                "underspecified_prompt": v["underspecified_prompt"],
                "severity": "delete",
                "criticality": "",
                "guessability": "",
                "priority_score": "",
                "predicted_difficulty": "",
                "ambiguity_class": v["ambiguity_class"],
                "information_dimension": json.dumps(v["information_dimension"]),
                "expected_questions": json.dumps(v["expected_questions"]),
            }
        )

    # Write instances.yaml (what --run reads)
    instances_path = output_dir / "instances.yaml"
    with open(instances_path, "w") as f:
        yaml.dump(instances, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Write original_instances.yaml (for baselines)
    original_path = output_dir / "original_instances.yaml"
    with open(original_path, "w") as f:
        yaml.dump(
            original_instances, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    # Write metadata CSV matching the pipeline's underspec_candidates.csv schema
    pd.DataFrame(rows).to_csv(output_dir / "underspec_candidates.csv", index=False)

    print(f"Wrote {len(instances)} variants to {instances_path}")
    print(f"Wrote {len(original_instances)} original tasks to {original_path}")
    print(f"Wrote {len(rows)} rows to underspec_candidates.csv")
    if skipped:
        print(f"({skipped} skipped — missing source data or invalid repo)")


if __name__ == "__main__":
    main()
