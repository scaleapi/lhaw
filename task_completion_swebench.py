"""
Task runner for SWE-bench Pro with underspecification support.

Uses the unified SyntheticPipeline for variant generation and
SWE-Agent for execution via generated instance YAML files.

Three action flags (at least one required):
  --generate  Run pipeline + uniform sample -> instances.yaml + underspec_candidates.csv
  --run       Run SWE-Agent trials on generated instances
  --freeze    Lock filtered variants for multi-model comparison

See experiments/swebench/README.md for full CLI reference, output structure,
pipeline stages, and multi-model experiment workflow.
See run_swebench_example.sh for a small reproducible end-to-end example.
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from constants import DEFAULTS, VARIANT_DELIMITER, resolve_model

# Add to path
REPO_ROOT = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))

# Load environment variables from .env
from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

# SWE-bench Pro paths
SWEBENCH_PRO_DIR = REPO_ROOT / "swebenchpro" / "SWE-bench_Pro-os"
SWEAGENT_DIR = SWEBENCH_PRO_DIR / "SWE-agent"

# Default output location (like experiments/agentcompany/runs/)
EXPERIMENTS_DIR = REPO_ROOT / "experiments" / "swebench" / "runs"

# Default trajectory directory for grounded extraction (mirrors TAC's golden_trajectories)
DEFAULT_TRAJECTORY_DIR = REPO_ROOT / "experiments" / "swebench" / "golden_trajectories"

# Ask-user infrastructure (inside vendored SWE-agent fork)
ASK_USER_TOOL_DIR = SWEAGENT_DIR / "tools" / "ask_user"
ASK_USER_CONFIG = SWEAGENT_DIR / "config" / "tool_use_with_ask_user.yaml"


def get_dockerhub_image_uri(uid: str, dockerhub_username: str, repo_name: str = "") -> str:
    """Generate Docker Hub image URI for a SWE-bench Pro instance."""
    if "/" not in repo_name:
        raise ValueError(f"repo_name must be in 'owner/name' format, got: {repo_name!r}")
    repo_base, repo_name_only = repo_name.lower().split("/", 1)
    hsh = uid.replace("instance_", "")

    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]

    return f"{dockerhub_username}/sweap-images:{tag}"


def load_tasks_from_file(tasks_file: Path) -> List[Dict[str, Any]]:
    """
    Load tasks from a JSON file.

    Supports two formats:
    1. {"tasks": [{"task_id": ..., ...}, ...]}  (10-task-test format)
    2. [{"instance_id": ..., ...}, ...]  (direct list format)
    """
    with open(tasks_file) as f:
        data = json.load(f)

    # Handle {"tasks": [...]} format
    if isinstance(data, dict) and "tasks" in data:
        tasks = data["tasks"]
        # Normalize task_id -> instance_id if needed
        for task in tasks:
            if "task_id" in task and "instance_id" not in task:
                task["instance_id"] = task["task_id"]
        return tasks

    # Handle direct list format
    if isinstance(data, list):
        # Support flat string list: ["instance_id_1", "instance_id_2", ...]
        if data and isinstance(data[0], str):
            return [{"instance_id": iid} for iid in data]
        return data

    raise ValueError(f"Unrecognized tasks file format: {tasks_file}")


def check_ask_user_setup() -> bool:
    """Verify ask_user tool infrastructure is in place."""
    required = [
        ASK_USER_TOOL_DIR / "bin" / "ask_user",
        ASK_USER_CONFIG,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("WARNING: ask_user infrastructure missing:")
        for p in missing:
            print(f"  - {p}")
        return False
    return True


class SWEBenchRunner:
    """Run SWE-bench Pro tasks with optional underspecification."""

    def __init__(
        self,
        dockerhub_username: str,
        trajectory_dir: str = str(DEFAULT_TRAJECTORY_DIR),
        suppress_traj_warning: bool = False,
    ):
        self.dockerhub_username = dockerhub_username
        self.trajectory_dir = trajectory_dir
        self._adapter = None
        self._pipeline = None
        self._dataset_cache = None

        traj_path = Path(trajectory_dir)
        if not traj_path.exists():
            if not suppress_traj_warning:
                print(f"WARNING: Trajectory directory not found: {traj_path}")
                print("  Segment extraction will run without trajectory grounding.")
                print("  For better results, provide baseline trajectories via --trajectory-dir.")
            traj_path.mkdir(parents=True, exist_ok=True)
        elif not any(traj_path.iterdir()) and not suppress_traj_warning:
            print(f"WARNING: Trajectory directory is empty: {traj_path}")
            print("  Segment extraction will run without trajectory grounding.")

    @property
    def adapter(self):
        """Lazy load SWEBenchProAdapter."""
        if self._adapter is None:
            from synthetic.adapters.swebench import SWEBenchProAdapter

            self._adapter = SWEBenchProAdapter(trajectory_dir=self.trajectory_dir)
        return self._adapter

    @property
    def pipeline(self):
        """Lazy load SyntheticPipeline."""
        if self._pipeline is None:
            from synthetic.pipeline import SyntheticPipeline

            self._pipeline = SyntheticPipeline(
                swebench_trajectory_dir=self.trajectory_dir,
            )
        return self._pipeline

    def get_tasks_from_hf(
        self, limit: Optional[int] = None, instance_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get raw SWE-bench Pro tasks from HuggingFace dataset."""
        if self._dataset_cache is None:
            from datasets import load_dataset

            self._dataset_cache = load_dataset("ScaleAI/SWE-bench_Pro", split="test")

        tasks = list(self._dataset_cache)

        if instance_ids:
            id_set = set(instance_ids)
            tasks = [t for t in tasks if t["instance_id"] in id_set]
        if limit:
            tasks = tasks[:limit]

        return tasks

    def get_tasks(
        self,
        tasks_file: Optional[Path] = None,
        limit: Optional[int] = None,
        instance_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get tasks from file or HuggingFace.

        Priority: tasks_file > instance_ids > HuggingFace (with limit)

        Note: Tasks file typically only contains task IDs. We fetch full data
        from HuggingFace to get repo, problem_statement, etc.
        """
        if tasks_file:
            file_tasks = load_tasks_from_file(tasks_file)
            if limit:
                file_tasks = file_tasks[:limit]
            # Extract instance IDs and fetch full data from HuggingFace
            ids_from_file = [t["instance_id"] for t in file_tasks]
            return self.get_tasks_from_hf(instance_ids=ids_from_file)

        return self.get_tasks_from_hf(limit=limit, instance_ids=instance_ids)

    @staticmethod
    def format_prompt(row: Dict) -> str:
        """Format problem statement with requirements and interface.

        Delegates to the adapter's static method for consistency.
        """
        from synthetic.adapters.swebench import SWEBenchProAdapter

        return SWEBenchProAdapter.format_prompt(row)

    # =========================================================================
    # STAGE 1: GENERATE
    # =========================================================================

    def run_generate(
        self,
        tasks: List[Dict],
        run_dir: Path,
        severity: str = DEFAULTS["severity"],
        max_level: int = DEFAULTS["max_level"],
        top_k_per_level: Optional[int] = None,
        target_variants: int = 200,
        include_originals: bool = True,
        ask_user: bool = False,
        generate_concurrency: int = 10,
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """Generate underspec variants: pipeline -> uniform sample -> save all metadata.

        This is Stage 1 of the SWE-bench underspec workflow. Produces:
        - instances.yaml (for SWE-Agent)
        - original_instances.yaml (baseline)
        - underspec_candidates.csv (full variant metadata, matches TAC schema)
        - pipeline_results.json (raw pipeline output)
        - variant_metadata.json (selected variant details)
        - task_definitions.json (if ask_user)

        Args:
            tasks: List of HF task rows to process
            run_dir: Output directory
            severity: Removal strategy (delete/vaguify/genericize)
            max_level: Max segments to remove together (1=single, 2=pairs)
            top_k_per_level: Limit combinations per level
            target_variants: Target number of underspec variants to select
            include_originals: Also generate baseline instances for original tasks
            ask_user: Generate task_definitions.json for ask_user mode

        Returns:
            (instances, task_definitions) where task_definitions is None if not ask_user
        """
        from synthetic.pipeline import SyntheticPipeline

        from synthetic import Severity

        run_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Run pipeline on each task (parallel)
        print(
            f"\n--- Phase 1: Running pipeline on {len(tasks)} tasks"
            f" (concurrency={generate_concurrency}) ---"
        )

        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        completed_count = 0
        count_lock = threading.Lock()

        def _process_one(row):
            nonlocal completed_count
            instance_id = row["instance_id"]
            try:
                result = self.pipeline.process_swebench_task(
                    instance_id=instance_id,
                    severity=Severity(severity),
                    max_level=max_level,
                    top_k_per_level=top_k_per_level,
                )
                with count_lock:
                    completed_count += 1
                    n = completed_count
                print(
                    f"[{n}/{len(tasks)}] {instance_id}"
                    f" -> {len(result.segments)} segments, {len(result.variants)} variants"
                )
                return result
            except Exception as e:
                with count_lock:
                    completed_count += 1
                    n = completed_count
                print(f"[{n}/{len(tasks)}] {instance_id} ERROR: {e}")
                return None

        pipeline_results = []
        n_errors = 0
        with ThreadPoolExecutor(max_workers=generate_concurrency) as pool:
            futures = {pool.submit(_process_one, row): row for row in tasks}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    pipeline_results.append(result)
                else:
                    n_errors += 1

        if not pipeline_results:
            raise RuntimeError("No pipeline results generated! All tasks failed.")

        if n_errors > 0:
            error_rate = n_errors / len(tasks)
            print(f"\nWARNING: {n_errors}/{len(tasks)} tasks failed ({error_rate:.0%})")
            if error_rate > 0.5:
                raise RuntimeError(
                    f"{n_errors}/{len(tasks)} tasks failed (>{50}%). "
                    f"Aborting — check errors above."
                )

        # Save raw pipeline results for debugging
        raw_results_path = run_dir / "pipeline_results.json"
        with open(raw_results_path, "w") as f:
            json.dump(
                [pr.to_dict() for pr in pipeline_results],
                f,
                indent=2,
                default=str,
            )
        print(f"Saved raw pipeline results: {raw_results_path}")

        # Phase 2: Uniform sample across all results
        print(f"\n--- Phase 2: Uniform sampling {target_variants} variants ---")
        selected = SyntheticPipeline.uniform_sample_variants(pipeline_results, target_variants)
        print(f"Selected {len(selected)} variants from {len(pipeline_results)} tasks")

        # Build a lookup from instance_id to HF row for Docker image URIs
        task_lookup = {row["instance_id"]: row for row in tasks}

        # Phase 3: Generate instances.yaml using variant.underspecified_prompt
        print("\n--- Phase 3: Generating output files ---")
        instances = []
        for task_id, variant in selected:
            row = task_lookup.get(task_id)
            if not row:
                print(f"  WARNING: No HF data for {task_id}, skipping")
                continue

            repo_name = row.get("repo", "")
            image_name = get_dockerhub_image_uri(task_id, self.dockerhub_username, repo_name)
            instance_id_out = f"{task_id}__{variant.id}"

            instances.append(
                {
                    "image_name": image_name,
                    "problem_statement": variant.underspecified_prompt,
                    "instance_id": instance_id_out,
                    "base_commit": row.get("base_commit", ""),
                    "repo_name": "app",
                }
            )

        # Write underspec instances
        instances_path = run_dir / "instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump(instances, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Generated {len(instances)} underspec instances: {instances_path}")

        # Generate original/baseline instances
        if include_originals:
            original_task_ids = list({task_id for task_id, _ in selected})
            original_instances = []
            for task_id in original_task_ids:
                row = task_lookup.get(task_id)
                if not row:
                    continue
                repo_name = row.get("repo", "")
                image_name = get_dockerhub_image_uri(task_id, self.dockerhub_username, repo_name)
                original_instances.append(
                    {
                        "image_name": image_name,
                        "problem_statement": self.format_prompt(row),
                        "instance_id": task_id,
                        "base_commit": row.get("base_commit", ""),
                        "repo_name": "app",
                    }
                )

            original_path = run_dir / "original_instances.yaml"
            with open(original_path, "w") as f:
                yaml.dump(
                    original_instances,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            print(f"Generated {len(original_instances)} original instances: {original_path}")

        # ── Save underspec_candidates.csv (matching TAC schema) ──
        # This is the key output for downstream process/filter/export scripts.
        # Matches columns from TAC's run_tac_underspec.py generate stage.
        candidates_rows = []
        for task_id, variant in selected:
            n_segs = len(variant.removed_segments)
            if n_segs == 1:
                seg = variant.removed_segments[0]
                dim_str = seg.dimension.value
                subdim_str = seg.subdimension
                removed_str = seg.value
                crit = seg.criticality
                guess = seg.guessability
                traj_used = seg.is_used_in_trajectory
                first_use = seg.first_use_pct
                cp_refs = ",".join(seg.checkpoint_refs) if seg.checkpoint_refs else ""
            else:
                dim_str = "+".join(s.dimension.value for s in variant.removed_segments)
                subdim_str = "+".join(
                    s.subdimension for s in variant.removed_segments if s.subdimension
                )
                removed_str = " | ".join(str(s.value) for s in variant.removed_segments)
                crit = max(s.criticality for s in variant.removed_segments)
                guess = sum(s.guessability for s in variant.removed_segments) / n_segs
                traj_used = any(s.is_used_in_trajectory for s in variant.removed_segments)
                first_use = min(
                    (s.first_use_pct for s in variant.removed_segments if s.first_use_pct),
                    default=None,
                )
                all_cp_refs = set()
                for s in variant.removed_segments:
                    all_cp_refs.update(s.checkpoint_refs or [])
                cp_refs = ",".join(sorted(all_cp_refs))

            priority = variant.predicted_difficulty
            removed_segments_json = json.dumps([s.to_dict() for s in variant.removed_segments])
            instance_id_out = f"{task_id}__{variant.id}"

            candidates_rows.append(
                {
                    "instance_id": instance_id_out,
                    "original_instance_id": task_id,
                    "variant_id": variant.id,
                    "dimension": dim_str,
                    "subdimension": subdim_str,
                    "removed_value": removed_str,
                    "removed_segments_json": removed_segments_json,
                    "num_segments": n_segs,
                    "original_prompt": variant.original_prompt,
                    "underspecified_prompt": variant.underspecified_prompt,
                    "severity": variant.severity.value if variant.severity else "delete",
                    "is_used_in_trajectory": traj_used,
                    "first_use_pct": first_use,
                    "checkpoint_refs": cp_refs,
                    "criticality": crit,
                    "guessability": guess,
                    "priority_score": priority,
                    "expected_questions": (
                        json.dumps(variant.expected_questions)
                        if variant.expected_questions
                        else "[]"
                    ),
                    "expected_failure_mode": variant.expected_failure_mode or "",
                    "predicted_difficulty": variant.predicted_difficulty,
                }
            )

        candidates_path = run_dir / "underspec_candidates.csv"
        if candidates_rows:
            fieldnames = list(candidates_rows[0].keys())
            with open(candidates_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(candidates_rows)
            print(
                f"Saved underspec_candidates.csv: {candidates_path} ({len(candidates_rows)} rows)"
            )

        # Generate task definitions for ask_user mode
        task_definitions = None
        if ask_user:
            task_definitions = self.generate_task_definitions(selected)
            td_path = run_dir / "task_definitions.json"
            with open(td_path, "w") as f:
                json.dump(task_definitions, f, indent=2, default=str)
            print(f"Generated task definitions for {len(task_definitions)} variants: {td_path}")

        # Save variant metadata (compact summary)
        variant_meta = []
        for task_id, variant in selected:
            variant_meta.append(
                {
                    "task_id": task_id,
                    "variant_id": variant.id,
                    "instance_id": f"{task_id}__{variant.id}",
                    "severity": variant.severity.value,
                    "num_segments": len(variant.removed_segments),
                    "dimensions": [s.dimension.value for s in variant.removed_segments],
                    "segment_values": [s.value for s in variant.removed_segments],
                    "predicted_difficulty": variant.predicted_difficulty,
                    "has_expected_questions": bool(variant.expected_questions),
                }
            )
        meta_path = run_dir / "variant_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(variant_meta, f, indent=2)
        print(f"Saved variant metadata: {meta_path}")

        return instances, task_definitions

    def generate_task_definitions(
        self, selected_variants: List[Tuple[str, Any]]
    ) -> Dict[str, Dict]:
        """Generate task_definitions.json from pipeline variant objects.

        Unlike run_benchmark_with_ask_user.py which reverse-engineers segments
        from variant IDs and generates template-based expected_questions, we use
        the real UnderspecVariant objects which carry all LLM-generated metadata.

        Args:
            selected_variants: List of (task_id, UnderspecVariant) tuples

        Returns:
            Dict mapping instance_id -> task definition
        """
        task_definitions = {}
        for task_id, variant in selected_variants:
            instance_id = f"{task_id}__{variant.id}"
            task_definitions[instance_id] = {
                "primary_task": variant.original_prompt,
                "underspecified_task": variant.underspecified_prompt,
                "removed_segments": [
                    {
                        "id": s.id,
                        "value": s.value,
                        "dimension": s.dimension.value,
                        "text": s.text,
                    }
                    for s in variant.removed_segments
                ],
                "expected_questions": variant.expected_questions,
                "expected_failure_mode": variant.expected_failure_mode,
                "severity": variant.severity.value,
                "original_instance_id": task_id,
            }
        return task_definitions

    # =========================================================================
    # STAGE 2: RUN
    # =========================================================================

    def run_trials(
        self,
        run_dir: Path,
        model: str = "gpt_5_2",
        num_trials: int = DEFAULTS["num_trials"],
        concurrency: int = DEFAULTS["concurrency"],
        startup_timeout: int = DEFAULTS["startup_timeout"],
        runtime_timeout: int = DEFAULTS["runtime_timeout"],
        reasoning_effort: Optional[str] = None,
        ask_user: bool = False,
        include_baseline: bool = True,
        dir_prefix: str = "exp",
    ):
        """Run SWE-Agent trials on generated instances (Stage 2).

        Reads instances.yaml from run_dir and runs num_trials trials.

        Args:
            run_dir: Directory with instances.yaml from generate stage
            model: Agent model (resolved identifier from constants.py)
            num_trials: Number of trials for pass@k evaluation
            concurrency: Number of parallel workers
            startup_timeout: Container startup timeout in seconds
            runtime_timeout: Container runtime timeout in seconds
            reasoning_effort: Reasoning effort level (low/medium/high)
            ask_user: Use ask_user config
            include_baseline: Also run baseline on original_instances.yaml
            dir_prefix: Directory prefix for trial output ("exp" or "baseline")
        """
        instances_yaml = run_dir / "instances.yaml"
        if not instances_yaml.exists():
            raise RuntimeError(
                f"instances.yaml not found in {run_dir}. "
                f"Run --generate first to create instances."
            )

        for exp_num in range(1, num_trials + 1):
            print(f"\n{'='*60}")
            print(f"{dir_prefix.upper()} {exp_num}/{num_trials}")
            print(f"{'='*60}")

            exp_dir = run_dir / f"{dir_prefix}_{exp_num}"
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Copy task_definitions.json into exp dir for ask_user hook auto-discovery
            if ask_user:
                import shutil

                _ensure_task_definitions(run_dir)
                td_src = run_dir / "task_definitions.json"
                if td_src.exists():
                    shutil.copy2(td_src, exp_dir / "task_definitions.json")

            success = self.run_sweagent(
                instances_yaml,
                exp_dir,
                model=model,
                num_workers=concurrency,
                startup_timeout=startup_timeout,
                runtime_timeout=runtime_timeout,
                reasoning_effort=reasoning_effort,
                ask_user=ask_user,
            )
            if not success:
                raise RuntimeError(
                    f"{dir_prefix}_{exp_num} failed. Aborting to avoid partial results."
                )

        # Run baseline on originals (same num_trials for pass@k)
        if include_baseline:
            original_yaml = run_dir / "original_instances.yaml"
            if not original_yaml.exists():
                raise RuntimeError(
                    f"original_instances.yaml not found in {run_dir}. "
                    f"Run --generate first to create baseline instances."
                )
            for exp_num in range(1, num_trials + 1):
                print(f"\n{'='*60}")
                print(f"BASELINE {exp_num}/{num_trials} (original tasks)")
                print(f"{'='*60}")
                success = self.run_sweagent(
                    original_yaml,
                    run_dir / f"baseline_{exp_num}",
                    model=model,
                    num_workers=concurrency,
                    startup_timeout=startup_timeout,
                    runtime_timeout=runtime_timeout,
                    reasoning_effort=reasoning_effort,
                    ask_user=False,
                )
                if not success:
                    raise RuntimeError(
                        f"baseline_{exp_num} failed. Aborting to avoid partial results."
                    )

    # =========================================================================
    # SWE-AGENT RUNNER
    # =========================================================================

    def _generate_sweagent_config(
        self,
        output_dir: Path,
        model: str,
        max_turns: int,
        api_key: Optional[str],
        api_base: Optional[str],
        reasoning_effort: str = "",
        enable_cache_control: bool = False,
        base_config_path: Optional[Path] = None,
    ) -> Path:
        """Generate a custom SWE-agent config with model + credentials baked in.

        Always used (rather than passing credentials as CLI args) so that
        API keys don't appear in the process table (ps aux).
        """
        # Load the base config (ask_user variant if provided, else standard)
        if base_config_path is None:
            base_config_path = SWEAGENT_DIR / "config" / "tool_use.yaml"
        with open(base_config_path) as f:
            config = yaml.safe_load(f)

        # Add/update the model configuration
        if "agent" not in config:
            config["agent"] = {}
        if "model" not in config["agent"]:
            config["agent"]["model"] = {}

        config["agent"]["model"]["name"] = model
        config["agent"]["model"]["per_instance_call_limit"] = max_turns
        config["agent"]["model"]["per_instance_cost_limit"] = 0
        config["agent"]["model"]["total_cost_limit"] = 0
        config["agent"]["model"]["top_p"] = None
        if reasoning_effort:
            config["agent"]["model"]["completion_kwargs"] = {
                "reasoning_effort": reasoning_effort,
            }

        if api_key:
            config["agent"]["model"]["api_key"] = api_key
        if api_base:
            config["agent"]["model"]["api_base"] = api_base

        # Enable Anthropic prompt caching
        if enable_cache_control:
            config["agent"]["history_processors"] = [
                {"type": "cache_control", "last_n_messages": 2}
            ]

        # Write to output directory
        custom_config_path = output_dir / "sweagent_config.yaml"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(custom_config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        features = []
        if reasoning_effort:
            features.append(f"reasoning_effort={reasoning_effort}")
        if enable_cache_control:
            features.append("cache_control")
        print(f"Generated custom config ({', '.join(features)}): {custom_config_path}")
        return custom_config_path

    def run_sweagent(
        self,
        instances_path: Path,
        output_dir: Path,
        model: str = "gpt_5_2",
        num_workers: int = 10,
        max_turns: int = 250,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        startup_timeout: int = DEFAULTS["startup_timeout"],
        runtime_timeout: int = DEFAULTS["runtime_timeout"],
        reasoning_effort: Optional[str] = None,
        ask_user: bool = False,
    ) -> bool:
        """
        Run SWE-Agent on instance file.

        Returns True if successful.
        """
        import os

        # Get API credentials from environment if not provided
        api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        api_base = api_base or os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL")

        # Ensure absolute paths since we run from SWEAGENT_DIR
        instances_path_abs = instances_path.resolve()
        output_dir_abs = output_dir.resolve()

        # Select config: ask_user or standard
        if ask_user:
            base_config_path = ASK_USER_CONFIG
        else:
            base_config_path = SWEAGENT_DIR / "config" / "tool_use.yaml"

        # Always generate a custom config to keep credentials out of the process table
        is_anthropic = "anthropic" in model.lower() or "claude" in model.lower()
        custom_config = self._generate_sweagent_config(
            output_dir_abs,
            model,
            max_turns,
            api_key,
            api_base,
            reasoning_effort or "",
            enable_cache_control=is_anthropic,
            base_config_path=base_config_path,
        )
        config_path = str(custom_config)

        sweagent_bin = str(Path(sys.executable).parent / "sweagent")
        cmd = [
            sweagent_bin,
            "run-batch",
            "--config",
            config_path,
            "--output_dir",
            str(output_dir_abs),
            "--num_workers",
            str(num_workers),
            "--instances.type",
            "file",
            "--instances.path",
            str(instances_path_abs),
            "--instances.deployment.type",
            "modal",
            "--instances.deployment.startup_timeout",
            str(startup_timeout),
            "--instances.deployment.runtime_timeout",
            str(runtime_timeout),
        ]

        print(f"\nRunning SWE-Agent: {' '.join(cmd[:10])}...")

        try:
            result = subprocess.run(cmd, cwd=SWEAGENT_DIR, check=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"SWE-Agent failed: {e}")
            return False
        except FileNotFoundError:
            print(
                "sweagent command not found. Install via: pip install -e swebenchpro/SWE-bench_Pro-os/SWE-agent"
            )
            return False


# =============================================================================
# HELPERS: Multi-model experiments
# =============================================================================


def _ensure_task_definitions(run_dir: Path):
    """Generate task_definitions.json from underspec_candidates.csv if missing.

    All required fields (original_prompt, underspecified_prompt, removed_segments,
    expected_questions, etc.) are already stored in the candidates CSV from the
    generate stage. This avoids requiring --ask-user during generation.
    """
    td_path = run_dir / "task_definitions.json"
    if td_path.exists():
        return

    csv_path = run_dir / "underspec_candidates.csv"
    if not csv_path.exists():
        print(
            f"WARNING: Cannot generate task_definitions.json — no underspec_candidates.csv in {run_dir}"
        )
        return

    import csv

    task_defs = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            iid = row.get("instance_id", "")
            if not iid or VARIANT_DELIMITER not in iid:
                continue
            segments_raw = row.get("removed_segments_json", "[]")
            try:
                segments = json.loads(segments_raw) if segments_raw else []
            except (json.JSONDecodeError, TypeError):
                segments = []

            eq_raw = row.get("expected_questions", "")
            try:
                expected_questions = json.loads(eq_raw) if eq_raw else []
            except (json.JSONDecodeError, TypeError):
                expected_questions = []

            task_defs[iid] = {
                "primary_task": row.get("original_prompt", ""),
                "underspecified_task": row.get("underspecified_prompt", ""),
                "removed_segments": segments,
                "expected_questions": expected_questions,
                "expected_failure_mode": row.get("expected_failure_mode", ""),
                "severity": row.get("severity", "delete"),
                "original_instance_id": row.get("original_instance_id", ""),
            }

    with open(td_path, "w") as f:
        json.dump(task_defs, f, indent=2, default=str)
    print(
        f"Auto-generated task_definitions.json ({len(task_defs)} variants) from underspec_candidates.csv"
    )


def _copy_instances_for_model(source_dir: Path, target_dir: Path):
    """Copy instance files from generate dir to per-model run dir.

    Prefers frozen_instances.yaml (Phase B) over instances.yaml.
    """
    import shutil

    # Instances: frozen > regular
    frozen_src = source_dir / "frozen_instances.yaml"
    instances_src = source_dir / "instances.yaml"
    if frozen_src.exists():
        shutil.copy2(frozen_src, target_dir / "instances.yaml")
        print("Copied frozen_instances.yaml -> instances.yaml")
    elif instances_src.exists():
        shutil.copy2(instances_src, target_dir / "instances.yaml")
        print("Copied instances.yaml")

    # Original instances for baseline
    orig_src = source_dir / "original_instances.yaml"
    if orig_src.exists():
        shutil.copy2(orig_src, target_dir / "original_instances.yaml")

    # Candidates CSV for downstream process script
    cand_src = source_dir / "underspec_candidates.csv"
    if cand_src.exists():
        shutil.copy2(cand_src, target_dir / "underspec_candidates.csv")

    # Task definitions for ask_user mode
    td_src = source_dir / "task_definitions.json"
    if td_src.exists():
        shutil.copy2(td_src, target_dir / "task_definitions.json")


def _freeze_instances(exp_dir: Path, filtered_csv: Path):
    """Create frozen_instances.yaml from filtered CSV."""
    import pandas as pd

    if not filtered_csv.exists():
        print(f"ERROR: Filtered CSV not found: {filtered_csv}")
        sys.exit(1)

    instances_path = exp_dir / "instances.yaml"
    if not instances_path.exists():
        print(f"ERROR: instances.yaml not found in {exp_dir}")
        sys.exit(1)

    df = pd.read_csv(filtered_csv)
    frozen_ids = set(df["instance_id"].tolist())

    with open(instances_path) as f:
        all_instances = yaml.safe_load(f) or []

    frozen = [inst for inst in all_instances if inst["instance_id"] in frozen_ids]
    missing = frozen_ids - {inst["instance_id"] for inst in frozen}

    frozen_path = exp_dir / "frozen_instances.yaml"
    with open(frozen_path, "w") as f:
        yaml.dump(frozen, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print("\nFREEZE COMPLETE")
    print(f"  Source instances: {len(all_instances)}")
    print(f"  Filtered CSV IDs: {len(frozen_ids)}")
    print(f"  Frozen instances: {len(frozen)}")
    if missing:
        print(f"  WARNING: {len(missing)} IDs not found in instances.yaml")
    print(f"  Output: {frozen_path}")
    print("\nNext: run per-model trials with --model-suffix")


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench Pro tasks with SWE-Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages (run separately or together):
  # 1. Prepare baselines (writes baseline_N/ dirs + exports golden trajectories)
  %(prog)s --prepare-baselines --run --format_name my_exp --limit 5 --num_trials 3

  # 2. Generate variants into the same directory
  %(prog)s --generate --format_name my_exp --runs-dir <exp_dir> --severity delete --target-variants 200 --ask-user

  # 3. Run underspec trials (baselines already present from step 1)
  %(prog)s --run --exp-dir <exp_dir> --num_trials 3 --skip-baseline

  # Freeze filtered variants for multi-model comparison
  %(prog)s --freeze --exp-dir <run_dir> --filtered-csv <run_dir>/underspec_results_filtered.csv

  # Per-model run on frozen variants
  %(prog)s --run --exp-dir <run_dir> --model-suffix sonnet_4_5 --backend_model sonnet_4_5
""",
    )

    # ── Stage selection ──
    stage_group = parser.add_argument_group("Stage selection (batch underspec)")
    stage_group.add_argument(
        "--generate",
        action="store_true",
        help="Stage 1: Run pipeline, uniform sample, generate instances.yaml + underspec_candidates.csv",
    )
    stage_group.add_argument(
        "--run",
        action="store_true",
        help="Stage 2: Run SWE-Agent trials on generated instances",
    )
    stage_group.add_argument(
        "--prepare-baselines",
        action="store_true",
        help="Create instances.yaml from F2P-filtered original tasks (no pipeline). "
        "Use with --run to get baseline trajectories for grounded extraction.",
    )
    stage_group.add_argument(
        "--exp-dir",
        type=str,
        help="Experiment directory for --run (reads instances.yaml from here)",
    )

    # ── Task source ──
    parser.add_argument(
        "--tasks-file",
        type=str,
        help="JSON file with task list (supports 10-task-test.json format)",
    )
    parser.add_argument("--instance-id", type=str, help="Specific instance ID from HuggingFace")
    parser.add_argument("--limit", type=int, help="Limit number of tasks (from HF or file)")

    # ── Experiment identification ──
    parser.add_argument(
        "--format_name",
        type=str,
        help="Experiment name (used in output directory). Required for --generate.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=DEFAULTS["num_trials"],
        help=f"Number of trials for pass@k evaluation (default: {DEFAULTS['num_trials']})",
    )

    # ── Model configuration ──
    parser.add_argument(
        "--backend_model",
        "--model",
        type=str,
        default="gpt_5_2",
        dest="backend_model",
        help="Agent model — short name (e.g. gpt_5_2, sonnet_4_6) or full LiteLLM identifier",
    )

    # ── Parallelization ──
    parser.add_argument(
        "--concurrency",
        "--num-workers",
        type=int,
        default=DEFAULTS["concurrency"],
        dest="concurrency",
        help=f"Number of parallel workers for SWE-Agent runs (default: {DEFAULTS['concurrency']})",
    )
    parser.add_argument(
        "--generate-concurrency",
        type=int,
        default=DEFAULTS["concurrency"],
        help=f"Number of parallel LLM workers for --generate stage (default: {DEFAULTS['concurrency']})",
    )

    # ── Container timeouts ──
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=DEFAULTS["startup_timeout"],
        help=f"Container startup timeout in seconds (default: {DEFAULTS['startup_timeout']})",
    )
    parser.add_argument(
        "--runtime-timeout",
        type=int,
        default=DEFAULTS["runtime_timeout"],
        help=f"Container runtime timeout in seconds (default: {DEFAULTS['runtime_timeout']})",
    )

    # ── Infrastructure ──
    parser.add_argument(
        "--dockerhub-username",
        type=str,
        default="jefzda",
        help="DockerHub username for SWE-bench Pro Docker images (default: jefzda)",
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        help="Override output directory (default: experiments/swebench/runs/run_<name>_<timestamp>)",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        default=str(DEFAULT_TRAJECTORY_DIR),
        help="Path to directory containing trajectory folders for grounded extraction "
        f"(default: {DEFAULT_TRAJECTORY_DIR.relative_to(REPO_ROOT)})",
    )

    # ── Variant generation options (for --generate) ──
    gen_group = parser.add_argument_group("Variant generation (--generate)")
    gen_group.add_argument(
        "--severity",
        default=DEFAULTS["severity"],
        choices=["delete", "vaguify", "genericize"],
        help=f"Removal strategy (default: {DEFAULTS['severity']})",
    )
    gen_group.add_argument(
        "--f2p-threshold",
        type=int,
        default=2,
        help="Min F2P test count for task selection (default: 2, i.e. F2P > 2)",
    )
    gen_group.add_argument(
        "--target-variants",
        type=int,
        default=200,
        help="Target number of underspec variants after uniform sampling (default: 200)",
    )
    gen_group.add_argument(
        "--max-level",
        type=int,
        default=DEFAULTS["max_level"],
        help=f"Max segments to remove together (1=single, 2=pairs). Default: {DEFAULTS['max_level']}",
    )
    gen_group.add_argument(
        "--top-k-per-level",
        type=int,
        default=None,
        help="Limit combinations per segment level (default: None = all)",
    )

    # ── Ask-user mode ──
    parser.add_argument(
        "--ask-user",
        action="store_true",
        help="Enable ask_user tool (generates task_definitions.json, uses ask_user config)",
    )

    # ── Model reasoning/thinking ──
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort level for models that support it",
    )

    # ── Execution control ──
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate files only, don't run SWE-Agent"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline runs (use when baselines already exist in the target dir)",
    )

    # ── Multi-model support ──
    multi_group = parser.add_argument_group("Multi-model experiments")
    multi_group.add_argument(
        "--model-suffix",
        type=str,
        default=None,
        help="Per-model run directory tag (e.g. gpt_5_2). Creates {exp_dir}_{suffix}/ with copied instances.",
    )
    multi_group.add_argument(
        "--freeze",
        action="store_true",
        help="Create frozen_instances.yaml from filtered CSV. Requires --exp-dir + --filtered-csv.",
    )
    multi_group.add_argument(
        "--filtered-csv",
        type=str,
        default=None,
        help="Path to underspec_results_filtered.csv (used with --freeze)",
    )

    args = parser.parse_args()

    # =========================================================================
    # Validation
    # =========================================================================

    # ── Multi-model validation ──
    if args.model_suffix:
        if not args.run:
            parser.error("--model-suffix requires --run")
        if not args.exp_dir:
            parser.error("--model-suffix requires --exp-dir")
        if args.generate:
            parser.error("--model-suffix cannot be used with --generate")

    if args.freeze:
        if not args.exp_dir:
            parser.error("--freeze requires --exp-dir")
        if not args.filtered_csv:
            parser.error("--freeze requires --filtered-csv")
        if args.generate or args.run:
            parser.error("--freeze cannot be used with --generate or --run")

    if args.prepare_baselines:
        if args.generate or args.freeze:
            parser.error("--prepare-baselines cannot be used with --generate or --freeze")
        if not args.format_name:
            parser.error("--format_name is required for --prepare-baselines")

    if not (args.generate or args.run or args.freeze or args.prepare_baselines):
        parser.error(
            "at least one of --generate, --run, --freeze, or --prepare-baselines is required"
        )

    if args.generate and not args.format_name:
        parser.error("--format_name is required for --generate")

    if args.run and not args.generate and not args.exp_dir and not args.prepare_baselines:
        parser.error(
            "--exp-dir is required when using --run without --generate or --prepare-baselines"
        )

    if args.run and not args.backend_model.strip():
        parser.error("--backend_model cannot be empty")

    # Resolve model shortcut (e.g. "gemini_3_flash" → "gemini/gemini-3-flash-preview")
    # and add litellm_proxy/ prefix when using a LiteLLM proxy
    args.backend_model = resolve_model(args.backend_model)

    if args.ask_user and not args.generate and not args.run:
        parser.error("--ask-user requires --generate or --run")

    # Validate ask_user infrastructure (hard error when running agents)
    if args.ask_user and args.run and not check_ask_user_setup():
        parser.error(
            "ask_user infrastructure is missing. Install the SWE-agent fork with ask_user support:\n"
            "  git checkout fork/yash/ask-user-host-interception (in SWE-agent dir)\n"
            "  See README.md 'SWE-Bench Pro > Setup' for full instructions."
        )

    # =========================================================================
    # Setup output directory
    # =========================================================================

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.backend_model.split("/")[-1].replace("-", "_")[:20]

    if args.freeze:
        # Freeze operates in-place on the generate dir
        run_dir = Path(args.exp_dir)
    elif args.model_suffix:
        # Per-model: create sibling directory with copied instances
        source_dir = Path(args.exp_dir)
        run_dir = source_dir.parent / f"{source_dir.name}_{args.model_suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        _copy_instances_for_model(source_dir, run_dir)
    elif args.prepare_baselines:
        run_name = f"run_{args.format_name}_{timestamp}"
        run_dir = EXPERIMENTS_DIR / run_name
    elif args.exp_dir and not args.generate:
        # --run only: use specified exp dir
        run_dir = Path(args.exp_dir)
    elif args.runs_dir:
        run_dir = Path(args.runs_dir)
    elif args.generate:
        # Generate is model-agnostic: omit SWE-Agent model from dir name
        run_name = f"run_{args.format_name}_{timestamp}"
        run_dir = EXPERIMENTS_DIR / run_name
    else:
        run_name = f"run_{args.format_name}_{model_short}_{timestamp}"
        run_dir = EXPERIMENTS_DIR / run_name

    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Freeze: early return ──
    if args.freeze:
        _freeze_instances(run_dir, Path(args.filtered_csv))
        return

    # ── Prepare baselines: create instances.yaml from original tasks, then optionally --run ──
    runner = None
    if args.prepare_baselines:
        runner = SWEBenchRunner(
            dockerhub_username=args.dockerhub_username,
            trajectory_dir=args.trajectory_dir,
            suppress_traj_warning=True,
        )
        if args.tasks_file or args.instance_id:
            tasks_file = Path(args.tasks_file) if args.tasks_file else None
            instance_ids = [args.instance_id] if args.instance_id else None
            tasks = runner.get_tasks(
                tasks_file=tasks_file, limit=args.limit, instance_ids=instance_ids
            )
        else:
            filtered_ids = runner.adapter.list_f2p_filtered_tasks(threshold=args.f2p_threshold)
            if args.limit:
                filtered_ids = filtered_ids[: args.limit]
            tasks = runner.get_tasks_from_hf(instance_ids=filtered_ids)

        instances = []
        for row in tasks:
            tid = row["instance_id"]
            repo_name = row.get("repo", "")
            instances.append(
                {
                    "image_name": get_dockerhub_image_uri(tid, args.dockerhub_username, repo_name),
                    "problem_statement": runner.format_prompt(row),
                    "instance_id": tid,
                    "base_commit": row.get("base_commit", ""),
                    "repo_name": "app",
                }
            )

        instances_path = run_dir / "instances.yaml"
        with open(instances_path, "w") as f:
            yaml.dump(instances, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"Generated {len(instances)} original instances: {instances_path}")
        print(f"Output: {run_dir}")

        if not args.run:
            print("\nTo run baselines:")
            print(
                f"  python task_completion_swebench.py --run --exp-dir {run_dir}"
                f" --backend_model {args.backend_model} --num_trials 3"
            )
            return

    # Save run configuration
    config = {
        "format_name": args.format_name,
        "backend_model": args.backend_model,
        "num_trials": args.num_trials,
        "concurrency": args.concurrency,
        "generate_concurrency": args.generate_concurrency,
        "startup_timeout": args.startup_timeout,
        "runtime_timeout": args.runtime_timeout,
        "reasoning_effort": args.reasoning_effort,
        "trajectory_dir": args.trajectory_dir,
        "tasks_file": args.tasks_file,
        "limit": args.limit,
        "instance_id": args.instance_id,
        "generate": args.generate,
        "run": args.run,
        "ask_user": args.ask_user,
        "severity": args.severity,
        "f2p_threshold": args.f2p_threshold if args.generate else None,
        "target_variants": args.target_variants if args.generate else None,
        "max_level": args.max_level if args.generate else None,
        "timestamp": timestamp,
        "run_dir": str(run_dir),
    }
    config_path = run_dir / "config.json"
    if args.generate and args.runs_dir and config_path.exists():
        # --generate reusing an existing dir: preserve original config.json
        # (which has the correct backend_model from step 1) and save
        # generate-specific settings separately.
        with open(run_dir / "generate_config.json", "w") as f:
            json.dump(config, f, indent=2)
    else:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    if runner is None:
        runner = SWEBenchRunner(
            dockerhub_username=args.dockerhub_username,
            trajectory_dir=args.trajectory_dir,
        )

    # =========================================================================
    # Execute: --generate and/or --run
    # =========================================================================

    # ── GENERATE STAGE ──
    if args.generate:
        if args.tasks_file or args.instance_id:
            tasks_file = Path(args.tasks_file) if args.tasks_file else None
            instance_ids = [args.instance_id] if args.instance_id else None
            tasks = runner.get_tasks(
                tasks_file=tasks_file, limit=args.limit, instance_ids=instance_ids
            )
        else:
            print(f"\n--- F2P filtering (threshold={args.f2p_threshold}) ---")
            filtered_ids = runner.adapter.list_f2p_filtered_tasks(threshold=args.f2p_threshold)
            if args.limit:
                filtered_ids = filtered_ids[: args.limit]
            print(f"F2P filtered: {len(filtered_ids)} tasks")
            tasks = runner.get_tasks_from_hf(instance_ids=filtered_ids)

        print(f"\n{'='*60}")
        print("SWE-bench Pro: GENERATE stage")
        print(f"{'='*60}")
        print(f"Format Name: {args.format_name}")
        print(f"Tasks: {len(tasks)}")
        print(f"Severity: {args.severity}")
        print(f"Max Level: {args.max_level}")
        print(f"Target Variants: {args.target_variants}")
        print(f"Trajectory Dir: {args.trajectory_dir or 'None'}")
        print(f"Generate Concurrency: {args.generate_concurrency}")
        print(f"Ask User: {args.ask_user}")
        print(f"Output: {run_dir}")
        print(f"{'='*60}")

        runner.run_generate(
            tasks=tasks,
            run_dir=run_dir,
            severity=args.severity,
            max_level=args.max_level,
            top_k_per_level=args.top_k_per_level,
            target_variants=args.target_variants,
            include_originals=True,
            ask_user=args.ask_user,
            generate_concurrency=args.generate_concurrency,
        )

        print(f"\nGenerate complete. Output: {run_dir}")
        print("\nGenerated files:")
        for p in sorted(run_dir.iterdir()):
            print(f"  {p.name}")

        if not args.run:
            if args.dry_run:
                print("\n[DRY RUN] Generate stage complete.")
            else:
                print("\nTo run trials:")
                print(
                    f"  python task_completion_swebench.py --run --exp-dir {run_dir} --num_trials 3"
                )
            return

    # ── RUN STAGE ──
    if args.run and not args.dry_run:
        exp_dir = run_dir

        print(f"\n{'='*60}")
        print("SWE-bench Pro: RUN stage")
        print(f"{'='*60}")
        print(f"Model: {args.backend_model}")
        print(f"Experiments: {args.num_trials}")
        print(f"Concurrency: {args.concurrency}")
        print(f"Exp Dir: {exp_dir}")
        print(f"{'='*60}")

        # --prepare-baselines writes to baseline_N/ so they coexist with
        # later --generate + --run which writes to exp_N/ in the same dir.
        trial_prefix = "baseline" if args.prepare_baselines else "exp"

        runner.run_trials(
            run_dir=exp_dir,
            model=args.backend_model,
            num_trials=args.num_trials,
            concurrency=args.concurrency,
            startup_timeout=args.startup_timeout,
            runtime_timeout=args.runtime_timeout,
            reasoning_effort=args.reasoning_effort,
            ask_user=args.ask_user,
            include_baseline=not args.skip_baseline and not args.prepare_baselines,
            dir_prefix=trial_prefix,
        )

        # Auto-export golden trajectories after --prepare-baselines --run
        if args.prepare_baselines:
            traj_dir = Path(args.trajectory_dir)
            traj_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            exported = 0
            for instance_dir in sorted((exp_dir / "baseline_1").iterdir()):
                if not instance_dir.is_dir():
                    continue
                traj_files = sorted(instance_dir.glob("*.traj"))
                if traj_files:
                    dest = traj_dir / instance_dir.name
                    dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(traj_files[0], dest / traj_files[0].name)
                    exported += 1
            print(f"\nExported {exported} golden trajectories to {traj_dir}")

        print(f"\n{'='*60}")
        print("DONE")
        print(f"Results: {exp_dir}")
        print("\nNext steps:")
        print("  # Evaluate + classify")
        print(f"  python scripts/process_swebench_underspec.py --exp-dir {exp_dir}")
        print("  # Filter")
        print(
            f"  python scripts/filter_swebench_samples.py --input {exp_dir}/underspec_results.csv"
        )
        print("  # Export benchmark JSON")
        print(
            f"  python scripts/export_swebench_dataset.py --input {exp_dir}/underspec_results_filtered.csv"
        )
        print(f"{'='*60}")

    elif args.run and args.dry_run:
        print("\n[DRY RUN] Skipping SWE-Agent execution.")


if __name__ == "__main__":
    main()
