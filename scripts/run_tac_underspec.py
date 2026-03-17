"""
Underspecified Task Evaluation
==============================

Generate underspecified variants and test if removing info causes task failure.

Quick E2E Test (single task, 1 trial - all-in-one):
    cd research/lhaw
    python scripts/run_tac_underspec.py --all --tasks sde_create_sqlite_database --experiment test --num-trials 1

Quick E2E Test (step-by-step):
    python scripts/run_tac_underspec.py --generate --tasks ds_coffee --experiment delete_exp --max-variants-per-level 3 --max-level 1
    python scripts/run_tac_underspec.py --run -e delete_exp --num-trials 1 --model gemini_3_flash
    python scripts/run_tac_underspec.py --results -e delete_exp

Variant count control:
    --max-segments N            Limit total segments considered (top-N by priority, default: all)
    --max-variants-per-level N  Max variants per level per task (default: all)
    --max-level L               Max segments removed together: 1=single, 2=pairs (default: 2)
    Example: 8 segments with --max-level 2 → 8 + C(8,2) = 36 variants
             8 segments with --max-variants-per-level 3 --max-level 1 → 3 variants
             8 segments with --max-segments 5 --max-level 2 → 5 + C(5,2) = 15 variants

If pass@N = 0%, the removed segment was OUTCOME-CRITICAL.

Parallelization:
    --generate-workers N    Concurrent tasks for --generate (parallel LLM calls, default: 4)
    --parallel-variants M   Concurrent variants during --run (default: 1)
                            Each variant runs its trials sequentially.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

# Paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import DEFAULTS, PATHS, get_model  # noqa: E402

SYNTH_OUTPUTS = REPO_ROOT / PATHS["synthetic_outputs"]
TAC_TASKS = REPO_ROOT / PATHS["tac_tasks"]
RUNS_DIR = REPO_ROOT / PATHS["runs_dir"]

# Cleanup interval from constants
CLEANUP_INTERVAL = DEFAULTS.get("cleanup_interval", 10)


def cleanup_docker_containers(experiment: str = None, age_minutes: int = 30) -> int:
    """Remove exited openhands containers to prevent resource exhaustion.

    Args:
        experiment: If provided, only clean containers matching this experiment name.
                   If None, cleans containers older than age_minutes.
        age_minutes: Only clean containers older than this many minutes (default: 30).
                    This avoids cleaning containers from concurrent runs.

    Returns number of containers removed.
    """
    try:
        # Get exited container info (ID, Name, Created time)
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                "name=openhands",
                "--filter",
                "status=exited",
                "--format",
                "{{.ID}}\t{{.Names}}\t{{.CreatedAt}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return 0

        lines = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
        if not lines:
            return 0

        cutoff_time = datetime.now() - timedelta(minutes=age_minutes)
        containers_to_remove = []

        for line in lines:
            parts = line.split("\t")
            if len(parts) < 2:
                continue

            container_id = parts[0]
            container_name = parts[1] if len(parts) > 1 else ""

            # If experiment specified, filter by name
            if experiment and experiment not in container_name:
                continue

            # If no experiment filter, use age-based filtering
            # Parse created time (format: "2026-01-16 00:22:33 +0000 UTC")
            if len(parts) > 2 and not experiment:
                try:
                    created_str = parts[2].split("+")[0].strip()
                    created_time = datetime.strptime(created_str, "%Y-%m-%d %H:%M:%S")
                    if created_time > cutoff_time:
                        continue  # Too recent, skip
                except (ValueError, IndexError):
                    pass  # Can't parse time, include it

            containers_to_remove.append(container_id)

        if not containers_to_remove:
            return 0

        # Remove in batches to avoid command line limits
        batch_size = 50
        total_removed = 0
        for i in range(0, len(containers_to_remove), batch_size):
            batch = containers_to_remove[i : i + batch_size]
            subprocess.run(["docker", "rm", "-f"] + batch, capture_output=True, timeout=60)
            total_removed += len(batch)

        return total_removed
    except Exception:
        return 0


def get_paths(experiment: str):
    """Get output paths for an experiment."""
    return SYNTH_OUTPUTS / experiment, TAC_TASKS / "underspecified" / experiment


def _process_single_task(args: tuple) -> List[Dict[str, Any]]:
    """Process a single task for variant generation (for parallel execution).

    Returns list of variant dicts, or empty list on error.
    """
    (
        task,
        use_trajectory,
        use_checkpoints,
        max_level,
        max_variants_per_level,
        max_segments,
        extraction_model,
        generation_model,
        severity_str,
        selection_strategy,
        trajectory_dir,
    ) = args

    # Import here to avoid issues with multiprocessing
    sys.path.insert(0, str(REPO_ROOT))
    from synthetic.adapters.tac import TACAdapter
    from synthetic.pipeline import Severity, SyntheticPipeline

    # Parse severity
    severity = Severity(severity_str) if severity_str else Severity.DELETE

    # Create fresh pipeline instance per task (thread-safe)
    kwargs = {}
    if extraction_model:
        kwargs["extraction_model"] = extraction_model
    if generation_model:
        kwargs["generation_model"] = generation_model
    pipeline = SyntheticPipeline(**kwargs)
    adapter = TACAdapter()

    variants = []
    try:
        trajectory = adapter.get_trajectory_path(task, trajectory_dir) if use_trajectory else None
        result = pipeline.process_tac_task(
            task,
            trajectory_path=trajectory,
            use_checkpoints=use_checkpoints,
            severity=severity,
            max_segments=max_segments,
            max_level=max_level,
            max_variants_per_level=max_variants_per_level,
            selection_strategy=selection_strategy,
        )

        # Show segment statistics
        total_segments = len(result.segments)
        n_traj_grounded = sum(1 for s in result.segments if s.is_used_in_trajectory)
        n_ckpt_grounded = sum(1 for s in result.segments if s.checkpoint_refs)
        avg_priority = (
            sum(s.priority_score for s in result.segments) / total_segments if total_segments else 0
        )
        print(
            f"✓ {task}: {total_segments} segments ({n_traj_grounded} traj, {n_ckpt_grounded} ckpt), "
            f"{len(result.variants)} variants (avg priority: {avg_priority:.2f})"
        )

        for i, v in enumerate(result.variants, 1):
            # Handle both single and multi-segment variants
            n_removed_segs = len(v.removed_segments)
            if n_removed_segs == 1:
                seg = v.removed_segments[0]
                dim_str = seg.dimension.value
                subdim_str = seg.subdimension
                removed_str = seg.value
                cp_refs = ",".join(seg.checkpoint_refs) if seg.checkpoint_refs else ""
                traj_used = seg.is_used_in_trajectory
                first_use = seg.first_use_pct
                crit = seg.criticality
                guess = seg.guessability
            else:
                # Multi-segment: aggregate info
                dim_str = "+".join(s.dimension.value for s in v.removed_segments)
                subdim_str = "+".join(s.subdimension for s in v.removed_segments if s.subdimension)
                removed_str = " | ".join(str(s.value) for s in v.removed_segments)
                all_cp_refs = set()
                for s in v.removed_segments:
                    all_cp_refs.update(s.checkpoint_refs or [])
                cp_refs = ",".join(sorted(all_cp_refs))
                traj_used = any(s.is_used_in_trajectory for s in v.removed_segments)
                first_use = min(
                    (s.first_use_pct for s in v.removed_segments if s.first_use_pct is not None),
                    default=None,
                )
                crit = max(s.criticality for s in v.removed_segments)  # Any critical part matters
                guess = sum(s.guessability for s in v.removed_segments) / len(
                    v.removed_segments
                )  # Average

            variants.append(
                {
                    "task_name": task,
                    "variant_id": f"{task}_V{i}_{dim_str}",
                    "dimension": dim_str,
                    "subdimension": subdim_str,
                    "removed_value": removed_str,
                    "removed_segments_json": json.dumps(
                        [s.to_dict() for s in v.removed_segments]
                    ),  # Source of truth with segment IDs
                    "num_segments": n_removed_segs,
                    "original_prompt": result.task.prompt,
                    "underspecified_prompt": v.underspecified_prompt,
                    "severity": v.severity.value if v.severity else "delete",
                    "is_used_in_trajectory": traj_used,
                    "first_use_pct": first_use,
                    "checkpoint_refs": cp_refs,
                    "criticality": crit,
                    "guessability": guess,
                    "priority_score": v.predicted_difficulty,
                    "expected_questions": (
                        json.dumps(v.expected_questions) if v.expected_questions else "[]"
                    ),
                    "expected_failure_mode": v.expected_failure_mode or "",
                }
            )
    except Exception as e:
        print(f"✗ {task}: ERROR - {e}")
        traceback.print_exc()

    return variants


def generate(
    tasks: List[str],
    experiment: str,
    max_variants_per_level: Optional[int] = None,
    max_segments: Optional[int] = None,
    max_level: int = 1,
    use_trajectory: bool = True,
    use_checkpoints: bool = True,
    extraction_model: Optional[str] = None,
    generation_model: Optional[str] = None,
    generate_workers: int = 4,
    severity: str = "delete",
    selection_strategy: str = "auto",
    trajectory_dir: Optional[str] = None,
) -> None:
    """Generate underspecified variants for tasks.

    Args:
        generate_workers: Concurrent tasks during variant generation (default: 4).
                          Each task makes LLM calls; this controls parallelism of those calls.
        severity: Removal strategy - delete (default), vaguify, or genericize.
        selection_strategy: Within-level combo ranking (cross-level uses mean priority):
            - auto: grounding-weighted with mean priority tiebreaker (RECOMMENDED)
            - priority_mean: mean of segment priorities
            - priority_max: max segment priority (legacy)
    """
    sys.path.insert(0, str(REPO_ROOT))

    print(
        f"Generating variants for {len(tasks)} tasks (workers={generate_workers}, severity={severity}, strategy={selection_strategy})"
    )

    # Build args for each task
    task_args = [
        (
            task,
            use_trajectory,
            use_checkpoints,
            max_level,
            max_variants_per_level,
            max_segments,
            extraction_model,
            generation_model,
            severity,
            selection_strategy,
            trajectory_dir,
        )
        for task in tasks
    ]

    all_variants = []

    if generate_workers > 1:
        # Parallel execution using ThreadPoolExecutor (I/O-bound LLM calls)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=generate_workers) as executor:
            futures = {executor.submit(_process_single_task, args): args[0] for args in task_args}
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    variants = future.result()
                    all_variants.extend(variants)
                except Exception as e:
                    print(f"✗ {task_name}: FAILED - {e}")
    else:
        # Sequential execution
        for args in task_args:
            variants = _process_single_task(args)
            all_variants.extend(variants)

    if not all_variants:
        print("No variants generated!")
        return

    # Save
    synth_out, task_out = get_paths(experiment)
    synth_out.mkdir(parents=True, exist_ok=True)
    task_out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_variants)
    df.to_csv(synth_out / "underspec_candidates.csv", index=False, encoding="utf-8")

    # Persist generation parameters for reproducibility
    metadata = {
        "stage": "generate",
        "experiment": experiment,
        "tasks": tasks,
        "severity": severity,
        "max_level": max_level,
        "max_variants_per_level": max_variants_per_level,
        "max_segments": max_segments,
        "use_trajectory": use_trajectory,
        "use_checkpoints": use_checkpoints,
        "trajectory_dir": trajectory_dir,
        "selection_strategy": selection_strategy,
        "extraction_model": extraction_model,
        "generation_model": generation_model,
        "n_variants": len(df),
        "n_tasks": int(df["task_name"].nunique()),
        "created_at": datetime.now().isoformat(),
    }
    with open(synth_out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("📊 GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Tasks processed: {df['task_name'].nunique()}")
    print(f"  Total variants: {len(df)}")
    print(f"  Trajectory-grounded: {df['is_used_in_trajectory'].sum():.0f}")
    print(f"  Checkpoint-grounded: {(df['checkpoint_refs'] != '').sum()}")
    print(f"  Avg priority score: {df['priority_score'].mean():.2f}")
    print(f"{'='*60}")
    print(f"✅ Saved to {synth_out / 'underspec_candidates.csv'}")

    # Create task directories
    for _, row in df.iterrows():
        _create_variant_dir(row.to_dict(), task_out)


def _create_variant_dir(variant: Dict[str, Any], task_out: Path) -> None:
    """Create variant directory with modified task.json."""
    task, vid = variant["task_name"], variant["variant_id"]
    orig = TAC_TASKS / task
    dest = task_out / vid / task

    if dest.parent.exists():
        shutil.rmtree(dest.parent)
    dest.mkdir(parents=True)

    # Modified task.json
    with open(orig / "task.json") as f:
        data = json.load(f)
    # Keep original task as primary_task (for user simulator ground truth)
    # Set underspecified_task to modified prompt (what agent sees when format_name contains "underspecified")
    data["primary_task"] = variant["original_prompt"]
    data["underspecified_task"] = variant["underspecified_prompt"]
    data["_variant_id"] = vid

    # Write removed_segments so the MCP user simulator knows what was removed
    raw_segments = variant.get("removed_segments_json", "")
    if raw_segments:
        try:
            full_segments = (
                json.loads(raw_segments) if isinstance(raw_segments, str) else raw_segments
            )
            data["removed_segments"] = [
                {"dimension": seg.get("dimension", ""), "value": seg.get("value", "")}
                for seg in full_segments
                if seg.get("value")
            ]
        except (json.JSONDecodeError, TypeError):
            pass

    with open(dest / "task.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Symlink other files
    for src in orig.iterdir():
        if src.name != "task.json":
            dst = dest / src.name
            if not dst.exists():
                dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())


def _run_single_variant(args: tuple) -> str:
    """Run a single variant (for parallel execution)."""
    (
        experiment,
        vid,
        task,
        variant_dir,
        model_full,
        num_trials,
        include_user_help,
        clarification_prompt,
        agent_strategy,
    ) = args

    if not variant_dir.exists():
        return f"SKIP {vid}: directory not found"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_suffix = (
        f"_{clarification_prompt}"
        if clarification_prompt and not experiment.endswith(f"_{clarification_prompt}")
        else ""
    )
    strategy_suffix = (
        f"_{agent_strategy}"
        if agent_strategy and not experiment.endswith(f"_{agent_strategy}")
        else ""
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / "task_completion_agentcompany.py"),
        "--format_name",
        f"underspecified_{experiment}{prompt_suffix}{strategy_suffix}_{vid}_{ts}",
        "--backend_model",
        model_full,
        "--num_trials",
        str(num_trials),
        "--task_names",
        task,
        "--tasks_dir",
        str(variant_dir),
    ]
    if include_user_help:
        cmd.append("--include_user_help")
    if clarification_prompt:
        cmd.extend(["--clarification_prompt", clarification_prompt])
    if agent_strategy:
        cmd.extend(["--agent_strategy", agent_strategy])

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    return f"✅ {vid}" if result.returncode == 0 else f"❌ {vid}: {result.returncode}"


def _count_sessions_in_run(run_dir: Path, task: str) -> int:
    """Count actual session directories (not evaluation files)."""
    local_results = run_dir / "local_results" / task
    if not local_results.exists():
        return 0
    session_count = 0
    try:
        for model_dir in local_results.iterdir():
            if not model_dir.is_dir():
                continue
            # Use regex to match YYYY-MM-DD format (future-proof, not hardcoded to 2026)
            session_count += sum(
                1
                for d in model_dir.iterdir()
                if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)
            )
    except Exception:
        pass  # Best-effort count
    return session_count


def _collect_completed_variants(
    df: pd.DataFrame,
    experiment: str,
    num_trials: int,
    clarification_prompt: Optional[str] = None,
    model_short: Optional[str] = None,
) -> tuple:
    """Scan RUNS_DIR for completed/incomplete variants.

    Args:
        df: DataFrame containing variant information.
        experiment: Experiment name.
        num_trials: Number of trials to run for each variant.
        clarification_prompt: Clarification prompt to use.
        model_short: If provided, only count evals under this model's subdirectory
                     (e.g. "claude-opus-4-5-20251101"). Prevents cross-model resume
                     collisions when the same experiment name is reused.

    Returns:
        (complete_vids, incomplete_vids, mismatched_vids) — all sets of variant IDs.
    """
    valid_vids = set(df["variant_id"].tolist())

    exp_patterns = [experiment]
    if clarification_prompt and not experiment.endswith(f"_{clarification_prompt}"):
        exp_patterns.append(f"{experiment}_{clarification_prompt}")

    # vid -> [(timestamp, run_dir, n_evals, n_sessions, task), ...]
    runs_by_vid = defaultdict(list)

    if not RUNS_DIR.exists():
        print("No previous runs directory found; resume has nothing to skip.")
        return set(), set(), set()

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        for exp_pattern in exp_patterns:
            meta = parse_run_dir(run_dir.name, exp_pattern, valid_vids)
            if meta:
                vid = meta["variant_id"]
                ts = meta["timestamp"]
                task = meta["base_task"]
                # Files are at: final_results/deterministic/{task}/{model}/evaluation_*.json
                eval_dir = run_dir / "final_results" / "deterministic" / task
                n_evals = 0
                if eval_dir.exists():
                    if model_short:
                        model_eval_dir = eval_dir / model_short
                        if model_eval_dir.exists():
                            n_evals = len(list(model_eval_dir.glob("evaluation_*.json")))
                    else:
                        n_evals = len(list(eval_dir.rglob("evaluation_*.json")))
                n_sessions = _count_sessions_in_run(run_dir, task)
                runs_by_vid[vid].append((ts, run_dir, n_evals, n_sessions, task))
                break

    # For each variant, find LATEST run and check completeness
    # A variant is complete only if BOTH evals >= num_trials AND sessions >= num_trials
    complete_vids = set()
    incomplete_vids = set()
    mismatched_vids = set()  # Sessions != evals (data integrity issue)

    for vid, runs in runs_by_vid.items():
        # Sort by timestamp descending, skip empty runs (0 sessions AND 0 evals)
        # This handles interrupted runs that create empty directories
        sorted_runs = sorted(runs, key=lambda x: x[0], reverse=True)
        latest = None
        for run in sorted_runs:
            if run[2] > 0 or run[3] > 0:  # n_evals > 0 or n_sessions > 0
                latest = run
                break

        # If all runs are empty, skip this variant (will be run fresh)
        if latest is None:
            continue

        n_evals = latest[2]
        n_sessions = latest[3]

        # Check for session/eval mismatch (data integrity issue)
        if n_sessions > 0 and n_sessions != n_evals:
            mismatched_vids.add(vid)
            # Treat as incomplete - needs re-run
            incomplete_vids.add(vid)
        elif n_sessions >= num_trials and n_evals >= num_trials:
            complete_vids.add(vid)
        elif n_sessions > 0 or n_evals > 0:
            incomplete_vids.add(vid)

    return complete_vids, incomplete_vids, mismatched_vids


def _filter_for_resume(
    df: pd.DataFrame,
    complete_vids: set,
    incomplete_vids: set,
    mismatched_vids: set,
    num_trials: int,
) -> Optional[pd.DataFrame]:
    """Filter out complete variants and print resume status messages.

    Returns the filtered DataFrame, or None if all variants are already complete.
    """
    if not (complete_vids or incomplete_vids or mismatched_vids):
        return df

    original_count = len(df)
    df = df[~df["variant_id"].isin(complete_vids)]
    skipped = original_count - len(df)
    if skipped > 0:
        print(f"⏭️  Skipping {skipped} complete variants ({num_trials}/{num_trials} sessions)")
    if mismatched_vids:
        # These have session/eval mismatch - data integrity issue, will re-run
        mismatched_in_df = mismatched_vids & set(df["variant_id"].tolist())
        if mismatched_in_df:
            print(f"⚠️  {len(mismatched_in_df)} variants have session/eval mismatch - will re-run")
    if incomplete_vids - mismatched_vids:
        # These have some trials but not all - will be re-run
        incomplete_only = (incomplete_vids - mismatched_vids) & set(df["variant_id"].tolist())
        if incomplete_only:
            print(f"🔄 {len(incomplete_only)} variants have incomplete trials - will re-run")

    if df.empty:
        print("✅ All variants already complete!")
        return None

    return df


def run(
    experiment: str,
    variant_id: Optional[str] = None,
    num_trials: int = 3,
    model: str = "opus_4_5",
    include_user_help: bool = False,
    parallel_variants: int = 1,
    clarification_prompt: Optional[str] = None,
    resume: bool = False,
    agent_strategy: Optional[str] = None,
) -> None:
    """Run agents on variants.

    Args:
        experiment: Experiment name.
        variant_id: Optional variant ID to run.
        num_trials: Number of trials to run for each variant, default is 3.
        model: Model to use, default is opus_4_5.
        include_user_help: Whether to include user help.
        parallel_variants: Number of variants to run in parallel, default is 1 (sequential).
                          Higher values speed up total time but require more resources.
        clarification_prompt: Prompt variant for clarification, default is None.
                             Only used if include_user_help=True.
        resume: If True, skip variants that already have all num_trials evaluations complete, default is False.
        agent_strategy: Agent prompting strategy (react, reflexion, plan_and_execute, none).
                       Copies strategy as AGENTS.md to workspace.
    """
    synth_out, task_out = get_paths(experiment)
    csv_path = synth_out / "underspec_candidates.csv"

    if not csv_path.exists():
        print(f"No candidates. Run --generate --experiment {experiment} first.")
        return

    df = pd.read_csv(csv_path)
    model_full = get_model(model)

    if variant_id:
        df = df[df["variant_id"] == variant_id]
        if df.empty:
            print(f"Variant '{variant_id}' not found.")
            return

    if resume:
        complete_vids, incomplete_vids, mismatched_vids = _collect_completed_variants(
            df,
            experiment,
            num_trials,
            clarification_prompt,
            model_short=model_full.split("/")[-1],
        )
        df = _filter_for_resume(df, complete_vids, incomplete_vids, mismatched_vids, num_trials)
        if df is None:
            return

    total_runs = len(df) * num_trials
    print(f"Running {len(df)} variants × {num_trials} trials = {total_runs} total runs")
    print(f"Model: {model}, Parallel variants: {parallel_variants}\n")

    # Persist run parameters for reproducibility
    run_metadata = {
        "stage": "run",
        "experiment": experiment,
        "model": model,
        "model_full": model_full,
        "num_trials": num_trials,
        "parallel_variants": parallel_variants,
        "include_user_help": include_user_help,
        "clarification_prompt": clarification_prompt,
        "agent_strategy": agent_strategy,
        "resume": resume,
        "n_variants": len(df),
        "started_at": datetime.now().isoformat(),
    }
    with open(synth_out / f"run_metadata_{model}.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    # Initial cleanup - only clean old containers (30+ min old) to avoid affecting other runs
    removed = cleanup_docker_containers(age_minutes=30)
    if removed > 0:
        print(f"🧹 Cleaned up {removed} old containers (>30 min) before starting")

    if parallel_variants > 1:
        # Parallel variant execution
        from concurrent.futures import ProcessPoolExecutor, as_completed

        args_list = [
            (
                experiment,
                row["variant_id"],
                row["task_name"],
                task_out / row["variant_id"],
                model_full,
                num_trials,
                include_user_help,
                clarification_prompt,
                agent_strategy,
            )
            for _, row in df.iterrows()
        ]

        completed_count = 0
        with ProcessPoolExecutor(max_workers=parallel_variants) as executor:
            futures = {
                executor.submit(_run_single_variant, args): args[1] for args in args_list
            }  # args[1] is vid
            pbar = tqdm(as_completed(futures), total=len(futures), desc="Variants", unit="var")
            for future in pbar:
                vid = futures[future]
                completed_count += 1
                try:
                    result = future.result()
                    status = "✅" if "✅" in result else "❌"
                    pbar.set_postfix_str(f"{status} {vid[:30]}")
                except Exception:
                    pbar.set_postfix_str(f"❌ {vid[:30]}")

                # Periodic cleanup - only clean containers from THIS experiment
                if completed_count % CLEANUP_INTERVAL == 0:
                    removed = cleanup_docker_containers(experiment=experiment, age_minutes=5)
                    if removed > 0:
                        tqdm.write(f"🧹 Cleaned up {removed} containers from {experiment}")
    else:
        # Sequential execution
        completed_count = 0
        pbar = tqdm(df.iterrows(), total=len(df), desc="Variants", unit="var")
        for _, row in pbar:
            vid = row["variant_id"]
            pbar.set_description(f"Running {vid[:35]}")
            args = (
                experiment,
                vid,
                row["task_name"],
                task_out / vid,
                model_full,
                num_trials,
                include_user_help,
                clarification_prompt,
                agent_strategy,
            )
            result = _run_single_variant(args)
            status = "✅" if "✅" in result else "❌"
            pbar.set_postfix_str(status)
            completed_count += 1

            # Periodic cleanup - only clean containers from THIS experiment
            if completed_count % CLEANUP_INTERVAL == 0:
                removed = cleanup_docker_containers(experiment=experiment, age_minutes=5)
                if removed > 0:
                    tqdm.write(f"🧹 Cleaned up {removed} containers from {experiment}")

    # Final cleanup - clean THIS experiment's containers
    removed = cleanup_docker_containers(experiment=experiment, age_minutes=1)
    if removed > 0:
        print(f"🧹 Final cleanup: removed {removed} containers from {experiment}")


def parse_run_dir(name: str, experiment: str, valid_vids: set) -> Optional[Dict[str, Any]]:
    """Parse run directory name to extract variant info.

    Expected format: run_underspecified_{experiment}[_{prompt}][_{strategy}]_{variant_id}_{YYYYMMDD}_{HHMMSS}_{YYYYMMDD}_{HHMMSS}[_{uuid8}]

    Finds variant IDs anywhere after the experiment prefix, so it works
    regardless of prompt/strategy tokens in the directory name.

    Args:
        name: Directory name to parse
        experiment: Required experiment name to filter by
        valid_vids: Set of valid variant IDs to match against

    Returns dict with variant_id, base_task, dimension, timestamp
    or None if not a valid run directory.
    """
    prefix = f"run_underspecified_{experiment}_"
    if not name.startswith(prefix):
        return None

    suffix = name[len(prefix) :]

    for vid in sorted(valid_vids, key=len, reverse=True):
        marker = f"{vid}_"
        idx = suffix.find(marker)
        if idx < 0:
            continue
        rest = suffix[idx + len(marker) :]
        m = re.match(r"(\d{8}_\d{6})_(\d{8}_\d{6}(?:_[0-9a-f]{8})?)$", rest)
        if m:
            parts = vid.rsplit("_", 2)
            if len(parts) >= 3:
                return {
                    "variant_id": vid,
                    "base_task": "_".join(parts[:-2]),
                    "dimension": parts[-1],
                    "timestamp": m.group(2),
                }
    return None


def results(experiment: str) -> None:
    """Show pass rates. 0% = outcome-critical."""
    synth_out, _ = get_paths(experiment)
    csv_path = synth_out / "underspec_candidates.csv"

    if not csv_path.exists():
        print(f"No candidates. Run --generate --experiment {experiment} first.")
        return

    valid = set(pd.read_csv(csv_path)["variant_id"].tolist())
    rows = []

    for run_dir in RUNS_DIR.glob(f"run_underspecified_{experiment}*"):
        parsed = parse_run_dir(run_dir.name, experiment, valid)
        if not parsed:
            continue

        vid, timestamp = parsed["variant_id"], parsed["timestamp"]

        summaries = list(run_dir.glob("final_results/summaries/deterministic_summary_*.csv"))
        if not summaries:
            continue

        for summary in summaries:
            for _, row in pd.read_csv(summary).iterrows():
                rows.append(
                    {
                        "variant_id": vid,
                        "task": row.get("task", "unknown"),
                        "percent": row.get("percent", 0),
                        "timestamp": timestamp,
                    }
                )

    if not rows:
        print("No results. Run --run first.")
        return

    df = pd.DataFrame(rows)

    # Keep latest per variant (sort by timestamp descending)
    df = df.sort_values("timestamp", ascending=False).drop_duplicates("variant_id", keep="first")

    print(f"Found results for {len(df)} variants")

    # Quick summary only - point to process_tac_underspec.py for full analysis
    print("\n⚠️  For detailed results with correct outcome-critical analysis, use:")
    print(f"   python scripts/process_tac_underspec.py -e {experiment}")
    print("\n   (This script only shows a quick task-level summary)\n")

    agg = df.groupby("task").agg({"percent": ["mean", "count"]}).reset_index()
    agg.columns = ["task", "avg_percent", "n_variants"]

    print(f"Task Summary ({len(agg)} tasks):")
    print(agg.to_string(index=False))


def load_filtered_samples(samples_path: str, experiment: str) -> None:
    """Load filtered (benchmark-ready) samples (skip LLM generation).

    Creates standalone task directories with:
    - task.json: primary_task (ground truth) + underspecified_task (agent sees) + _variant_info
    - evaluator.py: copied from original task
    - checkpoints.md: copied from original task

    Usage:
        python scripts/run_tac_underspec.py --load-samples filtered_samples.json -e benchmark
        python scripts/run_tac_underspec.py --run -e benchmark --include-user-help
    """
    samples_file = Path(samples_path)
    if not samples_file.exists():
        print(f"❌ Samples file not found: {samples_path}")
        return

    with open(samples_file) as f:
        samples = json.load(f)

    print(
        f"Loading {len(samples)} filtered (benchmark-ready) samples into experiment '{experiment}'"
    )

    synth_out, task_out = get_paths(experiment)
    synth_out.mkdir(parents=True, exist_ok=True)
    task_out.mkdir(parents=True, exist_ok=True)

    variants = []
    for sample in samples:
        required_keys = {
            "variant_id",
            "task",
            "agent_prompt",
            "ambiguity_type",
            "ambiguity_class",
            "removed_segments",
        }
        missing = sorted(k for k in required_keys if k not in sample)
        if missing:
            raise ValueError(f"Sample missing required keys: {', '.join(missing)}")

        vid = sample["variant_id"]
        task = sample["task"]
        orig_task_dir = TAC_TASKS / task

        if not orig_task_dir.exists():
            print(f"  ⚠️ Skip {vid}: task {task} not found")
            continue

        # Load original task.json for primary_task
        with open(orig_task_dir / "task.json") as f:
            orig_data = json.load(f)
        original_prompt = orig_data.get("primary_task", orig_data.get("task", ""))

        # Create variant directory: task_out/variant_id/task_name/
        dest = task_out / vid / task
        if dest.parent.exists():
            shutil.rmtree(dest.parent)
        dest.mkdir(parents=True)

        # Write task.json with both prompts
        # removed_segments at TOP LEVEL for mcp_server.py to extract removed values
        removed_segments = sample["removed_segments"]
        if not removed_segments:
            raise ValueError(f"Sample {vid} has no removed_segments")

        # Simplify to just dimension + value (mcp_server only needs these)
        removed_segments_simplified = [
            {"dimension": seg["dimension"], "value": seg["value"]}
            for seg in removed_segments
            if seg["value"]
        ]

        task_data = {
            "primary_task": original_prompt,
            "underspecified_task": sample["agent_prompt"],
            "_variant_id": vid,
            "removed_segments": removed_segments_simplified,  # TOP LEVEL for mcp_server.py
        }
        with open(dest / "task.json", "w", encoding="utf-8") as f:
            json.dump(task_data, f, indent=2)

        # Copy everything from original task EXCEPT task.json (we wrote our own)
        for item in orig_task_dir.iterdir():
            if item.name == "task.json" or item.name == "__pycache__":
                continue  # Skip task.json (we wrote our own) and pycache
            dst = dest / item.name
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)

        # Build removed_value string from segments
        removed_value = " | ".join(
            f"[{seg['dimension']}] {seg['value']}" for seg in removed_segments
        )

        criticality = sample.get("criticality")
        guessability = sample.get("guessability")
        priority_score = sample.get("priority_score")
        if criticality is None or guessability is None or priority_score is None:
            criticality = max((seg.get("criticality", 0) for seg in removed_segments), default=0)
            guessability = sum(seg.get("guessability", 0) for seg in removed_segments) / max(
                1, len(removed_segments)
            )
            seg_priorities = [
                seg.get(
                    "priority_score",
                    seg.get("criticality", 0) * (1 - seg.get("guessability", 0)),
                )
                for seg in removed_segments
            ]
            priority_score = sum(seg_priorities) / max(1, len(seg_priorities))

        variants.append(
            {
                "task_name": task,
                "variant_id": vid,
                "dimension": sample["ambiguity_type"],
                "original_prompt": original_prompt,
                "underspecified_prompt": sample["agent_prompt"],
                "ambiguity_class": sample["ambiguity_class"],
                "priority_score": priority_score,
                "criticality": criticality,
                "guessability": guessability,
                "removed_value": removed_value,
                "removed_segments_json": json.dumps(removed_segments),
            }
        )

    if not variants:
        print("❌ No valid samples were loaded; no candidates CSV written.")
        return

    df = pd.DataFrame(variants)
    df.to_csv(synth_out / "underspec_candidates.csv", index=False, encoding="utf-8")

    # Persist load parameters for reproducibility
    metadata = {
        "stage": "load_samples",
        "experiment": experiment,
        "source_file": str(samples_file),
        "n_variants": len(variants),
        "n_tasks": df["task_name"].nunique(),
        "created_at": datetime.now().isoformat(),
    }
    with open(synth_out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Loaded {len(variants)} variants")
    print(f"   CSV: {synth_out / 'underspec_candidates.csv'}")
    print(f"   Tasks: {task_out}")
    print(f"\nNext: python scripts/run_tac_underspec.py --run -e {experiment} --include-user-help")


def main():
    p = argparse.ArgumentParser(description="Underspec Evaluation")

    # Actions
    act = p.add_mutually_exclusive_group()
    act.add_argument("--all", action="store_true", help="Generate + run + results")
    act.add_argument("--generate", action="store_true", help="Generate variants")
    act.add_argument(
        "--load-samples",
        metavar="PATH",
        help="Load filtered (benchmark-ready) samples JSON (skip LLM generation)",
    )
    act.add_argument("--run", action="store_true", help="Run agents")
    act.add_argument("--results", action="store_true", help="Show results")

    # Common
    p.add_argument("-e", "--experiment", default="default", help="Experiment name")
    p.add_argument("--tasks", nargs="+", help="Tasks to process")

    # Generate
    p.add_argument(
        "--max-segments", type=int, help="Limit total segments considered (top-k by priority)"
    )
    p.add_argument("--max-variants-per-level", type=int, help="Max variants per level (per task)")
    p.add_argument(
        "--max-level",
        type=int,
        default=DEFAULTS["max_level"],
        help=f"Max segments to remove together (default: {DEFAULTS['max_level']})",
    )
    p.add_argument(
        "--trajectory-dir",
        help="Local directory with golden trajectory JSON files (from export_tac_golden_trajectories.py). "
        "Checked before falling back to S3.",
    )
    p.add_argument("--prompt-only", action="store_true", help="No grounding (ablation)")
    p.add_argument("--no-trajectory", action="store_true", help="No trajectory grounding")
    p.add_argument("--no-checkpoints", action="store_true", help="No checkpoint grounding")
    p.add_argument(
        "--extraction-model", help="Override extraction model (default: gemini-3-flash in pipeline)"
    )
    p.add_argument(
        "--generation-model", help="Override generation model (default: gemini-3-flash in pipeline)"
    )
    p.add_argument(
        "--severity",
        choices=["delete", "vaguify", "genericize"],
        default=DEFAULTS["severity"],
        help=f"Removal strategy (default: {DEFAULTS['severity']})",
    )
    p.add_argument(
        "--selection-strategy",
        choices=["auto", "priority_max", "priority_mean"],
        default="auto",
        help="Within-level combo ranking (default: auto). "
        "auto: grounding-weighted with mean priority tiebreaker (RECOMMENDED). "
        "priority_mean: mean of segment priorities. "
        "priority_max: max segment priority (legacy).",
    )
    # Run
    p.add_argument("--variant", help="Specific variant to run")
    p.add_argument(
        "--include-user-help", action="store_true", help="Include the 'request_clarification' tool"
    )
    p.add_argument(
        "--clarification-prompt",
        default="default",
        help="Clarification prompt key (default: 'default'). Use 'none' for no prompt.",
    )
    p.add_argument(
        "--agent-strategy",
        type=str,
        default=None,
        choices=["none", "react", "reflexion", "plan_and_execute"],
        help="Agent prompting strategy (for ablation_agent_prompt). Copies strategy as AGENTS.md. "
        "Options: none (baseline), react, reflexion, plan_and_execute.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: skip variants with complete trials (all num_trials evals exist)",
    )
    p.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULTS["num_trials"],
        help=f"Trials per variant (default: {DEFAULTS['num_trials']} for pass@k)",
    )
    p.add_argument(
        "--model", default=DEFAULTS["model"], help=f"Model (default: {DEFAULTS['model']})"
    )
    p.add_argument(
        "--generate-workers",
        type=int,
        default=4,
        help="Concurrent tasks during --generate (parallel LLM calls, default: 4).",
    )
    p.add_argument(
        "--parallel-variants",
        type=int,
        default=DEFAULTS["parallel_variants"],
        help=f"Concurrent variants during --run (default: {DEFAULTS['parallel_variants']}).",
    )
    args = p.parse_args()

    # Handle zsh word-splitting: $VAR without quotes passes as one string in zsh
    if args.tasks:
        args.tasks = [t for arg in args.tasks for t in arg.split()]

    use_traj = not (args.no_trajectory or args.prompt_only)
    use_ckpt = not (args.no_checkpoints or args.prompt_only)

    # Generate
    if args.generate or args.all:
        if not args.tasks:
            p.error("--generate/--all requires --tasks")
        generate(
            args.tasks,
            args.experiment,
            args.max_variants_per_level,
            args.max_segments,
            args.max_level,
            use_traj,
            use_ckpt,
            args.extraction_model,
            args.generation_model,
            args.generate_workers,
            args.severity,
            args.selection_strategy,
            (
                str(REPO_ROOT / args.trajectory_dir)
                if args.trajectory_dir and not Path(args.trajectory_dir).is_absolute()
                else args.trajectory_dir
            ),
        )

    # Load filtered (benchmark-ready) samples (alternative to --generate)
    if args.load_samples:
        load_filtered_samples(args.load_samples, args.experiment)

    # Run
    if args.run or args.all:
        run(
            args.experiment,
            args.variant,
            args.num_trials,
            args.model,
            args.include_user_help,
            args.parallel_variants,
            args.clarification_prompt if args.include_user_help else None,
            args.resume,
            args.agent_strategy,
        )

    # Results
    if args.results or args.all:
        results(args.experiment)


if __name__ == "__main__":
    main()
