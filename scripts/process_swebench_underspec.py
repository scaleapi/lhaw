#!/usr/bin/env python3
"""
Process SWE-bench underspecification experiment results.

This script processes results from SWE-bench experiments to:
1. Run evaluation on patches if needed (using swe_bench_pro_eval.py)
2. Compute terminal states from F2P test results (for divergence analysis)
3. Use F2P tests (only) for overall pass/fail determination
4. Classify variants into ambiguity classes (outcome-critical, divergent, benign, new_task)

The key difference from TAC (process_tac_underspec.py):
- TAC uses checkpoints for terminal states
- SWE-bench uses F2P (fail_to_pass) tests for terminal states
- SWE-bench uses F2P-only for overall pass/fail (P2P regressions are orthogonal)

Pipeline position:
    task_completion_swebench.py --generate  (produces underspec_candidates.csv + instances.yaml)
    task_completion_swebench.py --run       (produces exp_1/, exp_2/, exp_3/)
    >>> THIS SCRIPT <<<                     (produces underspec_results.csv)
    filter_swebench_samples.py              (produces underspec_results_filtered.csv)
    export_swebench_dataset.py              (produces lhaw_benchmark_swebench.json)

Usage:
    # Process results from existing experiment (evaluations must already exist)
    python scripts/process_swebench_underspec.py \
        --exp-dir experiments/swebench/runs/run_trial_v1_gpt_5_2_20260207/

    # Run evaluation first, then process
    python scripts/process_swebench_underspec.py \
        --exp-dir experiments/swebench/runs/run_trial_v1_gpt_5_2_20260207/ \
        --run-eval --dockerhub-username jefzda

    # With LLM judge for new_task detection
    python scripts/process_swebench_underspec.py \
        --exp-dir experiments/swebench/runs/run_trial_v1_gpt_5_2_20260207/ --judge
"""

import argparse
import ast
import csv
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

# Increase CSV field size limit (use safe constant to avoid C long overflow on some platforms)
csv.field_size_limit(10 * 1024 * 1024)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import parse_variant_id  # noqa: E402
from evaluation.pass_k_metrics import pass_at_k  # noqa: E402
from synthetic.llm import get_openai_client  # noqa: E402

# Default judge model
DEFAULT_JUDGE_MODEL = "gemini/gemini-3-flash-preview"

# SWE-bench Pro paths
SWEBENCH_PRO_DIR = REPO_ROOT / "swebenchpro" / "SWE-bench_Pro-os"
SWEBENCH_CSV = SWEBENCH_PRO_DIR / "swe_bench_pro_full.csv"
SCRIPTS_DIR = SWEBENCH_PRO_DIR / "run_scripts"


# =============================================================================
# Data Loading
# =============================================================================


def ensure_swebench_csv(csv_path: Path = SWEBENCH_CSV) -> Path:
    """Ensure swe_bench_pro_full.csv exists, downloading from HuggingFace if needed."""
    if csv_path.exists():
        return csv_path
    print(f"Downloading SWE-bench Pro dataset to {csv_path}...")
    from datasets import load_dataset

    ds = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
    df = ds.to_pandas()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  Saved {len(df)} tasks to {csv_path}")
    return csv_path


def load_swebench_pro_data() -> pd.DataFrame:
    """Load SWE-bench Pro dataset with F2P and P2P test info."""
    ensure_swebench_csv()
    df = pd.read_csv(SWEBENCH_CSV)
    df = df.fillna("")
    df = df.set_index("instance_id", drop=False)
    return df


def get_f2p_p2p_tests(instance_id: str, swebench_df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
    """Get F2P and P2P test names for an instance."""
    # Handle variant IDs - extract original instance ID
    original_id, _ = parse_variant_id(instance_id)

    # Handle instance_ prefix mismatch between dataset and pipeline IDs
    if original_id not in swebench_df.index:
        alt_id = original_id.removeprefix("instance_")
        if alt_id in swebench_df.index:
            original_id = alt_id
        elif f"instance_{original_id}" in swebench_df.index:
            original_id = f"instance_{original_id}"

    if original_id not in swebench_df.index:
        return set(), set()

    row = swebench_df.loc[original_id]
    try:
        f2p_tests = set(ast.literal_eval(row.get("fail_to_pass", "[]") or "[]"))
        p2p_tests = set(ast.literal_eval(row.get("pass_to_pass", "[]") or "[]"))
    except (SyntaxError, ValueError, TypeError):
        f2p_tests = set()
        p2p_tests = set()

    return f2p_tests, p2p_tests


def load_instances(exp_dir: Path) -> List[Dict[str, Any]]:
    """Load instance definitions from instances.yaml."""
    instances_path = exp_dir / "instances.yaml"
    if not instances_path.exists():
        raise FileNotFoundError(f"Instances file not found: {instances_path}")

    with open(instances_path) as f:
        return yaml.safe_load(f) or []


def load_candidates(exp_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load variant metadata from underspec_candidates.csv (from generate stage).

    This file matches the TAC schema from run_tac_underspec.py and contains
    all pipeline metadata: dimension, subdimension, removed_value, criticality,
    guessability, priority_score, removed_segments_json, expected_questions, etc.

    Returns:
        Dict mapping instance_id -> candidate row dict
    """
    candidates_path = exp_dir / "underspec_candidates.csv"
    if not candidates_path.exists():
        print(f"  Note: underspec_candidates.csv not found in {exp_dir}")
        print("  Variant metadata (dimension, criticality, etc.) will be unavailable.")
        print("  Run --generate stage to produce this file.")
        return {}

    candidates = {}
    with open(candidates_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inst_id = row.get("instance_id", "")
            if inst_id:
                candidates[inst_id] = row
    print(f"  Loaded {len(candidates)} candidates from underspec_candidates.csv")
    return candidates


def load_preds(exp_dir: Path, trial_num: int) -> Dict[str, Dict[str, Any]]:
    """Load predictions (patches) from a trial.

    Normalizes keys to have instance_ prefix (SWE-agent strips it from preds.json)
    to match instances.yaml and underspec_candidates.csv format.
    """
    preds_path = exp_dir / f"exp_{trial_num}" / "preds.json"
    if not preds_path.exists():
        return {}

    with open(preds_path) as f:
        preds = json.load(f)

    # Normalize keys: SWE-agent strips "instance_" prefix from preds.json keys
    normalized = {}
    for key, val in preds.items():
        if not key.startswith("instance_"):
            key = f"instance_{key}"
        normalized[key] = val
    return normalized


def load_final_steps_from_traj(
    exp_dir: Path, trial_num: int, instance_id: str, n_steps: int = 5
) -> str:
    """Load final steps from SWE-agent trajectory file.

    Mirrors TAC's _load_all_trial_responses methodology:
    - Extracts ALL agent messages from the trajectory
    - Filters to substantive messages (len > 30 or action == 'submit')
    - Takes the last N messages
    - Formats as: [action] message

    Args:
        exp_dir: Experiment directory
        trial_num: Trial number (1, 2, 3, etc.)
        instance_id: Instance ID to load trajectory for
        n_steps: Number of final steps to include (default: 5, same as TAC)

    Returns:
        String with last N assistant messages concatenated (same format as TAC)
    """
    # Find the trajectory file in the instance directory
    trial_dir = exp_dir / f"exp_{trial_num}"

    # Instance directories may be named "instance_{id}" or just "{id}"
    instance_dir = trial_dir / instance_id
    if not instance_dir.exists():
        instance_dir = trial_dir / f"instance_{instance_id}"

    if not instance_dir.exists():
        if instance_dir.is_symlink():
            instance_dir = instance_dir.resolve()
        else:
            return ""

    # Find .traj file (sorted for determinism when multiple exist)
    traj_files = sorted(instance_dir.glob("*.traj"))
    if not traj_files:
        return ""

    traj_file = traj_files[0]

    try:
        with open(traj_file) as f:
            traj_data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return ""

    # Extract ALL agent messages from history (mirrors TAC methodology)
    history = traj_data.get("history", [])
    agent_messages = []

    for entry in history:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") != "assistant":
            continue

        action = entry.get("action", "")
        message = action.strip() if action else ""
        if not message:
            message = entry.get("content", "").strip()

        if not message:
            continue

        # Truncate very long messages
        if len(message) > 2000:
            message = message[:2000] + "... [truncated]"

        # Filter to substantive messages (mirrors TAC: len > 30 or action == "finish")
        if len(message) > 30 or action == "submit":
            agent_messages.append(f"[{action or 'response'}] {message}")

    response = "\n\n".join(agent_messages[-n_steps:]) if agent_messages else ""
    return response


def load_eval_output(exp_dir: Path, trial_num: int, instance_id: str) -> Optional[Dict]:
    """Load evaluation output for a specific instance and trial.

    Handles variant IDs: eval results are stored under the original instance ID
    directory with a variant-aware prefix (e.g., trial1__V_S1_S2_delete_output.json).
    """
    # Extract original ID and variant suffix for path construction
    original_id, variant_suffix = parse_variant_id(instance_id)
    variant_prefix = (
        f"trial{trial_num}__V_{variant_suffix}" if variant_suffix else f"trial{trial_num}"
    )

    eval_paths = [
        # From run_combined_eval.py output
        exp_dir / f"exp_{trial_num}" / "variant_eval" / instance_id / "output.json",
        exp_dir / f"exp_{trial_num}" / "variant_eval" / instance_id / "result.json",
        # From swe_bench_pro_eval.py output (variant-aware: original_id dir, variant prefix)
        exp_dir
        / f"exp_{trial_num}"
        / "eval_results"
        / original_id
        / f"{variant_prefix}_output.json",
        # Legacy: direct instance_id lookup (for non-variant or older runs)
        exp_dir
        / f"exp_{trial_num}"
        / "eval_results"
        / instance_id
        / f"trial{trial_num}_output.json",
        exp_dir / f"exp_{trial_num}" / "eval_results" / instance_id / "_output.json",
    ]

    for path in eval_paths:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

    return None


def compute_test_results(
    eval_output: Optional[Dict],
    f2p_tests: Set[str],
    p2p_tests: Set[str],
) -> Dict[str, Any]:
    """Compute test pass/fail results from evaluation output.

    Returns:
    {
        "f2p_passed": set of passed F2P tests,
        "p2p_passed": set of passed P2P tests,
        "f2p_terminal_state": tuple of (1/0) for each F2P test (sorted),
        "overall_pass": True if all F2P tests pass (F2P-only),
        "f2p_pass_count": number of F2P tests passed,
        "p2p_pass_count": number of P2P tests passed,
    }
    """
    if not eval_output or "tests" not in eval_output:
        return {
            "f2p_passed": set(),
            "p2p_passed": set(),
            "f2p_terminal_state": tuple([0] * len(f2p_tests)) if f2p_tests else (0,),
            "overall_pass": False,
            "f2p_pass_count": 0,
            "p2p_pass_count": 0,
            "f2p_total": len(f2p_tests),
            "p2p_total": len(p2p_tests),
        }

    passed_tests = set()
    for test in eval_output.get("tests", []):
        if test.get("status") == "PASSED":
            passed_tests.add(test.get("name", ""))

    f2p_passed = f2p_tests & passed_tests
    p2p_passed = p2p_tests & passed_tests

    # Compute terminal state from F2P tests (sorted for consistency)
    f2p_sorted = sorted(f2p_tests)
    f2p_terminal_state = tuple(1 if t in f2p_passed else 0 for t in f2p_sorted)
    if not f2p_terminal_state:
        f2p_terminal_state = (1,) if not f2p_tests else (0,)

    # F2P-only: measures whether agent solved the right problem (underspec signal)
    # P2P regressions are orthogonal to information-removal treatment
    overall_pass = f2p_tests <= passed_tests

    return {
        "f2p_passed": f2p_passed,
        "p2p_passed": p2p_passed,
        "f2p_terminal_state": f2p_terminal_state,
        "overall_pass": overall_pass,
        "f2p_pass_count": len(f2p_passed),
        "p2p_pass_count": len(p2p_passed),
        "f2p_total": len(f2p_tests),
        "p2p_total": len(p2p_tests),
    }


# =============================================================================
# Evaluation Runner
# =============================================================================


def run_evaluation(
    exp_dir: Path,
    trial_num: int,
    dockerhub_username: str,
    num_workers: int = 50,
    redo: bool = False,
) -> bool:
    """Run SWE-bench Pro evaluation on a trial's predictions."""
    ensure_swebench_csv()
    preds_path = exp_dir / f"exp_{trial_num}" / "preds.json"
    eval_output_dir = exp_dir / f"exp_{trial_num}" / "eval_results"

    if not preds_path.exists():
        print(f"No predictions found: {preds_path}")
        return False

    if eval_output_dir.exists() and not redo:
        existing = list(eval_output_dir.glob("*/*_output.json"))
        if existing:
            print(f"Evaluation already exists for trial {trial_num}: {len(existing)} results")
            return True

    preds = load_preds(exp_dir, trial_num)

    patches = []
    for instance_id, pred_data in preds.items():
        patch = pred_data.get("model_patch", "")
        if patch:
            # Map variant ID to original for eval script (Docker image + test lookup)
            original_id, variant_suffix = parse_variant_id(instance_id)
            prefix = (
                f"trial{trial_num}__V_{variant_suffix}" if variant_suffix else f"trial{trial_num}"
            )
            patches.append(
                {
                    "instance_id": original_id,
                    "patch": patch,
                    "prefix": prefix,
                }
            )

    if not patches:
        print(f"No patches found in {preds_path}")
        return False

    patches_file = (exp_dir / f"exp_{trial_num}" / "patches_for_eval.json").resolve()
    with open(patches_file, "w") as f:
        json.dump(patches, f)

    eval_output_dir_abs = eval_output_dir.resolve()
    eval_script = SWEBENCH_PRO_DIR / "swe_bench_pro_eval.py"
    cmd = [
        sys.executable,
        str(eval_script),
        f"--raw_sample_path={SWEBENCH_CSV}",
        f"--patch_path={patches_file}",
        f"--output_dir={eval_output_dir_abs}",
        f"--dockerhub_username={dockerhub_username}",
        f"--scripts_dir={SCRIPTS_DIR}",
        f"--num_workers={num_workers}",
    ]
    if redo:
        cmd.append("--redo")

    print(f"\nRunning evaluation for trial {trial_num}...")
    print(f"  Command: {' '.join(str(c) for c in cmd[:6])}...")

    try:
        # Run from SWE-bench Pro dir (eval script uses relative paths to dockerfiles/)
        result = subprocess.run(cmd, check=True, cwd=str(SWEBENCH_PRO_DIR))
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
        return False


def run_baseline_evaluation(
    exp_dir: Path,
    baseline_num: int,
    dockerhub_username: str,
    num_workers: int = 50,
    redo: bool = False,
) -> bool:
    """Run SWE-bench Pro evaluation on a baseline's predictions.

    Baselines use original instance IDs (no variant suffix stripping needed).
    Output goes to baseline_N/eval_results/ with prefix 'baselineN'.
    """
    ensure_swebench_csv()
    baseline_dir = exp_dir / f"baseline_{baseline_num}"
    preds_path = baseline_dir / "preds.json"
    eval_output_dir = baseline_dir / "eval_results"

    if not preds_path.exists():
        print(f"No baseline predictions found: {preds_path}")
        return False

    if eval_output_dir.exists() and not redo:
        existing = list(eval_output_dir.glob("*/*_output.json"))
        if existing:
            print(
                f"Baseline eval already exists for baseline_{baseline_num}: {len(existing)} results"
            )
            return True

    with open(preds_path) as f:
        preds = json.load(f)

    patches = []
    for key, pred_data in preds.items():
        # Normalize key to have instance_ prefix
        instance_id = key if key.startswith("instance_") else f"instance_{key}"
        patch = pred_data.get("model_patch", "") if isinstance(pred_data, dict) else ""
        if patch:
            patches.append(
                {
                    "instance_id": instance_id,
                    "patch": patch,
                    "prefix": f"baseline{baseline_num}",
                }
            )

    if not patches:
        print(f"No patches found in {preds_path}")
        return False

    patches_file = (baseline_dir / "patches_for_eval.json").resolve()
    with open(patches_file, "w") as f:
        json.dump(patches, f)

    eval_output_dir_abs = eval_output_dir.resolve()
    eval_script = SWEBENCH_PRO_DIR / "swe_bench_pro_eval.py"
    cmd = [
        sys.executable,
        str(eval_script),
        f"--raw_sample_path={SWEBENCH_CSV}",
        f"--patch_path={patches_file}",
        f"--output_dir={eval_output_dir_abs}",
        f"--dockerhub_username={dockerhub_username}",
        f"--scripts_dir={SCRIPTS_DIR}",
        f"--num_workers={num_workers}",
    ]
    if redo:
        cmd.append("--redo")

    print(f"\nRunning baseline evaluation for baseline_{baseline_num}...")
    print(f"  Command: {' '.join(str(c) for c in cmd[:6])}...")

    try:
        result = subprocess.run(cmd, check=True, cwd=str(SWEBENCH_PRO_DIR))
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Baseline evaluation failed: {e}")
        return False


def _detect_baseline_dirs(exp_dir: Path) -> List[int]:
    """Detect baseline_N directories that have preds.json."""
    baseline_nums = []
    for d in sorted(exp_dir.glob("baseline_[0-9]*")):
        if d.is_dir() and d.name.startswith("baseline_"):
            num_str = d.name[len("baseline_") :]
            if num_str.isdigit() and (d / "preds.json").exists():
                baseline_nums.append(int(num_str))
    return baseline_nums


# =============================================================================
# LLM Judge
# =============================================================================


def get_new_task_judge_prompt(underspec_prompt: str, responses: List[str]) -> str:
    """Create the prompt for LLM judge of new-task eligibility."""
    cleaned_responses = [r.strip() for r in responses if r and r.strip()]
    if not cleaned_responses:
        cleaned_responses = ["[No response recorded]"]
    formatted_responses = "\n\n".join(
        f"Trial {idx + 1}:\n{response}" for idx, response in enumerate(cleaned_responses)
    )
    return f"""You are an impartial judge evaluating whether an AI assistant has successfully completed a task.

## Task Given to Assistant
{underspec_prompt}

## Assistant's Final Responses (All Trials)
{formatted_responses}

## Your Evaluation Task
Determine whether ALL trials reached the SAME answer/conclusion AND that answer completes the task AS SPECIFIED in the prompt above.

Note: the model should *NOT* make any assumptions about the task beyond what is explicitly stated in the prompt. If you see it making any connections that are not an explicit ask in the prompt then it should call task_completed as false.

## Response Format
Provide your evaluation in the following JSON format:
{{
    "task_completed": true/false,
    "reasoning": "Brief explanation of your verdict"
}}

Respond ONLY with the JSON object, no other text."""


def judge_new_task_sync(
    underspec_prompt: str,
    responses: List[str],
    model: str,
    max_retries: int = 2,
) -> Dict[str, Any]:
    """Use LLM to judge whether the underspecified task was completed."""
    try:
        client = get_openai_client()
    except (ImportError, ValueError) as e:
        return {"task_completed": False, "reasoning": f"Cannot initialize judge: {e}"}

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            prompt = get_new_task_judge_prompt(underspec_prompt, responses)
            if attempt > 0:
                prompt += "\n\nIMPORTANT: Your response MUST be valid JSON only."

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0 if attempt == 0 else 0.1,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content.strip()

            try:
                return json.loads(raw_content)
            except json.JSONDecodeError:
                json_match = re.search(r"\{[^{}]*\}", raw_content)
                if json_match:
                    return json.loads(json_match.group())

            last_error = f"Could not parse: {raw_content[:100]}"
        except Exception as exc:
            last_error = str(exc)

    return {"task_completed": False, "reasoning": f"Judge failed: {last_error}"}


# =============================================================================
# Main Processing
# =============================================================================


def process_experiment(
    exp_dir: Path,
    run_eval: bool = False,
    dockerhub_username: Optional[str] = None,
    num_workers: int = 50,
    judge_model: Optional[str] = None,
) -> pd.DataFrame:
    """Process SWE-bench experiment results and classify variants.

    Loads variant metadata from underspec_candidates.csv (generated by
    task_completion_swebench.py --generate) to include all pipeline metadata
    in the output (matching TAC's process_tac_underspec.py columns).
    """
    # Load SWE-bench Pro data
    print("Loading SWE-bench Pro data...")
    swebench_df = load_swebench_pro_data()

    # Load config
    config_path = exp_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    num_trials = config.get("num_trials", 3)

    # Load instances
    print("Loading instances...")
    instances = load_instances(exp_dir)
    print(f"  Found {len(instances)} instances")

    # Load variant metadata from underspec_candidates.csv
    print("Loading variant metadata...")
    candidates = load_candidates(exp_dir)

    # Create instance lookup (enriched with candidates data)
    instance_lookup = {}
    for inst in instances:
        inst_id = inst["instance_id"]

        # Start with data from candidates CSV (rich metadata)
        candidate = candidates.get(inst_id, {})

        _orig_id, _var_suffix = parse_variant_id(inst_id)
        instance_lookup[inst_id] = {
            "original_instance_id": candidate.get(
                "original_instance_id",
                _orig_id,
            ),
            "variant_id": candidate.get(
                "variant_id",
                ("V_" + _var_suffix) if _var_suffix else "original",
            ),
            # Pipeline metadata (from underspec_candidates.csv)
            "dimension": candidate.get("dimension", ""),
            "subdimension": candidate.get("subdimension", ""),
            "removed_value": candidate.get("removed_value", ""),
            "removed_segments_json": candidate.get("removed_segments_json", ""),
            "num_segments": (
                int(candidate.get("num_segments", 0)) if candidate.get("num_segments") else 0
            ),
            "severity": candidate.get("severity", "delete"),
            "criticality": (
                float(candidate.get("criticality", 0)) if candidate.get("criticality") else 0
            ),
            "guessability": (
                float(candidate.get("guessability", 0)) if candidate.get("guessability") else 0
            ),
            "priority_score": (
                float(candidate.get("priority_score", 0)) if candidate.get("priority_score") else 0
            ),
            "is_used_in_trajectory": candidate.get("is_used_in_trajectory", ""),
            "first_use_pct": candidate.get("first_use_pct", ""),
            "checkpoint_refs": candidate.get("checkpoint_refs", ""),
            "expected_questions": candidate.get("expected_questions", ""),
            "expected_failure_mode": candidate.get("expected_failure_mode", ""),
            "original_prompt": candidate.get("original_prompt", ""),
            "underspecified_prompt": candidate.get(
                "underspecified_prompt",
                inst.get("problem_statement", ""),
            ),
        }

    # Optionally run evaluation
    if run_eval:
        if not dockerhub_username:
            raise ValueError("--dockerhub-username required when running evaluation")
        eval_failures = []
        for trial_num in range(1, num_trials + 1):
            if not run_evaluation(exp_dir, trial_num, dockerhub_username, num_workers):
                eval_failures.append(f"exp_{trial_num}")
        baseline_nums = _detect_baseline_dirs(exp_dir)
        for bnum in baseline_nums:
            if not run_baseline_evaluation(exp_dir, bnum, dockerhub_username, num_workers):
                eval_failures.append(f"baseline_{bnum}")
        if eval_failures:
            raise RuntimeError(
                f"Evaluation failed for: {', '.join(eval_failures)}. "
                f"Fix errors above before processing."
            )

    # Check for evaluation results
    has_eval = any(
        (exp_dir / f"exp_{t}" / "eval_results").exists() for t in range(1, num_trials + 1)
    )

    if not has_eval:
        print("\n" + "=" * 70)
        print("WARNING: No evaluation results found!")
        print("=" * 70)
        print(f"Expected eval results in: {exp_dir}/exp_*/eval_results/")
        print("\nTo run evaluation, use:")
        print("  python scripts/process_swebench_underspec.py \\")
        print(f"    --exp-dir {exp_dir} \\")
        print("    --run-eval --dockerhub-username YOUR_USERNAME")
        print("\nProceeding with preds.json data only (no terminal state analysis)...")
        print("=" * 70 + "\n")

    # Collect results, patches, and final steps from all trials
    print("\nCollecting trial results...")
    results_by_instance = defaultdict(list)
    patches_by_instance = defaultdict(list)
    final_steps_by_instance = defaultdict(list)

    missing_trials = []
    for trial_num in range(1, num_trials + 1):
        preds = load_preds(exp_dir, trial_num)
        if not preds:
            print(f"  Trial {trial_num}: No preds.json found")
            missing_trials.append(trial_num)
            continue

        print(f"  Trial {trial_num}: {len(preds)} predictions")

        for inst_id, pred_data in preds.items():
            f2p_tests, p2p_tests = get_f2p_p2p_tests(inst_id, swebench_df)

            eval_output = load_eval_output(exp_dir, trial_num, inst_id)
            test_results = compute_test_results(eval_output, f2p_tests, p2p_tests)

            results_by_instance[inst_id].append(
                {
                    "trial": trial_num,
                    "has_eval": eval_output is not None,
                    **test_results,
                }
            )

            patch = pred_data.get("model_patch", "")
            patches_by_instance[inst_id].append(patch)

            final_steps = load_final_steps_from_traj(exp_dir, trial_num, inst_id, n_steps=5)
            final_steps_by_instance[inst_id].append(final_steps)

    if missing_trials:
        raise RuntimeError(
            f"Missing preds.json for trial(s) {missing_trials}. "
            f"Run all {num_trials} trials before processing."
        )

    # Build summary DataFrame
    no_trial_count = sum(1 for inst_id in instance_lookup if inst_id not in results_by_instance)
    if no_trial_count:
        missing_pct = no_trial_count / len(instance_lookup)
        print(f"\nWARNING: {no_trial_count}/{len(instance_lookup)} instances have no trial data")
        if missing_pct > 0.5:
            raise RuntimeError(
                f"{no_trial_count}/{len(instance_lookup)} instances ({missing_pct:.0%}) "
                f"have no trial data. This suggests incomplete runs."
            )
    print("\nBuilding summary...")
    rows = []

    for inst_id in tqdm(instance_lookup.keys(), desc="Processing"):
        inst_info = instance_lookup[inst_id]
        trials = results_by_instance.get(inst_id, [])

        row = {
            "instance_id": inst_id,
            "original_instance_id": inst_info["original_instance_id"],
            "variant_id": inst_info["variant_id"],
            # Pipeline metadata (from underspec_candidates.csv)
            "dimension": inst_info["dimension"],
            "subdimension": inst_info["subdimension"],
            "removed_value": inst_info["removed_value"],
            "removed_segments_json": inst_info["removed_segments_json"],
            "num_segments": inst_info["num_segments"],
            "severity": inst_info["severity"],
            "criticality": inst_info["criticality"],
            "guessability": inst_info["guessability"],
            "priority_score": inst_info["priority_score"],
            "is_used_in_trajectory": inst_info["is_used_in_trajectory"],
            "first_use_pct": inst_info["first_use_pct"],
            "checkpoint_refs": inst_info["checkpoint_refs"],
            "expected_questions": inst_info["expected_questions"],
            "expected_failure_mode": inst_info["expected_failure_mode"],
            "original_prompt": inst_info["original_prompt"],
            "underspecified_prompt": inst_info["underspecified_prompt"],
        }

        if not trials:
            rows.append(row)
            continue

        n = len(trials)

        # Overall pass/fail (F2P-only)
        full_passes = sum(1 for t in trials if t["overall_pass"])

        # F2P terminal states for divergence analysis
        f2p_terminal_states = [t["f2p_terminal_state"] for t in trials]
        unique_states = set(f2p_terminal_states)
        n_unique_states = len(unique_states)

        # Divergence metrics
        divergence_ratio = n_unique_states / n if n > 0 else 0

        # State entropy
        state_counts = Counter(f2p_terminal_states)
        state_entropy = 0.0
        if n > 0 and n_unique_states > 1:
            for count in state_counts.values():
                p = count / n
                if p > 0:
                    state_entropy -= p * math.log2(p)

        # F2P/P2P stats
        avg_f2p_passed = sum(t["f2p_pass_count"] for t in trials) / n
        avg_p2p_passed = sum(t["p2p_pass_count"] for t in trials) / n
        f2p_total = trials[0]["f2p_total"] if trials else 0
        p2p_total = trials[0]["p2p_total"] if trials else 0

        # Get final steps for judge
        final_steps = final_steps_by_instance.get(inst_id, [])
        final_steps = [s for s in final_steps if s and s.strip()]

        # Ambiguity classification (same logic as TAC)
        all_failed = full_passes == 0
        all_succeeded = full_passes == n
        some_succeeded = 0 < full_passes < n
        eligible_new_task = all_failed and n_unique_states == 1

        judge_verdict = None
        judge_reasoning = None

        if all_failed:
            if n_unique_states > 1:
                ambiguity_class = "outcome-critical"
            else:
                # Uniform test failure -> use judge to determine new_task vs outcome-critical
                if judge_model:
                    underspec_prompt = inst_info.get("underspecified_prompt", "")
                    responses = final_steps if final_steps else ["[No trajectory recorded]"]

                    judge_result = judge_new_task_sync(underspec_prompt, responses, judge_model)
                    judge_verdict = bool(judge_result.get("task_completed"))
                    judge_reasoning = judge_result.get("reasoning", "")

                    ambiguity_class = "new_task" if judge_verdict else "outcome-critical"
                else:
                    ambiguity_class = "new_task"  # Heuristic
        elif all_succeeded:
            ambiguity_class = "benign"
        elif some_succeeded:
            ambiguity_class = "divergent"
        else:
            ambiguity_class = "unknown"

        # Oracle decision (matches TAC: OC -> clarify, everything else -> proceed)
        oracle_decision = "clarify" if ambiguity_class == "outcome-critical" else "proceed"

        # pass@k metrics (pass_at_k handles n < k via fallback estimator)
        row.update(
            {
                "n_trials": n,
                "full_passes": full_passes,
                "success_rate": round(full_passes / n * 100, 1),
                "pass_at_1": round(pass_at_k(n, full_passes, 1), 3),
                "pass_at_3": round(pass_at_k(n, full_passes, 3), 3),
                "pass_at_N": round(pass_at_k(n, full_passes, n), 3),
                # F2P metrics
                "f2p_total": f2p_total,
                "p2p_total": p2p_total,
                "avg_f2p_passed": round(avg_f2p_passed, 1),
                "avg_p2p_passed": round(avg_p2p_passed, 1),
                "f2p_pass_rate": round(avg_f2p_passed / f2p_total, 3) if f2p_total else 0,
                "p2p_pass_rate": round(avg_p2p_passed / p2p_total, 3) if p2p_total else 0,
                # Terminal state analysis
                "n_unique_states": n_unique_states,
                "divergence_ratio": round(divergence_ratio, 3),
                "state_entropy": round(state_entropy, 3),
                "terminal_states": str(list(unique_states)),
                # Ambiguity classification
                "outcome_critical": full_passes == 0,
                "new_task_eligible": eligible_new_task,
                "judge_verdict": judge_verdict,
                "judge_reasoning": judge_reasoning,
                "ambiguity_class": ambiguity_class,
                "oracle_decision": oracle_decision,
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    if "n_trials" not in df.columns:
        print("\nNo evaluated variants found (n_trials column missing).")
        return

    evaluated = df[df["n_trials"].notna()]

    if evaluated.empty:
        print("\nNo evaluated variants found.")
        return

    print("\n" + "=" * 70)
    print("SWE-BENCH UNDERSPEC RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nTotal variants: {len(df)}")
    print(f"Evaluated: {len(evaluated)}")
    print(f"Total trials: {int(evaluated['n_trials'].sum())}")

    # Ambiguity class breakdown
    if "ambiguity_class" in evaluated.columns:
        print("\n--- Ambiguity Classification ---")
        class_counts = evaluated["ambiguity_class"].value_counts()
        for cls in ["outcome-critical", "divergent", "benign", "new_task"]:
            if cls in class_counts.index:
                count = class_counts[cls]
                pct = count / len(evaluated) * 100
                print(f"  {cls:20s}: {count:4d} ({pct:5.1f}%)")

    # Oracle decision
    if "oracle_decision" in evaluated.columns:
        print("\n--- Oracle Decision ---")
        for dec, count in evaluated["oracle_decision"].value_counts().items():
            print(f"  {dec:20s}: {count:4d} ({count/len(evaluated)*100:5.1f}%)")

    # pass@k summary
    print("\n--- Pass@k Summary ---")
    print(f"  pass@1: {evaluated['pass_at_1'].mean():.3f}")
    print(f"  pass@3: {evaluated['pass_at_3'].mean():.3f}")
    print(f"  Success rate: {evaluated['success_rate'].mean():.1f}%")

    # Divergence summary
    print("\n--- Divergence Analysis (F2P Terminal States) ---")
    print(f"  Avg unique states: {evaluated['n_unique_states'].mean():.2f}")
    print(f"  Avg divergence ratio: {evaluated['divergence_ratio'].mean():.3f}")
    print(f"  Avg state entropy: {evaluated['state_entropy'].mean():.3f}")

    # By dimension
    if "dimension" in evaluated.columns:
        print("\n--- By Dimension ---")
        for dim in evaluated["dimension"].unique():
            if not dim:
                continue
            dim_df = evaluated[evaluated["dimension"] == dim]
            print(
                f"  {dim or 'unknown':20s}: {len(dim_df):4d} variants, "
                f"pass@3={dim_df['pass_at_3'].mean():.3f}, "
                f"states={dim_df['n_unique_states'].mean():.2f}"
            )

    # By ambiguity class: per-class pass rates
    if "ambiguity_class" in evaluated.columns:
        print("\n--- Per-Class Performance ---")
        for cls in ["outcome-critical", "divergent", "benign", "new_task"]:
            cls_df = evaluated[evaluated["ambiguity_class"] == cls]
            if not cls_df.empty:
                print(
                    f"  {cls:20s}: pass@1={cls_df['pass_at_1'].mean():.3f}, "
                    f"pass@3={cls_df['pass_at_3'].mean():.3f}, "
                    f"n={len(cls_df)}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Process SWE-bench underspec experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Experiment directory containing exp_1/, exp_2/, exp_3/",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output CSV path (default: {exp_dir}/underspec_results.csv)",
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run evaluation on predictions before processing",
    )
    parser.add_argument(
        "--dockerhub-username",
        type=str,
        help="Docker Hub username (required with --run-eval)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=50,
        help="Workers for evaluation (default: 50)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation, skip classification and CSV output",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Use LLM judge for new_task classification",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"LLM model for judging (default: {DEFAULT_JUDGE_MODEL})",
    )

    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)

    judge_model = args.judge_model if args.judge else None

    # eval-only mode: run evaluation and exit (skip classification)
    if args.eval_only:
        if not args.run_eval:
            print("Error: --eval-only requires --run-eval")
            sys.exit(1)
        if not args.dockerhub_username:
            print("Error: --dockerhub-username required with --run-eval")
            sys.exit(1)
        ensure_swebench_csv()
        # Detect num trials from exp_N dirs (only match exp_1, exp_2, etc.)
        num_trials = len(
            [
                d
                for d in exp_dir.glob("exp_[0-9]*")
                if d.is_dir() and d.name.startswith("exp_") and d.name[4:].isdigit()
            ]
        )
        # Detect baseline dirs
        baseline_nums = _detect_baseline_dirs(exp_dir)

        if num_trials == 0 and not baseline_nums:
            print(f"Error: No exp_* or baseline_* directories found in {exp_dir}")
            sys.exit(1)

        eval_failures = []
        if num_trials > 0:
            print(f"Running variant evaluation ({num_trials} trials)")
            for trial_num in range(1, num_trials + 1):
                if not run_evaluation(
                    exp_dir, trial_num, args.dockerhub_username, args.num_workers
                ):
                    eval_failures.append(f"exp_{trial_num}")

        if baseline_nums:
            print(f"Running baseline evaluation ({len(baseline_nums)} baselines)")
            for bnum in baseline_nums:
                if not run_baseline_evaluation(
                    exp_dir, bnum, args.dockerhub_username, args.num_workers
                ):
                    eval_failures.append(f"baseline_{bnum}")

        if eval_failures:
            print(
                f"\nEvaluation failed for: {', '.join(eval_failures)}",
                file=sys.stderr,
            )
            sys.exit(1)

        print("\nEvaluation complete.")
        sys.exit(0)

    df = process_experiment(
        exp_dir=exp_dir,
        run_eval=args.run_eval,
        dockerhub_username=args.dockerhub_username,
        num_workers=args.num_workers,
        judge_model=judge_model,
    )

    # Print summary
    print_summary(df)

    # Save results
    output_path = Path(args.output) if args.output else exp_dir / "underspec_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n{'='*70}")
    print(f"Saved results to: {output_path}")

    # Also save JSON for filtering
    json_path = output_path.with_suffix(".json")
    df.to_json(json_path, orient="records", indent=2)
    print(f"Saved JSON to: {json_path}")

    # Print next steps
    print("\nNext steps:")
    print(f"  python scripts/filter_swebench_samples.py --input {output_path}")
    print("  python scripts/export_swebench_dataset.py --input <filtered_output>")


if __name__ == "__main__":
    main()
