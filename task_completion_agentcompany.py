"""
End-to-end orchestrator for AgentCompany tasks using OpenHands.

Workflow:
1) Build a temporary local tasks directory with brief.txt for each task
2) Populate each task's inputs/ with shared tac-openhands data
3) Create a tasks.txt file listing task names
4) Invoke OpenHands runner to generate local_results
5) Format local_results into per-session JSONs and emit transcripts under tasks/

Inputs come from task_pairs_agentcompany/<task_name>/task.json where each JSON
contains primary_task (and optionally underspecified_task) strings.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

from constants import DEFAULTS, PATHS, resolve_model
from evaluation.pass_k_metrics import write_pass_k_summary_csv

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
AGENTCOMPANY_DIR = os.path.join(REPO_ROOT, "experiments", "agentcompany")
DEFAULT_TASKS_DIR = os.path.join(REPO_ROOT, PATHS["tac_tasks"])
OPENHANDS_DIR = os.path.join(AGENTCOMPANY_DIR, "openhands")
TAC_OPENHANDS_DIR = os.path.join(AGENTCOMPANY_DIR, "tac-openhands")
SCRIPTS_DIR = os.path.join(AGENTCOMPANY_DIR, "scripts")

# Clarification prompt variants for user simulator experiments.
# "none" = no prompt (baseline), "default" = paper default.

CLARIFICATION_PROMPTS = {
    "none": "",
    "default": """IMPORTANT: Your output will be checked by an auto-grader looking for exact answers.
This task may be missing critical information.
Use the ask_user tool to ask the user for any missing details.""",
}


def load_task_definition(task_name: str, tasks_dir: str = None) -> dict:
    """Load task definition from JSON.

    Expects file at tasks_dir/<task_name>/task.json.
    Returns a dictionary containing task fields.
    """
    tasks_dir = tasks_dir or DEFAULT_TASKS_DIR
    json_path = os.path.join(tasks_dir, task_name, "task.json")
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found. Using empty tasks for {task_name}.")
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Failed to parse {json_path}: {e}. Using empty tasks for {task_name}.")
        return {}


def safe_rmtree(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def copy_tree(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def parse_workspaces_from_report(report_path: str) -> List[Tuple[str, str]]:
    """Extract (task_name, workspace_path) pairs from OpenHands JSON report."""
    session_pairs: List[Tuple[str, str]] = []

    try:
        with open(report_path) as f:
            report = json.load(f)

        for key, result in report.get("task_results", {}).items():
            task_id = result.get("task_id", "")
            workspace = result.get("workspace", "")

            if task_id and workspace:
                # workspace is full path to workspace/ dir, parent is model_run_dir
                model_run_dir = os.path.dirname(workspace)
                print(f"Workspace for {task_id}: {model_run_dir}")
                session_pairs.append((task_id, model_run_dir))

    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: failed to parse report {report_path}: {e}")

    return session_pairs


def _resolve_evaluator_path(tasks_dir: str, task_name: str) -> str:
    """Resolve the evaluator script path, following symlinks if necessary."""
    evaluator_path = os.path.join(tasks_dir, task_name, "evaluator.py")
    if os.path.islink(evaluator_path):
        evaluator_path = os.path.realpath(evaluator_path)
    evaluator_path = os.path.abspath(evaluator_path)
    if not os.path.exists(evaluator_path):
        raise FileNotFoundError(f"Evaluator path not found: {evaluator_path}")
    return evaluator_path


def _find_trajectory_path(
    run_base_dir: str, task_name: str, trial_idx: int, session_dir: str = None
) -> Optional[str]:
    """Find the trajectory JSON for a given task trial in final_results.

    Tries to match by session UUID (extracted from session_dir's state directory)
    for exact pairing. Falls back to index-based lookup.
    """
    final_task_dir = os.path.join(run_base_dir, "final_results", task_name)
    if not os.path.isdir(final_task_dir):
        return None
    try:
        session_files = sorted(
            (
                f
                for f in os.listdir(final_task_dir)
                if f.endswith(".json") and f.startswith("session_")
            ),
            key=lambda f: (
                int(f.removesuffix(".json").split("_")[1])
                if f.removesuffix(".json").split("_")[1].isdigit()
                else 0
            ),
        )
        if not session_files:
            return None

        if session_dir:
            state_sessions = os.path.join(session_dir, "state", "sessions")
            if os.path.isdir(state_sessions):
                uuid_dirs = [
                    d
                    for d in os.listdir(state_sessions)
                    if os.path.isdir(os.path.join(state_sessions, d))
                ]
                if uuid_dirs:
                    session_uuid = uuid_dirs[0]
                    for sf in session_files:
                        if session_uuid in sf:
                            return os.path.join(final_task_dir, sf)

        if trial_idx < len(session_files):
            return os.path.join(final_task_dir, session_files[trial_idx])
    except Exception:
        pass
    return None


def _run_eval_with_retries(
    cmd: list, session_dir: str, env: dict, result_path: str, max_retries: int = 3
) -> bool:
    """Run the evaluation subprocess with retries on transient failures."""
    for eval_attempt in range(max_retries):
        try:
            subprocess.run(cmd, check=True, cwd=session_dir, env=env, timeout=600)  # 10 min timeout
            return True
        except subprocess.TimeoutExpired as e:
            print(
                f"Warning: deterministic eval timed out for {session_dir} (attempt {eval_attempt + 1}/{max_retries}): {e}"
            )
            if eval_attempt < max_retries - 1:
                time.sleep(2)  # Brief pause before retry
        except subprocess.CalledProcessError as e:
            print(
                f"Warning: deterministic eval failed for {session_dir} (attempt {eval_attempt + 1}/{max_retries}): {e}"
            )
            if eval_attempt < max_retries - 1:
                time.sleep(2)  # Brief pause before retry
    return False


def run_deterministic_evaluation(
    session_pairs: List[Tuple[str, str]],
    tasks_dir: str,
    repo_root: str,
    run_base_dir: str,
    model_short: str,
    format_name: str,
    original_tasks_dir: str = None,
) -> tuple[int, int]:
    """Run deterministic eval for each (task_name, session_dir) and write summaries.

    Args:
        session_pairs: List of (task_name, session_dir) tuples
        tasks_dir: Directory containing task definitions (may be underspec variant dir)
        repo_root: Repository root path
        run_base_dir: Base directory for this run
        model_short: Short model name
        format_name: Format name for the run
        original_tasks_dir: Original tasks directory for evaluator assets (uses tasks_dir if None)
    """
    # For underspec evaluation, evaluator should come from original tasks dir
    # (symlinks point there, but we resolve to be safe)
    eval_assets_dir = original_tasks_dir or tasks_dir

    # Deduplicate sessions and track per-task indices
    # Key: session_dir (unique), Value: (task_name, trial_idx)
    seen_sessions: set = set()
    task_trial_counts: dict = {}  # task_name -> next trial index

    # Sort session_pairs by (task_name, session_dir) for consistent trial numbering across runs
    sorted_session_pairs = sorted(session_pairs, key=lambda x: (x[0], x[1]))

    evaluated_count = 0
    failed_count = 0

    for task_name, session_dir in sorted_session_pairs:
        # Skip duplicate sessions (same session shouldn't be evaluated twice)
        if session_dir in seen_sessions:
            print(f"Skipping duplicate session: {session_dir}")
            continue
        seen_sessions.add(session_dir)

        # Skip if session already has a result.json (already evaluated)
        result_path = os.path.join(session_dir, "result.json")
        if os.path.exists(result_path):
            print(f"Skipping already evaluated session: {session_dir}")
            evaluated_count += 1
            # Still need to copy to deterministic dir if not already there
            deterministic_dir = os.path.join(
                run_base_dir, "final_results", "deterministic", task_name, model_short
            )
            trial_idx = task_trial_counts.get(task_name, 0)
            task_trial_counts[task_name] = trial_idx + 1
            dest_path = os.path.join(deterministic_dir, f"evaluation_{trial_idx}.json")
            if not os.path.exists(dest_path):
                os.makedirs(deterministic_dir, exist_ok=True)
                try:
                    shutil.copy2(result_path, dest_path)
                    print(f"Copied existing result to: {dest_path}")
                except Exception as copy_exc:
                    print(f"Warning: failed to copy existing result: {copy_exc}")
            continue

        # Get trial index for this task (0-based, per-task)
        trial_idx = task_trial_counts.get(task_name, 0)
        task_trial_counts[task_name] = trial_idx + 1

        evaluator_path = _resolve_evaluator_path(tasks_dir, task_name)

        cmd = [
            sys.executable,
            os.path.join(repo_root, "evaluation", "tac_eval.py"),
            "--evaluator_path",
            evaluator_path,
            "--workspace_path",
            session_dir,
            "--lhaw_root",
            repo_root,
            "--result_path",
            result_path,
        ]
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = repo_root + (":" + pythonpath if pythonpath else "")
        # Provide evaluators with output root (workspace/output) and optional eval assets root
        output_root = os.path.join(session_dir, "workspace", "output")
        env["MRT_OUTPUT_ROOT"] = output_root
        # Point evaluators to the original task directory for assets (handles symlinks)
        env["MRT_EVAL_ASSETS_ROOT"] = os.path.join(eval_assets_dir, task_name)
        trajectory_path = _find_trajectory_path(run_base_dir, task_name, trial_idx, session_dir)
        if trajectory_path:
            cmd.extend(["--trajectory_path", trajectory_path])

        print(f"Running deterministic eval (trial {trial_idx}):", " ".join(cmd))

        max_eval_retries = 3
        eval_success = _run_eval_with_retries(cmd, session_dir, env, result_path, max_eval_retries)

        if eval_success:
            evaluated_count += 1
            try:
                deterministic_dir = os.path.join(
                    run_base_dir, "final_results", "deterministic", task_name, model_short
                )
                os.makedirs(deterministic_dir, exist_ok=True)
                dest_path = os.path.join(deterministic_dir, f"evaluation_{trial_idx}.json")
                shutil.copy2(result_path, dest_path)
                print(f"Wrote deterministic summary: {dest_path}")
            except Exception as copy_exc:
                print(f"Warning: failed to copy deterministic result for {session_dir}: {copy_exc}")
        else:
            failed_count += 1
            print(
                f"Error: deterministic eval failed after {max_eval_retries} attempts for {session_dir}"
            )

    return evaluated_count, failed_count


def write_deterministic_summary_csv(
    final_results_dir: str, task_names: List[str], format_name: str, model_short: str
) -> None:
    """Aggregate deterministic evaluation JSONs into a single CSV per run."""
    try:
        import csv

        summary_dir = os.path.join(final_results_dir, "summaries")
        os.makedirs(summary_dir, exist_ok=True)
        summary_csv = os.path.join(
            summary_dir, f"deterministic_summary_{format_name}_{model_short}.csv"
        )
        rows: List[List[str]] = [["task", "session", "score", "total", "percent", "result_path"]]
        for task_name in task_names:
            deterministic_dir = os.path.join(
                final_results_dir, "deterministic", task_name, model_short
            )
            if not os.path.exists(deterministic_dir):
                continue
            for fname in sorted(os.listdir(deterministic_dir)):
                if not fname.endswith(".json"):
                    continue
                fpath = os.path.join(deterministic_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                    result = data.get("final_score", {}).get("result", None)
                    total = data.get("final_score", {}).get("total", None)
                    percent = (
                        (100.0 * result / total)
                        if (
                            isinstance(result, (int, float))
                            and isinstance(total, (int, float))
                            and total
                        )
                        else ""
                    )
                    session_id = os.path.splitext(fname)[0]
                    rows.append(
                        [
                            task_name,
                            session_id,
                            str(result),
                            str(total),
                            f"{percent:.1f}" if percent != "" else "",
                            fpath,
                        ]
                    )
                except Exception:
                    print(f"Warning: failed to read {fpath}", file=sys.stderr)
                    rows.append([task_name, os.path.splitext(fname)[0], "", "", "", fpath])
        with open(summary_csv, "w", encoding="utf-8", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerows(rows)
        print(f"Wrote deterministic summary CSV: {summary_csv}")
    except Exception as e:
        print(f"Warning: failed to write summary CSV: {e}")


def build_local_tasks_dir(
    task_names: List[str],
    format_name: str,
    base_dir: str,
    tasks_dir: str = None,
    clarification_prompt: str = "",
) -> str:
    """Create local_tasks directory structure for OpenHands from task definitions.

    Args:
        task_names: List of task names to include
        format_name: Format name for the run
        base_dir: Base directory for run outputs
        clarification_prompt: Optional prompt suffix to encourage clarification
        tasks_dir: Custom tasks directory (for underspec variants). If None, uses DEFAULT_TASKS_DIR.
    """
    tasks_dir = tasks_dir or DEFAULT_TASKS_DIR
    local_tasks = os.path.join(base_dir, f"local_tasks_{format_name}")
    safe_rmtree(local_tasks)
    os.makedirs(local_tasks, exist_ok=True)

    for task_name in task_names:
        task_dir = os.path.join(local_tasks, task_name)
        inputs_dir = os.path.join(task_dir, "inputs")
        os.makedirs(inputs_dir, exist_ok=True)

        task_def = load_task_definition(task_name, tasks_dir)

        if "underspecified" in format_name:
            if "underspecified_task" not in task_def or not task_def["underspecified_task"]:
                raise ValueError(f"`underspecified_task` key not found or empty for {task_name}")
            brief = task_def["underspecified_task"]
        else:
            # Note: We don't use secondary task within TAC experiments for LHAW
            if "primary_task" not in task_def or not task_def["primary_task"]:
                raise ValueError(f"`primary_task` key not found or empty for {task_name}")
            brief = task_def["primary_task"]

        # Add clarification prompt suffix if specified (for user simulator experiments)
        if clarification_prompt:
            brief = brief + "\n\n" + clarification_prompt

        with open(os.path.join(task_dir, "brief.txt"), "w", encoding="utf-8") as f:
            f.write(brief)

        # Write full task definition for MCP server's user simulator.
        # Downstream: setup_local_task_data -> prepare_workspace -> task.py
        # base64-encodes this and passes it as an env var to the MCP subprocess.
        if "primary_task" in task_def and task_def["primary_task"]:
            with open(os.path.join(task_dir, "primary_task.json"), "w", encoding="utf-8") as f:
                json.dump(task_def, f, indent=2)

    return local_tasks


def populate_inputs(local_tasks_dir: str) -> None:
    if not os.path.exists(TAC_OPENHANDS_DIR):
        raise FileNotFoundError(f"Missing shared inputs directory: {TAC_OPENHANDS_DIR}")

    for task_name in os.listdir(local_tasks_dir):
        task_dir = os.path.join(local_tasks_dir, task_name)
        if not os.path.isdir(task_dir):
            continue
        inputs_dir = os.path.join(task_dir, "inputs")
        # clean destination inputs/
        for item in os.listdir(inputs_dir):
            item_path = os.path.join(inputs_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        copy_tree(TAC_OPENHANDS_DIR, inputs_dir)


def write_tasks_file(task_names: List[str], base_dir: str) -> str:
    tasks_file = os.path.join(base_dir, "tasks.txt")
    with open(tasks_file, "w", encoding="utf-8") as f:
        for name in sorted(task_names):
            f.write(name + "\n")
    return tasks_file


def _cleanup_exited_containers():
    """Remove exited openhands containers to free Docker resources between trials.

    Note: On shared hosts with parallel experiments, this may remove exited containers
    from other runs. Only `status=exited` containers are affected (never running ones).
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "name=openhands", "--filter", "status=exited"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        container_ids = result.stdout.strip().split()
        if container_ids and container_ids[0]:
            subprocess.run(
                ["docker", "rm", "-f"] + container_ids,
                capture_output=True,
                timeout=30,
            )
            print(f"Cleaned up {len(container_ids)} exited containers")
    except Exception:
        pass


def run_openhands(
    tasks_file: str,
    local_tasks_dir: str,
    local_results_dir: str,
    model: str,
    max_iterations: int,
    api_key: str = None,
    base_url: str = None,
    include_user_help: bool = False,
    user_simulator_model: str = None,
    agent_strategy: str = None,
) -> tuple[Optional[str], int]:
    """Run OpenHands and return path to the JSON report (or None if not found)."""
    run_id = uuid.uuid4().hex[:12]  # Short unique ID
    cmd = [
        sys.executable,
        os.path.join(OPENHANDS_DIR, "main.py"),
        "--tasks-file",
        tasks_file,
        "--local-tasks-dir",
        local_tasks_dir,
        "--local-results-dir",
        local_results_dir,
        "--run-id",
        run_id,
        "--model-config",
        os.path.join(OPENHANDS_DIR, "model_config.json"),
        "--model",
        model,
        "--max-workers",
        "1",
        "--max-iterations",
        str(max_iterations),
    ]

    if base_url:
        cmd.extend(["--base-url", base_url])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if include_user_help:
        cmd.append("--include-user-help")
    if user_simulator_model:
        cmd.extend(["--user-simulator-model", user_simulator_model])
    if agent_strategy:
        cmd.extend(["--agent-strategy", agent_strategy])

    print("Running OpenHands:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=AGENTCOMPANY_DIR, check=False)

    if result.returncode != 0:
        print(f"\n⚠️  OpenHands returned exit code {result.returncode} (some tasks may have failed)")
        print("Continuing to evaluation phase for completed tasks...")

    # Find our report by searching for matching run_id in JSON
    reports_dir = os.path.join(AGENTCOMPANY_DIR, "reports")
    if not os.path.isdir(reports_dir):
        return None, result.returncode

    for filename in os.listdir(reports_dir):
        if not filename.endswith(".json"):
            continue
        report_path = os.path.join(reports_dir, filename)
        try:
            with open(report_path) as f:
                data = json.load(f)
            if data.get("summary", {}).get("run_id") == run_id:
                return report_path, result.returncode
        except (json.JSONDecodeError, OSError):
            continue

    return None, result.returncode


def format_results(
    local_results_dir: str,
    final_results_dir: str,
    transcripts_root: Optional[str],
    format_name: str,
    model_name: str,
) -> None:
    formatter = os.path.join(SCRIPTS_DIR, "format_results.py")
    cmd = [
        sys.executable,
        formatter,
        "--local-results-dir",
        local_results_dir,
        "--final-results-dir",
        final_results_dir,
    ]
    # Only add transcript args if transcripts_root is provided
    if transcripts_root:
        cmd.extend(
            [
                "--transcripts-root",
                transcripts_root,
                "--format-name",
                format_name,
                "--model-name",
                model_name,
            ]
        )
    print("Formatting results:", " ".join(cmd))
    # Ensure LHAW project root is on PYTHONPATH so formatter can import modules
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = REPO_ROOT + (":" + pythonpath if pythonpath else "")
    subprocess.run(cmd, cwd=AGENTCOMPANY_DIR, check=True, env=env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgentCompany tasks end-to-end with OpenHands")
    parser.add_argument(
        "--format_name", required=True, help="Experiment format name (e.g., benign_v1)"
    )
    # Support both --task_names and --task_folder as synonyms (wrapper may use either)
    parser.add_argument("--task_names", nargs="+", help="Space-separated list of task names")
    parser.add_argument(
        "--task_folder", nargs="+", dest="task_names_alt", help="Alias for --task_names"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=DEFAULTS["num_trials"],
        help=f"Trials per task for pass@k (default: {DEFAULTS['num_trials']})",
    )
    parser.add_argument(
        "--backend_model",
        required=True,
        help="Model identifier (e.g., anthropic/claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=DEFAULTS["max_iterations"],
        help=f"Max agent iterations (default: {DEFAULTS['max_iterations']})",
    )
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--api_key", default=None)
    parser.add_argument(
        "--runs_dir",
        default=None,
        help="Directory to store run outputs (default: experiments/agentcompany/runs)",
    )
    parser.add_argument(
        "--tasks_dir",
        default=None,
        help="Custom tasks directory (for underspec variants). Default: task_pairs_agentcompany",
    )
    parser.add_argument(
        "--include_user_help",
        action="store_true",
        default=False,
        help="Enable MCP request_clarification tool for agent to request clarification (default: disabled)",
    )
    parser.add_argument(
        "--clarification_prompt",
        type=str,
        default="none",
        choices=list(CLARIFICATION_PROMPTS.keys()),
        help="Clarification prompt key: 'none' (no prompt) or 'default' (paper default).",
    )
    parser.add_argument(
        "--user_simulator_model",
        type=str,
        default="openai/gpt-4.1-2025-04-14",
        help="Model for user simulator LLM (must be fast, <30s response). "
        "Default: openai/gpt-4.1-2025-04-14",
    )
    parser.add_argument(
        "--agent_strategy",
        type=str,
        default=None,
        choices=["none", "react", "reflexion", "plan_and_execute"],
        help="Agent prompting strategy. Copies strategy as AGENTS.md to workspace. "
        "Options: none (baseline), react, reflexion, plan_and_execute.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Resolve task names from either flag
    task_names = (
        args.task_names
        if args.task_names
        else (args.task_names_alt if getattr(args, "task_names_alt", None) else None)
    )
    if not task_names:
        print("Error: You must provide task names via --task_names or --task_folder")
        return 1

    args.backend_model = resolve_model(args.backend_model)

    # Short model name (used for internal subdirectory structure, not in run dir name)
    model_short = args.backend_model.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]

    # Workspace for local tasks/results inside agentcompany/runs
    runs_dir = args.runs_dir if args.runs_dir else os.path.join(AGENTCOMPANY_DIR, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    run_base_dir = os.path.join(runs_dir, f"run_{args.format_name}_{timestamp}")
    os.makedirs(run_base_dir, exist_ok=True)

    # Use custom tasks_dir if provided (for underspec variants)
    tasks_dir = args.tasks_dir or DEFAULT_TASKS_DIR

    # Resolve clarification prompt (use predefined key or custom string)
    clarification_prompt = args.clarification_prompt
    if clarification_prompt in CLARIFICATION_PROMPTS:
        clarification_prompt = CLARIFICATION_PROMPTS[clarification_prompt]

    local_tasks_dir = build_local_tasks_dir(
        task_names, args.format_name, run_base_dir, tasks_dir, clarification_prompt
    )
    populate_inputs(local_tasks_dir)

    tasks_file = write_tasks_file(task_names, run_base_dir)

    local_results_dir = os.path.join(run_base_dir, "local_results")
    safe_rmtree(local_results_dir)
    os.makedirs(local_results_dir, exist_ok=True)

    # Persist run parameters for reproducibility
    run_metadata = {
        "format_name": args.format_name,
        "backend_model": args.backend_model,
        "model_short": model_short,
        "num_trials": args.num_trials,
        "max_iterations": args.max_iterations,
        "include_user_help": args.include_user_help,
        "clarification_prompt": args.clarification_prompt,
        "user_simulator_model": args.user_simulator_model,
        "agent_strategy": args.agent_strategy,
        "tasks": task_names,
        "tasks_dir": tasks_dir,
        "started_at": timestamp,
    }
    with open(os.path.join(run_base_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    # Execute runner for the requested number of trials
    session_pairs: List[Tuple[str, str]] = []
    openhands_failures = 0
    missing_reports = 0

    # Use OpenAI/LiteLLM proxy (same for all trials)
    api_key = args.api_key or os.environ.get("LLM_API_KEY")
    base_url = args.base_url or os.environ.get("LLM_BASE_URL")

    for trial_idx in range(args.num_trials):
        print(f"\n=== Running trial {trial_idx + 1}/{args.num_trials} ===")

        # Clean up exited containers between trials to free resources
        _cleanup_exited_containers()

        report_path, trial_rc = run_openhands(
            tasks_file=tasks_file,
            local_tasks_dir=os.path.relpath(local_tasks_dir, AGENTCOMPANY_DIR),
            local_results_dir=os.path.relpath(local_results_dir, AGENTCOMPANY_DIR),
            model=args.backend_model,
            max_iterations=args.max_iterations,
            api_key=api_key,
            base_url=base_url,
            include_user_help=args.include_user_help,
            user_simulator_model=args.user_simulator_model,
            agent_strategy=args.agent_strategy,
        )
        if trial_rc != 0:
            openhands_failures += 1

        # Extract workspace paths from OpenHands report
        if report_path:
            pairs = parse_workspaces_from_report(report_path)
            session_pairs.extend(pairs)
        else:
            missing_reports += 1
            print(f"Warning: no report found for trial {trial_idx + 1}")

    # Format results (transcripts go only to final_results, not task_pairs_agentcompany)
    final_results_dir = os.path.join(run_base_dir, "final_results")
    format_results(
        local_results_dir=os.path.relpath(local_results_dir, AGENTCOMPANY_DIR),
        final_results_dir=os.path.relpath(final_results_dir, AGENTCOMPANY_DIR),
        transcripts_root=None,  # Don't write transcripts to task_pairs_agentcompany
        format_name=args.format_name,
        model_name=model_short,
    )

    # Run deterministic evaluation on discovered sessions
    n_eval_ok, n_eval_failed = run_deterministic_evaluation(
        session_pairs=session_pairs,
        tasks_dir=tasks_dir,
        repo_root=REPO_ROOT,
        run_base_dir=run_base_dir,
        model_short=model_short,
        format_name=args.format_name,
        original_tasks_dir=DEFAULT_TASKS_DIR if args.tasks_dir else None,
    )

    # Check for failures BEFORE writing summaries. Summary CSVs are used as
    # completion sentinels by --resume, so they must only exist for clean runs.
    if not session_pairs:
        print("Error: no sessions discovered from OpenHands reports")
        return 1
    if openhands_failures > 0:
        print(f"Error: {openhands_failures}/{args.num_trials} OpenHands trials returned non-zero")
        return 1
    if missing_reports > 0:
        print(f"Error: missing reports in {missing_reports}/{args.num_trials} trials")
        return 1
    if n_eval_failed > 0:
        print(f"Error: deterministic evaluation failed for {n_eval_failed} sessions")
        return 1
    if n_eval_ok == 0:
        print("Error: deterministic evaluation produced zero successful session evaluations")
        return 1

    # Write per-run summary CSV for deterministic results
    write_deterministic_summary_csv(
        final_results_dir=final_results_dir,
        task_names=task_names,
        format_name=args.format_name,
        model_short=model_short,
    )

    # Write pass^k summary CSV (for reliability analysis, k=3 by default)
    write_pass_k_summary_csv(
        final_results_dir=final_results_dir,
        task_names=task_names,
        format_name=args.format_name,
        model_short=model_short,
        k=args.num_trials,  # Use num_trials as k for pass@k
    )

    print("\nAll done.")
    print(f"Run directory: {run_base_dir}")
    print(f"Formatted results: {final_results_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
