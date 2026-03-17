"""
Process underspecification experiment results from /synthetic/outputs/{exp_name}.

Usage:
    python scripts/process_tac_underspec.py -e v1_ds                    # Single experiment
    python scripts/process_tac_underspec.py -e v1_ds,v1_finance,v1_hr   # Multiple experiments
    python scripts/process_tac_underspec.py -e "v1_*"                   # Pattern match
    python scripts/process_tac_underspec.py -e v1_ds --preview          # Preview variants
    python scripts/process_tac_underspec.py -e v1_ds --export           # Export CSV/JSON
    python scripts/process_tac_underspec.py -e v1_ds --html             # Generate HTML review
    python scripts/process_tac_underspec.py -e v1_ds --judge            # Run LLM judge on new_task candidates
    python scripts/process_tac_underspec.py --list                      # List experiments
"""

import argparse
import csv
import fnmatch
import html as html_module
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import DEFAULTS, PATHS, get_model  # noqa: E402
from evaluation.pass_k_metrics import pass_at_k  # noqa: E402
from scripts.run_tac_underspec import parse_run_dir  # noqa: E402
from synthetic.llm import get_openai_client  # noqa: E402

SYNTH_OUTPUTS = REPO_ROOT / PATHS["synthetic_outputs"]
RUNS_DIR = REPO_ROOT / PATHS["runs_dir"]

DEFAULT_JUDGE_MODEL = get_model("gemini_3_flash")

# =============================================================================
# LLM Judge for new_task detection
# Uses synthetic/llm.py for consistent LLM access
# =============================================================================


def get_new_task_judge_prompt(
    underspec_prompt: str,
    responses: list[str],
) -> str:
    """Create the prompt for LLM judge of new-task eligibility.

    The judge determines if all trials reached the SAME outcome AND that outcome
    completes a coherent task (even if different from the original intent).
    """
    cleaned_responses = [
        response.strip() if response and response.strip() else "[No response recorded]"
        for response in responses
    ]
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


def extract_json_from_response(content: str) -> dict[str, Any] | None:
    """Extract JSON from LLM response, handling common formats."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    if "```" in content:
        code_block_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
        for match in re.findall(code_block_pattern, content):
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

    json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    for match in re.findall(json_pattern, content, re.DOTALL):
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = content[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def judge_new_task_sync(
    underspec_prompt: str,
    responses: list[str],
    model: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    """Use LLM to judge whether the underspecified task was completed.

    Uses get_openai_client() from synthetic/llm.py for consistent LLM access.

    Returns:
        dict with keys:
        - task_completed: bool - True if all trials completed a coherent task
        - reasoning: str - Explanation of the verdict
    """
    try:
        client = get_openai_client()
    except (ImportError, ValueError) as e:
        return {
            "task_completed": False,
            "reasoning": f"Cannot initialize judge client: {e}",
        }

    last_error = None
    raw_content = None
    for attempt in range(max_retries + 1):
        try:
            prompt = get_new_task_judge_prompt(underspec_prompt, responses)
            if attempt > 0:
                prompt += "\n\nIMPORTANT: Your response MUST be valid JSON only. No markdown, no extra text."

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0 if attempt == 0 else 0.1,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            raw_content = response.choices[0].message.content.strip()
            parsed = extract_json_from_response(raw_content)
            if parsed is not None:
                return parsed
            last_error = f"Could not parse JSON: {raw_content[:200]}"
        except Exception as exc:
            last_error = str(exc)

    return {
        "task_completed": False,
        "reasoning": f"Judge failed: {last_error or 'Unknown error'}",
        "raw_response": raw_content,
    }


def resolve_experiments(pattern: str) -> list[str]:
    """Resolve experiment pattern to list of experiment names.

    Supports:
    - Single: "v1_ds"
    - Multiple: "v1_ds,v1_finance,v1_hr"
    - Glob: "v1_*"
    """
    if "," in pattern:
        return [e.strip() for e in pattern.split(",")]

    if "*" in pattern or "?" in pattern:
        matches = []
        for d in sorted(SYNTH_OUTPUTS.iterdir()):
            if d.is_dir() and (d / "underspec_candidates.csv").exists():
                if fnmatch.fnmatch(d.name, pattern):
                    matches.append(d.name)
        return matches

    return [pattern]


# =============================================================================
# Loaders
# =============================================================================


def load_variants(experiment: str, include_prompts: bool = False) -> dict:
    """Load variants from underspec_candidates.csv.

    Args:
        experiment: Experiment name (prompt suffixes like _default/_none are
            stripped automatically if the exact path doesn't exist)
        include_prompts: If True, include original/underspecified prompts (larger memory)
    """
    csv_path = SYNTH_OUTPUTS / experiment / "underspec_candidates.csv"
    if not csv_path.exists():
        for suffix in ("_default", "_none"):
            if experiment.endswith(suffix):
                base = experiment[: -len(suffix)]
                alt = SYNTH_OUTPUTS / base / "underspec_candidates.csv"
                if alt.exists():
                    csv_path = alt
                    break
    if not csv_path.exists():
        return {}

    variants = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            v = {
                "task": row["task_name"],
                "dimension": row["dimension"],
                "subdimension": row.get("subdimension", ""),  # For diversity filtering
                "removed_value": row.get("removed_value", ""),  # For similarity dedup
                # Load pre-computed ambiguity_class from filtered CSV (used by process_tac_usersim.py)
                # Note: build_summary_df() recomputes this from trial results, but we need
                # the original class for user-sim experiments that run on pre-classified variants
                "ambiguity_class": row.get("ambiguity_class", "unknown"),
                "criticality": float(row.get("criticality") or 0),
                "guessability": float(row.get("guessability") or 0),
                "priority_score": float(row.get("priority_score") or 0),
                # Always include underspecified_prompt for LLM judge
                "underspecified_prompt": row.get("underspecified_prompt", ""),
            }
            if include_prompts:
                v.update(
                    {
                        "original_prompt": row.get("original_prompt", ""),
                        "expected_questions": row.get("expected_questions", ""),
                        "expected_failure_mode": row.get("expected_failure_mode", ""),
                        "removed_segments_json": row.get("removed_segments_json", ""),
                    }
                )
            variants[row["variant_id"]] = v
    return variants


def _load_all_trial_responses(
    run_dir: Path, task: str, include_tool_calls: bool = True, max_events: int = 30
) -> list[str]:
    """Load agent responses for ALL trials from trajectory events.

    Each trial has its own date_sessionId directory. We sort alphabetically
    to match evaluation_0.json, evaluation_1.json, etc.

    Args:
        run_dir: Path to experiment run directory
        task: Task name
        include_tool_calls: If True, include tool calls and results (for judge context)
        max_events: Maximum events to include per trial (last N events)

    Returns list of response strings (one per trial, in order).
    """
    local_results = run_dir / "local_results" / task
    if not local_results.exists():
        return []

    responses = []
    try:
        # Find model dir
        for model_dir in sorted(local_results.iterdir()):
            if not model_dir.is_dir():
                continue

            # Alphabetical sort matches run_deterministic_evaluation's sort order
            # (sorted by path in task_completion_agentcompany.py line 214)
            date_dirs = sorted(
                d
                for d in model_dir.iterdir()
                if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)
            )

            for date_dir in date_dirs:
                sessions_dir = date_dir / "state" / "sessions"
                if not sessions_dir.exists():
                    responses.append("")
                    continue

                session_dirs = sorted([d for d in sessions_dir.iterdir() if d.is_dir()])
                if not session_dirs:
                    responses.append("")
                    continue

                events_dir = session_dirs[0] / "events"
                if not events_dir.exists():
                    responses.append("")
                    continue

                # Load events and extract agent's work/conclusions
                event_files = sorted(
                    [f for f in events_dir.glob("*.json") if f.stem.isdigit()],
                    key=lambda x: int(x.stem),
                )

                trajectory_parts = []
                for ev_file in event_files:
                    try:
                        ev = json.loads(ev_file.read_text(encoding="utf-8").strip())
                        source = ev.get("source", "")
                        action = ev.get("action", "")
                        message = ev.get("message", "")
                        args = ev.get("args", {})

                        # Include tool calls with their arguments (shows what agent did)
                        if include_tool_calls and action and action not in ("", "null"):
                            if action == "run":
                                # Command execution
                                cmd = args.get("command", "")[:200] if args else ""
                                if cmd:
                                    trajectory_parts.append(f"[TOOL:run] {cmd}")
                            elif action == "browse":
                                url = args.get("url", "")[:100] if args else ""
                                if url:
                                    trajectory_parts.append(f"[TOOL:browse] {url}")
                            elif action == "write":
                                path = args.get("path", "")[:80] if args else ""
                                if path:
                                    trajectory_parts.append(f"[TOOL:write] {path}")
                            elif action == "finish":
                                # Always include finish with full message
                                if message:
                                    trajectory_parts.append(f"[FINISH] {message.strip()}")

                        # Capture agent messages (reasoning/conclusions)
                        if source == "agent" and message:
                            msg = message.strip()
                            # Include substantive messages (skip very short ones)
                            if len(msg) > 30 or action == "finish":
                                # Avoid duplicating finish message
                                if action != "finish":
                                    trajectory_parts.append(f"[AGENT] {msg[:500]}")
                    except (json.JSONDecodeError, KeyError):
                        continue

                # Take last N events to capture the agent's final work
                # This gives the judge full context of what the agent actually did
                response = "\n\n".join(trajectory_parts[-max_events:]) if trajectory_parts else ""
                responses.append(response)

            break  # Only process first model dir
    except Exception:
        pass  # Best-effort response loading

    return responses


def _load_trials(
    run_dir: Path, task: str, model: str = None, skip_responses: bool = False
) -> list[dict]:
    """Load trials from evaluation JSONs with per-trial agent responses.

    Args:
        run_dir: Path to experiment run directory
        task: Task name
        model: Model subdirectory name. If None, auto-detects (each run has one model).
        skip_responses: If True, skip loading agent responses (faster for reports).
    """
    det_task = run_dir / "final_results" / "deterministic" / task
    if model:
        det_root = det_task / model
    else:
        model_dirs = (
            sorted(d for d in det_task.iterdir() if d.is_dir()) if det_task.exists() else []
        )
        det_root = model_dirs[0] if model_dirs else det_task / "_missing"
    if not det_root.exists():
        return []

    # Load responses for ALL trials (one per date_sessionId directory)
    # Skip for faster loading when responses aren't needed
    all_responses = [] if skip_responses else _load_all_trial_responses(run_dir, task)

    trials = []
    eval_files = sorted(
        det_root.glob("evaluation_*.json"),
        key=lambda f: int(f.stem.rsplit("_", 1)[-1]) if f.stem.rsplit("_", 1)[-1].isdigit() else -1,
    )

    for i, f in enumerate(eval_files):
        try:
            content = f.read_text(encoding="utf-8").strip()
            if not content:
                continue  # Skip empty files
            data = json.loads(content)

            # Use final_score for accurate pass/fail (not individual checkpoints)
            final_score = data.get("final_score", {})
            score = int(final_score.get("result", 0) or 0)
            total = int(final_score.get("total", 1) or 1)

            cps = data.get("checkpoints", [])
            cp_passed = [
                1 if int(cp.get("result", 0) or 0) == int(cp.get("total", 1) or 1) else 0
                for cp in cps
            ]
            weights = [1 if cp.get("total") in (None, "") else int(cp.get("total")) for cp in cps]
            terminal_state = tuple(r for r, w in zip(cp_passed, weights) if w > 0)
            checkpoint_weights = tuple(w for w in weights if w > 0)  # Match terminal_state length

            # Use the eval file's trial index (not enumerate index) to handle
            # gaps from failed evaluations (e.g. evaluation_0, evaluation_2)
            trial_num = int(f.stem.rsplit("_", 1)[-1]) if f.stem.rsplit("_", 1)[-1].isdigit() else i
            response = all_responses[trial_num] if trial_num < len(all_responses) else ""

            trials.append(
                {
                    "score": score,
                    "total": total,
                    "percent": score / total * 100 if total else 0,
                    "full_pass": score == total and total > 0,
                    "session": f.name,
                    "terminal_state": terminal_state,
                    "checkpoint_weights": checkpoint_weights,  # Filtered to match terminal_state
                    "response": response,  # Per-trial agent response
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return trials


GOLDEN_TRAJECTORY_DIR = REPO_ROOT / PATHS["tac_golden_trajectories"]

_golden_trajectory_cache: dict[str, list[dict]] = {}


def _load_golden_trajectory(task: str, max_events: int = 100) -> list[dict]:
    """Load golden trajectory for a task from local golden_trajectories/ directory.

    Golden trajectories are exported by export_tac_golden_trajectories.py (Stage 1).
    Results are cached in memory.
    """
    if task in _golden_trajectory_cache:
        return _golden_trajectory_cache[task]

    data = None

    local_path = GOLDEN_TRAJECTORY_DIR / f"{task}.json"
    if local_path.exists():
        try:
            data = json.loads(local_path.read_text(encoding="utf-8").strip())
        except (json.JSONDecodeError, OSError):
            pass

    if not data:
        print(f"  No golden trajectory found for {task} at {local_path}, returning empty list")
        _golden_trajectory_cache[task] = []
        return []

    try:
        events = []
        for ev in data[:max_events]:
            source = ev.get("source", "")
            action = ev.get("action", "")
            if action == "system" or source == "":
                continue
            events.append(
                {
                    "id": ev.get("id", 0),
                    "source": source,
                    "action": action,
                    "message": ev.get("message", ""),
                }
            )
        _golden_trajectory_cache[task] = events
        return events
    except (KeyError, TypeError):
        _golden_trajectory_cache[task] = []
        return []


def _format_trajectory_event(ev: dict) -> dict:
    """Format an event for display. Returns dict with type, content, is_clar, step."""
    action = ev.get("action", "")
    observation = ev.get("observation", "")
    source = ev.get("source", "")
    step = ev.get("id", ev.get("_id", 0))

    result = {"step": step, "type": action, "content": "", "is_clar": False, "show": True}

    if action == "call_tool_mcp":
        args = ev.get("args", {})
        result["type"] = "mcp_call"
        tool_name = args.get("name", "") or ev.get("tool_call_metadata", {}).get(
            "function_name", ""
        )
        func_args = args.get("arguments", {})
        if isinstance(func_args, str):
            try:
                func_args = json.loads(func_args)
            except json.JSONDecodeError:
                pass
        result["content"] = func_args.get(
            "question",
            json.dumps(func_args, indent=2) if isinstance(func_args, dict) else str(func_args),
        )
        result["is_clar"] = "ask_user" in tool_name.lower() or "clarification" in tool_name.lower()

    elif observation == "mcp":
        result["type"] = "mcp_response"
        tool_name = ev.get("tool_call_metadata", {}).get("function_name", "") or ev.get(
            "extras", {}
        ).get("name", "")
        msg = ev.get("message", "") or ev.get("content", "")
        if isinstance(msg, str):
            try:
                msg_data = json.loads(msg)
                result["content"] = msg_data.get("structuredContent", {}).get("result", msg)
            except json.JSONDecodeError:
                result["content"] = msg
        else:
            result["content"] = str(msg)
        result["is_clar"] = "ask_user" in tool_name.lower() or "clarification" in tool_name.lower()

    elif action == "run":
        result["type"] = "bash"
        result["content"] = ev.get("args", {}).get("command", "")

    elif action == "read":
        result["type"] = "read"
        result["content"] = ev.get("args", {}).get("path", "")

    elif action in ("edit", "write"):
        result["type"] = action
        result["content"] = ev.get("args", {}).get("path", "")

    elif action == "finish":
        result["type"] = "finish"
        result["content"] = ev.get("args", {}).get("outputs", {}).get("content", "")

    elif action == "message" and source == "user":
        result["type"] = "user_msg"
        result["content"] = ev.get("message", "")

    elif action is None or action == "":
        obs = ev.get("observation", "")
        content = ev.get("content", "")
        if obs or content:
            result["type"] = "output"
            result["content"] = content[:500] + "..." if len(str(content)) > 500 else content
        else:
            result["show"] = False

    else:
        result["show"] = False

    return result


def _load_trajectory_events(run_dir: Path, task: str, max_events: int = 100) -> list[dict]:
    """Load trajectory events from a run's local_results.

    Handles both old structure (single date dir) and new structure (date_sessionId per trial).
    Returns formatted events ready for display.
    """
    local_results = run_dir / "local_results" / task
    if not local_results.exists():
        return []

    events = []
    try:
        for model_dir in sorted(local_results.iterdir()):
            if not model_dir.is_dir():
                continue

            for date_dir in sorted(
                d
                for d in model_dir.iterdir()
                if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)
            ):
                sessions_dir = date_dir / "state" / "sessions"
                if not sessions_dir.exists():
                    continue

                session_dirs = sorted([d for d in sessions_dir.iterdir() if d.is_dir()])
                if not session_dirs:
                    continue

                events_dir = session_dirs[0] / "events"
                if not events_dir.exists():
                    continue

                event_files = sorted(
                    [f for f in events_dir.glob("*.json") if f.stem.isdigit()],
                    key=lambda x: int(x.stem),
                )

                for ev_file in event_files[:max_events]:
                    try:
                        content = ev_file.read_text(encoding="utf-8").strip()
                        if not content:
                            continue
                        ev = json.loads(content)
                        ev["id"] = int(ev_file.stem)  # Use file number as ID
                        formatted = _format_trajectory_event(ev)
                        if formatted["show"]:
                            events.append(formatted)
                    except (json.JSONDecodeError, KeyError):
                        continue

                if events:
                    return events
    except Exception:
        pass  # Best-effort event loading
    return events


def load_results(experiment: str, latest_only: bool = True) -> dict:
    """Load evaluation results scoped by experiment + variant_id."""
    variants = load_variants(experiment)
    if not variants:
        return {}

    valid_vids = set(variants.keys())
    runs_by_vid = defaultdict(list)

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        meta = parse_run_dir(run_dir.name, experiment, valid_vids)
        if meta:
            runs_by_vid[meta["variant_id"]].append((meta["timestamp"], run_dir, meta))

    results = defaultdict(list)
    for vid, runs in runs_by_vid.items():
        # Sort by timestamp descending (newest first)
        sorted_runs = sorted(runs, key=lambda x: x[0], reverse=True)

        for ts, run_dir, meta in sorted_runs:
            task = meta["base_task"]
            trials = _load_trials(run_dir, task)
            if trials:
                # Detect model from deterministic dir structure
                det_task = run_dir / "final_results" / "deterministic" / task
                model_dirs = (
                    sorted(d.name for d in det_task.iterdir() if d.is_dir())
                    if det_task.exists()
                    else []
                )
                model_name = model_dirs[0] if model_dirs else "unknown"
                for t in trials:
                    t.update(
                        {
                            "task": task,
                            "dimension": meta["dimension"],
                            "variant_id": vid,
                            "model": model_name,
                            "run_dir": run_dir.name,
                            "timestamp": ts,
                        }
                    )
                    results[vid].append(t)

                # If latest_only, stop after first non-empty run
                if latest_only:
                    break
            # If this run is empty, continue to next (older) run

    return dict(results)


# =============================================================================
# DataFrames
# =============================================================================


def _load_cached_judge_results(experiment: str) -> dict[str, dict]:
    """Load cached judge verdicts from existing results CSV.

    Returns dict: variant_id -> {judge_verdict, judge_reasoning, ambiguity_class}
    """
    csv_path = SYNTH_OUTPUTS / experiment / f"underspec_results_{experiment}.csv"
    if not csv_path.exists():
        return {}

    cache = {}
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            vid = row.get("variant_id")
            # Only cache if we have a judge verdict (not NaN)
            if vid and pd.notna(row.get("judge_verdict")):
                cache[vid] = {
                    "judge_verdict": row["judge_verdict"],
                    "judge_reasoning": row.get("judge_reasoning", ""),
                    "ambiguity_class": row.get("ambiguity_class", ""),
                }
    except Exception:
        pass  # Start fresh without cache
    return cache


def build_summary_df(
    experiments: str | list[str],
    target_k: int = DEFAULTS["num_trials"],
    judge_model: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """One row per variant with aggregated metrics. Supports multiple experiments.

    Args:
        experiments: Experiment name(s)
        target_k: Target k for pass@k calculation (default: DEFAULTS["num_trials"])
        judge_model: LLM model for new-task judging. If provided, will call LLM to
                     determine if "new_task eligible" variants actually completed a
                     coherent task (just different from original).
        show_progress: If True, show progress bar when processing variants
    """
    if isinstance(experiments, str):
        experiments = [experiments]

    all_rows = []
    for experiment in experiments:
        variants = load_variants(experiment)
        results = load_results(experiment)

        # Load cached judge results to avoid re-calling LLM
        judge_cache = _load_cached_judge_results(experiment)
        if judge_cache:
            print(f"  📋 Using {len(judge_cache)} cached judge results for {experiment}")

        variant_ids = sorted(set(variants.keys()) | set(results.keys()))
        iterator = (
            tqdm(variant_ids, desc=f"Processing {experiment}") if show_progress else variant_ids
        )

        for vid in iterator:
            v = variants.get(vid, {})
            trials = results.get(vid, [])

            row = {
                "experiment": experiment,
                "variant_id": vid,
                "task": v.get("task", trials[0]["task"] if trials else ""),
                "dimension": v.get("dimension", trials[0]["dimension"] if trials else ""),
                "subdimension": v.get("subdimension", ""),  # For diversity filtering
                "removed_value": v.get("removed_value", ""),  # For similarity dedup
                "criticality": v.get("criticality"),
                "guessability": v.get("guessability"),
                "priority_score": v.get("priority_score"),
            }

            if trials:
                n, c = len(trials), sum(1 for t in trials if t["full_pass"])
                total = trials[0]["total"]
                avg = sum(t["score"] for t in trials) / n
                # pass_at_k handles n < k internally via fallback estimator

                # Terminal state divergence analysis
                terminal_states = [t["terminal_state"] for t in trials if t.get("terminal_state")]
                n_unique_states = len(set(terminal_states))

                # Outcome divergence: multiple different terminal states
                # High divergence = agents reach different outcomes = ambiguity matters
                outcome_divergence = n_unique_states > 1
                divergence_ratio = n_unique_states / n if n > 0 else 0

                # Terminal state entropy (higher = more divergence)
                state_counts = Counter(terminal_states)
                state_entropy = 0.0
                if n > 0 and n_unique_states > 1:
                    for count in state_counts.values():
                        p = count / n
                        if p > 0:
                            state_entropy -= p * math.log2(p)

                # Outcome classification with LLM judge for new_task detection
                # - outcome-critical: all trials failed
                # - benign: all trials succeeded
                # - divergent: some trials succeeded
                all_failed = c == 0
                all_succeeded = c == n
                some_succeeded = 0 < c < n
                eligible_new_task = all_failed and n_unique_states == 1
                judge_verdict = None
                judge_reasoning = None

                if all_failed:
                    if n_unique_states == 1:  # new_task candidate
                        # Check cache first to avoid re-calling LLM
                        cached = judge_cache.get(vid)
                        if cached and cached.get("judge_verdict") is not None:
                            judge_verdict = cached["judge_verdict"]
                            judge_reasoning = cached.get("judge_reasoning", "(cached)")
                            ambiguity_class = cached.get("ambiguity_class", "")
                        elif judge_model:
                            # Call LLM judge to distinguish new_task vs outcome-critical
                            underspec_prompt = v.get("underspecified_prompt", "")
                            responses = [t.get("response", "") for t in trials]

                            judge_result = judge_new_task_sync(
                                underspec_prompt,
                                responses,
                                judge_model,
                            )
                            judge_verdict = bool(judge_result.get("task_completed"))
                            judge_reasoning = judge_result.get("reasoning", "")
                            ambiguity_class = "new_task" if judge_verdict else "outcome-critical"
                        else:
                            # No judge requested — default to outcome-critical
                            ambiguity_class = "outcome-critical"

                    else:
                        ambiguity_class = "outcome-critical"  # Divergent failures
                elif all_succeeded:
                    ambiguity_class = "benign"
                elif some_succeeded:
                    ambiguity_class = "divergent"
                else:
                    ambiguity_class = "unknown"

                row.update(
                    {
                        "n_trials": n,
                        "full_passes": c,
                        "max_score": total,
                        "avg_score": round(avg, 2),
                        "avg_pct": round(avg / total * 100, 1) if total else 0,
                        "success_rate": round(c / n * 100, 1),
                        "pass_at_1": round(
                            pass_at_k(n, c, 1), 3
                        ),  # Unbiased estimator (equals c/n for k=1)
                        "pass_at_3": round(pass_at_k(n, c, 3), 3),  # Unbiased estimator for k=3
                        "pass_at_N": round(pass_at_k(n, c, target_k), 3),
                        "target_k": target_k,
                        # Terminal state analysis
                        "n_unique_states": n_unique_states,
                        "outcome_divergence": outcome_divergence,
                        "divergence_ratio": round(divergence_ratio, 3),
                        "state_entropy": round(state_entropy, 3),
                        "terminal_states": str(sorted(set(terminal_states))),
                        # Ambiguity classification (single source of truth)
                        "ambiguity_class": ambiguity_class,
                        "new_task_eligible": eligible_new_task,
                        "judge_verdict": judge_verdict,
                        "judge_reasoning": judge_reasoning,
                        "judge_model": judge_model if eligible_new_task else None,
                        "model": trials[0]["model"],
                    }
                )
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def build_trials_df(experiments: str | list[str]) -> pd.DataFrame:
    """One row per trial. Supports multiple experiments."""
    if isinstance(experiments, str):
        experiments = [experiments]

    all_rows = []
    for experiment in experiments:
        variants = load_variants(experiment)
        results = load_results(experiment)

        for vid, trials in sorted(results.items()):
            v = variants.get(vid, {})
            for i, t in enumerate(trials):
                all_rows.append(
                    {
                        "experiment": experiment,
                        "variant_id": vid,
                        "task": t["task"],
                        "dimension": t["dimension"],
                        "model": t["model"],
                        "trial_idx": i,
                        "run_dir": t["run_dir"],
                        "score": t["score"],
                        "max_score": t["total"],
                        "percent": t["percent"],
                        "full_pass": t["full_pass"],
                        "criticality": v.get("criticality"),
                        "guessability": v.get("guessability"),
                    }
                )
    return pd.DataFrame(all_rows)


# =============================================================================
# Visualization
# =============================================================================


def _plot_pass_rate_by_task(ax, df, colors):
    """Plot grouped horizontal bar chart of pass rates by task. Returns k_val."""
    task_stats = df.groupby("task").agg(
        {
            "avg_pct": "mean",
            "pass_at_1": "mean",
            "pass_at_N": "mean",
            "n_trials": "mean",
            "ambiguity_class": "count",
        }
    )
    task_stats.columns = [
        "avg_pct",
        "pass_at_1",
        "pass_at_N",
        "n_trials",
        "n_variants",
    ]
    task_stats["n_critical"] = df.groupby("task")["ambiguity_class"].apply(
        lambda x: (x == "outcome-critical").sum()
    )
    task_stats = task_stats.fillna(0)

    y = range(len(task_stats))
    bar_height = 0.25
    k_val = (
        int(task_stats["target_k"].mean())
        if "target_k" in task_stats.columns and len(task_stats)
        else DEFAULTS["num_trials"]
    )

    ax.barh(
        [i - bar_height for i in y],
        task_stats["avg_pct"],
        height=bar_height,
        color=colors["primary"],
        alpha=0.85,
        label="Avg Checkpoint %",
    )
    ax.barh(
        [i for i in y],
        task_stats["pass_at_1"] * 100,
        height=bar_height,
        color=colors["warning"],
        alpha=0.85,
        label="Pass@1 (≥1 success)",
    )
    ax.barh(
        [i + bar_height for i in y],
        task_stats["pass_at_N"] * 100,
        height=bar_height,
        color=colors["success"],
        alpha=0.85,
        label=f"Pass@{k_val} (≥1 in {k_val} runs)",
    )

    ax.set_yticks(y)
    ax.set_yticklabels([t[:28] for t in task_stats.index], fontsize=8)
    ax.set_xlabel("Percentage")
    ax.set_title("Performance by Task (avg across variants)", fontweight="bold")
    ax.set_xlim(0, 115)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (t, row) in enumerate(task_stats.iterrows()):
        max_val = max(row["avg_pct"], row["pass_at_1"] * 100, row["pass_at_N"] * 100)
        n_var = int(row["n_variants"])
        n_crit = int(row["n_critical"])
        label = f"n={n_var}, {n_crit} fail"
        ax.text(max_val + 2, i, label, va="center", fontsize=7, color=colors["neutral"])

    return k_val


def _plot_pass_rate_by_dimension(ax, df, colors):
    """Plot bar chart of pass rates by dimension."""
    df["base_dim"] = df["dimension"].apply(lambda x: x.split("+")[0] if "+" in str(x) else str(x))
    base_dims = ["goal", "input", "constraint", "context"]
    base_dims = [d for d in base_dims if d in df["base_dim"].values]

    dim_stats = df.groupby("base_dim").agg({"avg_pct": ["mean", "count"]}).round(1)
    dim_stats.columns = ["avg", "n"]
    dim_stats = dim_stats.reindex([d for d in base_dims if d in dim_stats.index])

    dim_colors = {
        "goal": colors["danger"],
        "input": colors["primary"],
        "constraint": colors["success"],
        "context": colors["secondary"],
    }
    ax.bar(
        range(len(dim_stats)),
        dim_stats["avg"],
        color=[dim_colors.get(d, colors["neutral"]) for d in dim_stats.index],
        alpha=0.85,
        edgecolor="white",
        linewidth=1,
    )
    ax.set_xticks(range(len(dim_stats)))
    ax.set_xticklabels([f"{d}\n(n={int(dim_stats.loc[d, 'n'])})" for d in dim_stats.index])
    ax.set_ylabel("Avg Pass %")
    ax.set_title("Pass Rate by Dimension", fontweight="bold")
    ax.set_ylim(0, max(dim_stats["avg"]) * 1.3 if len(dim_stats) else 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, (d, row) in enumerate(dim_stats.iterrows()):
        ax.text(
            i, row["avg"] + 1, f"{row['avg']:.1f}%", ha="center", fontsize=10, fontweight="bold"
        )


def _plot_outcome_critical_breakdown(ax, df, colors):
    """Plot stacked bar chart of outcome-critical vs non-critical by experiment."""
    exp_total = df.groupby("experiment").size().rename("total")
    exp_critical = (
        df[df["ambiguity_class"] == "outcome-critical"]
        .groupby("experiment")
        .size()
        .rename("critical")
    )
    exp_stats = pd.concat([exp_critical, exp_total], axis=1).fillna(0).astype(int)
    exp_stats["non_critical"] = exp_stats["total"] - exp_stats["critical"]

    x = range(len(exp_stats))
    ax.bar(x, exp_stats["critical"], label="Outcome-Critical", color=colors["danger"], alpha=0.85)
    ax.bar(
        x,
        exp_stats["non_critical"],
        bottom=exp_stats["critical"],
        label="Other Classes",
        color=colors["success"],
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("v1_", "") for e in exp_stats.index], rotation=0)
    ax.set_ylabel("Variants")
    ax.set_title("Outcome-Critical by Experiment", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_segments_summary(ax, df, k_val, colors):
    """Plot summary table of results by number of segments removed."""
    ax.axis("off")

    df["n_segments"] = df["dimension"].apply(lambda x: str(x).count("+") + 1)
    df["zero_pass"] = df["pass_at_1"] == 0
    seg_stats = df.groupby("n_segments").agg(
        {
            "avg_pct": "mean",
            "pass_at_1": "mean",
            "pass_at_N": "mean",
            "zero_pass": "sum",
            "variant_id": "count",
        }
    )
    seg_stats.columns = ["avg_ckpt_pct", "avg_pass1", "avg_passN", "n_zero", "total"]

    col_labels = [
        "# Seg",
        "n",
        "Avg Ckpt %",
        "pass@1\n(≥1 success)",
        f"pass@{k_val}\n(≥1 in {k_val} runs)",
        f"# Fail\n(0% in {k_val} runs)",
    ]
    table_data = []
    for n in seg_stats.index:
        row = seg_stats.loc[n]
        table_data.append(
            [
                f"{n}",
                f"{int(row['total'])}",
                f"{row['avg_ckpt_pct']:.1f}%",
                f"{row['avg_pass1']:.1%}",
                f"{row['avg_passN']:.1%}",
                f"{int(row['n_zero'])}",
            ]
        )

    total_variants = int(seg_stats["total"].sum())
    total_zero = int(seg_stats["n_zero"].sum())
    avg_ckpt_all = df["avg_pct"].mean()
    avg_pass1_all = df["pass_at_1"].mean()
    avg_passN_all = df["pass_at_N"].mean()
    table_data.append(
        [
            "All",
            f"{total_variants}",
            f"{avg_ckpt_all:.1f}%",
            f"{avg_pass1_all:.1%}",
            f"{avg_passN_all:.1%}",
            f"{total_zero}",
        ]
    )

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.3, 2.2)

    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor(colors["primary"])
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    n_rows = len(table_data)
    for i in range(1, n_rows + 1):
        for j in range(len(col_labels)):
            if i == n_rows:
                table[(i, j)].set_facecolor("#1e40af")
                table[(i, j)].set_text_props(color="white", fontweight="bold")
            elif i % 2 == 0:
                table[(i, j)].set_facecolor(colors["light"])

    ax.set_title("Results by # Segments Removed", fontsize=12, fontweight="bold", pad=20)


def plot_results(experiments: str | list[str], save_path: Path = None):
    """Generate underspec results visualization. Supports multiple experiments."""
    if isinstance(experiments, str):
        experiments = [experiments]

    df = build_summary_df(experiments)
    if df.empty or "n_trials" not in df.columns:
        print("No results to visualize")
        return

    df = df[df["n_trials"].notna()].copy()
    if df.empty:
        print("No evaluated variants")
        return

    title = ", ".join(experiments) if len(experiments) <= 3 else f"{len(experiments)} experiments"

    colors = {
        "primary": "#2563eb",
        "secondary": "#7c3aed",
        "success": "#059669",
        "warning": "#d97706",
        "danger": "#dc2626",
        "neutral": "#6b7280",
        "light": "#f3f4f6",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Underspecification Results: {title}", fontsize=14, fontweight="bold")

    k_val = _plot_pass_rate_by_task(axes[0, 0], df, colors)
    _plot_pass_rate_by_dimension(axes[0, 1], df, colors)
    _plot_outcome_critical_breakdown(axes[1, 0], df, colors)
    _plot_segments_summary(axes[1, 1], df, k_val, colors)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()

    return df  # Return df for further analysis


# =============================================================================
# HTML Review Export
# =============================================================================


def _esc(s: str) -> str:
    """Escape HTML special characters."""
    return html_module.escape(str(s)) if s else ""


def _terminal_state_str(ts: tuple, weights: tuple = None) -> str:
    """Format terminal state tuple as string like (1,0,1) or (1×3,0×2,1×1) if weights vary.

    Args:
        ts: Terminal state tuple of pass/fail values (1 = passed, 0 = failed)
        weights: Optional tuple of checkpoint weights (must match ts length).
    """
    if not ts:
        return "()"

    # If weights provided and they match length, check if they vary
    if weights and len(weights) == len(ts):
        unique_weights = set(weights)
        if len(unique_weights) > 1:  # Weights vary - show weighted format
            # Format: (1×3,0×2,1×1) where p=pass/fail (1 or 0), w=weight
            return "(" + ",".join(f"{p}×{w}" for p, w in zip(ts, weights)) + ")"

    # Simple format (uniform weights, no weights, or length mismatch)
    return "(" + ",".join(str(x) for x in ts) + ")"


def _build_review_html_styles():
    """Return the CSS <style> block for the review HTML page."""
    return """    <style>
        * { box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #0f0f1a; color: #e0e0e0; }
        .header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 24px; margin: -20px -20px 20px -20px; border-bottom: 2px solid #4299e1; }
        .header h1 { margin: 0 0 12px 0; font-size: 28px; }
        .stats { display: flex; gap: 16px; flex-wrap: wrap; }
        .stat { background: rgba(255,255,255,0.08); padding: 10px 16px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); }
        .stat-value { font-size: 24px; font-weight: bold; color: #4299e1; }
        .stat-label { font-size: 11px; opacity: 0.7; margin-top: 2px; }
        .controls { background: #1a1a2e; padding: 14px; border-radius: 10px; margin-bottom: 20px; display: flex; gap: 12px; align-items: center; border: 1px solid #2d3748; }
        .btn { background: #4299e1; color: white; border: none; padding: 10px 16px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; }
        .btn:hover { background: #3182ce; }
        .btn.secondary { background: #2d3748; }
        input { padding: 10px 14px; border: 1px solid #2d3748; border-radius: 6px; min-width: 250px; background: #0f0f1a; color: #e0e0e0; }
        .variant { background: #1a1a2e; border-radius: 10px; margin-bottom: 16px; border: 1px solid #2d3748; overflow: hidden; }
        .variant-header { background: #2d3748; color: white; padding: 16px 20px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; }
        .variant-header:hover { background: #3d4758; }
        .variant-id { font-family: 'SF Mono', Monaco, monospace; font-size: 13px; }
        .badge { padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; margin-left: 10px; }
        .badge.goal { background: #48bb78; color: white; }
        .badge.input { background: #4299e1; color: white; }
        .badge.constraint { background: #ed8936; color: white; }
        .badge.pass { background: #48bb78; color: white; }
        .badge.fail { background: #f56565; color: white; }
        .badge.norun { background: #4a5568; color: #a0aec0; }
        .variant-body { padding: 20px; display: none; }
        .variant.expanded .variant-body { display: block; }
        .section { margin-bottom: 20px; }
        .section-header { font-weight: 600; margin-bottom: 10px; font-size: 14px; color: #a0aec0; }
        .removed { background: rgba(220, 38, 38, 0.1); padding: 12px 16px; border-radius: 8px; border-left: 4px solid #dc2626; font-family: monospace; font-size: 12px; color: #fca5a5; }
        .prompts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
        .prompt-box { padding: 14px; border-radius: 8px; font-family: 'SF Mono', Monaco, monospace; font-size: 11px; white-space: pre-wrap; word-break: break-word; max-height: 350px; overflow-y: auto; line-height: 1.5; }
        .prompt-box.original { background: rgba(46, 125, 50, 0.1); border: 1px solid rgba(46, 125, 50, 0.3); color: #a5d6a7; }
        .prompt-box.underspec { background: rgba(230, 81, 0, 0.1); border: 1px solid rgba(230, 81, 0, 0.3); color: #ffcc80; }
        .prompt-box h4 { margin: 0 0 10px 0; font-size: 12px; font-weight: 600; }
        .meta { display: flex; gap: 20px; font-size: 12px; color: #a0aec0; }
        .trials-table { width: 100%; border-collapse: collapse; font-size: 11px; margin-top: 12px; }
        .trials-table th, .trials-table td { padding: 6px 8px; text-align: center; border-bottom: 1px solid #2d3748; }
        .trials-table th { background: #2d3748; color: #a0aec0; font-weight: 600; }
        .trials-table tr:hover { background: rgba(255,255,255,0.03); }
        .terminal-state { font-family: 'SF Mono', Monaco, monospace; font-size: 10px; }
        .trajectory-card { background: rgba(255,255,255,0.03); border-radius: 8px; padding: 14px; border: 1px solid #2d3748; border-left: 3px solid #ed8936; max-height: 600px; overflow-y: auto; margin-top: 16px; }
        .trajectory-card h5 { margin: 0 0 12px 0; font-size: 13px; color: #a0aec0; }
        .event { padding: 6px 10px; margin-bottom: 4px; border-radius: 4px; font-size: 11px; }
        .event.agent { background: rgba(66, 153, 225, 0.1); border-left: 2px solid #4299e1; }
        .event.user { background: rgba(72, 187, 120, 0.1); border-left: 2px solid #48bb78; }
        .event .event-header { display: flex; justify-content: space-between; margin-bottom: 2px; }
        .event .event-id { color: #718096; font-size: 10px; }
        .event .event-action { color: #9f7aea; font-weight: 500; font-size: 10px; }
        .event .event-msg { color: #e0e0e0; font-family: 'SF Mono', Monaco, monospace; font-size: 10px; white-space: pre-wrap; word-break: break-word; }
        .ev-mcp { padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 11px; background: rgba(236,201,75,0.15); border-left: 3px solid #ecc94b; }
        .ev-mcp b { color: #ecc94b; }
        .ev-response { padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 11px; background: rgba(56,178,172,0.15); border-left: 3px solid #38b2ac; }
        .ev-response b { color: #38b2ac; }
        .ev-bash { padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 11px; background: rgba(160,174,192,0.08); border-left: 3px solid #718096; }
        .ev-read { padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 11px; background: rgba(99,179,237,0.08); border-left: 3px solid #4299e1; }
        .ev-edit { padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 11px; background: rgba(237,137,54,0.08); border-left: 3px solid #ed8936; }
        .ev-output { padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 11px; background: rgba(0,0,0,0.3); border-left: 3px solid #4a5568; }
        .ev-finish { padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 11px; background: rgba(159,122,234,0.08); border-left: 3px solid #805ad5; }
        .ev-action { padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 11px; background: rgba(66,153,225,0.08); border-left: 3px solid #4299e1; }
        .ev-mcp pre, .ev-response pre, .ev-bash pre, .ev-read pre, .ev-edit pre, .ev-output pre { background: rgba(0,0,0,0.3); padding: 8px; border-radius: 4px; font-size: 10px; white-space: pre-wrap; word-break: break-word; max-height: 200px; overflow-y: auto; margin: 6px 0 0 0; font-family: 'SF Mono', Monaco, monospace; }
        .no-data { color: #718096; font-style: italic; padding: 12px; text-align: center; }
        .tab-bar { display: flex; gap: 4px; margin-bottom: 16px; }
        .tab { padding: 8px 16px; border-radius: 6px 6px 0 0; cursor: pointer; font-size: 12px; background: #2d3748; color: #a0aec0; border: 1px solid #3d4758; border-bottom: none; }
        .tab.active { background: #1a1a2e; color: #e0e0e0; border-color: #4299e1; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>"""


def _render_trajectory_events(events, _esc_func):
    """Render trajectory events as HTML string."""
    if not events:
        return ""
    html = (
        f'<div style="font-size:10px;color:#718096;margin-bottom:8px;">{len(events)} events</div>'
    )
    for ev in events:
        step = ev.get("step", ev.get("id", 0))
        ev_type = ev.get("type", "")
        content = _esc_func(str(ev.get("content", ev.get("message", ""))))
        is_clar = ev.get("is_clar", False)

        if ev_type == "mcp_call":
            label = "ASK USER" if is_clar else "MCP"
            css_class = "ev-mcp" if is_clar else "ev-action"
            html += f'<div class="{css_class}"><b>[{step}] {label}</b><pre>{content}</pre></div>'
        elif ev_type == "mcp_response":
            label = "USER RESPONSE" if is_clar else "MCP RESULT"
            css_class = "ev-response" if is_clar else "ev-output"
            html += f'<div class="{css_class}"><b>[{step}] {label}</b><pre>{content}</pre></div>'
        elif ev_type == "bash":
            html += f'<div class="ev-bash"><b>[{step}] RUN</b><pre>{content}</pre></div>'
        elif ev_type == "read":
            html += f'<div class="ev-read"><b>[{step}] READ</b><pre>{content}</pre></div>'
        elif ev_type in ("edit", "write"):
            html += (
                f'<div class="ev-edit"><b>[{step}] {ev_type.upper()}</b><pre>{content}</pre></div>'
            )
        elif ev_type == "output":
            html += f'<div class="ev-output"><b>[{step}] OUTPUT</b><pre>{content}</pre></div>'
        elif ev_type == "finish":
            html += f'<div class="ev-finish"><b>[{step}] FINISH</b></div>'
        else:
            source = ev.get("source", "agent")
            action = ev.get("action", "")
            msg = _esc_func(ev.get("message", ""))
            html += f'<div class="event {source}"><b>[{step}] {_esc_func(action)}</b><pre>{msg}</pre></div>'
    return html


def _build_variant_card_html(v, i, _esc_func):
    """Build HTML for a single variant card. Returns an HTML string."""
    expanded = "expanded" if i == 0 else ""
    dim_class = v["dimension"].split("+")[0] if "+" in v["dimension"] else v["dimension"]
    removed_preview = _esc_func(
        v["removed_value"][:300] + "..." if len(v["removed_value"]) > 300 else v["removed_value"]
    )
    orig = _esc_func(v["original_prompt"]) if v["original_prompt"] else "(empty)"
    under = _esc_func(v["underspecified_prompt"]) if v["underspecified_prompt"] else "(empty)"
    search_data = f"{v['variant_id']} {v['task']} {v['dimension']}".lower()

    trials = v["trials"]
    if trials:
        n_pass = sum(1 for t in trials if t["full_pass"])
        is_any_pass = n_pass > 0
        total = trials[0]["total"]
        avg_score = sum(t["score"] for t in trials) / len(trials)
        run_badge = f'<span class="badge {"pass" if is_any_pass else "fail"}">{avg_score:.1f}/{total} ckpts | {n_pass}/{len(trials)} trials</span>'
    else:
        run_badge = '<span class="badge norun">No Run</span>'

    trials_html = ""
    if trials:
        rows = "".join(
            [
                f"""<tr>
                    <td>#{j}</td>
                    <td>{t["score"]}/{t["total"]} ({t["percent"]:.0f}%)</td>
                    <td class="terminal-state">{_terminal_state_str(t.get("terminal_state", ()), t.get("checkpoint_weights"))}</td>
                    <td>{"✅" if t["full_pass"] else "❌"}</td>
                </tr>"""
                for j, t in enumerate(trials[:8])
            ]
        )
        more = (
            f'<div style="color:#718096;font-size:11px;margin-top:8px;">...and {len(trials)-8} more</div>'
            if len(trials) > 8
            else ""
        )
        trials_html = f"""
                <h5 style="color:#a0aec0;">📊 All Trials ({len(trials)} runs)</h5>
                <table class="trials-table">
                    <thead><tr><th>Trial</th><th>Score</th><th>Terminal State</th><th>Status</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
                {more}
            """
    else:
        trials_html = '<div class="no-data">No evaluation results</div>'

    gt_traj = v.get("golden_trajectory", [])
    us_traj = v.get("trajectory", [])

    gt_html = (
        _render_trajectory_events(gt_traj, _esc_func)
        if gt_traj
        else '<div class="no-data">No GT trajectory</div>'
    )
    us_html = (
        _render_trajectory_events(us_traj, _esc_func)
        if us_traj
        else '<div class="no-data">Trajectory not preserved</div>'
    )

    traj_html = f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px;">
                <div class="trajectory-card" style="border-left-color:#48bb78;">
                    <h5>🟢 GT Trajectory (baseline task)</h5>
                    {gt_html}
                </div>
                <div class="trajectory-card">
                    <h5>🟠 Underspec Trajectory (Sample)</h5>
                    {us_html}
                </div>
            </div>
        """

    return f"""
    <div class="variant {expanded}" data-search="{_esc_func(search_data)}">
        <div class="variant-header" onclick="this.parentElement.classList.toggle('expanded')">
            <div>
                <span class="variant-id">{_esc_func(v["variant_id"])}</span>
                <span class="badge {dim_class}">{_esc_func(v["dimension"])}</span>
                {run_badge}
            </div>
            <div class="meta">
                <span>Crit: {v["criticality"]:.1f}</span>
                <span>Guess: {v["guessability"]:.1f}</span>
            </div>
        </div>
        <div class="variant-body">
            <div class="tab-bar">
                <div class="tab active" onclick="showTab(this, 'prompts-{i}')">📝 Prompts</div>
                <div class="tab" onclick="showTab(this, 'results-{i}')">📊 Results</div>
            </div>
            
            <div id="prompts-{i}" class="tab-content active">
                <div class="section">
                    <div class="section-header">🗑️ Removed Segment</div>
                    <div class="removed">{removed_preview}</div>
                </div>
                <div class="prompts">
                    <div class="prompt-box original"><h4>🟢 Original Prompt</h4>{orig}</div>
                    <div class="prompt-box underspec"><h4>🟠 Underspecified Prompt</h4>{under}</div>
                </div>
            </div>
            
            <div id="results-{i}" class="tab-content">
                {trials_html}
                {traj_html}
            </div>
        </div>
    </div>
"""


def generate_review_html(
    experiment: str,
    output_path: Path = None,
    max_variants: int = 50,
) -> str:
    """Generate an HTML page with collapsible variant cards for review.

    Includes: prompts, checkpoint results, terminal states, sample trajectory.
    """
    if output_path is None:
        output_path = SYNTH_OUTPUTS / experiment / f"variants_review_{experiment}.html"

    # Load variants with prompts
    variants = load_variants(experiment, include_prompts=True)
    if not variants:
        print(f"❌ No variants found for experiment: {experiment}")
        return None

    # Load results (includes run_dir, trials with terminal states)
    valid_vids = set(variants.keys())
    runs_by_vid = {}
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        meta = parse_run_dir(run_dir.name, experiment, valid_vids)
        if meta:
            vid = meta["variant_id"]
            # Keep latest run per variant
            if vid not in runs_by_vid or meta["timestamp"] > runs_by_vid[vid][0]:
                runs_by_vid[vid] = (meta["timestamp"], run_dir, meta)

    # Build variant info list
    variant_list = []
    for vid in list(variants.keys())[:max_variants]:
        v = variants[vid]
        run_info = runs_by_vid.get(vid)

        task = v["task"]
        info = {
            "variant_id": vid,
            "task": task,
            "dimension": v["dimension"],
            "criticality": v["criticality"],
            "guessability": v["guessability"],
            "original_prompt": v.get("original_prompt", ""),
            "underspecified_prompt": v.get("underspecified_prompt", ""),
            "removed_value": v.get("removed_value", ""),
            "removed_segments_json": v.get("removed_segments_json", ""),
            "trials": [],
            "trajectory": [],
            "golden_trajectory": _load_golden_trajectory(task),
            "has_run": False,
        }

        if run_info:
            ts, run_dir, meta = run_info
            trials = _load_trials(run_dir, meta["base_task"])
            if trials:
                info["trials"] = trials
                info["has_run"] = True
            info["trajectory"] = _load_trajectory_events(run_dir, meta["base_task"])

        variant_list.append(info)

    # Count stats
    n_with_runs = sum(1 for v in variant_list if v["has_run"])

    # Build HTML
    styles = _build_review_html_styles()
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Variant Review: {_esc(experiment)}</title>
    {styles}
</head>
<body>
    <div class="header">
        <h1>📋 Variant Review: {_esc(experiment)}</h1>
        <div class="stats">
            <div class="stat"><div class="stat-value">{len(variant_list)}</div><div class="stat-label">Variants</div></div>
            <div class="stat"><div class="stat-value">{n_with_runs}</div><div class="stat-label">With Runs</div></div>
        </div>
    </div>
    <div class="controls">
        <button class="btn" onclick="toggleAll(true)">Expand All</button>
        <button class="btn secondary" onclick="toggleAll(false)">Collapse All</button>
        <input id="filter" placeholder="Filter by variant ID or task..." oninput="applyFilter()" />
    </div>
"""

    for i, v in enumerate(variant_list):
        html += _build_variant_card_html(v, i, _esc)

    html += """
    <script>
        function toggleAll(expand) {
            document.querySelectorAll('.variant').forEach(v => {
                if (expand) v.classList.add('expanded');
                else v.classList.remove('expanded');
            });
        }
        function applyFilter() {
            const q = document.getElementById('filter').value.toLowerCase();
            document.querySelectorAll('.variant').forEach(v => {
                const s = v.getAttribute('data-search') || '';
                v.style.display = s.includes(q) ? '' : 'none';
            });
        }
        function showTab(tabEl, contentId) {
            const parent = tabEl.closest('.variant-body');
            parent.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            parent.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tabEl.classList.add('active');
            document.getElementById(contentId).classList.add('active');
        }
    </script>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"✅ Generated: {output_path}")
    print(f"   Variants: {len(variant_list)}, With runs: {n_with_runs}")

    return str(output_path)


# =============================================================================
# Display & Export
# =============================================================================


def print_summary(
    experiments: str | list[str],
    show_variants: bool = True,
    judge_model: str | None = None,
):
    """Print experiment summary. Supports multiple experiments."""
    if isinstance(experiments, str):
        experiments = [experiments]

    title = ", ".join(experiments) if len(experiments) <= 3 else f"{len(experiments)} experiments"
    print(f"\n{'='*60}\n{title.upper()}\n{'='*60}")

    df = build_summary_df(experiments, judge_model=judge_model)
    if df.empty:
        print("No variants found")
        return df

    evaluated = df[df["n_trials"].notna()]
    if evaluated.empty:
        print(f"📊 Variants: {len(df)} | No evaluation results yet.")
        return df

    total_trials = int(evaluated["n_trials"].sum())
    critical = int((evaluated["ambiguity_class"] == "outcome-critical").sum())

    print(f"\n📊 Variants: {len(df)} | Evaluated: {len(evaluated)}")
    print(f"📈 Trials: {total_trials} | Outcome-Critical: {critical}/{len(evaluated)}")

    # Show ambiguity class breakdown
    if "ambiguity_class" in evaluated.columns:
        print("\n--- Ambiguity Classification ---")
        class_counts = evaluated["ambiguity_class"].value_counts()
        for cls in ["outcome-critical", "divergent", "benign", "new_task"]:
            if cls in class_counts.index:
                count = class_counts[cls]
                pct = count / len(evaluated) * 100
                print(f"  {cls}: {count} ({pct:.1f}%)")

        # Show judge stats if judge was used
        if "judge_verdict" in evaluated.columns:
            judged = evaluated[evaluated["judge_verdict"].notna()]
            if not judged.empty:
                true_new = (judged["judge_verdict"] == True).sum()  # noqa: E712
                false_new = (judged["judge_verdict"] == False).sum()  # noqa: E712
                print("\n  Judge results (on new_task candidates):")
                print(f"    task_completed=True (new_task): {true_new}")
                print(f"    task_completed=False (outcome-critical): {false_new}")

    # By task summary - average per-variant metrics
    task_agg = (
        evaluated.groupby("task")
        .agg(
            {
                "n_trials": "mean",  # avg trials per variant (should be ~8)
                "pass_at_1": "mean",  # avg pass@1 across variants
                "pass_at_N": "mean",  # avg pass@N across variants
                "avg_pct": "mean",
                "variant_id": "count",  # number of variants
            }
        )
        .rename(columns={"variant_id": "n_variants"})
    )

    k_val = (
        int(evaluated["target_k"].max())
        if "target_k" in evaluated.columns and len(evaluated) > 0
        else DEFAULTS["num_trials"]
    )
    print(f"\n--- Task Summary ({len(task_agg)} tasks, avg across variants) ---")
    print(f"{'task':<40} {'avg_cp%':>8} {'pass@1':>8} {'pass@'+str(k_val):>8} {'variants':>8}")
    print("-" * 76)
    for task, row in task_agg.iterrows():
        print(
            f"{task:<40} {row['avg_pct']:>7.1f}% {row['pass_at_1']:>8.3f} {row['pass_at_N']:>8.3f} {int(row['n_variants']):>8}"
        )

    # Variant-level table
    if show_variants:
        print(f"\n--- Variant Details ({len(evaluated)} variants) ---")
        print(
            f"{'variant_id':<50} {'dim':<10} {'avg%':>6} {'p@1':>6} {'p@N':>6} {'n':>4} {'crit':>4}"
        )
        print("-" * 92)

        for _, row in evaluated.sort_values(["task", "variant_id"]).iterrows():
            vid = row["variant_id"][:49]
            dim = str(row["dimension"])[:9]
            avg = row["avg_pct"]
            p1 = row["pass_at_1"]
            pN = row["pass_at_N"] if pd.notna(row.get("pass_at_N")) else 0
            n = int(row["n_trials"])
            crit = "✗" if row["ambiguity_class"] == "outcome-critical" else ""
            print(f"{vid:<50} {dim:<10} {avg:>5.1f}% {p1:>6.3f} {pN:>6.3f} {n:>4} {crit:>4}")

    return df


def preview_variants(experiments: str | list[str], show_full: bool = False, limit: int = None):
    """Preview generated variants. Supports multiple experiments."""
    if isinstance(experiments, str):
        experiments = [experiments]

    all_dfs = []
    for exp in experiments:
        csv_path = SYNTH_OUTPUTS / exp / "underspec_candidates.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["experiment"] = exp
            all_dfs.append(df)

    if not all_dfs:
        print("No candidates found")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    title = ", ".join(experiments) if len(experiments) <= 3 else f"{len(experiments)} experiments"

    print(f"\n{'='*60}\n{title.upper()} (PREVIEW)\n{'='*60}")
    print(f"\n📊 Variants: {len(df)} | Tasks: {df['task_name'].nunique()}")
    print(f"📁 Dimensions: {dict(df['dimension'].value_counts())}")

    if "priority_score" in df.columns:
        high = (df["priority_score"] >= 0.8).sum()
        med = ((df["priority_score"] >= 0.4) & (df["priority_score"] < 0.8)).sum()
        low = (df["priority_score"] < 0.4).sum()
        print(f"⚡ Priority: high={high}, medium={med}, low={low}")

    print(f"\n{'Variant':<50} {'Dim':<12} {'Removed':<25} {'Pri':>5}")
    print("-" * 95)

    for _, row in (df.head(limit) if limit else df).iterrows():
        vid = row["variant_id"][:49]
        dim = row["dimension"][:11]
        removed = str(row.get("removed_value", ""))[:24]
        pri = row.get("priority_score", 0)
        print(f"{vid:<50} {dim:<12} {removed:<25} {pri:>5.2f}")

        if show_full:
            print(f"\n  ORIGINAL: {row['original_prompt'][:150]}...")
            print(f"  UNDERSPEC: {row['underspecified_prompt'][:150]}...")
            print("-" * 95)


def export_experiment(experiments: str | list[str], output_dir: Path = None):
    """Export to CSV/JSON. Supports multiple experiments."""
    if isinstance(experiments, str):
        experiments = [experiments]

    name = experiments[0] if len(experiments) == 1 else "combined"
    output_dir = Path(output_dir) if output_dir else SYNTH_OUTPUTS / name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = build_summary_df(experiments)
    trials_df = build_trials_df(experiments)

    summary_df.to_csv(output_dir / f"underspec_summary_{name}.csv", index=False, encoding="utf-8")
    trials_df.to_json(output_dir / f"underspec_trials_{name}.json", orient="records", indent=2)

    n_crit = (
        (summary_df["ambiguity_class"] == "outcome-critical").sum()
        if "ambiguity_class" in summary_df
        else 0
    )
    print(f"✅ Exported {len(summary_df)} variants, {len(trials_df)} trials ({n_crit} critical)")


def list_experiments():
    """List available experiments."""
    print("\nExperiments:")
    for d in sorted(SYNTH_OUTPUTS.iterdir()):
        if d.is_dir() and (d / "underspec_candidates.csv").exists():
            n = len(load_variants(d.name))
            print(f"  {d.name}: {n} variants")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Process underspec results")
    parser.add_argument(
        "--experiment", "-e", help="Experiment(s): name, comma-separated, or glob pattern"
    )
    parser.add_argument("--preview", "-p", action="store_true", help="Preview variants")
    parser.add_argument("--full", action="store_true", help="Show full prompts")
    parser.add_argument("--limit", "-n", type=int, help="Limit rows")
    parser.add_argument("--export", "-x", action="store_true", help="Export CSV/JSON")
    parser.add_argument("--viz", "-v", action="store_true", help="Generate visualization")
    parser.add_argument("--html", action="store_true", help="Generate HTML review page")
    parser.add_argument("--save", "-o", help="Output path for viz/export/html")
    parser.add_argument("--list", "-l", action="store_true", help="List experiments")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Run LLM judge on new_task candidates to distinguish new_task vs outcome-critical",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"LLM model for new_task judging (default: {DEFAULT_JUDGE_MODEL})",
    )
    args = parser.parse_args()

    if args.list:
        list_experiments()
        return

    if not args.experiment:
        parser.print_help()
        return

    experiments = resolve_experiments(args.experiment)
    if not experiments:
        print(f"No experiments matching '{args.experiment}'")
        return

    print(f"📦 Experiments: {', '.join(experiments)}")

    # Determine judge model
    judge_model = args.judge_model if args.judge else None
    if judge_model:
        print(f"🧑‍⚖️ Using LLM judge: {judge_model}")

    # Determine output name and directory
    name = "+".join(experiments) if len(experiments) <= 4 else f"combined_{len(experiments)}"
    output_dir = SYNTH_OUTPUTS / experiments[0] if len(experiments) == 1 else SYNTH_OUTPUTS
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.preview:
        preview_variants(experiments, args.full, args.limit)
    elif args.export:
        export_experiment(experiments, Path(args.save) if args.save else output_dir)
    elif args.viz:
        save_path = Path(args.save) if args.save else output_dir / f"underspec_viz_{name}.png"
        plot_results(experiments, save_path)
    elif args.html:
        for exp in experiments:
            if args.save and len(experiments) > 1:
                save_path = Path(args.save).parent / f"variants_review_{exp}.html"
            elif args.save:
                save_path = Path(args.save)
            else:
                save_path = None
            generate_review_html(exp, output_path=save_path, max_variants=args.limit or 50)
    else:
        df = print_summary(experiments, judge_model=judge_model)
        # Auto-generate viz and save results
        viz_path = output_dir / f"underspec_viz_{name}.png"
        csv_path = output_dir / f"underspec_results_{name}.csv"
        plot_results(experiments, viz_path)
        # Save summary CSV
        if df is not None and not df.empty:
            df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"📄 Saved results to {csv_path}")
            # Write per-experiment CSVs so judge cache works across steps
            if len(experiments) > 1 and "experiment" in df.columns:
                for exp in experiments:
                    exp_slice = df[df["experiment"] == exp]
                    if not exp_slice.empty:
                        exp_csv = SYNTH_OUTPUTS / exp / f"underspec_results_{exp}.csv"
                        exp_csv.parent.mkdir(parents=True, exist_ok=True)
                        exp_slice.to_csv(exp_csv, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
