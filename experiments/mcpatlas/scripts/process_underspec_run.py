"""
Process underspecification experiment runs and build summary DataFrames.

Usage:
    from process_underspec_run import build_summary_df, load_variants, load_results
    
    variants = load_variants("experiments/underspec_datasets/intersection_delete_v1_20260120_110000")
    results = load_results(
        "experiments/underspec_datasets/intersection_delete_v1_20260120_110000",
        "run_underspec_larger_gemini-3-pro-preview_20260121_014321"
    )
    df = build_summary_df(variants, results)
"""

import csv
import json
import math
import os
import re
import sys
import warnings
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Any

import litellm
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

LHAW_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if str(LHAW_ROOT) not in sys.path:
    sys.path.insert(0, str(LHAW_ROOT))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

csv.field_size_limit(10 * 1024 * 1024)
warnings.filterwarnings("ignore", category=UserWarning, message="Pydantic serializer warnings:.*")

from evaluation.pass_k_metrics import pass_at_k  # noqa: E402
from utils import _scores_dir_sort_key  # noqa: E402


def configure_litellm() -> None:
    """Configure LiteLLM using environment variables.

    Accepts LLM_API_KEY (canonical) or LITELLM_API_KEY (legacy alias).
    """
    load_dotenv(override=True)
    api_key = os.getenv("LLM_API_KEY") or os.getenv("LITELLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY (or LITELLM_API_KEY) is not set")

    api_base = os.getenv("LLM_BASE_URL") or os.getenv("LITELLM_BASE_URL")

    litellm.api_key = api_key
    if api_base:
        litellm.api_base = api_base


def get_new_task_judge_prompt(
    underspec_prompt: str,
    responses: list[str],
) -> str:
    """Create the prompt for LLM judge of new-task eligibility."""
    cleaned_responses = [
        response.strip() for response in responses if response and response.strip()
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
    """Use LLM to judge whether the underspecified task was completed."""
    last_error = None
    raw_content = None
    for attempt in range(max_retries + 1):
        try:
            prompt = get_new_task_judge_prompt(underspec_prompt, responses)
            if attempt > 0:
                prompt += "\n\nIMPORTANT: Your response MUST be valid JSON only. No markdown, no extra text."
            llm_response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0 if attempt == 0 else 0.1,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            raw_content = llm_response.choices[0].message.content.strip()
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


def _resolve_json_dir(dataset_dir: Path) -> Path:
    candidates = [dataset_dir / "underspec_generation" / "json", dataset_dir / "json"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _normalize_prompt(prompt: str) -> str:
    """Normalize prompt for matching.

    Handles:
    - Null bytes from encoding corruption
    - Unicode accents (é → e) for fuzzy matching
    - Whitespace normalization
    """
    import unicodedata

    # Remove null bytes (encoding corruption)
    prompt = prompt.replace("\x00", "")
    # Normalize Unicode and strip accents (é → e)
    prompt = unicodedata.normalize("NFD", prompt)
    prompt = "".join(c for c in prompt if unicodedata.category(c) != "Mn")
    # Normalize whitespace
    return " ".join(prompt.split())


def _flatten_or_list(values: list[Any]) -> Any:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return json.dumps(values)


def load_variants(dataset_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Load variant information from task JSON files."""
    dataset_dir = Path(dataset_dir)
    json_dir = _resolve_json_dir(dataset_dir)

    variants: dict[str, dict[str, Any]] = {}
    for json_path in json_dir.glob("*.json"):
        task_id = json_path.stem
        data = json.loads(json_path.read_text())

        task_info = data["task"]
        for variant in data.get("variants", []):
            removed_segments = variant.get("removed_segments", [])
            dimensions = [seg.get("dimension") for seg in removed_segments if seg.get("dimension")]
            subdimensions = [
                seg.get("subdimension") for seg in removed_segments if seg.get("subdimension")
            ]
            removed_values = [seg.get("value") for seg in removed_segments if seg.get("value")]
            criticalities = [
                seg.get("criticality")
                for seg in removed_segments
                if seg.get("criticality") is not None
            ]
            guessabilities = [
                seg.get("guessability")
                for seg in removed_segments
                if seg.get("guessability") is not None
            ]
            priority_scores = [
                seg.get("priority_score")
                for seg in removed_segments
                if seg.get("priority_score") is not None
            ]

            variant_id = variant.get("id", "")
            variant_key = f"{task_id}::{variant_id}" if variant_id else task_id
            underspec_prompt = variant.get("underspecified_prompt", "")

            removed_segment_ids = [seg.get("id") for seg in removed_segments if seg.get("id")]

            variants[variant_key] = {
                "task": task_id,
                "variant_id": variant_id,
                "prompt": task_info.get("prompt", ""),
                "underspec_prompt": underspec_prompt,
                "underspec_prompt_normalized": _normalize_prompt(underspec_prompt),
                "dimension": _flatten_or_list(dimensions),
                "subdimension": _flatten_or_list(subdimensions),
                "removed_value": _flatten_or_list(removed_values),
                "criticality": _flatten_or_list(criticalities),
                "guessability": _flatten_or_list(guessabilities),
                "priority_score": _flatten_or_list(priority_scores),
                "removed_segment_ids": removed_segment_ids,
            }

    return variants


def _build_prompt_index(variants: dict[str, dict[str, Any]]) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}
    for variant_key, info in variants.items():
        task_id = info.get("task")
        prompt = info.get("underspec_prompt_normalized") or _normalize_prompt(
            info.get("underspec_prompt", "")
        )
        if task_id and prompt:
            key = (task_id, prompt)
            if key in index:
                raise ValueError(
                    f"Duplicate (task_id, prompt) pair found in variants.\n"
                    f"  task_id: {task_id}\n"
                    f"  existing variant: {index[key]}\n"
                    f"  new variant: {variant_key}\n"
                    f"  prompt:\n    {prompt}"
                )
            index[key] = variant_key
    return index


def load_results(
    dataset_dir: str | Path,
    run_dir: str,
    variants: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load trial results from evaluation JSON."""
    dataset_dir = Path(dataset_dir)
    run_path = Path(run_dir)
    if not run_path.is_absolute() and not run_path.exists():
        run_path = dataset_dir / run_dir

    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    eval_files = list(run_path.rglob("evaluation_*.json"))
    if not eval_files:
        raise FileNotFoundError(f"No evaluation_*.json found in {run_path}")
    eval_path = eval_files[0]
    eval_data = json.loads(eval_path.read_text())

    model = eval_data["model"]
    results: dict[str, list[dict[str, Any]]] = {}
    completions = load_completions(run_path)
    variants = variants or load_variants(dataset_dir)
    prompt_index = _build_prompt_index(variants)

    for task_id, task_data in eval_data["tasks"].items():
        trials: list[dict[str, Any]] = []
        for i, result in enumerate(task_data["results"]):
            final_score = result["final_score"]
            response = ""
            prompt = ""
            claim_outcomes: tuple[int, ...] = ()
            if task_id in completions and i < len(completions[task_id]):
                response = completions[task_id][i]["response"]
                prompt = completions[task_id][i]["prompt"]
                claim_outcomes = completions[task_id][i]["claim_outcomes"]

            normalized_prompt = _normalize_prompt(prompt)
            lookup_key = (task_id, normalized_prompt)
            if lookup_key not in prompt_index:
                available_variants = [k for k, v in variants.items() if v.get("task") == task_id]
                available_prompts = [
                    variants[k].get("underspec_prompt_normalized", "")[:200] + "..."
                    for k in available_variants
                ]
                raise ValueError(
                    f"No variant match found for result.\n"
                    f"  task_id: {task_id}\n"
                    f"  trial: {i}\n"
                    f"  normalized prompt:\n    {normalized_prompt}\n"
                    f"  Available variants for this task: {available_variants}\n"
                    f"  Their prompts (first 200 chars each):\n"
                    + "\n".join(
                        f"    {v}: {p}" for v, p in zip(available_variants, available_prompts)
                    )
                )
            variant_key = prompt_index[lookup_key]

            trials.append(
                {
                    "task": task_id,
                    "trial": i,
                    "score": final_score["result"],
                    "total": final_score["total"],
                    "full_pass": final_score["result"] == final_score["total"],
                    "terminal_state": claim_outcomes,
                    "response": response,
                    "prompt": prompt,
                    "model": model,
                    "dimension": "",
                    "variant_key": variant_key,
                }
            )
        for trial in trials:
            results.setdefault(trial["variant_key"], []).append(trial)

    return results


def load_completions(run_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Load agent responses and coverage details from scored CSV files."""
    completions: dict[str, list[dict[str, Any]]] = {}

    scores_dirs = sorted(run_path.glob("scores_*"), key=_scores_dir_sort_key)
    if not scores_dirs:
        raise ValueError(
            f"No scores_* directories found in {run_path}. "
            f"Scored CSVs are required for coverage details."
        )

    for scores_dir in scores_dirs:
        for csv_path in scores_dir.glob("scored_*.csv"):
            csv_text = csv_path.read_text(encoding="utf-8", errors="replace")
            reader = csv.DictReader(StringIO(csv_text))
            for row in reader:
                task_id = row.get("TASK", "")
                prompt = row.get("PROMPT", "")
                response = row.get("script_model_response", "")
                coverage_json = row.get("coverage_details_json", "")

                # Parse coverage details to extract per-claim binary outcomes
                claim_outcomes: tuple[int, ...] = ()
                if coverage_json:
                    try:
                        coverage = json.loads(coverage_json)
                        per_claim = coverage.get("per_claim", [])
                        claim_outcomes = tuple(
                            1 if claim.get("covered", False) else 0 for claim in per_claim
                        )
                    except (json.JSONDecodeError, TypeError):
                        pass

                if task_id:
                    completions.setdefault(task_id, []).append(
                        {
                            "prompt": prompt,
                            "response": response,
                            "claim_outcomes": claim_outcomes,
                        }
                    )

    return completions


def build_summary_df(
    variants: dict[str, dict],
    results: dict[str, list[dict]],
    segment_priority: dict[str, tuple[str | None, str | None]] | None = None,
    baseline_metrics: dict[str, dict[str, float]] | None = None,
    target_k: int = 8,
    judge_model: str | None = None,
) -> pd.DataFrame:
    """One row per variant with aggregated metrics.

    Args:
        variants: Dict from load_variants()
        results: Dict from load_results()
        segment_priority: Dict from load_segment_priorities() for removal_condition classification
        baseline_metrics: Dict from load_baseline_metrics() for baseline comparison
        target_k: Target k for pass@k calculation (default: 8)
        judge_model: LLM model for new-task eligibility judging
    """
    if judge_model:
        configure_litellm()

    segment_priority = segment_priority or {}
    baseline_metrics = baseline_metrics or {}

    all_rows = []
    variant_ids_with_results = sorted(set(variants.keys()) & set(results.keys()))
    skipped = len(set(variants.keys()) - set(results.keys()))
    if skipped:
        print(f"Skipping {skipped} variants with no run results (not in this run)")
    for vid in tqdm(variant_ids_with_results, desc="Processing variants"):
        v = variants[vid]
        trials = results[vid]

        task_id = v["task"]
        removed_segment_ids = v["removed_segment_ids"]
        removal_condition = classify_removal_condition(
            task_id, removed_segment_ids, segment_priority
        )
        task_baseline = baseline_metrics.get(task_id, {})
        baseline_success_rate = task_baseline.get("success_rate")
        baseline_pass_at_3 = task_baseline.get("pass_at_3")

        row = {
            "variant_id": vid,
            "variant_local_id": v["variant_id"],
            "task": task_id,
            "dimension": v["dimension"],
            "subdimension": v["subdimension"],
            "removed_value": v["removed_value"],
            "criticality": v["criticality"],
            "guessability": v["guessability"],
            "priority_score": v["priority_score"],
            "removal_condition": removal_condition,
            "baseline_success_rate": baseline_success_rate,
            "baseline_pass_at_3": baseline_pass_at_3,
        }

        n, c = len(trials), sum(1 for t in trials if t["full_pass"])
        total = trials[0]["total"]
        avg = sum(t["score"] for t in trials) / n
        k_for_pass_n = min(target_k, n)

        # Terminal state divergence analysis
        terminal_states = [t["terminal_state"] for t in trials]
        unique_terminal_states = set(terminal_states)
        n_unique_states = len(unique_terminal_states)

        outcome_divergence = n_unique_states > 1
        divergence_ratio = n_unique_states / n if n > 0 else 0

        # Terminal state entropy
        state_counts = Counter(terminal_states)
        state_entropy = 0.0
        if n > 0 and n_unique_states > 1:
            for count in state_counts.values():
                p = count / n
                if p > 0:
                    state_entropy -= p * math.log2(p)

        # Outcome classification
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
            if n_unique_states == 1:  # we might be a new_task candidate
                underspec_prompt = v["underspec_prompt"]
                responses = [t["response"] for t in trials]
                judge_result = judge_new_task_sync(
                    underspec_prompt,
                    responses,
                    judge_model,
                )  # This should judge that *all* trials reached the *same outcome*, but *without* making any assumptions
                judge_verdict = bool(judge_result.get("task_completed"))
                judge_reasoning = judge_result.get("reasoning", "")

                if judge_verdict is True:
                    ambiguity_class = "new_task"
                else:
                    ambiguity_class = "outcome-critical"
            else:
                ambiguity_class = "outcome-critical"
        elif all_succeeded:  # if pass rate is >= original pass rate
            ambiguity_class = "benign"
        elif some_succeeded:
            ambiguity_class = "divergent"  # same as partial ???
        else:
            raise ValueError(f"Unexpected outcome: {all_failed}, {all_succeeded}, {some_succeeded}")

        row.update(
            {
                "n_trials": n,
                "full_passes": c,
                "max_score": total,
                "avg_score": round(avg, 2),
                "avg_pct": round(avg / total * 100, 1) if total else 0,
                "success_rate": round(c / n * 100, 1),
                "pass_at_1": round(pass_at_k(n, c, 1), 3),
                "pass_at_3": round(pass_at_k(n, c, 3), 3) if n >= 3 else None,
                "pass_at_N": round(pass_at_k(n, c, k_for_pass_n), 3),
                "target_k": target_k,
                "n_unique_states": n_unique_states,
                "outcome_divergence": outcome_divergence,
                "divergence_ratio": round(divergence_ratio, 3),
                "state_entropy": round(state_entropy, 3),
                "terminal_states": str(terminal_states),
                "outcome_critical": c == 0,
                "new_task_eligible": eligible_new_task,
                "judge_verdict": judge_verdict,
                "judge_reasoning": judge_reasoning,
                "judge_model": judge_model,
                "ambiguity_class": ambiguity_class,
                "model": trials[0]["model"],
            }
        )
        all_rows.append(row)

    return pd.DataFrame(all_rows)


def load_segment_priorities(dataset_dir: str | Path) -> dict[str, tuple[str | None, str | None]]:
    """Load segment priorities to determine top1/top2 segments per task.

    Returns:
        Dict mapping task_id -> (top1_segment_id, top2_segment_id)
    """
    dataset_dir = Path(dataset_dir)
    json_dir = _resolve_json_dir(dataset_dir)

    segment_priority: dict[str, tuple[str | None, str | None]] = {}
    for json_path in json_dir.glob("*.json"):
        data = json.loads(json_path.read_text())
        task_id = data.get("task", {}).get("id") or json_path.stem
        segments = data.get("segments", [])

        # Sort by priority_score (desc), then criticality (desc)
        segments_sorted = sorted(
            segments,
            key=lambda s: (s.get("priority_score", 0), s.get("criticality", 0)),
            reverse=True,
        )
        top1 = segments_sorted[0]["id"] if len(segments_sorted) > 0 else None
        top2 = segments_sorted[1]["id"] if len(segments_sorted) > 1 else None
        segment_priority[task_id] = (top1, top2)

    return segment_priority


def load_baseline_metrics(
    dataset_dir: str | Path,
    model_prefix: str = "gemini",
) -> dict[str, dict[str, float]]:
    """Load baseline metrics per task from baseline_results.

    Args:
        dataset_dir: Path to dataset directory
        model_prefix: Model prefix to match (e.g., 'gemini', 'gpt', 'claude')

    Returns:
        Dict mapping task_id -> {success_rate: float, pass_at_3: float}
        success_rate is 0-100 percentage, pass_at_3 is 0-1 proportion
    """
    dataset_dir = Path(dataset_dir)
    baseline_dir = dataset_dir / "baseline_results" / "per_model"

    if not baseline_dir.exists():
        return {}

    # Find matching pass_k file
    pass_k_files = list(baseline_dir.glob(f"{model_prefix}*pass_k_per_task.csv"))
    if not pass_k_files:
        return {}

    baseline_metrics: dict[str, dict[str, float]] = {}
    df = pd.read_csv(pass_k_files[0])
    for _, row in df.iterrows():
        task_id = row["task_id"]
        baseline_metrics[task_id] = {
            # Convert pass_rate (0-1) to percentage (0-100) to match success_rate
            "success_rate": row["pass_rate"] * 100,
            "pass_at_3": row["pass@3"],
        }

    return baseline_metrics


def classify_removal_condition(
    task_id: str,
    removed_segment_ids: list[str],
    segment_priority: dict[str, tuple[str | None, str | None]],
) -> str:
    """Classify a variant by which segments were removed.

    Returns one of: 'top1_removal', 'top2_removal', 'top1_and_top2_removal', 'other'
    """
    top1, top2 = segment_priority.get(task_id, (None, None))
    removed_set = set(removed_segment_ids)

    if len(removed_segment_ids) == 1:
        if removed_segment_ids[0] == top1:
            return "top1_removal"
        if removed_segment_ids[0] == top2:
            return "top2_removal"
        return "other"
    if len(removed_segment_ids) == 2 and removed_set == {top1, top2}:
        return "top1_and_top2_removal"
    return "other"


def plot_summary_charts(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Plot bar charts showing mean success rate by condition and ambiguity class.

    Expects df to have 'removal_condition', 'ambiguity_class', 'success_rate',
    and optionally 'baseline_success_rate' columns.
    """
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)

    # Filter to rows with success_rate data
    df_valid = df[df["success_rate"].notna()].copy()

    # Calculate baseline mean if available
    has_baseline = (
        "baseline_success_rate" in df_valid.columns
        and df_valid["baseline_success_rate"].notna().any()
    )
    baseline_mean = df_valid["baseline_success_rate"].mean() if has_baseline else None
    baseline_count = df_valid["baseline_success_rate"].notna().sum() if has_baseline else 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Mean success rate by removal condition (with baseline)
    ax1 = axes[0]

    # Start with baseline bar
    x_cond = []
    y_cond = []
    colors_cond = []
    labels_cond = []

    if has_baseline and baseline_mean is not None:
        x_cond.append("Baseline")
        y_cond.append(baseline_mean)
        colors_cond.append("#7f8c8d")  # Gray for baseline
        labels_cond.append(f"n={baseline_count}")

    condition_order = ["top1_removal", "top2_removal", "top1_and_top2_removal", "other"]
    condition_labels = ["Top 1", "Top 2", "Top 1 & 2", "Other"]
    condition_colors = ["#3498db", "#2ecc71", "#e74c3c", "#95a5a6"]
    condition_stats = df_valid.groupby("removal_condition")["success_rate"].agg(
        ["mean", "std", "count"]
    )

    for cond, label, color in zip(condition_order, condition_labels, condition_colors):
        if cond in condition_stats.index:
            x_cond.append(label)
            y_cond.append(condition_stats.loc[cond, "mean"])
            colors_cond.append(color)
            labels_cond.append(f"n={int(condition_stats.loc[cond, 'count'])}")

    bars1 = ax1.bar(x_cond, y_cond, color=colors_cond)
    ax1.set_ylabel("Mean Success Rate (%)")
    ax1.set_title("Mean Success Rate by Removal Condition")
    ax1.set_ylim(0, 110)
    ax1.axhline(y=100, color="gray", linestyle="--", alpha=0.3)

    # Add count labels on bars
    for bar, label in zip(bars1, labels_cond):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Mean success rate by ambiguity class (with baseline)
    ax2 = axes[1]

    x_class = []
    y_class = []
    colors_class = []
    labels_class = []

    if has_baseline and baseline_mean is not None:
        x_class.append("Baseline")
        y_class.append(baseline_mean)
        colors_class.append("#7f8c8d")  # Gray for baseline
        labels_class.append(f"n={baseline_count}")

    class_order = ["benign", "divergent", "outcome-critical", "new_task"]
    class_labels = ["Benign", "Divergent", "Outcome-Critical", "New Task"]
    class_colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6"]

    df_class = df_valid[df_valid["ambiguity_class"].notna()]
    class_stats = df_class.groupby("ambiguity_class")["success_rate"].agg(["mean", "std", "count"])

    for cls, label, color in zip(class_order, class_labels, class_colors):
        if cls in class_stats.index:
            x_class.append(label)
            y_class.append(class_stats.loc[cls, "mean"])
            colors_class.append(color)
            labels_class.append(f"n={int(class_stats.loc[cls, 'count'])}")

    bars2 = ax2.bar(x_class, y_class, color=colors_class)
    ax2.set_ylabel("Mean Success Rate (%)")
    ax2.set_title("Mean Success Rate by Ambiguity Class")
    ax2.set_ylim(0, 110)
    ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.3)

    # Add count labels on bars
    for bar, label in zip(bars2, labels_class):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # Save figure
    plot_path = output_dir / "summary_charts.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved charts to: {plot_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process underspecification run")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", help="Output directory (defaults to dataset-dir)")
    parser.add_argument(
        "--judge-model",
        default="openai/gemini/gemini-3-flash-preview",
        help="LLM model for new-task judging (only used when eligible)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip generating plots")
    parser.add_argument("--head", type=int, help="Process only the first N variants (for testing)")

    args = parser.parse_args()

    # Default output_dir to dataset_dir
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.dataset_dir)
    if output_dir.suffix:  # Has file extension, likely a file path
        raise ValueError(f"--output-dir must be a directory, not a file: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = load_variants(args.dataset_dir)
    results = load_results(args.dataset_dir, args.run_dir, variants)
    segment_priority = load_segment_priorities(args.dataset_dir)
    baseline_metrics = load_baseline_metrics(args.dataset_dir)

    if baseline_metrics:
        print(f"Loaded baseline metrics for {len(baseline_metrics)} tasks")
    else:
        print("No baseline metrics found")

    # Limit to first N variants if --head is specified
    if args.head:
        variant_keys = sorted(variants.keys())[: args.head]
        variants = {k: variants[k] for k in variant_keys}
        results = {k: results[k] for k in variant_keys if k in results}
        print(f"Processing first {args.head} variants (--head)")

    df = build_summary_df(
        variants,
        results,
        segment_priority=segment_priority,
        baseline_metrics=baseline_metrics,
        judge_model=args.judge_model,
    )

    print(f"\n=== Summary ({len(df)} variants) ===")
    print(df["ambiguity_class"].value_counts())

    # Save CSV
    run_name = Path(args.run_dir).name
    csv_path = output_dir / f"summary_{run_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to: {csv_path}")

    # Generate plots
    if not args.no_plot:
        plot_summary_charts(df, output_dir)
