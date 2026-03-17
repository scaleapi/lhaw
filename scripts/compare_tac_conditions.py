#!/usr/bin/env python3
"""
Comprehensive comparison across all experimental conditions.

Compares three conditions:
1. Baseline - No user simulator (agent must guess)
2. User-sim (none) - Tool available but not prompted (natural behavior)
3. User-sim (prompted) - Tool available + encouraged to use

Also supports ablation analysis:
- Agent prompting strategies (ReAct, Reflexion, Plan-and-Execute)

Key Analysis:
- Distribution shift by ambiguity class
- Asking behavior by condition and model
- Clarification ROI (does asking actually help?)
- Cost-benefit analysis (tokens vs performance)
- Per-task breakdown

Usage:
    python scripts/compare_tac_conditions.py --experiment my_exp
    python scripts/compare_tac_conditions.py --experiment my_exp --models sonnet_4_5,gemini_3_flash
    
    # Ablation: agent prompting strategies
    python scripts/compare_tac_conditions.py --experiment my_exp --ablation agent_prompt --models gpt_5_2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import PATHS  # noqa: E402
from scripts.process_tac_usersim import build_summary_df  # noqa: E402

SYNTH_OUTPUTS = REPO_ROOT / PATHS["synthetic_outputs"]
REPORTS_DIR = SYNTH_OUTPUTS

ORIGINAL_BASELINE_PATH = REPO_ROOT / PATHS["runs_dir"] / "aggregated_results_all_models.csv"
RUNS_DIR_PATH = REPO_ROOT / PATHS["runs_dir"]


def _load_baseline_from_pass_k_summaries(tasks: list[str], models: list[str]) -> dict:
    """Fallback: scan per-run pass_k_summary CSVs when aggregated CSV doesn't exist."""
    result = {}
    task_set = set(tasks)
    csvs = list(RUNS_DIR_PATH.glob("run_*/final_results/summaries/pass_k_summary_*.csv"))
    if not csvs:
        return {}

    for csv_path in csvs:
        run_dir_name = csv_path.parents[2].name
        if "underspec" in run_dir_name:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        for _, row in df.iterrows():
            task = row.get("task", "")
            if task == "AGGREGATE" or task not in task_set:
                continue
            for model in sorted(models, key=len, reverse=True):
                if f"_{model}_" in f"_{run_dir_name}_":
                    p3 = row.get("pass@3", 0)
                    avg_pct = row.get("avg_percent", 0)
                    result.setdefault(task, {})
                    existing = result[task].get(model)
                    if existing is None or p3 > existing["pass_at_3"]:
                        result[task][model] = {
                            "pass_at_3": p3,
                            "avg_ckpt_acc": avg_pct / 100.0,
                        }
                    break
    return result


def load_original_baseline(tasks: list[str], models: list[str]) -> dict:
    """Load original well-specified task performance from TAC baselines."""
    if not ORIGINAL_BASELINE_PATH.exists():
        result = _load_baseline_from_pass_k_summaries(tasks, models)
        if result:
            print(
                f"  📋 Loaded original baseline from per-run pass_k_summary CSVs ({sum(len(v) for v in result.values())} task-model pairs)"
            )
        return result

    try:
        df = pd.read_csv(ORIGINAL_BASELINE_PATH)
        result = {}

        for task in tasks:
            result[task] = {}
            for model in models:
                task_df = df[(df["task"] == task) & (df["model"] == model)]
                if not task_df.empty:
                    result[task][model] = {
                        "pass_at_3": task_df["pass@3"].iloc[0],
                        "avg_ckpt_acc": task_df["avg_ckpt_acc"].iloc[0],
                    }
        return result
    except Exception as e:
        print(f"  ⚠️ Could not load original baseline: {e}")
        return {}


def load_condition_data(
    experiment: str,
    models: list[str],
    min_evals: int = 3,
    usersim_suffix: str = "default",
) -> dict:
    """Load data for all conditions and models.

    Args:
        usersim_suffix: Suffix for the user-sim experiment name.
            e.g. 'default' → user_sim_{experiment}_{model}_default
                 'med_plain' → user_sim_{experiment}_{model}_med_plain
    """
    conditions = {
        "baseline": [],
        "usersim_prompted": [],
    }

    for model in models:
        baseline_exp = f"baseline_{experiment}_{model}"
        try:
            df = build_summary_df(baseline_exp, min_evals=min_evals, skip_responses=True)
            if not df.empty and df["has_run"].sum() > 0:
                df["model"] = model
                df["condition"] = "baseline"
                conditions["baseline"].append(df)
        except Exception as e:
            print(f"  ⚠️ Could not load {baseline_exp}: {e}")

        prompted_exp = f"user_sim_{experiment}_{model}_{usersim_suffix}"
        try:
            df = build_summary_df(prompted_exp, min_evals=min_evals, skip_responses=True)
            if not df.empty and df["has_run"].sum() > 0:
                df["model"] = model
                df["condition"] = "usersim_prompted"
                conditions["usersim_prompted"].append(df)
        except Exception as e:
            print(f"  ⚠️ Could not load {prompted_exp}: {e}")

    result = {}
    models_by_cond = {}
    for cond, dfs in conditions.items():
        if dfs:
            result[cond] = pd.concat(dfs, ignore_index=True)
            models_by_cond[cond] = sorted(result[cond]["model"].unique().tolist())

    if models_by_cond:
        baseline_models = set(models_by_cond.get("baseline", []))
        prompted_models = set(models_by_cond.get("usersim_prompted", []))
        common_models = baseline_models & prompted_models
        result["_common_models"] = sorted(common_models)
        result["_models_by_cond"] = models_by_cond

        if baseline_models != prompted_models:
            print("  ⚠️  Model coverage differs between baseline and prompted:")
            print(f"      baseline: {sorted(baseline_models)}")
            print(f"      prompted: {sorted(prompted_models)}")
            print(f"      Common: {sorted(common_models)}")

    return result


def load_ablation_data(
    ablation_type: str, experiment: str, models: list[str], min_evals: int = 3
) -> dict:
    """Load data for ablation experiments.

    Args:
        ablation_type: Type of ablation (e.g., 'agent_prompt')
        experiment: Experiment name (e.g., 'my_exp')
        models: List of model shortnames
        min_evals: Minimum evaluations required

    Returns:
        Dict mapping condition names to DataFrames
    """
    if ablation_type == "agent_prompt":
        strategies = ["none", "react", "reflexion", "plan_and_execute"]
        strategy_labels = {
            "none": "No Strategy",
            "react": "ReAct",
            "reflexion": "Reflexion",
            "plan_and_execute": "Plan & Execute",
        }
        # Two conditions: baseline (no user sim) and usersim (with user sim)
        exp_conditions = ["baseline", "usersim"]
        condition_labels = {
            "baseline": "Baseline (No User)",
            "usersim": "With User Simulator",
        }
    else:
        print(f"❌ Unknown ablation type: {ablation_type}")
        return {}

    conditions = {}

    for model in models:
        for exp_cond in exp_conditions:
            for strategy in strategies:
                cond_key = f"{exp_cond}_{model}_{strategy}"
                df = None

                # For "none" strategy, use existing experiments as baseline
                if strategy == "none":
                    if exp_cond == "baseline":
                        existing_exp = f"baseline_{experiment}_{model}"
                    else:
                        existing_exp = f"user_sim_{experiment}_{model}_default"

                    try:
                        df = build_summary_df(
                            existing_exp, min_evals=min_evals, skip_responses=True
                        )
                        if df.empty or df["has_run"].sum() == 0:
                            df = None
                        else:
                            print(
                                f"  ✓ {existing_exp} (as 'none' strategy): {df['has_run'].sum()} variants"
                            )
                    except Exception:
                        pass
                else:
                    exp_name = (
                        f"ablation_{ablation_type}_{exp_cond}_{experiment}_{model}_{strategy}"
                    )

                    try:
                        df = build_summary_df(exp_name, min_evals=min_evals, skip_responses=True)
                        if df.empty or df["has_run"].sum() == 0:
                            df = None
                        else:
                            print(f"  ✓ {exp_name}: {df['has_run'].sum()} variants")
                    except Exception:
                        pass

                if df is not None and not df.empty and df["has_run"].sum() > 0:
                    df["model"] = model
                    df["strategy"] = strategy
                    df["strategy_label"] = strategy_labels.get(strategy, strategy)
                    df["exp_condition"] = exp_cond
                    df["exp_condition_label"] = condition_labels.get(exp_cond, exp_cond)
                    df["condition"] = cond_key

                    if cond_key not in conditions:
                        conditions[cond_key] = df
                    else:
                        conditions[cond_key] = pd.concat(
                            [conditions[cond_key], df], ignore_index=True
                        )

    # Add metadata
    conditions["_ablation_type"] = ablation_type
    conditions["_strategies"] = strategies
    conditions["_strategy_labels"] = strategy_labels
    conditions["_exp_conditions"] = exp_conditions
    conditions["_condition_labels"] = condition_labels

    return conditions


def _compute_asking_stats(sub_df: pd.DataFrame) -> dict:
    """Compute asking/clarification statistics for a DataFrame slice."""
    n_v = len(sub_df)
    total_trials = (
        float(sub_df["n_trials"].sum()) if "n_trials" in sub_df.columns else float(n_v * 3)
    )
    if "asked_any" in sub_df.columns and "n_trials" in sub_df.columns:
        asked_trials = float((sub_df["asked_any"] * sub_df["n_trials"]).sum())
    elif "asked_any" in sub_df.columns:
        asked_trials = float(sub_df["asked_any"].mean() * total_trials)
    else:
        asked_trials = 0.0
    if "n_questions_total" in sub_df.columns:
        total_q = float(sub_df["n_questions_total"].sum())
    elif "n_questions" in sub_df.columns and "n_trials" in sub_df.columns:
        total_q = float((sub_df["n_questions"] * sub_df["n_trials"]).sum())
    else:
        total_q = 0.0
    return {
        "n_samples_asked": (
            int((sub_df["asked_any"] > 0).sum()) if "asked_any" in sub_df.columns else 0
        ),
        "pct_samples_asked": (
            (sub_df["asked_any"] > 0).sum() / n_v
            if "asked_any" in sub_df.columns and n_v > 0
            else 0
        ),
        "pct_trials_asked": (asked_trials / total_trials) if total_trials > 0 else 0,
        "total_questions": total_q,
        "avg_q_per_sample": (total_q / n_v) if n_v > 0 else 0,
        "avg_q_per_trial": (total_q / total_trials) if total_trials > 0 else 0,
        "avg_q_per_traj": (total_q / asked_trials) if asked_trials > 0 else 0,
    }


def _compute_token_stats(evaluated: pd.DataFrame) -> dict:
    """Compute token usage statistics."""
    total_tokens = (
        int(evaluated["total_tokens"].sum())
        if "total_tokens" in evaluated.columns
        else (
            int(evaluated["total_tokens_sum"].sum())
            if "total_tokens_sum" in evaluated.columns
            else 0
        )
    )
    tokens_input = (
        int(evaluated["tokens_user_read_sum"].sum())
        if "tokens_user_read_sum" in evaluated.columns
        else 0
    )
    tokens_output = (
        int(evaluated["tokens_user_wrote_sum"].sum())
        if "tokens_user_wrote_sum" in evaluated.columns
        else 0
    )
    return {
        "total_tokens": total_tokens,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
    }


def _compute_by_class_metrics(evaluated: pd.DataFrame) -> dict:
    """Compute metrics grouped by ambiguity class."""
    by_class = {}
    for cls in ["outcome-critical", "divergent", "benign"]:
        cls_df = evaluated[evaluated["ambiguity_class"] == cls]
        if len(cls_df) > 0:
            cls_asking = _compute_asking_stats(cls_df)
            by_class[cls] = {
                "n": len(cls_df),
                "pass_at_3": cls_df["pass_at_3"].mean(),
                "avg_pct": cls_df["avg_pct"].mean(),
                **{
                    k: cls_asking[k]
                    for k in (
                        "n_samples_asked",
                        "pct_samples_asked",
                        "pct_trials_asked",
                        "avg_q_per_sample",
                        "avg_q_per_trial",
                        "avg_q_per_traj",
                    )
                },
            }
    return by_class


def _compute_by_task_metrics(evaluated: pd.DataFrame) -> dict:
    """Compute metrics grouped by task."""
    by_task = {}
    for task in evaluated["task"].unique():
        task_df = evaluated[evaluated["task"] == task]
        task_n = len(task_df)
        task_asking = _compute_asking_stats(task_df)
        by_task[task] = {
            "n": task_n,
            "pass_at_3": task_df["pass_at_3"].mean(),
            "avg_pct": task_df["avg_pct"].mean(),
            "n_samples_asked": task_asking["n_samples_asked"],
            "pct_samples_asked": task_asking["pct_samples_asked"],
            "pct_trials_asked": task_asking["pct_trials_asked"],
        }
    return by_task


def compute_condition_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate metrics for a condition."""
    evaluated = df[df["has_run"]].copy()
    if evaluated.empty:
        return {}

    # Core metrics
    n = len(evaluated)
    n_trials = int(evaluated["n_trials"].sum())

    # Data quality: track incomplete variants (< 3 evals in LATEST run)
    if "n_evals_latest" in evaluated.columns:
        incomplete = evaluated[evaluated["n_evals_latest"] < 3]
    else:
        incomplete = evaluated[evaluated["n_trials"] < 3]
    n_incomplete = len(incomplete)

    # Performance
    avg_pass1 = evaluated["pass_at_1"].mean()
    avg_pass3 = evaluated["pass_at_3"].mean()
    avg_ckpt = evaluated["avg_pct"].mean()

    asking_stats = _compute_asking_stats(evaluated)
    token_stats = _compute_token_stats(evaluated)
    by_class = _compute_by_class_metrics(evaluated)

    # By segment count (parse from dimension: "goal+input" → 2 segments)
    if "dimension" in evaluated.columns:
        evaluated["n_segments"] = evaluated["dimension"].apply(
            lambda x: str(x).count("+") + 1 if pd.notna(x) else 1
        )
        by_segments = {}
        for n_seg in sorted(evaluated["n_segments"].unique()):
            seg_df = evaluated[evaluated["n_segments"] == n_seg]
            seg_n = len(seg_df)
            seg_asking = _compute_asking_stats(seg_df)
            by_segments[n_seg] = {
                "n": seg_n,
                "pass_at_3": seg_df["pass_at_3"].mean(),
                "avg_pct": seg_df["avg_pct"].mean(),
                "pct_samples_asked": seg_asking["pct_samples_asked"],
                "pct_trials_asked": seg_asking["pct_trials_asked"],
                "avg_q_per_traj": seg_asking["avg_q_per_traj"],
            }
    else:
        by_segments = {}

    by_task = _compute_by_task_metrics(evaluated)

    # By model (if available)
    by_model = {}
    if "model" in evaluated.columns:
        for model in evaluated["model"].unique():
            model_df = evaluated[evaluated["model"] == model]
            model_n = len(model_df)
            if "n_evals_latest" in model_df.columns:
                model_incomplete = len(model_df[model_df["n_evals_latest"] < 3])
            else:
                model_incomplete = len(model_df[model_df["n_trials"] < 3])
            model_asking = _compute_asking_stats(model_df)
            by_model[model] = {
                "n": model_n,
                "n_trials": int(model_df["n_trials"].sum()),
                "n_incomplete": model_incomplete,
                "pass_at_3": model_df["pass_at_3"].mean(),
                "avg_pct": model_df["avg_pct"].mean(),
                **{
                    k: model_asking[k]
                    for k in (
                        "n_samples_asked",
                        "pct_samples_asked",
                        "pct_trials_asked",
                        "avg_q_per_sample",
                        "avg_q_per_traj",
                    )
                },
            }

    return {
        "n_variants": n,
        "n_trials": n_trials,
        "n_incomplete": n_incomplete,
        "pass_at_1": avg_pass1,
        "pass_at_3": avg_pass3,
        "avg_ckpt_pct": avg_ckpt,
        **asking_stats,
        **token_stats,
        "by_class": by_class,
        "by_segments": by_segments,
        "by_task": by_task,
        "by_model": by_model,
    }


MODEL_DISPLAY_NAMES = {
    # Anthropic
    "opus_4_5": "Opus-4.5",
    "sonnet_4_5": "Sonnet-4.5",
    "opus_4_6": "Opus-4.6",
    "sonnet_4_6": "Sonnet-4.6",
    "sonnet_4": "Sonnet-4",
    "haiku_4_5": "Haiku-4.5",
    # OpenAI
    "gpt_5_2": "GPT-5.2",
    "gpt_5_1": "GPT-5.1",
    "gpt_5": "GPT-5",
    "o3_pro": "o3-Pro",
    "o3": "o3",
    "gpt_4_1_mini": "GPT-4.1-Mini",
    # Google
    "gemini_3_pro": "Gemini-3-Pro",
    "gemini_3_flash": "Gemini-3-Flash",
    "gemini_3_1_pro": "Gemini-3.1-Pro",
    "gemini_3_1_flash_lite": "Gemini-3.1-Flash-Lite",
    # Other
    "kimi_k2": "Kimi-K2",
    "qwen3_235b": "Qwen3-235B",
    "llama4_maverick": "Llama4-Maverick",
    "glm_4p5_air": "GLM-4.5-Air",
    "nova_2_lite": "Nova-2-Lite",
}

DIMENSION_ORDER = ["goal", "constraint", "input", "context"]
DIMENSION_SHORT = {"goal": "Goal", "constraint": "Constr.", "input": "Input", "context": "Context"}


def _get_primary_dimensions(dim_str: str) -> list[str]:
    """Split compound dimension string (e.g. 'goal+input') into individual dimensions."""
    if pd.isna(dim_str) or not dim_str:
        return []
    return [d.strip().lower() for d in str(dim_str).split("+")]


def _compute_slice_metrics(df: pd.DataFrame) -> dict:
    """Compute pass@3, ckpt%, and asking stats for a DataFrame slice."""
    evaluated = df[df["has_run"]].copy() if "has_run" in df.columns else df.copy()
    if evaluated.empty:
        return {}

    n = len(evaluated)
    asking = _compute_asking_stats(evaluated)

    # Recompute intermediates that callers need as raw values
    total_trials = (
        float(evaluated["n_trials"].sum()) if "n_trials" in evaluated.columns else float(n * 3)
    )
    if "asked_any" in evaluated.columns and "n_trials" in evaluated.columns:
        asked_trials = float((evaluated["asked_any"] * evaluated["n_trials"]).sum())
    elif "asked_any" in evaluated.columns:
        asked_trials = float(evaluated["asked_any"].mean() * total_trials)
    else:
        asked_trials = 0.0

    return {
        "n": n,
        "pass_at_3": evaluated["pass_at_3"].mean(),
        "avg_ckpt_pct": evaluated["avg_pct"].mean(),
        "pct_trials_asked": asking["pct_trials_asked"],
        "asked_trials": asked_trials,
        "total_trials": total_trials,
        "total_questions": asking["total_questions"],
        "avg_q_per_traj": asking["avg_q_per_traj"],
    }


def _gain_per_q(gain_pct: float, total_questions: float) -> float:
    """Compute Gain/Q = percentage-point gain / total questions asked."""
    if total_questions <= 0:
        return 0.0
    return gain_pct / total_questions


def print_table3(
    data: dict,
    original_baseline: dict,
    models: list[str],
    dataset_label: str = "TAC",
    n_tasks: int | None = None,
    n_variants: int | None = None,
):
    """Print Table 3 (main results): per-model performance summary.

    Matches the paper format:
      Model | Baseline Pass@3 Ckpt | User Behavior Ask% Avg/Traj |
             Pass@3% Score (+gain) Gain/Q | Ckpt% Score (+gain) Gain/Q
    Also shows Total Q for context on Gain/Q denominators.
    """
    baseline_df = data.get("baseline")
    prompted_df = data.get("usersim_prompted")
    if baseline_df is None and prompted_df is None:
        print("  (no data for Table 3)")
        return

    meta = dataset_label
    if n_tasks is not None:
        meta += f", n_tasks={n_tasks}"
    if n_variants is not None:
        meta += f", n_variants={n_variants}"

    print(f"\n{'='*130}")
    print(f"  Table 3: Overall Performance Summary ({meta})")
    print(f"{'='*130}")
    sub = (
        f"{'':18} "
        f"{'─── Baseline ───':>16}  "
        f"{'── User Behavior ──':>19}  "
        f"{'──── Pass@3 % ────':>22}  "
        f"{'──── Ckpt % ────':>22}  "
        f"{'─ Questions ─':>13}"
    )
    hdr = (
        f"{'Model':<18} "
        f"{'Pass@3':>7} {'Ckpt':>7}  "
        f"{'Ask%':>5} {'Avg/T':>6} {'TotQ':>6}  "
        f"{'Score (+gain)':>16} {'Gain/Q':>6}  "
        f"{'Score (+gain)':>16} {'Gain/Q':>6}  "
    )
    print(sub)
    print(hdr)
    print("─" * 130)

    for model in models:
        display = MODEL_DISPLAY_NAMES.get(model, model)

        # Original well-specified baseline (average across tasks for this model)
        orig_p3_vals = []
        orig_ckpt_vals = []
        for task_data in original_baseline.values():
            md = task_data.get(model)
            if md:
                if md.get("pass_at_3") is not None:
                    orig_p3_vals.append(md["pass_at_3"])
                if md.get("avg_ckpt_acc") is not None:
                    orig_ckpt_vals.append(md["avg_ckpt_acc"])
        orig_p3 = (sum(orig_p3_vals) / len(orig_p3_vals) * 100) if orig_p3_vals else None
        orig_ckpt = (sum(orig_ckpt_vals) / len(orig_ckpt_vals) * 100) if orig_ckpt_vals else None

        # Underspec baseline (no user tool)
        bl_m = (
            _compute_slice_metrics(baseline_df[baseline_df["model"] == model])
            if baseline_df is not None and "model" in baseline_df.columns
            else {}
        )

        # Underspec prompted (with user tool)
        pr_m = (
            _compute_slice_metrics(prompted_df[prompted_df["model"] == model])
            if prompted_df is not None and "model" in prompted_df.columns
            else {}
        )

        if not bl_m and not pr_m:
            continue

        bl_p3 = bl_m.get("pass_at_3", 0) * 100 if bl_m else None
        bl_ckpt = bl_m.get("avg_ckpt_pct", 0) if bl_m else None
        pr_p3 = pr_m.get("pass_at_3", 0) * 100 if pr_m else None
        pr_ckpt = pr_m.get("avg_ckpt_pct", 0) if pr_m else None

        # Asking stats from the prompted condition
        ask_pct = pr_m.get("pct_trials_asked", 0) * 100 if pr_m else 0
        avg_q = pr_m.get("avg_q_per_traj", 0) if pr_m else 0
        total_q = pr_m.get("total_questions", 0) if pr_m else 0

        # Gains (only when both conditions have data)
        has_both = bl_p3 is not None and pr_p3 is not None
        gain_p3 = (pr_p3 - bl_p3) if has_both else 0
        gain_ckpt = ((pr_ckpt or 0) - (bl_ckpt or 0)) if has_both else 0

        # Gain/Q = percentage-point gain / total questions
        gq_p3 = _gain_per_q(gain_p3, total_q)
        gq_ckpt = _gain_per_q(gain_ckpt, total_q)

        # Format baseline (use original well-spec if available, else underspec baseline)
        bl_p3_v = bl_p3 if bl_p3 is not None else 0
        bl_ckpt_v = bl_ckpt if bl_ckpt is not None else 0
        orig_p3_str = f"{orig_p3:>6.1f}%" if orig_p3 is not None else f"{bl_p3_v:>6.1f}%"
        orig_ckpt_str = f"{orig_ckpt:>6.1f}%" if orig_ckpt is not None else f"{bl_ckpt_v:>6.1f}%"

        # Format scores with gain in parentheses
        if has_both:
            sign_p3 = "+" if gain_p3 >= 0 else ""
            sign_ck = "+" if gain_ckpt >= 0 else ""
            p3_score = f"{bl_p3_v:.1f} ({sign_p3}{gain_p3:.1f})"
            ck_score = f"{bl_ckpt_v:.1f} ({sign_ck}{gain_ckpt:.1f})"
        else:
            p3_score = f"{bl_p3_v:.1f}" if bl_p3 is not None else "—"
            ck_score = f"{bl_ckpt_v:.1f}" if bl_ckpt is not None else "—"

        # Gain/Q formatting
        gq_p3_str = f"{gq_p3:+.2f}" if has_both and total_q > 0 else "N/A"
        gq_ckpt_str = f"{gq_ckpt:+.2f}" if has_both and total_q > 0 else "N/A"

        total_q_str = f"{int(total_q)}"

        print(
            f"{display:<18} "
            f"{orig_p3_str} {orig_ckpt_str}  "
            f"{ask_pct:>4.0f}% {avg_q:>5.2f} {total_q_str:>6}  "
            f"{p3_score:>16} {gq_p3_str:>6}  "
            f"{ck_score:>16} {gq_ckpt_str:>6}  "
        )


def print_tables_6_7(
    data: dict,
    original_baseline: dict,
    models: list[str],
    dataset_label: str = "TAC",
):
    """Print Tables 6 & 7: per-model x per-dimension Pass@3 and Ckpt%.

    Shows Base, All (with +gain), then per-dimension (with +gain).
    """
    baseline_df = data.get("baseline")
    prompted_df = data.get("usersim_prompted")
    if baseline_df is None and prompted_df is None:
        print("  (no data for Tables 6/7)")
        return

    for metric_name, metric_col, table_num in [
        ("Pass@3 (%)", "pass_at_3", 6),
        ("Avg. Checkpoint Progress (%)", "avg_pct", 7),
    ]:
        print(f"\n{'='*100}")
        print(f"  Table {table_num}: {metric_name} by Information Dimension ({dataset_label})")
        print(f"{'='*100}")
        dim_headers = "".join(f"{DIMENSION_SHORT.get(d, d):>14}" for d in DIMENSION_ORDER)
        hdr = f"{'Model':<18} {'Base':>7} {'All':>14}{dim_headers}"
        print(hdr)
        print("─" * len(hdr))

        for model in models:
            display = MODEL_DISPLAY_NAMES.get(model, model)

            # Original baseline for this model
            orig_vals = []
            for task_data in original_baseline.values():
                md = task_data.get(model)
                if md:
                    if metric_col == "pass_at_3" and md.get("pass_at_3") is not None:
                        orig_vals.append(md["pass_at_3"])
                    elif metric_col == "avg_pct" and md.get("avg_ckpt_acc") is not None:
                        orig_vals.append(md["avg_ckpt_acc"])
            base_val = (sum(orig_vals) / len(orig_vals) * 100) if orig_vals else None

            # Compute All (underspec baseline vs prompted)
            bl_all = (
                _compute_slice_metrics(baseline_df[baseline_df["model"] == model])
                if baseline_df is not None and "model" in baseline_df.columns
                else {}
            )
            pr_all = (
                _compute_slice_metrics(prompted_df[prompted_df["model"] == model])
                if prompted_df is not None and "model" in prompted_df.columns
                else {}
            )

            if not bl_all and not pr_all:
                continue

            if metric_col == "pass_at_3":
                bl_val = bl_all.get("pass_at_3", 0) * 100 if bl_all else None
                pr_val = pr_all.get("pass_at_3", 0) * 100 if pr_all else None
            else:
                bl_val = bl_all.get("avg_ckpt_pct", 0) if bl_all else None
                pr_val = pr_all.get("avg_ckpt_pct", 0) if pr_all else None

            base_str = f"{base_val:>6.1f}%" if base_val is not None else f"{'—':>7}"
            if bl_val is not None and pr_val is not None:
                gain_all = pr_val - bl_val
                sign_all = "+" if gain_all >= 0 else ""
                all_str = f"{pr_val:.1f} ({sign_all}{gain_all:.1f})"
            elif pr_val is not None:
                all_str = f"{pr_val:>6.1f}%"
            elif bl_val is not None:
                all_str = f"{bl_val:>6.1f}%"
            else:
                all_str = f"{'—':>7}"

            # Per-dimension breakdown
            dim_strs = []
            for dim in DIMENSION_ORDER:
                bl_dim_df = (
                    _filter_by_dimension(baseline_df, model, dim)
                    if baseline_df is not None
                    else pd.DataFrame()
                )
                pr_dim_df = (
                    _filter_by_dimension(prompted_df, model, dim)
                    if prompted_df is not None
                    else pd.DataFrame()
                )

                bl_dim_m = _compute_slice_metrics(bl_dim_df) if not bl_dim_df.empty else {}
                pr_dim_m = _compute_slice_metrics(pr_dim_df) if not pr_dim_df.empty else {}

                if not bl_dim_m and not pr_dim_m:
                    dim_strs.append(f"{'–':>14}")
                    continue

                if metric_col == "pass_at_3":
                    bl_d = bl_dim_m.get("pass_at_3", 0) * 100 if bl_dim_m else None
                    pr_d = pr_dim_m.get("pass_at_3", 0) * 100 if pr_dim_m else None
                else:
                    bl_d = bl_dim_m.get("avg_ckpt_pct", 0) if bl_dim_m else None
                    pr_d = pr_dim_m.get("avg_ckpt_pct", 0) if pr_dim_m else None

                if bl_d is not None and pr_d is not None:
                    gain_d = pr_d - bl_d
                    sign_d = "+" if gain_d >= 0 else ""
                    dim_strs.append(f"{pr_d:.1f} ({sign_d}{gain_d:.1f})")
                elif pr_d is not None:
                    dim_strs.append(f"{pr_d:>6.1f}%")
                elif bl_d is not None:
                    dim_strs.append(f"{bl_d:>6.1f}%")
                else:
                    dim_strs.append(f"{'–':>14}")

            dim_vals = "".join(f"{s:>14}" for s in dim_strs)
            print(f"{display:<18} {base_str} {all_str:>14}{dim_vals}")

        print()


def _filter_by_dimension(df: pd.DataFrame, model: str, dimension: str) -> pd.DataFrame:
    """Filter DataFrame to rows matching a model and containing a specific dimension."""
    if df is None or df.empty or "model" not in df.columns or "dimension" not in df.columns:
        return pd.DataFrame()
    model_df = df[df["model"] == model].copy()
    mask = model_df["dimension"].apply(lambda d: dimension in _get_primary_dimensions(d))
    return model_df[mask]


def print_table_by_class(
    data: dict,
    models: list[str],
    dataset_label: str = "TAC",
):
    """Print performance breakdown by ambiguity class (Tables 8-10 style)."""
    baseline_df = data.get("baseline")
    prompted_df = data.get("usersim_prompted")
    if baseline_df is None and prompted_df is None:
        print("  (no data for by-class table)")
        return

    print(f"\n{'='*80}")
    print(f"  Pass@3 (%) by Ambiguity Class ({dataset_label})")
    print(f"{'='*80}")
    hdr = f"{'Class':<24} {'Model':<18} {'Base':>7} {'Under.':>7} {'+User':>7}"
    print(hdr)
    print("─" * len(hdr))

    for cls in ["outcome-critical", "divergent", "benign"]:
        # Collect unique tasks in this class from either condition
        cls_tasks = set()
        for cond_df in [baseline_df, prompted_df]:
            if cond_df is not None and not cond_df.empty:
                cls_mask = cond_df["ambiguity_class"] == cls
                cls_tasks.update(cond_df.loc[cls_mask, "task"].unique())

        first_in_cls = True
        for model in models:
            display = MODEL_DISPLAY_NAMES.get(model, model)

            # Original baseline scoped to tasks in this ambiguity class
            orig_vals = []
            for task, task_data in data.get("_original_baseline", {}).items():
                if task not in cls_tasks:
                    continue
                md = task_data.get(model)
                if md and md.get("pass_at_3") is not None:
                    orig_vals.append(md["pass_at_3"])
            orig_p3 = (sum(orig_vals) / len(orig_vals) * 100) if orig_vals else None

            # Underspec baseline (no user)
            bl_cls_df = _filter_by_class(baseline_df, model, cls)
            bl_m = _compute_slice_metrics(bl_cls_df) if not bl_cls_df.empty else {}

            # Underspec prompted (with user)
            pr_cls_df = _filter_by_class(prompted_df, model, cls)
            pr_m = _compute_slice_metrics(pr_cls_df) if not pr_cls_df.empty else {}

            if not bl_m and not pr_m:
                continue

            base_str = f"{orig_p3:>6.1f}%" if orig_p3 is not None else f"{'—':>7}"
            bl_p3_str = f"{bl_m.get('pass_at_3', 0) * 100:>6.1f}%" if bl_m else f"{'—':>7}"
            pr_p3_str = f"{pr_m.get('pass_at_3', 0) * 100:>6.1f}%" if pr_m else f"{'—':>7}"

            n = bl_m.get("n", 0) or pr_m.get("n", 0)
            cls_display = f"{cls} (n={n})" if first_in_cls else ""
            first_in_cls = False
            print(f"{cls_display:<24} {display:<18} " f"{base_str} {bl_p3_str} {pr_p3_str}")
        if cls != "benign":
            print()


def _filter_by_class(df: pd.DataFrame, model: str, cls: str) -> pd.DataFrame:
    """Filter DataFrame to rows matching a model and ambiguity class."""
    if df is None or df.empty:
        return pd.DataFrame()
    mask = (df["model"] == model) & (df["ambiguity_class"] == cls)
    return df[mask]


def print_table11_ablation(
    data: dict,
    models: list[str],
    dataset_label: str = "TAC",
):
    """Print Table 11: Agentic Prompting Ablation.

    By class x strategy showing Pass@3, Ckpt%, %Ask, Avg/Traj.
    Values in parentheses show improvement with user tool.
    """
    strategies = data.get("_strategies", [])
    strategy_labels = data.get("_strategy_labels", {})

    print(f"\n{'='*100}")
    print(f"  Table 11: Agentic Prompting Ablation ({dataset_label}, {', '.join(models)})")
    print(f"{'='*100}")
    hdr = (
        f"{'Class':<20} {'Strategy':<22} {'n':>5} "
        f"{'Pass@3':>14} {'Ckpt%':>14} {'%Ask':>5} {'Avg/T':>6}"
    )
    print(hdr)
    print("─" * len(hdr))

    for cls_label, cls_filter in [
        ("Overall", None),
        ("Outcome-Critical", "outcome-critical"),
        ("Divergent", "divergent"),
        ("Benign", "benign"),
    ]:
        first_in_class = True
        for strategy in strategies:
            display_strategy = strategy_labels.get(strategy, strategy)
            if strategy == "none":
                display_strategy = "CodeAct (Original)"

            bl_slices = []
            us_slices = []
            for model in models:
                bl_key = f"baseline_{model}_{strategy}"
                us_key = f"usersim_{model}_{strategy}"
                bl_df = data.get(bl_key)
                us_df = data.get(us_key)

                if isinstance(bl_df, pd.DataFrame) and not bl_df.empty:
                    sub = bl_df[bl_df["ambiguity_class"] == cls_filter] if cls_filter else bl_df
                    if not sub.empty:
                        bl_slices.append(sub)
                if isinstance(us_df, pd.DataFrame) and not us_df.empty:
                    sub = us_df[us_df["ambiguity_class"] == cls_filter] if cls_filter else us_df
                    if not sub.empty:
                        us_slices.append(sub)

            bl_combined = pd.concat(bl_slices, ignore_index=True) if bl_slices else pd.DataFrame()
            us_combined = pd.concat(us_slices, ignore_index=True) if us_slices else pd.DataFrame()

            bl_m = _compute_slice_metrics(bl_combined) if not bl_combined.empty else {}
            us_m = _compute_slice_metrics(us_combined) if not us_combined.empty else {}

            if not bl_m and not us_m:
                continue

            n_bl = bl_m.get("n", 0)
            n_us = us_m.get("n", 0)

            if bl_m:
                bl_p3 = bl_m.get("pass_at_3", 0) * 100
                bl_ckpt = bl_m.get("avg_ckpt_pct", 0)
            else:
                bl_p3 = None
                bl_ckpt = None

            us_p3 = us_m.get("pass_at_3", 0) * 100 if us_m else 0
            us_ckpt = us_m.get("avg_ckpt_pct", 0) if us_m else 0

            if bl_p3 is not None and us_m:
                gain_p3 = us_p3 - bl_p3
                gain_ckpt = us_ckpt - bl_ckpt
            else:
                gain_p3 = 0
                gain_ckpt = 0

            ask_pct = us_m.get("pct_trials_asked", 0) * 100 if us_m else 0
            avg_q = us_m.get("avg_q_per_traj", 0) if us_m else 0

            if bl_p3 is not None:
                sign_p3 = "+" if gain_p3 >= 0 else ""
                sign_ck = "+" if gain_ckpt >= 0 else ""
                p3_str = f"{bl_p3:.1f} ({sign_p3}{gain_p3:.1f})"
                ck_str = f"{bl_ckpt:.1f} ({sign_ck}{gain_ckpt:.1f})"
            else:
                p3_str = "—"
                ck_str = "—"

            n_str = f"{n_bl}/{n_us}" if n_bl != n_us else str(n_bl)
            cls_display = cls_label if first_in_class else ""
            first_in_class = False

            print(
                f"{cls_display:<20} {display_strategy:<22} {n_str:>5} "
                f"{p3_str:>14} {ck_str:>14} {ask_pct:>4.0f}% {avg_q:>5.1f}"
            )
        if cls_label != "Benign":
            print()


def main():
    parser = argparse.ArgumentParser(description="Compare all experimental conditions")
    parser.add_argument("--experiment", "-e", required=True, help="Experiment name")
    parser.add_argument(
        "--models",
        "-m",
        default="sonnet_4_5,gpt_5_2,opus_4_5,gemini_3_flash,gemini_3_pro",
        help="Comma-separated model short names (from constants.py MODELS)",
    )
    parser.add_argument(
        "--min-evals",
        type=int,
        default=3,
        help="Minimum evals required in latest run to include a variant",
    )
    parser.add_argument(
        "--usersim-suffix",
        "-s",
        default="default",
        help="Suffix for user-sim experiment name (e.g., 'default', 'med_plain'). "
        "Experiment name becomes: user_sim_{experiment}_{model}_{suffix}",
    )
    parser.add_argument(
        "--ablation",
        type=str,
        choices=["agent_prompt"],
        help="Run ablation analysis instead of standard comparison. Options: agent_prompt",
    )
    args = parser.parse_args()

    # Support "all" as shorthand for all models used in paper exps
    ALL_MODELS = ["sonnet_4_5", "gpt_5_2", "opus_4_5", "gemini_3_flash", "gemini_3_pro"]
    if args.models.lower() == "all":
        models = ALL_MODELS
    else:
        models = [m.strip() for m in args.models.split(",")]
    experiment = args.experiment

    # Handle ablation mode
    if args.ablation:
        print(f"📊 Loading ablation data: {args.ablation} for {experiment}...")
        print(f"   Models: {', '.join(models)}")

        data = load_ablation_data(args.ablation, experiment, models, min_evals=args.min_evals)

        if not data or len([k for k in data.keys() if not k.startswith("_")]) == 0:
            print("❌ No ablation data found!")
            return 1

        print("\n📈 Computing metrics...")
        metrics = {}
        for cond, df in data.items():
            if cond.startswith("_") or not isinstance(df, pd.DataFrame):
                continue
            metrics[cond] = compute_condition_metrics(df)
            n = metrics[cond].get("n_variants", 0)
            p3 = metrics[cond].get("pass_at_3", 0) * 100
            print(f"   {cond}: {n} variants, {p3:.1f}% pass@3")

        # Count unique variants from any condition
        all_variants = set()
        all_tasks = set()
        for cond, df in data.items():
            if isinstance(df, pd.DataFrame) and "variant_id" in df.columns:
                all_variants.update(df["variant_id"].unique())
            if isinstance(df, pd.DataFrame) and "task" in df.columns:
                all_tasks.update(df["task"].unique())

        print_table11_ablation(
            data,
            models,
            dataset_label=f"TAC {experiment}, n_tasks={len(all_tasks)}, n_variants={len(all_variants)}",
        )

        return 0

    # Standard comparison mode
    usersim_suffix = args.usersim_suffix
    print(f"📊 Loading data for {experiment}...")
    print(f"   Models: {', '.join(models)}")
    print(f"   User-sim suffix: {usersim_suffix}")

    # Load all condition data
    data = load_condition_data(
        experiment, models, min_evals=args.min_evals, usersim_suffix=usersim_suffix
    )

    if not data:
        print("❌ No data found for any condition!")
        return 1

    print("\n📈 Computing metrics...")
    metrics = {}
    for cond, df in data.items():
        if cond.startswith("_"):
            continue
        metrics[cond] = compute_condition_metrics(df)

    # Load original well-specified TAC baseline for terminal summary
    all_tasks_in_data = set()
    for cond_df in data.values():
        if isinstance(cond_df, pd.DataFrame) and "task" in cond_df.columns:
            all_tasks_in_data.update(cond_df["task"].unique().tolist())
    original_baseline = load_original_baseline(list(all_tasks_in_data), models)

    # Count unique tasks and variants for metadata
    n_unique_tasks = len(all_tasks_in_data)
    n_unique_variants = 0
    for cond_df in data.values():
        if isinstance(cond_df, pd.DataFrame) and "variant_id" in cond_df.columns:
            n_unique_variants = max(n_unique_variants, cond_df["variant_id"].nunique())

    dataset_label = f"TAC {experiment}"

    # Store original_baseline in data dict for by-class table
    data["_original_baseline"] = original_baseline

    # ── Table 3: Per-Model Performance Summary ──
    print_table3(
        data,
        original_baseline,
        models,
        dataset_label=dataset_label,
        n_tasks=n_unique_tasks,
        n_variants=n_unique_variants,
    )

    # ── Tables 6 & 7: Per-Dimension Breakdown ──
    print_tables_6_7(data, original_baseline, models, dataset_label=dataset_label)

    # ── By Ambiguity Class ──
    print_table_by_class(data, models, dataset_label=dataset_label)

    # ── Per-Task Breakdown (compact) ──
    _print_per_task_breakdown(data, metrics, original_baseline, models)

    print()

    return 0


def _print_per_task_breakdown(
    data: dict,
    metrics: dict,
    original_baseline: dict,
    models: list[str],
):
    """Print compact per-task breakdown showing baseline vs prompted pass@3."""
    active_conds = [c for c in ["baseline", "usersim_prompted"] if c in metrics]
    if not active_conds:
        return

    all_tasks = set()
    for cond in active_conds:
        by_task = metrics.get(cond, {}).get("by_task", {})
        all_tasks.update(by_task.keys())
    if not all_tasks:
        return

    has_orig = bool(original_baseline)
    cond_labels = {"baseline": "No User", "usersim_prompted": "+User"}

    print(f"\n{'='*80}")
    print("  Per-Task Breakdown")
    print(f"{'='*80}")
    task_header = f"{'Task':<40} "
    if has_orig:
        task_header += f"{'Orig':>6} "
    for cond in active_conds:
        task_header += f"{cond_labels.get(cond, cond):>8} {'Ckpt':>5} "
    if len(active_conds) > 1:
        task_header += f"{'Δ P@3':>7} {'Δ Ckpt':>7}"
    print(task_header)
    print("─" * len(task_header))

    for task in sorted(all_tasks):
        short_task = task[:38] + ".." if len(task) > 40 else task
        print(f"{short_task:<40} ", end="")
        if has_orig:
            task_orig = original_baseline.get(task, {})
            orig_vals = [v.get("pass_at_3", 0) for v in task_orig.values()]
            if orig_vals:
                print(f"{sum(orig_vals)/len(orig_vals)*100:>5.0f}% ", end="")
            else:
                print(f"{'—':>6} ", end="")
        task_p3s = []
        task_ckpts = []
        for cond in active_conds:
            td = metrics.get(cond, {}).get("by_task", {}).get(task, {})
            if td:
                tp3 = td.get("pass_at_3", 0) * 100
                tavg = td.get("avg_pct", 0)
                task_p3s.append(tp3)
                task_ckpts.append(tavg)
                print(f"{tp3:>7.0f}% {tavg:>4.0f}% ", end="")
            else:
                task_p3s.append(None)
                task_ckpts.append(None)
                print(f"{'—':>8} {'—':>5} ", end="")
        if len(active_conds) > 1:
            if all(v is not None for v in task_p3s):
                delta_p3 = task_p3s[-1] - task_p3s[0]
                sign = "+" if delta_p3 >= 0 else ""
                delta_str = f"{sign}{delta_p3:.0f}pp"
                print(f"{delta_str:>7}", end="")
            else:
                print(f"{'—':>7}", end="")
            if all(v is not None for v in task_ckpts):
                delta_ckpt = task_ckpts[-1] - task_ckpts[0]
                sign = "+" if delta_ckpt >= 0 else ""
                delta_str = f"{sign}{delta_ckpt:.0f}pp"
                print(f"{delta_str:>7}", end="")
            else:
                print(f"{'—':>7}", end="")
        print()


if __name__ == "__main__":
    sys.exit(main())
