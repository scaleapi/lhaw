"""
Process user simulator benchmark results with clarification-specific metrics.

Extends process_tac_underspec.py with user-simulator-specific analysis:
- Distribution Shift: Compare pass@3 with user help vs baseline underspec
- Cost Metrics: Questions asked, tokens read/written by user
- Efficiency: 1/(1+N) for benign tasks (penalizes over-asking)
- Strategic Score: Correct ask/abstain decisions
- Clarification ROI: pass@3 delta when asked vs not asked

Key Research Question (RQ-2c):
What clarification strategies do agents employ, and which types of ambiguity
are more or less likely to be successfully resolved versus hallucinated through?

Usage:
    python scripts/process_tac_usersim.py -e user_sim_v1_sonnet_default
    python scripts/process_tac_usersim.py -e user_sim_v1_gpt_default --export
    python scripts/process_tac_usersim.py -e user_sim_v1_sonnet_default --html
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import shared utilities from process_tac_underspec
from evaluation.pass_k_metrics import pass_at_k  # noqa: E402
from scripts.process_tac_underspec import RUNS_DIR, SYNTH_OUTPUTS, _load_trials  # noqa: E402
from scripts.process_tac_underspec import load_variants as _load_variants_base  # noqa: E402
from scripts.run_tac_underspec import parse_run_dir  # noqa: E402

# =============================================================================
# Loaders (extend base with clarification-specific fields)
# =============================================================================


def load_variants(experiment: str) -> dict:
    """Load variants with ambiguity_class defaulted."""
    base = _load_variants_base(experiment, include_prompts=True)
    for vid, v in base.items():
        v.setdefault("ambiguity_class", "unknown")
    return base


def _extract_tool_name(ev: dict) -> str:
    """Best-effort extraction of tool name from event metadata."""
    args = ev.get("args", {}) or {}
    tool_name = args.get("name") or ev.get("tool_call_metadata", {}).get("function_name", "")
    if not tool_name:
        tool_name = ev.get("extras", {}).get("name", "")
    return str(tool_name)


def _extract_tool_args(ev: dict) -> dict:
    """Best-effort extraction of tool args from event payload."""
    args = ev.get("args", {}) or {}
    tool_args = args.get("arguments", {})
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            return {}
    if isinstance(tool_args, dict):
        return tool_args
    return {}


def _extract_question_from_event(ev: dict) -> str:
    """Extract ask_user question from tool args or message."""
    tool_args = _extract_tool_args(ev)
    if isinstance(tool_args, dict) and tool_args.get("question"):
        return str(tool_args.get("question", "")).strip()

    message = ev.get("message", "") or ev.get("content", "")
    if not isinstance(message, str):
        return ""

    for pattern in [
        r"'question':\s*'([^']*)'",
        r'"question":\s*"([^"]*)"',
        r'question["\']?\s*[:=]\s*["\']([^"\']+)',
    ]:
        match = re.search(pattern, message)
        if match:
            return match.group(1)
    return ""


def _extract_response_from_event(ev: dict) -> str:
    """Extract user response for ask_user from MCP response event."""
    message = ev.get("message", "") or ev.get("content", "")
    if isinstance(message, dict):
        return str(message.get("structuredContent", {}).get("result", "")).strip()
    if isinstance(message, str) and "structuredContent" in message:
        try:
            resp_data = json.loads(message)
            return str(resp_data.get("structuredContent", {}).get("result", "")).strip()
        except json.JSONDecodeError:
            return ""
    return ""


def _load_clarifications_by_trial(run_dir: Path, task: str) -> list[dict]:
    """Load clarification events per trial (aligned to trial ordering).

    Returns list of dicts with: questions, responses, session.
    """
    local_results = run_dir / "local_results" / task
    if not local_results.exists():
        return []

    per_trial = []
    try:
        for model_dir in sorted(d for d in local_results.iterdir() if d.is_dir()):

            # Alphabetical sort matches run_deterministic_evaluation's sort order
            # (sorted by (task_name, session_dir) in task_completion_agentcompany.py line 214)
            date_dirs = sorted(
                d
                for d in model_dir.iterdir()
                if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)
            )

            for date_dir in date_dirs:
                sessions_dir = date_dir / "state" / "sessions"
                if not sessions_dir.exists():
                    per_trial.append({"questions": [], "responses": [], "session": ""})
                    continue

                session_dirs = sorted([d for d in sessions_dir.iterdir() if d.is_dir()])
                if not session_dirs:
                    per_trial.append({"questions": [], "responses": [], "session": ""})
                    continue

                session_dir = session_dirs[0]
                events_dir = session_dir / "events"
                if not events_dir.exists():
                    per_trial.append(
                        {"questions": [], "responses": [], "session": session_dir.name}
                    )
                    continue

                event_files = sorted(
                    [f for f in events_dir.glob("*.json") if f.stem.isdigit()],
                    key=lambda x: int(x.stem),
                )

                questions = []
                responses = []
                pending = []
                for ev_file in event_files:
                    try:
                        ev = json.loads(ev_file.read_text(encoding="utf-8").strip())
                    except (json.JSONDecodeError, KeyError):
                        continue

                    action = ev.get("action", "")
                    tool_name = _extract_tool_name(ev).lower()

                    if action == "call_tool_mcp" and "ask_user" in tool_name:
                        question = _extract_question_from_event(ev)
                        if question:
                            questions.append(question)
                            pending.append(question)
                        continue

                    if ev.get("observation") == "mcp" and pending:
                        response = _extract_response_from_event(ev)
                        if response:
                            responses.append(response)
                            pending.pop(0)

                per_trial.append(
                    {"questions": questions, "responses": responses, "session": session_dir.name}
                )

            break  # Only process first model dir
    except Exception as e:
        print(f"  Warning: failed loading clarifications from {local_results}: {e}")

    return per_trial


def _count_tokens(text: str) -> int:
    """Approximate token count using ~4 chars per token heuristic.

    Follow tau-bench's heuristic for token counting:
    https://github.com/sierra-research/tau-bench/blob/main/tau_bench/model_utils/model/utils.py#L141

    For more accurate counts, would need tiktoken or similar.
    """
    if not text:
        return 0
    return len(text) // 4


def _compute_efficiency(n_questions: int, ambiguity_class: str) -> float:
    """Compute Efficiency = 1/(1+N) for benign tasks.

    Parallels Reciprocal Rank: measures how much user effort is wasted.
    For benign tasks, ideal is N=0 (efficiency=1.0, no unnecessary questions).
    For critical tasks, asking is expected so we don't penalize.

    Returns efficiency score in [0, 1].
    """
    if ambiguity_class == "benign":
        # Penalize asking on benign tasks (should infer, not ask)
        return 1.0 / (1.0 + n_questions)
    elif ambiguity_class in ("outcome-critical", "divergent"):
        # For critical/partial, asking is expected - return 1.0 if asked
        return 1.0 if n_questions > 0 else 0.0
    else:
        return 1.0  # Unknown class, no penalty


def _count_sessions(run_dir: Path, task: str) -> int:
    """Count actual session directories (not evaluation files).

    Sessions are the date_sessionId directories in local_results that represent
    actual agent runs. This is the true trial count.
    """
    local_results = run_dir / "local_results" / task
    if not local_results.exists():
        return 0

    session_count = 0
    try:
        for model_dir in sorted(d for d in local_results.iterdir() if d.is_dir()):
            # Count date_sessionId directories (actual agent sessions)
            # Use regex to match YYYY-MM-DD format (future-proof, not hardcoded to 2026)
            session_count += sum(
                1
                for d in model_dir.iterdir()
                if d.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}", d.name)
            )
    except Exception:
        pass
    return session_count


def _process_single_variant(args: tuple) -> tuple[str, dict] | None:
    """Process a single variant for parallel loading."""
    vid, runs, variants, skip_responses, min_evals = args

    # Sort runs by timestamp (latest first), skip empty runs
    all_runs_sorted = sorted(runs, key=lambda x: x[0], reverse=True)

    def _find_det_root(rd: Path, task: str) -> Path | None:
        """Find the model subdirectory under final_results/deterministic/{task}/.

        Each run has exactly one model, so there should be exactly one subdirectory.
        Returns None if the directory structure doesn't exist.
        """
        det_task = rd / "final_results" / "deterministic" / task
        if not det_task.exists():
            return None
        model_dirs = sorted(d for d in det_task.iterdir() if d.is_dir())
        if not model_dirs:
            return None
        if len(model_dirs) > 1:
            print(f"Warning: multiple model dirs in {det_task}, using {model_dirs[0].name}")
        return model_dirs[0]

    # Find the latest NON-EMPTY run (skip runs with 0 sessions AND 0 evals)
    latest_ts, latest_rd, latest_meta = None, None, None
    for ts, rd, meta in all_runs_sorted:
        task = meta["base_task"]
        det_root = _find_det_root(rd, task)
        n_evals = len(list(det_root.glob("evaluation_*.json"))) if det_root else 0
        n_sessions = _count_sessions(rd, task)
        if n_evals > 0 or n_sessions > 0:
            latest_ts, latest_rd, latest_meta = ts, rd, meta
            break

    # If all runs are empty, skip this variant
    if latest_rd is None:
        return None

    # Get latest run and count its evaluations AND sessions
    # (Re-count for the selected run to use below)
    task = latest_meta["base_task"]
    det_root = _find_det_root(latest_rd, task)
    n_evals_latest_run = len(list(det_root.glob("evaluation_*.json"))) if det_root else 0

    # Count actual sessions (the true trial count)
    n_sessions_latest_run = _count_sessions(latest_rd, task)

    # Skip variants where latest run has insufficient SESSIONS (not just evals)
    # This ensures we have actual agent runs, not just repeated evaluations
    if n_sessions_latest_run < min_evals:
        return None

    # Warn if session/eval mismatch (indicates data integrity issue)
    if n_sessions_latest_run != n_evals_latest_run and n_sessions_latest_run > 0:
        print(
            f"Warning: {vid} has {n_sessions_latest_run} sessions but {n_evals_latest_run} evals (mismatch)"
        )

    # Use latest non-empty run for all metrics
    ts, run_dir, meta = latest_ts, latest_rd, latest_meta

    task = meta["base_task"]
    # Load trials - but limit to actual session count to avoid counting repeated evals
    trials = _load_trials(run_dir, task, skip_responses=skip_responses)
    # Only use as many trials as we have actual sessions
    if len(trials) > n_sessions_latest_run and n_sessions_latest_run > 0:
        trials = trials[:n_sessions_latest_run]

    clarifications_by_trial = _load_clarifications_by_trial(run_dir, task)

    v = variants.get(vid, {})
    ambiguity_class = v.get("ambiguity_class", "unknown")

    # Align clarifications list length to trials length
    n_trials = len(trials)
    n_clarification_sessions = len(clarifications_by_trial)
    sessions_missing = n_trials - n_clarification_sessions

    if sessions_missing > 0:
        clarifications_by_trial.extend(
            [{"questions": [], "responses": [], "session": "", "_missing": True}] * sessions_missing
        )
    clarifications_by_trial = clarifications_by_trial[:n_trials]

    # Per-trial clarification metrics
    per_trial_questions = [len(c.get("questions", [])) for c in clarifications_by_trial]
    per_trial_asked = [1 if q > 0 else 0 for q in per_trial_questions]
    per_trial_tokens_read = [
        sum(_count_tokens(q) for q in c.get("questions", [])) for c in clarifications_by_trial
    ]
    per_trial_tokens_wrote = [
        sum(_count_tokens(r) for r in c.get("responses", [])) for c in clarifications_by_trial
    ]
    per_trial_responses = [len(c.get("responses", [])) for c in clarifications_by_trial]

    # Aggregate per-trial (normalized) metrics
    n_questions_avg = sum(per_trial_questions) / n_trials if n_trials else 0
    asked_any_rate = sum(per_trial_asked) / n_trials if n_trials else 0
    n_trials_asked = sum(per_trial_asked)
    avg_q_per_traj = sum(per_trial_questions) / n_trials_asked if n_trials_asked else 0
    # Compute Efficiency per trial (average)
    efficiency = (
        sum(_compute_efficiency(q, ambiguity_class) for q in per_trial_questions) / n_trials
        if n_trials
        else 1.0
    )

    # Track session/eval mismatch for data quality
    session_eval_mismatch = n_evals_latest_run != n_sessions_latest_run

    result = {
        "variant_id": vid,
        "task": task,
        "dimension": meta["dimension"],
        "model": det_root.name if det_root else "unknown",
        "ambiguity_class": ambiguity_class,
        # Trial metrics
        "trials": trials,
        "n_trials": len(trials),
        "sessions_missing": sessions_missing,
        "session_eval_mismatch": session_eval_mismatch,
        "full_passes": sum(1 for t in trials if t["full_pass"]),
        # Clarification metrics (per-trial normalized)
        "n_questions": n_questions_avg,
        "n_questions_total": int(sum(per_trial_questions)),
        "n_responses_total": int(sum(per_trial_responses)),
        "asked_any": asked_any_rate,
        "n_trials_asked": n_trials_asked,
        "avg_q_per_traj": round(avg_q_per_traj, 2),
        # Token costs (totals across all trials)
        "tokens_user_read_sum": int(sum(per_trial_tokens_read)),
        "tokens_user_wrote_sum": int(sum(per_trial_tokens_wrote)),
        "total_tokens": int(sum(per_trial_tokens_read) + sum(per_trial_tokens_wrote)),
        "efficiency": efficiency,
    }
    return vid, result


def load_user_sim_results(
    experiment: str, min_evals: int = 3, skip_responses: bool = False
) -> dict:
    """Load evaluation results with clarification metrics (per-trial normalized).

    Args:
        skip_responses: If True, skip loading agent responses (faster for reports).
    """
    variants = load_variants(experiment)
    if not variants:
        return {}

    valid_vids = set(variants.keys())
    runs_by_vid = defaultdict(list)

    # Use glob to only scan directories matching this experiment
    for run_dir in RUNS_DIR.glob(f"*{experiment}*"):
        if not run_dir.is_dir():
            continue
        meta = parse_run_dir(run_dir.name, experiment, valid_vids)
        if meta:
            runs_by_vid[meta["variant_id"]].append((meta["timestamp"], run_dir, meta))

    # Process variants in parallel for faster loading
    results = {}
    work_items = [
        (vid, runs, variants, skip_responses, min_evals) for vid, runs in runs_by_vid.items()
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process_single_variant, item) for item in work_items]
        for future in as_completed(futures):
            result = future.result()
            if result:
                vid, data = result
                results[vid] = data

    return results


# =============================================================================
# DataFrames
# =============================================================================


def build_summary_df(
    experiment: str, min_evals: int = 3, skip_responses: bool = False
) -> pd.DataFrame:
    """Build summary DataFrame with user-sim metrics.

    Args:
        skip_responses: If True, skip loading agent responses (faster for reports).
    """
    results = load_user_sim_results(experiment, min_evals=min_evals, skip_responses=skip_responses)
    variants = load_variants(experiment)

    rows = []
    for vid in sorted(set(results.keys()) | set(variants.keys())):
        r = results.get(vid, {})
        v = variants.get(vid, {})

        trials = r.get("trials", [])
        n = len(trials)
        c = sum(1 for t in trials if t["full_pass"]) if trials else 0

        # Calculate pass@k using unbiased estimator
        p1 = pass_at_k(n, c, 1) if n > 0 else 0.0  # pass@1
        p3 = pass_at_k(n, c, 3) if n > 0 else 0.0  # pass@3 (for baseline comparison)

        # Calculate avg checkpoint percentage
        ckpt_pcts = [t["percent"] for t in trials] if trials else []
        avg_pct = sum(ckpt_pcts) / len(ckpt_pcts) if ckpt_pcts else 0

        row = {
            "variant_id": vid,
            "task": r.get("task", v.get("task", "")),
            "dimension": r.get("dimension", v.get("dimension", "")),
            "model": r.get("model", ""),
            "ambiguity_class": r.get("ambiguity_class", v.get("ambiguity_class", "unknown")),
            # Performance
            "n_trials": n,
            "full_passes": c,
            "pass_at_1": p1,
            "pass_at_3": p3,
            "avg_pct": avg_pct,
            # Clarification behavior (per-trial normalized)
            "asked_any": r.get("asked_any", 0.0),
            "n_trials_asked": r.get("n_trials_asked", 0),
            "n_questions": r.get("n_questions", 0),
            "n_questions_total": r.get("n_questions_total", 0),
            "n_responses_total": r.get("n_responses_total", 0),
            "avg_q_per_traj": r.get("avg_q_per_traj", 0),
            # Token costs (totals across all trials)
            "tokens_user_read_sum": r.get("tokens_user_read_sum", 0),
            "tokens_user_wrote_sum": r.get("tokens_user_wrote_sum", 0),
            "total_tokens": r.get("total_tokens", 0),
            # Efficiency (1/(1+N) for benign)
            "efficiency": r.get("efficiency", 1.0),
            # Data quality
            "sessions_missing": r.get("sessions_missing", 0),
            "session_eval_mismatch": r.get("session_eval_mismatch", False),
            # Metadata
            "criticality": v.get("criticality", 0),
            "guessability": v.get("guessability", 0),
            "has_run": bool(r),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_aggregate_metrics(df: pd.DataFrame) -> dict:
    """Compute aggregate metrics across all variants."""
    if df.empty:
        return {}

    evaluated = df[df["has_run"]].copy()
    if evaluated.empty:
        return {"n_variants": len(df), "n_evaluated": 0}

    # Strategic Behavior: Did agent do the RIGHT thing (trial-weighted)?
    # Critical: should ask. Benign: should abstain.
    critical = evaluated[evaluated["ambiguity_class"] == "outcome-critical"]
    benign = evaluated[evaluated["ambiguity_class"] == "benign"]
    divergent = evaluated[evaluated["ambiguity_class"] == "divergent"]

    critical_trials = int(critical["n_trials"].sum()) if len(critical) else 0
    benign_trials = int(benign["n_trials"].sum()) if len(benign) else 0
    critical_asked_trials = int(critical["n_trials_asked"].sum()) if len(critical) else 0
    benign_abstained_trials = (
        int((benign["n_trials"] - benign["n_trials_asked"]).sum()) if len(benign) else 0
    )
    critical_score = critical_asked_trials / critical_trials if critical_trials > 0 else 0
    benign_score = benign_abstained_trials / benign_trials if benign_trials > 0 else 0
    total_decisive = critical_trials + benign_trials
    strategic_score = (
        (critical_asked_trials + benign_abstained_trials) / total_decisive
        if total_decisive > 0
        else 0
    )

    # Clarification ROI: trial-weighted approximation using asked/not-asked trial counts.
    # LIMITATION: pass_at_3 is aggregated across ALL trials for a variant, so mixed
    # variants (some trials asked, some didn't) contribute the same pass_at_3 to both
    # buckets. This metric is only cleanly separable when variants are all-asked or
    # all-not-asked. A proper fix requires per-trial pass/fail × asked status pairing.
    ask_weights = evaluated["n_trials_asked"].clip(lower=0)
    abstain_weights = (evaluated["n_trials"] - evaluated["n_trials_asked"]).clip(lower=0)
    pass3_when_asked = (
        (evaluated["pass_at_3"] * ask_weights).sum() / ask_weights.sum()
        if ask_weights.sum() > 0
        else 0
    )
    pass3_when_not_asked = (
        (evaluated["pass_at_3"] * abstain_weights).sum() / abstain_weights.sum()
        if abstain_weights.sum() > 0
        else 0
    )
    clarification_roi = pass3_when_asked - pass3_when_not_asked

    # Resolution Rate: How many outcome-critical tasks became solvable?
    # This is the KEY distribution shift metric
    critical_resolved = (critical["pass_at_3"] > 0).sum() if len(critical) else 0
    divergent_resolved = (divergent["pass_at_3"] > 0).sum() if len(divergent) else 0
    benign_resolved = (benign["pass_at_3"] > 0).sum() if len(benign) else 0

    total_trials = int(evaluated["n_trials"].sum())
    total_trials_asked = (
        int(evaluated["n_trials_asked"].sum()) if "n_trials_asked" in evaluated.columns else 0
    )
    total_questions = int(evaluated["n_questions_total"].sum())
    total_responses = int(evaluated["n_responses_total"].sum())
    total_tokens_read = int(evaluated["tokens_user_read_sum"].sum())
    total_tokens_wrote = int(evaluated["tokens_user_wrote_sum"].sum())

    # Data quality: check for missing sessions
    total_sessions_missing = (
        int(evaluated["sessions_missing"].sum()) if "sessions_missing" in evaluated.columns else 0
    )

    # Paper metrics: Ask% = fraction of trials invoking user tool
    pct_trials_asked = round(total_trials_asked / total_trials * 100, 1) if total_trials > 0 else 0
    # Paper metrics: Avg/Traj = mean questions per trajectory (only among trajectories with calls)
    avg_q_per_traj = round(total_questions / total_trials_asked, 2) if total_trials_asked > 0 else 0

    avg_questions_per_trial = round(total_questions / total_trials, 2) if total_trials > 0 else 0

    # Overall metrics
    metrics = {
        "n_variants": len(df),
        "n_evaluated": len(evaluated),
        "n_trials_total": total_trials,
        "n_sessions_missing": total_sessions_missing,
        # Performance
        "avg_pass_at_1": round(evaluated["pass_at_1"].mean() * 100, 1),
        "avg_pass_at_3": round(evaluated["pass_at_3"].mean() * 100, 1),
        "avg_ckpt_pct": round(evaluated["avg_pct"].mean(), 1),
        # Clarification behavior (paper Table 3 metrics)
        "pct_asked": pct_trials_asked,  # Ask% (trial-level)
        "avg_q_per_traj": avg_q_per_traj,  # Avg/Traj
        "avg_questions": avg_questions_per_trial,  # Avg Q/Trial (trial-level)
        "total_questions": total_questions,
        # Cost metrics (user effort)
        "total_tokens_user_read": total_tokens_read,
        "total_tokens_user_wrote": total_tokens_wrote,
        "total_tokens": total_tokens_read + total_tokens_wrote,
        "avg_tokens_per_question": round(total_tokens_read / max(1, total_questions), 1),
        "avg_tokens_per_response": round(total_tokens_wrote / max(1, total_responses), 1),
        # Strategic behavior (key metric!)
        "strategic_score": round(strategic_score * 100, 1),  # % correct decisions
        "critical_asked_correctly_pct": round(critical_score * 100, 1),
        "critical_total": len(critical),
        "benign_abstained_correctly_pct": round(benign_score * 100, 1),
        "benign_total": len(benign),
        # Clarification ROI (using pass@3)
        "pass3_when_asked": round(pass3_when_asked * 100, 1),
        "pass3_when_not_asked": round(pass3_when_not_asked * 100, 1),
        "clarification_roi": round(clarification_roi * 100, 1),  # Delta
        # Resolution Rate (KEY distribution shift metric)
        "critical_resolved": int(critical_resolved),
        "critical_resolution_rate": (
            round(critical_resolved / len(critical) * 100, 1) if len(critical) else 0
        ),
        "divergent_resolved": int(divergent_resolved),
        "divergent_resolution_rate": (
            round(divergent_resolved / len(divergent) * 100, 1) if len(divergent) else 0
        ),
        "benign_resolved": int(benign_resolved),
        "benign_resolution_rate": (
            round(benign_resolved / len(benign) * 100, 1) if len(benign) else 0
        ),
        # Efficiency (1/(1+N) for benign tasks)
        "avg_efficiency": round(evaluated["efficiency"].mean(), 3),
    }

    # By ambiguity class
    for cls in ["outcome-critical", "divergent", "benign", "new_task"]:
        cls_df = evaluated[evaluated["ambiguity_class"] == cls]
        if len(cls_df) > 0:
            key = cls.replace("-", "_")
            metrics[f"{key}_n"] = len(cls_df)
            metrics[f"{key}_pass_at_1"] = round(cls_df["pass_at_1"].mean() * 100, 1)
            metrics[f"{key}_pass_at_3"] = round(cls_df["pass_at_3"].mean() * 100, 1)
            metrics[f"{key}_avg_pct"] = round(
                cls_df["avg_pct"].mean(), 1
            )  # Avg checkpoint accuracy
            cls_trials = int(cls_df["n_trials"].sum())
            cls_trials_asked = (
                int(cls_df["n_trials_asked"].sum()) if "n_trials_asked" in cls_df.columns else 0
            )
            cls_total_q = int(cls_df["n_questions_total"].sum())
            metrics[f"{key}_pct_asked"] = (
                round(cls_trials_asked / cls_trials * 100, 1) if cls_trials > 0 else 0
            )
            metrics[f"{key}_avg_q_per_traj"] = (
                round(cls_total_q / cls_trials_asked, 2) if cls_trials_asked > 0 else 0
            )
            metrics[f"{key}_avg_questions"] = (
                round(cls_total_q / cls_trials, 2) if cls_trials > 0 else 0
            )
            metrics[f"{key}_total_tokens"] = int(cls_df["total_tokens"].sum())
            if cls == "benign":
                metrics[f"{key}_efficiency"] = round(cls_df["efficiency"].mean(), 3)

    # By task breakdown
    task_stats = (
        evaluated.groupby("task")
        .agg(
            {
                "pass_at_1": "mean",
                "pass_at_3": "mean",
                "avg_pct": "mean",  # Average checkpoint accuracy
                "asked_any": "mean",
                "n_questions": "mean",
                "variant_id": "count",
            }
        )
        .rename(columns={"variant_id": "n_variants"})
    )
    metrics["task_breakdown"] = task_stats.to_dict(orient="index")

    return metrics


# =============================================================================
# Display
# =============================================================================


def print_summary(experiment: str):
    """Print comprehensive summary of user-sim results."""
    df = build_summary_df(experiment)
    metrics = compute_aggregate_metrics(df)

    print(f"\n{'='*70}")
    print(f"USER SIMULATOR RESULTS: {experiment}")
    print(f"{'='*70}")

    if not metrics.get("n_evaluated"):
        print("No evaluated variants found.")
        return

    print("\n📊 Overview")
    print(f"   Variants: {metrics['n_variants']} | Evaluated: {metrics['n_evaluated']}")
    print(f"   Trials: {metrics['n_trials_total']}")

    # Data quality warning
    sessions_missing = metrics.get("n_sessions_missing", 0)
    if sessions_missing > 0:
        print(
            f"   ⚠️  {sessions_missing} sessions missing (clarification metrics may be undercounted)"
        )

    pass3 = metrics.get("avg_pass_at_3", 0)

    print("\n📈 PERFORMANCE (Key Metrics)")
    print(f"   pass@3:     {pass3:.1f}%")
    print(f"   pass@1:     {metrics['avg_pass_at_1']:.1f}%")
    print(f"   Avg Ckpt %: {metrics['avg_ckpt_pct']:.1f}%")

    print("\n💬 Clarification Behavior (Paper Table 3)")
    print(f"   Ask%%:       {metrics['pct_asked']:.1f}%  (fraction of trials invoking user tool)")
    print(
        f"   Avg/Traj:   {metrics['avg_q_per_traj']:.2f}  (mean questions per trajectory, among those that asked)"
    )
    print(f"   Avg Q/Trial: {metrics['avg_questions']:.2f}  (across all trials)")
    print(f"   Total Questions Asked: {metrics['total_questions']}")
    print("   User Token Cost:")
    print(f"     - Tokens read (questions):  {metrics.get('total_tokens_user_read', 0):,}")
    print(f"     - Tokens wrote (responses): {metrics.get('total_tokens_user_wrote', 0):,}")
    print(f"     - Avg tokens/question:      {metrics.get('avg_tokens_per_question', 0):.1f}")
    print(f"     - Avg tokens/response:      {metrics.get('avg_tokens_per_response', 0):.1f}")

    # Strategic behavior - the KEY metric
    print("\n🎯 STRATEGIC BEHAVIOR (Key Metric)")
    print(f"   Strategic Score: {metrics['strategic_score']:.1f}%")
    print(
        f"   ├─ Critical: asked correctly {metrics['critical_asked_correctly_pct']:.1f}% "
        f"(n={metrics['critical_total']})"
    )
    print(
        f"   └─ Benign:   abstained correctly {metrics['benign_abstained_correctly_pct']:.1f}% "
        f"(n={metrics['benign_total']})"
    )

    # Clarification ROI (using pass@3)
    print("\n📊 Clarification ROI (pass@3)")
    print(f"   pass@3 when asked:     {metrics['pass3_when_asked']:.1f}%")
    print(f"   pass@3 when not asked: {metrics['pass3_when_not_asked']:.1f}%")
    print(f"   ROI (delta):           {metrics['clarification_roi']:+.1f}%")

    # KEY METRIC: Resolution Rate (distribution shift)
    print("\n🔄 RESOLUTION RATE (Distribution Shift)")
    print("   'How many tasks became solvable with clarification?'")
    print(
        f"   ├─ Outcome-critical (was 0% pass): {metrics['critical_resolved']}/{metrics['critical_total']} = {metrics['critical_resolution_rate']:.1f}%"
    )
    print(
        f"   ├─ Divergent:                      {metrics.get('divergent_resolved', 0)}/{metrics.get('divergent_n', 0)} = {metrics.get('divergent_resolution_rate', 0):.1f}%"
    )
    print(
        f"   └─ Benign:                         {metrics['benign_resolved']}/{metrics['benign_total']} = {metrics['benign_resolution_rate']:.1f}%"
    )

    print(
        f"\n⚡ Efficiency (benign): {metrics.get('benign_efficiency', 0):.3f} (1.0 = no wasted questions)"
    )

    # By ambiguity class
    print(f"\n{'─'*90}")
    print("BY AMBIGUITY CLASS")
    print(f"{'─'*90}")
    print(
        f"{'Class':<18} {'n':>5} {'Pass@3':>7} {'AvgAcc':>7} {'Ask%':>7} {'Avg/Traj':>9} {'Tokens':>8} {'Correct':>8}"
    )
    print(f"{'─'*90}")

    for cls in ["outcome-critical", "divergent", "benign"]:
        key = cls.replace("-", "_")
        if f"{key}_n" in metrics:
            if cls == "outcome-critical":
                correct = f"{metrics['critical_asked_correctly_pct']:.1f}%"
            elif cls == "benign":
                correct = f"{metrics['benign_abstained_correctly_pct']:.1f}%"
            else:
                correct = "n/a"

            print(
                f"{cls:<18} {metrics[f'{key}_n']:>5} "
                f"{metrics[f'{key}_pass_at_3']:>6.1f}% "
                f"{metrics.get(f'{key}_avg_pct', 0):>6.1f}% "
                f"{metrics[f'{key}_pct_asked']:>6.1f}% "
                f"{metrics.get(f'{key}_avg_q_per_traj', 0):>9.2f} "
                f"{metrics.get(f'{key}_total_tokens', 0):>8} "
                f"{correct:>8}"
            )

    # By task breakdown
    print(f"\n{'─'*80}")
    print("BY TASK")
    print(f"{'─'*80}")
    print(f"{'Task':<40} {'n':>3} {'Pass@3':>7} {'AvgAcc':>7} {'Asked%':>8}")
    print(f"{'─'*80}")

    task_breakdown = metrics.get("task_breakdown", {})
    for task, stats in sorted(task_breakdown.items(), key=lambda x: -x[1]["pass_at_3"]):
        print(
            f"{task:<40} {stats['n_variants']:>3} "
            f"{stats['pass_at_3']*100:>6.1f}% "
            f"{stats.get('avg_pct', 0):>6.1f}% "
            f"{stats['asked_any']*100:>7.1f}%"
        )

    # Key insights
    print(f"\n{'─'*70}")
    print("KEY INSIGHTS")
    print(f"{'─'*70}")

    strategic = metrics.get("strategic_score", 0)
    benign_asked = metrics.get("benign_pct_asked", 0)
    roi = metrics.get("clarification_roi", 0)

    if strategic >= 70:
        print(f"✅ Good strategic behavior ({strategic:.0f}% correct decisions)")
    elif strategic >= 50:
        print(f"⚠️  Moderate strategic behavior ({strategic:.0f}% correct decisions)")
    else:
        print(f"❌ Poor strategic behavior ({strategic:.0f}% correct decisions)")

    if benign_asked > 50:
        print(f"⚠️  Over-asking on benign tasks ({benign_asked:.0f}%) - wasting user time")

    if roi > 0:
        print(f"✅ Clarification helps! (+{roi:.1f}% pass rate when asked)")
    else:
        print(f"⚠️  Clarification not helping ({roi:+.1f}% delta) - questions not useful?")


def print_variant_details(experiment: str, limit: int = 20):
    """Print per-variant details."""
    df = build_summary_df(experiment)
    evaluated = df[df["has_run"]].sort_values(["ambiguity_class", "task", "variant_id"])

    print(f"\n{'─'*110}")
    print(f"VARIANT DETAILS (showing {min(limit, len(evaluated))} of {len(evaluated)})")
    print(f"{'─'*110}")
    print(
        f"{'Variant':<45} {'Class':<10} {'p@3':>5} {'AvgAcc':>7} {'Asked%':>7} {'#Q':>4} {'Tokens':>7}"
    )
    print(f"{'─'*110}")

    for _, row in evaluated.head(limit).iterrows():
        vid = row["variant_id"][:44]
        cls = row["ambiguity_class"][:9]
        pass3 = row["pass_at_3"] * 100
        avg_acc = row.get("avg_pct", 0)
        asked = row["asked_any"] * 100
        n_q = row["n_questions"]
        tokens = int(row["total_tokens"])
        print(
            f"{vid:<45} {cls:<10} {pass3:>4.0f}% {avg_acc:>6.1f}% "
            f"{asked:>6.1f}% {n_q:>4.2f} {tokens:>7}"
        )


# =============================================================================
# Export
# =============================================================================


def export_results(experiment: str, output_dir: Path = None):
    """Export results to CSV/JSON."""
    df = build_summary_df(experiment)
    metrics = compute_aggregate_metrics(df)

    output_dir = output_dir or SYNTH_OUTPUTS / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export summary CSV
    df.to_csv(output_dir / f"user_sim_summary_{experiment}.csv", index=False, encoding="utf-8")

    # Export aggregate metrics JSON
    with open(output_dir / f"user_sim_metrics_{experiment}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Exported to {output_dir}")
    print(f"   - user_sim_summary_{experiment}.csv ({len(df)} variants)")
    print(f"   - user_sim_metrics_{experiment}.json")


def generate_html_report(experiment: str, output_path: Path = None) -> str:
    """Generate HTML report with visualizations."""
    df = build_summary_df(experiment)
    metrics = compute_aggregate_metrics(df)

    if not metrics.get("n_evaluated"):
        print(f"❌ No evaluated variants for {experiment}")
        return None

    output_path = output_path or SYNTH_OUTPUTS / experiment / f"user_sim_report_{experiment}.html"

    # Determine card classes based on thresholds
    strategic_class = (
        "success"
        if metrics.get("strategic_score", 0) >= 70
        else "warning" if metrics.get("strategic_score", 0) >= 50 else "danger"
    )

    # Build HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>User Simulator Report: {experiment}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 0; padding: 20px; background: #0f0f1a; color: #e0e0e0; }}
        .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                  padding: 24px; margin: -20px -20px 20px; border-bottom: 2px solid #4299e1; }}
        .header h1 {{ margin: 0 0 8px; font-size: 28px; }}
        .header .subtitle {{ opacity: 0.7; font-size: 14px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); 
                        gap: 16px; margin-bottom: 24px; }}
        .metric-card {{ background: #1a1a2e; border-radius: 12px; padding: 20px; 
                       border: 1px solid #2d3748; }}
        .metric-card .value {{ font-size: 28px; font-weight: bold; color: #4299e1; }}
        .metric-card .label {{ font-size: 11px; opacity: 0.7; margin-top: 4px; }}
        .metric-card .sublabel {{ font-size: 10px; opacity: 0.5; }}
        .metric-card.success .value {{ color: #48bb78; }}
        .metric-card.warning .value {{ color: #ed8936; }}
        .metric-card.danger .value {{ color: #f56565; }}
        .section {{ background: #1a1a2e; border-radius: 12px; padding: 24px; 
                   margin-bottom: 20px; border: 1px solid #2d3748; }}
        .section h2 {{ margin: 0 0 16px; font-size: 18px; color: #a0aec0; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #2d3748; }}
        th {{ background: #2d3748; color: #a0aec0; font-weight: 600; }}
        tr:hover {{ background: rgba(255,255,255,0.03); }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 20px; 
                 font-size: 11px; font-weight: 600; }}
        .badge.critical {{ background: #f56565; color: white; }}
        .badge.divergent {{ background: #ed8936; color: white; }}
        
        .badge.benign {{ background: #48bb78; color: white; }}
        .badge.outcome {{ background: #f56565; color: white; }}
        .progress {{ background: #2d3748; border-radius: 4px; height: 8px; overflow: hidden; }}
        .progress-bar {{ height: 100%; background: #4299e1; }}
        .progress-bar.green {{ background: #48bb78; }}
        .progress-bar.orange {{ background: #ed8936; }}
        .insight {{ background: rgba(66, 153, 225, 0.1); border-left: 4px solid #4299e1; 
                   padding: 12px 16px; margin: 12px 0; border-radius: 0 8px 8px 0; }}
        .insight.warning {{ background: rgba(237, 137, 54, 0.1); border-color: #ed8936; }}
        .insight.success {{ background: rgba(72, 187, 120, 0.1); border-color: #48bb78; }}
        .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 User Simulator Report</h1>
        <div class="subtitle">{experiment} | {metrics['n_evaluated']} variants | {metrics['n_trials_total']} trials</div>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="value">{metrics.get('avg_pass_at_3', 0):.1f}%</div>
            <div class="label">pass@3</div>
            <div class="sublabel">Unbiased estimator (k=3)</div>
        </div>
        <div class="metric-card">
            <div class="value">{metrics.get('avg_pass_at_1', 0):.1f}%</div>
            <div class="label">pass@1</div>
        </div>
        <div class="metric-card">
            <div class="value">{metrics.get('avg_ckpt_pct', 0):.1f}%</div>
            <div class="label">Avg Checkpoint %</div>
            <div class="sublabel">Partial credit (score/total)</div>
        </div>
        <div class="metric-card {strategic_class}">
            <div class="value">{metrics.get('strategic_score', 0):.0f}%</div>
            <div class="label">Strategic Score</div>
            <div class="sublabel">{metrics.get('critical_asked_correctly_pct', 0):.1f}% crit + {metrics.get('benign_abstained_correctly_pct', 0):.1f}% benign</div>
        </div>
        <div class="metric-card">
            <div class="value">{metrics['pct_asked']:.0f}%</div>
            <div class="label">Ask%</div>
            <div class="sublabel">Fraction of trials invoking user tool</div>
        </div>
        <div class="metric-card">
            <div class="value">{metrics.get('avg_q_per_traj', 0):.2f}</div>
            <div class="label">Avg/Traj</div>
            <div class="sublabel">Mean questions per trajectory (when asked)</div>
        </div>
        <div class="metric-card">
            <div class="value">{metrics.get('total_questions', 0)}</div>
            <div class="label">Total Questions</div>
        </div>
        <div class="metric-card">
            <div class="value">{metrics.get('total_tokens', 0):,}</div>
            <div class="label">Total User Tokens</div>
            <div class="sublabel">~chars/4 | Read: {metrics.get('total_tokens_user_read', 0):,} | Wrote: {metrics.get('total_tokens_user_wrote', 0):,}</div>
        </div>
        <div class="metric-card {'success' if metrics.get('critical_resolution_rate', 0) > 20 else 'warning'}">
            <div class="value">{metrics.get('critical_resolution_rate', 0):.0f}%</div>
            <div class="label">Critical Resolution Rate</div>
            <div class="sublabel">{metrics.get('critical_resolved', 0)}/{metrics.get('critical_total', 0)} outcome-critical → solved</div>
        </div>
    </div>
"""

    # By class breakdown
    html += """
    <div class="section">
        <h2>📊 By Ambiguity Class</h2>
        <table>
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Variants</th>
                    <th>pass@3</th>
                    <th>Avg Acc</th>
                    <th>% Asked</th>
                    <th>Avg Questions</th>
                    <th>Total Tokens</th>
                </tr>
            </thead>
            <tbody>
"""

    for cls in ["outcome-critical", "divergent", "benign"]:
        key = cls.replace("-", "_")
        if f"{key}_n" in metrics:
            badge_cls = cls.split("-")[0] if "-" in cls else cls
            pass3_val = metrics.get(f"{key}_pass_at_3", 0)
            avg_acc = metrics.get(f"{key}_avg_pct", 0)
            html += f"""
                <tr>
                    <td><span class="badge {badge_cls}">{cls}</span></td>
                    <td>{metrics[f'{key}_n']}</td>
                    <td>
                        <div class="progress" style="width:100px;display:inline-block;vertical-align:middle;">
                            <div class="progress-bar" style="width:{pass3_val}%"></div>
                        </div>
                        {pass3_val:.1f}%
                    </td>
                    <td>{avg_acc:.1f}%</td>
                    <td>{metrics[f'{key}_pct_asked']:.0f}%</td>
                    <td>{metrics[f'{key}_avg_questions']:.2f}</td>
                    <td>{metrics.get(f'{key}_total_tokens', 0):,}</td>
                </tr>
"""

    html += """
            </tbody>
        </table>
    </div>
"""

    # Metric formulas section
    html += """
    <div class="section">
        <h2>📐 Metric Formulas</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div style="background: rgba(255,255,255,0.03); padding: 16px; border-radius: 8px;">
                <h3 style="color: #4299e1; margin: 0 0 12px 0; font-size: 14px;">Cost Metrics (User Effort)</h3>
                <div style="font-family: monospace; font-size: 12px; line-height: 1.8;">
                    <div><strong># Questions</strong> = count of ask_user calls</div>
                    <div style="color: #718096; font-size: 11px; margin-left: 20px;">↳ "How many times was the user interrupted?"</div>
                    <div style="margin-top: 8px;"><strong>Tokens</strong> ≈ len(text) // 4</div>
                    <div style="color: #718096; font-size: 11px; margin-left: 20px;">↳ τ-bench heuristic (~4 chars per token)</div>
                    <div style="margin-top: 8px;"><strong>Tokens Read</strong> = sum of question token counts</div>
                    <div style="color: #718096; font-size: 11px; margin-left: 20px;">↳ "How much did user have to read?"</div>
                    <div style="margin-top: 8px;"><strong>Tokens Wrote</strong> = sum of response token counts</div>
                    <div style="color: #718096; font-size: 11px; margin-left: 20px;">↳ "How much did user have to write?"</div>
                </div>
                <div style="margin-top: 12px; padding: 8px; background: rgba(66, 153, 225, 0.1); border-radius: 4px; font-size: 11px;">
                    <strong>Goal:</strong> Minimize user effort while maximizing task success.
                </div>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 16px; border-radius: 8px;">
                <h3 style="color: #48bb78; margin: 0 0 12px 0; font-size: 14px;">Strategic Score & Efficiency</h3>
                <div style="font-family: monospace; font-size: 12px; line-height: 1.8;">
                    <div><strong>Strategic Score</strong> = (critical_asked_trials + benign_abstained_trials) / (critical_trials + benign_trials)</div>
                    <div style="color: #718096; font-size: 11px; margin-left: 20px;">↳ "% of correct ask/abstain decisions"</div>
                    <div style="margin-top: 8px;"><strong>Clarification ROI</strong> = weighted pass@3(asked trials) − weighted pass@3(non-asked trials)</div>
                    <div style="color: #718096; font-size: 11px; margin-left: 20px;">↳ "Does asking actually help pass rate?"</div>
                    <div style="margin-top: 8px;"><strong>Efficiency</strong> = 1 / (1 + N) for benign tasks</div>
                    <div style="color: #718096; font-size: 11px; margin-left: 20px;">↳ Parallels Reciprocal Rank; penalizes wasted questions</div>
                </div>
                <div style="margin-top: 12px; padding: 8px; background: rgba(72, 187, 120, 0.1); border-radius: 4px; font-size: 11px;">
                    <strong>Ideal:</strong> Ask on critical (100%), abstain on benign (efficiency = 1.0).
                </div>
            </div>
        </div>
    </div>
"""

    # Task breakdown and insights side by side
    html += """
    <div class="two-col">
"""

    # Task breakdown
    task_breakdown = metrics.get("task_breakdown", {})
    if task_breakdown:
        html += """
        <div class="section">
            <h2>📋 By Task</h2>
            <table>
                <thead><tr><th>Task</th><th>Variants</th><th>pass@3</th><th>Avg Acc</th><th>Asked%</th></tr></thead>
                <tbody>
"""
        for task, stats in sorted(task_breakdown.items(), key=lambda x: -x[1].get("pass_at_3", 0)):
            pass3_pct = stats.get("pass_at_3", 0) * 100
            avg_acc = stats.get("avg_pct", 0)
            asked_pct = stats["asked_any"] * 100
            html += f"""
                    <tr>
                        <td style="font-size:11px;">{task[:35]}</td>
                        <td>{int(stats['n_variants'])}</td>
                        <td>{pass3_pct:.0f}%</td>
                        <td>{avg_acc:.1f}%</td>
                        <td>{asked_pct:.0f}%</td>
                    </tr>
"""
        html += """
                </tbody>
            </table>
        </div>
"""

    # Key insights
    benign_asked = metrics.get("benign_pct_asked", 0)
    strategic = metrics.get("strategic_score", 0)
    roi = metrics.get("clarification_roi", 0)

    html += """
        <div class="section">
            <h2>💡 Key Insights</h2>
"""

    # Strategic score insight
    if strategic >= 70:
        html += f"""
            <div class="insight success">
                <strong>✅ Good strategic behavior ({strategic:.0f}%)</strong><br>
                Agent makes correct ask/abstain decisions most of the time.
            </div>
"""
    elif strategic >= 50:
        html += f"""
            <div class="insight warning">
                <strong>⚠️ Moderate strategic behavior ({strategic:.0f}%)</strong><br>
                Agent could better distinguish critical vs benign ambiguity.
            </div>
"""
    else:
        html += f"""
            <div class="insight warning">
                <strong>❌ Poor strategic behavior ({strategic:.0f}%)</strong><br>
                Agent fails to correctly identify when to ask for clarification.
            </div>
"""

    # Clarification ROI insight
    if roi > 5:
        html += f"""
            <div class="insight success">
                <strong>✅ Clarification helps! (+{roi:.1f}%)</strong><br>
                Pass rate is higher when agent asks questions.
            </div>
"""
    elif roi < -5:
        html += f"""
            <div class="insight warning">
                <strong>⚠️ Clarification not helping ({roi:+.1f}%)</strong><br>
                Questions may not be targeted at useful information.
            </div>
"""

    # Specific class insights
    if benign_asked > 50:
        html += f"""
            <div class="insight warning">
                <strong>⚠️ Over-asking on benign ({benign_asked:.0f}%)</strong><br>
                Agent wastes user time on inferable tasks.
            </div>
"""

    html += """
        </div>
    </div>
"""

    # Variant table
    evaluated = df[df["has_run"]].sort_values(
        ["ambiguity_class", "pass_at_3"], ascending=[True, False]
    )

    html += f"""
    <div class="section">
        <h2>📋 Variant Details ({len(evaluated)} variants)</h2>
        <table>
            <thead>
                <tr>
                    <th>Variant</th>
                    <th>Class</th>
                    <th>pass@3</th>
                    <th>Avg Acc</th>
                    <th>Asked%</th>
                    <th>#Q</th>
                    <th>Tokens</th>
                </tr>
            </thead>
            <tbody>
"""

    for _, row in evaluated.head(50).iterrows():
        badge_cls = (
            row["ambiguity_class"].split("-")[0]
            if "-" in row["ambiguity_class"]
            else row["ambiguity_class"]
        )
        asked_pct = row["asked_any"] * 100
        avg_acc = row.get("avg_pct", 0)
        html += f"""
                <tr>
                    <td style="font-family:monospace;font-size:11px;">{row['variant_id'][:50]}</td>
                    <td><span class="badge {badge_cls}">{row['ambiguity_class']}</span></td>
                    <td>{row['pass_at_3']*100:.0f}%</td>
                    <td>{avg_acc:.1f}%</td>
                    <td>{asked_pct:.0f}%</td>
                    <td>{row['n_questions']:.2f}</td>
                    <td>{int(row['total_tokens'])}</td>
                </tr>
"""

    html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"✅ Generated: {output_path}")
    return str(output_path)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Process user simulator benchmark results")
    parser.add_argument(
        "--experiment",
        "-e",
        required=True,
        help="Experiment name (e.g., user_sim_v1_sonnet_default)",
    )
    parser.add_argument("--export", "-x", action="store_true", help="Export CSV/JSON")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--details", "-d", action="store_true", help="Show variant details")
    parser.add_argument("--limit", "-n", type=int, default=20, help="Limit variant details")
    parser.add_argument("--output", "-o", help="Output path")
    args = parser.parse_args()

    experiment = args.experiment

    print(f"📦 Experiment: {experiment}")

    if args.export:
        export_results(experiment, Path(args.output) if args.output else None)
    elif args.html:
        generate_html_report(experiment, Path(args.output) if args.output else None)
    else:
        print_summary(experiment)
        if args.details:
            print_variant_details(experiment, args.limit)


if __name__ == "__main__":
    main()
