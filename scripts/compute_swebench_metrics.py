#!/usr/bin/env python3
"""
Compute SWE-Bench Pro metrics for ICML tables.

Uses Phase A classifications (from GPT 5.2) as ground truth labels,
then computes per-model metrics from eval results and ask-user trajectories.

Produces:
  Table 3: Main results (Baseline, User Behavior, Pass@3, Ckpt%)
  Table: Pass@3 by Information Dimension
  Table: Ckpt% by Information Dimension
  Table: Pass@3 by Ambiguity Class
"""
import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import DEFAULTS, parse_variant_id  # noqa: E402
from evaluation.pass_k_metrics import pass_at_k  # noqa: E402
from scripts.process_swebench_underspec import ensure_swebench_csv  # noqa: E402

# =============================================================================
# Configuration
# =============================================================================

SWEBENCH_CSV = REPO_ROOT / "swebenchpro" / "SWE-bench_Pro-os" / "swe_bench_pro_full.csv"

NUM_TRIALS = DEFAULTS["num_trials"]


def auto_detect_models(base_dir: Path) -> list:
    """Auto-detect model configs from sibling directories.

    Returns list of (display_name, regular_suffix, ask_suffix) tuples.
    The base dir (first model) always has suffix "".
    """
    base_name = base_dir.name
    parent = base_dir.parent

    suffixes = set()
    for d in sorted(parent.iterdir()):
        if d.is_dir() and d.name.startswith(base_name + "_"):
            suffixes.add(d.name[len(base_name) :])

    ask_suffixes = {s for s in suffixes if s.endswith("_ask")}
    regular_suffixes = sorted(s for s in suffixes if not s.endswith("_ask"))

    # Base model: infer name from config.json
    base_model_name = "Model-0"
    config_path = base_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            bm = cfg.get("backend_model", "")
            base_model_name = bm.split("/")[-1].replace("-", "_")[:30]
        except (json.JSONDecodeError, IOError):
            pass

    # Find ask suffix for base model: any _*_ask suffix whose model part
    # is not a regular (non-ask) suffix
    base_ask = ""
    for a in sorted(ask_suffixes):
        model_part = a[1:-4]  # strip leading "_" and trailing "_ask"
        if f"_{model_part}" not in regular_suffixes:
            base_ask = a
            ask_suffixes.discard(a)
            break

    models = [(base_model_name, "", base_ask)]

    for reg in regular_suffixes:
        name = reg.lstrip("_")
        expected_ask = f"{reg}_ask"
        ask = expected_ask if expected_ask in ask_suffixes else ""
        ask_suffixes.discard(expected_ask)
        models.append((name, reg, ask))

    return models


# =============================================================================
# Data Loading
# =============================================================================


def load_phase_a_labels(filtered_csv: Path) -> pd.DataFrame:
    """Load Phase A classifications (frozen 100 variants)."""
    df = pd.read_csv(filtered_csv)
    print(f"Loaded {len(df)} frozen variants from Phase A")
    print(f"  Classes: {dict(df['ambiguity_class'].value_counts())}")
    print(f"  Unique originals: {df['original_instance_id'].nunique()}")
    return df


def _normalize_id(iid: str) -> str:
    """Strip 'instance_' prefix for consistent key matching across data sources."""
    return iid.removeprefix("instance_")


def load_f2p_tests() -> Dict[str, Set[str]]:
    """Load F2P test lists per original instance from swe_bench_pro_full.csv.

    Keys are normalized (instance_ prefix stripped) so they match
    normalized original_instance_id values regardless of source format.
    """
    ensure_swebench_csv(SWEBENCH_CSV)
    swe_df = pd.read_csv(SWEBENCH_CSV)
    f2p_map = {}
    for _, row in swe_df.iterrows():
        iid = _normalize_id(str(row["instance_id"]))
        try:
            f2p = set(ast.literal_eval(row["fail_to_pass"]))
        except (ValueError, SyntaxError, TypeError):
            f2p = set()
        f2p_map[iid] = f2p
    return f2p_map


def load_eval_output(eval_dir: Path, original_id: str, prefix: str) -> Optional[dict]:
    """Load a single eval output file."""
    output_file = eval_dir / original_id / f"{prefix}_output.json"
    if output_file.exists():
        try:
            with open(output_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def get_variant_eval_results(
    model_dir: Path,
    trial_num: int,
    variant_id: str,
    original_id: str,
    f2p_tests: Set[str],
) -> Dict:
    """Get eval results for one variant in one trial.

    Returns dict with f2p_passed count, f2p_total, overall_pass.
    """
    eval_dir = model_dir / f"exp_{trial_num}" / "eval_results"
    # The variant suffix is encoded in the prefix: trialN__V_<suffix>
    _, variant_suffix = parse_variant_id(variant_id)
    prefix = f"trial{trial_num}__V_{variant_suffix}" if variant_suffix else f"trial{trial_num}"

    output = load_eval_output(eval_dir, original_id, prefix)
    if not output or "tests" not in output:
        return {"f2p_passed": 0, "f2p_total": len(f2p_tests), "overall_pass": False}

    passed = {t.get("name", "") for t in output["tests"] if t.get("status") == "PASSED"}
    f2p_passed = f2p_tests & passed
    overall_pass = f2p_tests <= passed  # All F2P tests pass

    return {
        "f2p_passed": len(f2p_passed),
        "f2p_total": len(f2p_tests),
        "overall_pass": overall_pass,
    }


def get_baseline_eval_results(
    model_dir: Path,
    trial_num: int,
    original_id: str,
    f2p_tests: Set[str],
) -> Dict:
    """Get eval results for one baseline trial."""
    eval_dir = model_dir / f"baseline_{trial_num}" / "eval_results"
    prefix = f"baseline{trial_num}"

    output = load_eval_output(eval_dir, original_id, prefix)
    if not output or "tests" not in output:
        return {"f2p_passed": 0, "f2p_total": len(f2p_tests), "overall_pass": False}

    passed = {t.get("name", "") for t in output["tests"] if t.get("status") == "PASSED"}
    f2p_passed = f2p_tests & passed
    overall_pass = f2p_tests <= passed

    return {
        "f2p_passed": len(f2p_passed),
        "f2p_total": len(f2p_tests),
        "overall_pass": overall_pass,
    }


def count_ask_user_calls(traj_path: Path) -> int:
    """Count ask_user tool invocations in a trajectory."""
    try:
        with open(traj_path) as f:
            traj = json.load(f)
        count = 0
        for msg in traj.get("trajectory", []):
            action = msg.get("action", "")
            if isinstance(action, str) and action.startswith("ask_user"):
                count += 1
        return count
    except (json.JSONDecodeError, IOError):
        return 0


def find_trajectory(model_dir: Path, trial_num: int, variant_id: str) -> Optional[Path]:
    """Find the trajectory file for a variant in a trial."""
    exp_dir = model_dir / f"exp_{trial_num}"
    for candidate in [exp_dir / variant_id, exp_dir / f"instance_{variant_id}"]:
        if candidate.exists():
            traj_files = sorted(candidate.glob("*.traj"))
            if traj_files:
                return traj_files[0]
    return None


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_model_metrics(
    model_name: str,
    regular_dir: Path,
    ask_dir: Path,
    labels_df: pd.DataFrame,
    f2p_map: Dict[str, Set[str]],
    relevant_originals: Set[str],
) -> Dict:
    """Compute all metrics for one model."""
    print(f"\n{'='*60}")
    print(f"Computing metrics for: {model_name}")
    print(f"  Regular dir: {regular_dir.name}")
    print(f"  Ask dir: {ask_dir.name}")

    # ----- BASELINE METRICS -----
    baseline_pass_results = {}  # original_id -> [pass_trial1, pass_trial2, pass_trial3]
    baseline_ckpt_results = {}  # original_id -> [f2p_rate_t1, f2p_rate_t2, f2p_rate_t3]

    for orig_id in relevant_originals:
        f2p = f2p_map.get(_normalize_id(orig_id), set())
        if not f2p:
            continue
        passes = []
        ckpts = []
        for t in range(1, NUM_TRIALS + 1):
            res = get_baseline_eval_results(regular_dir, t, orig_id, f2p)
            passes.append(1 if res["overall_pass"] else 0)
            ckpts.append(res["f2p_passed"] / res["f2p_total"] if res["f2p_total"] > 0 else 0)
        baseline_pass_results[orig_id] = passes
        baseline_ckpt_results[orig_id] = ckpts

    # Baseline pass@3 per original task, then average
    baseline_pass3_values = []
    for orig_id, passes in baseline_pass_results.items():
        n = len(passes)
        c = sum(passes)
        baseline_pass3_values.append(pass_at_k(n, c, 3))

    baseline_pass3 = (
        sum(baseline_pass3_values) / len(baseline_pass3_values) * 100
        if baseline_pass3_values
        else 0
    )
    # Baseline ckpt: per-task average, then average across tasks (macro-average)
    baseline_ckpt_values = []
    for orig_id, ckpts in baseline_ckpt_results.items():
        if ckpts:
            baseline_ckpt_values.append(sum(ckpts) / len(ckpts))
    baseline_ckpt = (
        sum(baseline_ckpt_values) / len(baseline_ckpt_values) * 100 if baseline_ckpt_values else 0
    )

    print(f"  Baseline: pass@3={baseline_pass3:.1f}%, ckpt={baseline_ckpt:.1f}%")
    print(f"    ({len(baseline_pass_results)} originals with eval results)")

    # ----- VARIANT METRICS (without user) -----
    variant_results = []  # List of per-variant metrics
    for _, row in labels_df.iterrows():
        variant_id = row["instance_id"]
        orig_id = row["original_instance_id"]
        f2p = f2p_map.get(_normalize_id(orig_id), set())
        if not f2p:
            continue

        passes = []
        ckpts = []
        for t in range(1, NUM_TRIALS + 1):
            res = get_variant_eval_results(regular_dir, t, variant_id, orig_id, f2p)
            passes.append(1 if res["overall_pass"] else 0)
            ckpts.append(res["f2p_passed"] / res["f2p_total"] if res["f2p_total"] > 0 else 0)

        n = len(passes)
        c = sum(passes)
        variant_results.append(
            {
                "variant_id": variant_id,
                "original_id": orig_id,
                "ambiguity_class": row["ambiguity_class"],
                "dimension": row["dimension"],
                "pass_at_3": pass_at_k(n, c, 3),
                "avg_ckpt": sum(ckpts) / len(ckpts) if ckpts else 0,
                "n_trials": n,
                "n_pass": c,
            }
        )

    # ----- VARIANT METRICS (with user) -----
    ask_results = []
    trials_with_ask = 0
    total_trials = 0
    total_questions = 0

    for _, row in labels_df.iterrows():
        variant_id = row["instance_id"]
        orig_id = row["original_instance_id"]
        f2p = f2p_map.get(_normalize_id(orig_id), set())
        if not f2p:
            continue

        passes = []
        ckpts = []
        trial_asks = []
        for t in range(1, NUM_TRIALS + 1):
            res = get_variant_eval_results(ask_dir, t, variant_id, orig_id, f2p)
            passes.append(1 if res["overall_pass"] else 0)
            ckpts.append(res["f2p_passed"] / res["f2p_total"] if res["f2p_total"] > 0 else 0)

            # Count ask_user calls
            traj_path = find_trajectory(ask_dir, t, variant_id)
            n_asks = count_ask_user_calls(traj_path) if traj_path else 0
            trial_asks.append(n_asks)
            total_trials += 1
            if n_asks > 0:
                trials_with_ask += 1
                total_questions += n_asks

        n = len(passes)
        c = sum(passes)
        ask_results.append(
            {
                "variant_id": variant_id,
                "original_id": orig_id,
                "ambiguity_class": row["ambiguity_class"],
                "dimension": row["dimension"],
                "pass_at_3": pass_at_k(n, c, 3),
                "avg_ckpt": sum(ckpts) / len(ckpts) if ckpts else 0,
                "n_trials": n,
                "n_pass": c,
                "ask_calls": trial_asks,
            }
        )

    # ----- AGGREGATE METRICS -----
    # Pass@3 (no user): average across variants
    no_user_pass3 = (
        sum(v["pass_at_3"] for v in variant_results) / len(variant_results) * 100
        if variant_results
        else 0
    )
    no_user_ckpt = (
        sum(v["avg_ckpt"] for v in variant_results) / len(variant_results) * 100
        if variant_results
        else 0
    )

    # Pass@3 (+user): average across variants
    with_user_pass3 = (
        sum(v["pass_at_3"] for v in ask_results) / len(ask_results) * 100 if ask_results else 0
    )
    with_user_ckpt = (
        sum(v["avg_ckpt"] for v in ask_results) / len(ask_results) * 100 if ask_results else 0
    )

    # User behavior
    ask_pct = trials_with_ask / total_trials * 100 if total_trials > 0 else 0
    avg_per_traj = total_questions / trials_with_ask if trials_with_ask > 0 else 0

    # Gain/Q
    pass3_gain = with_user_pass3 - no_user_pass3
    ckpt_gain = with_user_ckpt - no_user_ckpt
    pass3_gain_per_q = pass3_gain / total_questions if total_questions > 0 else 0
    ckpt_gain_per_q = ckpt_gain / total_questions if total_questions > 0 else 0

    print(f"  Underspec (no user): pass@3={no_user_pass3:.1f}%, ckpt={no_user_ckpt:.1f}%")
    print(f"  Underspec (+user):   pass@3={with_user_pass3:.1f}%, ckpt={with_user_ckpt:.1f}%")
    print(f"  Ask%={ask_pct:.0f}%, Avg/Traj={avg_per_traj:.2f}, Total Qs={total_questions}")

    return {
        "model": model_name,
        "baseline_pass3": baseline_pass3,
        "baseline_ckpt": baseline_ckpt,
        "ask_pct": ask_pct,
        "avg_per_traj": avg_per_traj,
        "total_questions": total_questions,
        "no_user_pass3": no_user_pass3,
        "no_user_ckpt": no_user_ckpt,
        "with_user_pass3": with_user_pass3,
        "with_user_ckpt": with_user_ckpt,
        "pass3_gain": pass3_gain,
        "ckpt_gain": ckpt_gain,
        "pass3_gain_per_q": pass3_gain_per_q,
        "ckpt_gain_per_q": ckpt_gain_per_q,
        "variant_results": variant_results,
        "ask_results": ask_results,
        "baseline_pass_results": baseline_pass_results,
        "n_baseline_originals": len(baseline_pass_results),
    }


# =============================================================================
# Table Formatters
# =============================================================================


def get_primary_dims(dimension_str: str) -> List[str]:
    """Extract primary dimensions from compound dimension string (e.g. 'goal+constraint' -> ['goal', 'constraint'])."""
    if not dimension_str or pd.isna(dimension_str):
        return []
    return [d.strip() for d in dimension_str.split("+")]


def fmt_gain(val: float) -> str:
    """Format a gain value with sign."""
    if val >= 0:
        return f"(+{val:.1f})"
    return f"({val:.1f})"


def print_table3(all_metrics: List[Dict]):
    """Print Table 3: Main results."""
    n_variants = len(all_metrics[0].get("variant_results", [])) if all_metrics else 0
    n_originals = all_metrics[0]["n_baseline_originals"] if all_metrics else 0

    print(f"\n{'='*110}")
    print(f"TABLE 3: SWE-Bench Pro (n_tasks={n_originals}, n_variants={n_variants})")
    print(f"{'='*110}")
    header = f"{'Model':<18} {'Base P@3':>8} {'Base Ckpt':>9} {'Ask%':>5} {'Avg/T':>6} {'P@3 Score':>14} {'G/Q':>5} {'Ckpt Score':>14} {'G/Q':>5}"
    print(header)
    print("-" * len(header))

    for m in all_metrics:
        has_ask = m.get("has_ask_data", False)
        if has_ask:
            p3_score = f"{m['no_user_pass3']:.1f} {fmt_gain(m['pass3_gain'])}"
            ck_score = f"{m['no_user_ckpt']:.1f} {fmt_gain(m['ckpt_gain'])}"
            ask_pct = f"{m['ask_pct']:>4.0f}%"
            avg_t = f"{m['avg_per_traj']:>5.2f}"
            if m["total_questions"] > 0:
                gq_p3 = f"{m['pass3_gain_per_q']:>5.2f}"
                gq_ck = f"{m['ckpt_gain_per_q']:>5.2f}"
            else:
                gq_p3 = f"{'--':>5}"
                gq_ck = f"{'--':>5}"
        else:
            p3_score = f"{m['no_user_pass3']:.1f}{'--':>7}"
            ck_score = f"{m['no_user_ckpt']:.1f}{'--':>7}"
            ask_pct = f"{'--':>5}"
            avg_t = f"{'--':>5}"
            gq_p3 = f"{'--':>5}"
            gq_ck = f"{'--':>5}"
        print(
            f"{m['model']:<18} "
            f"{m['baseline_pass3']:>7.1f}% "
            f"{m['baseline_ckpt']:>8.1f}% "
            f"{ask_pct} "
            f"{avg_t} "
            f"{p3_score:>14} "
            f"{gq_p3} "
            f"{ck_score:>14} "
            f"{gq_ck}"
        )


def print_dimension_table(all_metrics: List[Dict], metric_key: str, title: str):
    """Print per-dimension table (pass@3 or ckpt)."""
    dims = ["goal", "constraint", "input", "context"]

    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")
    header = f"{'Model':<18} {'Base':>6} {'All':>12} {'Goal':>12} {'Constr.':>12} {'Input':>12} {'Context':>12}"
    print(header)
    print("-" * len(header))

    for m in all_metrics:
        # Group variants by dimension (double-counting for multi-dim)
        no_user_by_dim = defaultdict(list)
        ask_by_dim = defaultdict(list)

        for v in m["variant_results"]:
            primary_dims = get_primary_dims(v["dimension"])
            val = v["pass_at_3"] if metric_key == "pass3" else v["avg_ckpt"]
            for d in primary_dims:
                no_user_by_dim[d].append(val)

        for v in m["ask_results"]:
            primary_dims = get_primary_dims(v["dimension"])
            val = v["pass_at_3"] if metric_key == "pass3" else v["avg_ckpt"]
            for d in primary_dims:
                ask_by_dim[d].append(val)

        # Overall
        all_no_user = [
            v["pass_at_3"] if metric_key == "pass3" else v["avg_ckpt"] for v in m["variant_results"]
        ]
        all_ask = [
            v["pass_at_3"] if metric_key == "pass3" else v["avg_ckpt"] for v in m["ask_results"]
        ]
        overall_no = sum(all_no_user) / len(all_no_user) * 100 if all_no_user else 0
        overall_ask = sum(all_ask) / len(all_ask) * 100 if all_ask else 0
        gain_all = overall_ask - overall_no

        base = m["baseline_pass3"] if metric_key == "pass3" else m["baseline_ckpt"]

        has_ask = m.get("has_ask_data", False)
        if has_ask:
            gain_str = fmt_gain(gain_all)
        else:
            gain_str = f"{'--':>8}"
        parts = [f"{m['model']:<18} {base:>5.1f}% {overall_no:>5.1f} {gain_str:>8}"]

        for d in dims:
            no_vals = no_user_by_dim.get(d, [])
            if no_vals:
                no_avg = sum(no_vals) / len(no_vals) * 100
                if has_ask:
                    ask_vals = ask_by_dim.get(d, [])
                    ask_avg = sum(ask_vals) / len(ask_vals) * 100 if ask_vals else no_avg
                    gain = ask_avg - no_avg
                    parts.append(f"{no_avg:>5.1f} {fmt_gain(gain):>8}")
                else:
                    parts.append(f"{no_avg:>5.1f} {'--':>8}")
            else:
                parts.append(f"{'--':>14}")

        print(" ".join(parts))


def print_ambiguity_class_table(all_metrics: List[Dict]):
    """Print Pass@3 by ambiguity class table."""
    classes = ["outcome-critical", "divergent", "benign"]

    print(f"\n{'='*80}")
    n_variants = len(all_metrics[0].get("variant_results", [])) if all_metrics else 0
    print(f"Pass@3 by Ambiguity Class on SWE-Bench Pro (n={n_variants})")
    print(f"{'='*80}")

    for cls in classes:
        # Count variants in this class
        n_cls = sum(1 for v in all_metrics[0]["variant_results"] if v["ambiguity_class"] == cls)
        print(f"\n--- {cls} (n={n_cls}) ---")
        header = f"  {'Model':<18} {'Base':>6} {'Under.':>7} {'+User':>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for m in all_metrics:
            # Filter to this class
            no_user_cls = [v for v in m["variant_results"] if v["ambiguity_class"] == cls]
            ask_cls = [v for v in m["ask_results"] if v["ambiguity_class"] == cls]

            if no_user_cls:
                under = sum(v["pass_at_3"] for v in no_user_cls) / len(no_user_cls)
            else:
                under = 0
            if ask_cls:
                with_user = sum(v["pass_at_3"] for v in ask_cls) / len(ask_cls)
            else:
                with_user = 0

            # Per-class baseline: only originals that generated variants in this class
            class_originals = {v["original_id"] for v in no_user_cls}
            baseline_data = m.get("baseline_pass_results", {})
            class_baseline_vals = []
            for orig_id in class_originals:
                if orig_id in baseline_data:
                    passes = baseline_data[orig_id]
                    n = len(passes)
                    c = sum(passes)
                    class_baseline_vals.append(pass_at_k(n, c, 3))
            base = sum(class_baseline_vals) / len(class_baseline_vals) if class_baseline_vals else 0

            has_ask = m.get("has_ask_data", False)
            user_str = f"{with_user:>7.2f}" if has_ask else f"{'--':>7}"
            print(f"  {m['model']:<18} {base:>5.2f} {under:>7.2f} {user_str}")


def print_table3_latex(all_metrics: List[Dict]):
    """Print LaTeX for Table 3 SWE-Bench section."""
    n_originals = all_metrics[0]["n_baseline_originals"] if all_metrics else 0
    n_variants = len(all_metrics[0].get("variant_results", [])) if all_metrics else 0

    print(f"\n{'='*80}")
    print("LaTeX: Table 3 SWE-Bench Pro section")
    print(f"{'='*80}")
    print(
        f"\\multirow{{{len(all_metrics)}}}{{*}}{{\\shortstack[]{{SWE-Bench\\\\Pro\\\\\\\\n\\_tasks={n_originals}\\\\n\\_variants={n_variants}}}}}"
    )

    for m in all_metrics:
        has_ask = m.get("has_ask_data", False)
        if has_ask:
            gain_p3 = m["pass3_gain"]
            gain_ck = m["ckpt_gain"]
            g_p3 = f"+{gain_p3:.1f}" if gain_p3 >= 0 else f"$-${abs(gain_p3):.1f}"
            g_ck = f"+{gain_ck:.1f}" if gain_ck >= 0 else f"$-${abs(gain_ck):.1f}"
            p3_str = f"{m['no_user_pass3']:.1f} {{\\scriptsize({g_p3})}}"
            ck_str = f"{m['no_user_ckpt']:.1f} {{\\scriptsize({g_ck})}}"
            gq_p3 = f"{m['pass3_gain_per_q']:.2f}" if m["total_questions"] > 0 else "--"
            gq_ck = f"{m['ckpt_gain_per_q']:.2f}" if m["total_questions"] > 0 else "--"
            print(
                f"& {m['model']:<18} & {m['baseline_pass3']:>5.1f} & {m['baseline_ckpt']:>5.1f} "
                f"& {m['ask_pct']:.0f} & {m['avg_per_traj']:.2f} "
                f"& {p3_str} & {gq_p3} "
                f"& {ck_str} & {gq_ck} \\\\"
            )
        else:
            p3_str = f"{m['no_user_pass3']:.1f} {{\\scriptsize(--)}}"
            ck_str = f"{m['no_user_ckpt']:.1f} {{\\scriptsize(--)}}"
            print(
                f"& {m['model']:<18} & {m['baseline_pass3']:>5.1f} & {m['baseline_ckpt']:>5.1f} "
                f"& -- & -- "
                f"& {p3_str} & -- "
                f"& {ck_str} & -- \\\\"
            )

    # Also print dimension table LaTeX
    print(f"\n{'='*80}")
    print("LaTeX: Pass@3 by Dimension (SWE-Bench Pro)")
    print(f"{'='*80}")
    dims = ["goal", "constraint", "input", "context"]
    print(
        f"\\multirow{{{len(all_metrics)}}}{{*}}{{\\shortstack[]{{SWE-Bench\\\\Pro\\\\\\\\n\\_tasks={n_originals}\\\\n\\_variants={n_variants}}}}}"
    )

    for m in all_metrics:
        has_ask = m.get("has_ask_data", False)
        no_user_by_dim = defaultdict(list)
        ask_by_dim = defaultdict(list)
        for v in m["variant_results"]:
            for d in get_primary_dims(v["dimension"]):
                no_user_by_dim[d].append(v["pass_at_3"])
        if has_ask:
            for v in m["ask_results"]:
                for d in get_primary_dims(v["dimension"]):
                    ask_by_dim[d].append(v["pass_at_3"])

        all_no = [v["pass_at_3"] for v in m["variant_results"]]
        overall_no = sum(all_no) / len(all_no) * 100 if all_no else 0
        base = m["baseline_pass3"]

        if has_ask:
            all_ask = [v["pass_at_3"] for v in m["ask_results"]]
            overall_ask = sum(all_ask) / len(all_ask) * 100 if all_ask else 0
            gain_all = overall_ask - overall_no
            g = f"+{gain_all:.1f}" if gain_all >= 0 else f"$-${abs(gain_all):.1f}"
        else:
            g = "--"
        parts = [f"& {m['model']:<18} & {base:.1f} & {overall_no:.1f} {{\\scriptsize({g})}}"]

        for d in dims:
            no_vals = no_user_by_dim.get(d, [])
            if no_vals:
                no_avg = sum(no_vals) / len(no_vals) * 100
                if has_ask:
                    ask_vals = ask_by_dim.get(d, [])
                    ask_avg = sum(ask_vals) / len(ask_vals) * 100 if ask_vals else no_avg
                    gain = ask_avg - no_avg
                    g = f"+{gain:.1f}" if gain >= 0 else f"$-${abs(gain):.1f}"
                else:
                    g = "--"
                parts.append(f"& {no_avg:.1f} {{\\scriptsize({g})}}")
            else:
                parts.append("& --")
        print(" ".join(parts) + " \\\\")

    # Ckpt% by dimension LaTeX
    print(f"\n{'='*80}")
    print("LaTeX: Ckpt% by Dimension (SWE-Bench Pro)")
    print(f"{'='*80}")
    print(
        f"\\multirow{{{len(all_metrics)}}}{{*}}{{\\shortstack[]{{SWE-Bench\\\\Pro\\\\\\\\n\\_tasks={n_originals}\\\\n\\_variants={n_variants}}}}}"
    )

    for m in all_metrics:
        has_ask = m.get("has_ask_data", False)
        no_user_by_dim = defaultdict(list)
        ask_by_dim = defaultdict(list)
        for v in m["variant_results"]:
            for d in get_primary_dims(v["dimension"]):
                no_user_by_dim[d].append(v["avg_ckpt"])
        if has_ask:
            for v in m["ask_results"]:
                for d in get_primary_dims(v["dimension"]):
                    ask_by_dim[d].append(v["avg_ckpt"])

        all_no = [v["avg_ckpt"] for v in m["variant_results"]]
        overall_no = sum(all_no) / len(all_no) * 100 if all_no else 0
        base = m["baseline_ckpt"]

        if has_ask:
            all_ask = [v["avg_ckpt"] for v in m["ask_results"]]
            overall_ask = sum(all_ask) / len(all_ask) * 100 if all_ask else 0
            gain_all = overall_ask - overall_no
            g = f"+{gain_all:.1f}" if gain_all >= 0 else f"$-${abs(gain_all):.1f}"
        else:
            g = "--"
        parts = [f"& {m['model']:<18} & {base:.1f} & {overall_no:.1f} {{\\scriptsize({g})}}"]

        for d in dims:
            no_vals = no_user_by_dim.get(d, [])
            if no_vals:
                no_avg = sum(no_vals) / len(no_vals) * 100
                if has_ask:
                    ask_vals = ask_by_dim.get(d, [])
                    ask_avg = sum(ask_vals) / len(ask_vals) * 100 if ask_vals else no_avg
                    gain = ask_avg - no_avg
                    g = f"+{gain:.1f}" if gain >= 0 else f"$-${abs(gain):.1f}"
                else:
                    g = "--"
                parts.append(f"& {no_avg:.1f} {{\\scriptsize({g})}}")
            else:
                parts.append("& --")
        print(" ".join(parts) + " \\\\")

    # Ambiguity class LaTeX
    print(f"\n{'='*80}")
    print("LaTeX: Pass@3 by Ambiguity Class (SWE-Bench Pro)")
    print(f"{'='*80}")
    classes = [
        ("outcome-critical", "Outcome-\\\\Critical"),
        ("divergent", "Divergent"),
        ("benign", "Benign"),
    ]
    for cls, cls_label in classes:
        n_cls = sum(1 for v in all_metrics[0]["variant_results"] if v["ambiguity_class"] == cls)
        print(
            f"\\multirow{{{len(all_metrics)}}}{{*}}{{\\shortstack[l]{{{cls_label}\\\\(n={n_cls})}}}}"
        )
        for m in all_metrics:
            no_user_cls = [v for v in m["variant_results"] if v["ambiguity_class"] == cls]
            ask_cls = [v for v in m["ask_results"] if v["ambiguity_class"] == cls]
            under = (
                sum(v["pass_at_3"] for v in no_user_cls) / len(no_user_cls) if no_user_cls else 0
            )
            has_ask = m.get("has_ask_data", False)
            if has_ask:
                with_user = sum(v["pass_at_3"] for v in ask_cls) / len(ask_cls) if ask_cls else 0
                user_str = f"{with_user:.2f}"
            else:
                user_str = "--"
            class_originals = {v["original_id"] for v in no_user_cls}
            baseline_data = m.get("baseline_pass_results", {})
            class_baseline_vals = []
            for orig_id in class_originals:
                if orig_id in baseline_data:
                    passes = baseline_data[orig_id]
                    class_baseline_vals.append(pass_at_k(len(passes), sum(passes), 3))
            base = sum(class_baseline_vals) / len(class_baseline_vals) if class_baseline_vals else 0
            print(f"& {m['model']:<18} & {base:.2f} & {under:.2f} & {user_str}\\\\")
        print("\\midrule")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Compute SWE-Bench Pro metrics for ICML tables",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Base experiment directory (e.g. experiments/swebench/runs/run_my_exp_20260101_120000)",
    )
    args = parser.parse_args()

    base_run_dir = Path(args.exp_dir)
    if not base_run_dir.exists():
        print(f"Error: Experiment directory not found: {base_run_dir}")
        raise SystemExit(1)

    filtered_csv = base_run_dir / "underspec_results_filtered.csv"
    if not filtered_csv.exists():
        print(f"Error: Filtered CSV not found: {filtered_csv}")
        raise SystemExit(1)

    # Read num_trials from experiment config
    global NUM_TRIALS
    config_path = base_run_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            NUM_TRIALS = cfg.get("num_trials", NUM_TRIALS)
        except (json.JSONDecodeError, IOError):
            pass

    print("Loading data...")
    labels_df = load_phase_a_labels(filtered_csv)
    f2p_map = load_f2p_tests()
    relevant_originals = set(labels_df["original_instance_id"].unique())

    print(f"\nRelevant originals for baseline: {len(relevant_originals)}")
    print(f"F2P test map loaded: {len(f2p_map)} instances")

    models = auto_detect_models(base_run_dir)
    print(f"\nAuto-detected {len(models)} model(s):")
    for name, reg, ask in models:
        reg_label = "(base dir)" if reg == "" else reg
        ask_label = ask if ask else "(none)"
        print(f"  {name}: regular={reg_label}, ask={ask_label}")

    all_metrics = []

    for model_name, regular_suffix, ask_suffix in models:
        regular_dir = base_run_dir.parent / f"{base_run_dir.name}{regular_suffix}"
        ask_dir = base_run_dir.parent / f"{base_run_dir.name}{ask_suffix}" if ask_suffix else None

        if regular_suffix == "":
            regular_dir = base_run_dir

        has_ask_data = ask_dir is not None and ask_dir.exists()
        if not has_ask_data:
            if ask_suffix:
                print(
                    f"  WARNING: ask-user dir not found for {model_name} "
                    f"(expected {ask_suffix}). +user columns will show '--'."
                )
            ask_dir = regular_dir

        metrics = compute_model_metrics(
            model_name, regular_dir, ask_dir, labels_df, f2p_map, relevant_originals
        )
        metrics["has_ask_data"] = has_ask_data
        all_metrics.append(metrics)

    # Print all tables
    print_table3(all_metrics)
    print_dimension_table(all_metrics, "pass3", "Pass@3 (%) by Information Dimension")
    print_dimension_table(
        all_metrics, "ckpt", "Avg. Checkpoint Progress (%) by Information Dimension"
    )
    print_ambiguity_class_table(all_metrics)

    # Print LaTeX for Table 3 (SWE-Bench section only)
    print_table3_latex(all_metrics)

    # Save raw data
    output_path = base_run_dir / "icml_metrics.json"
    save_data = []
    for m in all_metrics:
        save_m = {
            k: v
            for k, v in m.items()
            if k not in ("variant_results", "ask_results", "baseline_pass_results")
        }
        save_data.append(save_m)
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nRaw metrics saved to: {output_path}")


if __name__ == "__main__":
    main()
