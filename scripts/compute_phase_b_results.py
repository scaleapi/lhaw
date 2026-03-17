#!/usr/bin/env python3
"""
Compute Phase B cross-model comparison results.

Reads underspec_results.csv from each model dir + baseline eval results.
Produces a comprehensive comparison table.
"""
import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from constants import DEFAULTS, parse_variant_id  # noqa: E402
from evaluation.pass_k_metrics import pass_at_k  # noqa: E402
from scripts.compute_swebench_metrics import auto_detect_models  # noqa: E402
from scripts.process_swebench_underspec import ensure_swebench_csv  # noqa: E402

SWEBENCH_CSV = REPO_ROOT / "swebenchpro" / "SWE-bench_Pro-os" / "swe_bench_pro_full.csv"


def _resolve_dir(base_dir: Path, suffix: str) -> Path:
    """Resolve a model suffix to its experiment directory."""
    if suffix == "":
        return base_dir
    return base_dir.parent / f"{base_dir.name}{suffix}"


def load_frozen_ids(base_dir: Path):
    """Load frozen instance IDs."""
    frozen_path = base_dir / "frozen_instances.yaml"
    if frozen_path.exists():
        import yaml

        with open(frozen_path) as f:
            instances = yaml.safe_load(f) or []
        return {inst["instance_id"] for inst in instances}
    return set()


def load_variant_results(model_name, model_dir, is_main=False, frozen_ids=None):
    """Load underspec_results.csv for a model."""
    csv_path = model_dir / "underspec_results.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)

    if is_main and frozen_ids:
        df = df[df["instance_id"].isin(frozen_ids)]
        print(f"  {model_name}: Filtered to {len(df)} frozen variants (from 200)")
    else:
        print(f"  {model_name}: {len(df)} variants")

    return df


def compute_baseline_pass_rate(model_dir, swebench_df, num_trials=None):
    """Compute baseline pass rate from eval results.

    Returns list of dicts with n_resolved, n_total, and per_instance
    (mapping instance_id -> 1/0 for per-task pass@k).
    """
    results = []
    _num_trials = num_trials or DEFAULTS["num_trials"]

    for b in range(1, _num_trials + 1):
        baseline_dir = model_dir / f"baseline_{b}"
        eval_dir = baseline_dir / "eval_results"
        preds_path = baseline_dir / "preds.json"

        if not preds_path.exists():
            continue

        with open(preds_path) as f:
            preds = json.load(f)

        n_total = len(preds)
        n_resolved = 0
        per_instance = {}

        if eval_dir.exists():
            for key in preds:
                inst_id = key if key.startswith("instance_") else f"instance_{key}"
                original_id, _ = parse_variant_id(inst_id)
                resolved = 0

                for pattern in [
                    eval_dir / original_id / f"baseline{b}_output.json",
                    eval_dir / inst_id / f"baseline{b}_output.json",
                    eval_dir / original_id / "output.json",
                ]:
                    if pattern.exists():
                        try:
                            with open(pattern) as f:
                                result = json.load(f)

                            # Get F2P tests for this instance
                            if original_id in swebench_df.index:
                                row = swebench_df.loc[original_id]
                                f2p_tests = set(
                                    ast.literal_eval(row.get("fail_to_pass", "[]") or "[]")
                                )

                                passed_tests = {
                                    t.get("name", "")
                                    for t in result.get("tests", [])
                                    if t.get("status") == "PASSED"
                                }
                                if f2p_tests <= passed_tests:
                                    n_resolved += 1
                                    resolved = 1
                        except (json.JSONDecodeError, KeyError, ValueError, SyntaxError, TypeError):
                            pass
                        break
                per_instance[original_id] = resolved

        results.append(
            {
                "trial": b,
                "n_total": n_total,
                "n_resolved": n_resolved,
                "per_instance": per_instance,
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute Phase B cross-model comparison results",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Base experiment directory",
    )
    args = parser.parse_args()

    base_dir = Path(args.exp_dir)
    if not base_dir.exists():
        print(f"Error: Experiment directory not found: {base_dir}")
        raise SystemExit(1)

    # Read num_trials from experiment config
    num_trials = DEFAULTS["num_trials"]
    config_path = base_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            num_trials = cfg.get("num_trials", num_trials)
        except (json.JSONDecodeError, IOError):
            pass

    models = auto_detect_models(base_dir)

    print("=" * 80)
    print("PHASE B CROSS-MODEL COMPARISON")
    print("=" * 80)

    # Load frozen IDs
    frozen_ids = load_frozen_ids(base_dir)
    print(f"\nFrozen variants: {len(frozen_ids)}")

    # Load SWE-bench data for baseline eval
    ensure_swebench_csv(SWEBENCH_CSV)
    swebench_df = pd.read_csv(SWEBENCH_CSV).fillna("").set_index("instance_id", drop=False)

    # ======================================================================
    # 1. VARIANT RESULTS (Regular runs - no ask_user)
    # ======================================================================
    print("\n" + "=" * 80)
    print("1. VARIANT RESULTS (Regular runs)")
    print("=" * 80)

    regular_results = {}
    for i, (name, reg_suffix, _ask_suffix) in enumerate(models):
        model_dir = _resolve_dir(base_dir, reg_suffix)
        print(f"\nLoading {name}...")
        df = load_variant_results(name, model_dir, is_main=(i == 0), frozen_ids=frozen_ids)
        if df is not None:
            regular_results[name] = df

    print("\n--- Pass Rate Summary ---")
    print(f"{'Model':<20} {'pass@1':>8} {'pass@3':>8} {'Success%':>10} {'Variants':>10}")
    print("-" * 60)
    for model_name, df in regular_results.items():
        if "pass_at_1" in df.columns and "pass_at_3" in df.columns:
            p1 = df["pass_at_1"].mean()
            p3 = df["pass_at_3"].mean()
            sr = df["success_rate"].mean() if "success_rate" in df.columns else 0
            print(f"{model_name:<20} {p1:>8.3f} {p3:>8.3f} {sr:>9.1f}% {len(df):>10}")

    print("\n--- Ambiguity Classification ---")
    classes = ["outcome-critical", "divergent", "benign", "new_task"]
    print(f"{'Model':<20}", end="")
    for cls in classes:
        print(f" {cls:>16}", end="")
    print()
    print("-" * 90)
    for model_name, df in regular_results.items():
        if "ambiguity_class" in df.columns:
            print(f"{model_name:<20}", end="")
            for cls in classes:
                count = len(df[df["ambiguity_class"] == cls])
                print(f" {count:>16}", end="")
            print()

    print("\n--- Per-Class Pass Rates ---")
    for cls in classes:
        print(f"\n  {cls}:")
        print(f"  {'Model':<20} {'pass@1':>8} {'pass@3':>8} {'N':>6}")
        print(f"  {'-'*44}")
        for model_name, df in regular_results.items():
            cls_df = df[df["ambiguity_class"] == cls]
            if not cls_df.empty and "pass_at_1" in cls_df.columns:
                p1 = cls_df["pass_at_1"].mean()
                p3 = cls_df["pass_at_3"].mean()
                print(f"  {model_name:<20} {p1:>8.3f} {p3:>8.3f} {len(cls_df):>6}")

    # ======================================================================
    # 2. ASK-USER RESULTS
    # ======================================================================
    print("\n" + "=" * 80)
    print("2. ASK-USER RESULTS")
    print("=" * 80)

    ask_results = {}
    for name, _reg_suffix, ask_suffix in models:
        if not ask_suffix:
            continue
        ask_dir = _resolve_dir(base_dir, ask_suffix)
        ask_name = f"{name}_ask"
        print(f"\nLoading {ask_name}...")
        df = load_variant_results(ask_name, ask_dir)
        if df is not None:
            ask_results[ask_name] = df

    print("\n--- Pass Rate Summary ---")
    print(f"{'Model':<25} {'pass@1':>8} {'pass@3':>8} {'Success%':>10} {'Variants':>10}")
    print("-" * 65)
    for model_name, df in ask_results.items():
        if "pass_at_1" in df.columns and "pass_at_3" in df.columns:
            p1 = df["pass_at_1"].mean()
            p3 = df["pass_at_3"].mean()
            sr = df["success_rate"].mean() if "success_rate" in df.columns else 0
            print(f"{model_name:<25} {p1:>8.3f} {p3:>8.3f} {sr:>9.1f}% {len(df):>10}")

    # ======================================================================
    # 3. REGULAR vs ASK-USER COMPARISON
    # ======================================================================
    print("\n" + "=" * 80)
    print("3. REGULAR vs ASK-USER COMPARISON (Delta)")
    print("=" * 80)

    print(
        f"\n{'Model':<20} {'Reg p@1':>8} {'Ask p@1':>8} {'Delta':>8} {'Reg p@3':>8} {'Ask p@3':>8} {'Delta':>8}"
    )
    print("-" * 75)
    for name, _reg_suffix, ask_suffix in models:
        if not ask_suffix:
            continue
        ask_name = f"{name}_ask"
        if name in regular_results and ask_name in ask_results:
            reg_df = regular_results[name]
            ask_df = ask_results[ask_name]
            rp1 = reg_df["pass_at_1"].mean()
            rp3 = reg_df["pass_at_3"].mean()
            ap1 = ask_df["pass_at_1"].mean()
            ap3 = ask_df["pass_at_3"].mean()
            base = name.replace("_", " ")
            print(
                f"{base:<20} {rp1:>8.3f} {ap1:>8.3f} {ap1-rp1:>+8.3f} {rp3:>8.3f} {ap3:>8.3f} {ap3-rp3:>+8.3f}"
            )

    # ======================================================================
    # 4. BASELINE PASS RATES
    # ======================================================================
    print("\n" + "=" * 80)
    print("4. BASELINE PASS RATES (Original tasks, no underspec)")
    print("=" * 80)

    print(f"\n{'Model':<20} {'b1':>10} {'b2':>10} {'b3':>10} {'Avg':>10} {'pass@3':>10}")
    print("-" * 72)
    for name, reg_suffix, _ask_suffix in models:
        model_dir = _resolve_dir(base_dir, reg_suffix)
        bl_results = compute_baseline_pass_rate(model_dir, swebench_df, num_trials)
        if bl_results:
            rates = []
            for r in bl_results:
                rate = r["n_resolved"] / r["n_total"] if r["n_total"] else 0
                rates.append(rate)
            avg_rate = sum(rates) / len(rates) if rates else 0

            # Correct pass@3: compute per-task, then average
            per_task = {}
            for r in bl_results:
                for iid, resolved in r.get("per_instance", {}).items():
                    per_task.setdefault(iid, []).append(resolved)
            if per_task:
                task_pass3 = []
                for iid, trial_passes in per_task.items():
                    n_t = len(trial_passes)
                    c_t = sum(trial_passes)
                    task_pass3.append(pass_at_k(n_t, c_t, 3))
                p3 = sum(task_pass3) / len(task_pass3)
            else:
                c = sum(r["n_resolved"] for r in bl_results)
                n = sum(r["n_total"] for r in bl_results)
                p3 = pass_at_k(n, c, 3) if n > 0 else 0

            line = f"{name:<20}"
            for r in bl_results:
                rate = r["n_resolved"] / r["n_total"] if r["n_total"] else 0
                line += f" {r['n_resolved']}/{r['n_total']}={rate:.0%}".rjust(11)
            line += f" {avg_rate:>9.1%}"
            line += f" {p3:>9.3f}"
            print(line)

    # ======================================================================
    # 5. BASELINE vs UNDERSPEC DROP
    # ======================================================================
    print("\n" + "=" * 80)
    print("5. BASELINE vs UNDERSPEC DROP")
    print("=" * 80)

    print(
        f"\n{'Model':<20} {'Baseline':>10} {'Underspec':>10} {'Ask-User':>10} {'Drop(R)':>10} {'Drop(A)':>10}"
    )
    print("-" * 72)
    for name, reg_suffix, ask_suffix in models:
        model_dir = _resolve_dir(base_dir, reg_suffix)
        bl_results = compute_baseline_pass_rate(model_dir, swebench_df, num_trials)
        bl_avg = 0
        if bl_results:
            rates = [r["n_resolved"] / r["n_total"] for r in bl_results if r["n_total"]]
            bl_avg = sum(rates) / len(rates) if rates else 0

        reg_p1 = regular_results[name]["pass_at_1"].mean() if name in regular_results else 0
        ask_name = f"{name}_ask"
        ask_p1 = ask_results[ask_name]["pass_at_1"].mean() if ask_name in ask_results else 0

        drop_r = reg_p1 - bl_avg
        drop_a = ask_p1 - bl_avg

        print(
            f"{name:<20} {bl_avg:>9.1%} {reg_p1:>9.3f} {ask_p1:>9.3f} {drop_r:>+9.3f} {drop_a:>+9.3f}"
        )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
