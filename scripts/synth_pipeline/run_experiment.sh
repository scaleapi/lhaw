#!/bin/bash
# Run synthetic underspecification experiments
#
# Pipeline Phases:
#   1. GENERATE: Extract segments → create underspecified variants → save candidates CSV
#   2. RUN TRIALS: Execute agent on each variant N times → collect pass/fail outcomes
#
# Usage:
#   ./run_experiment.sh <experiment> [options]
#
# Examples:
#   ./run_experiment.sh v4                    # Full pipeline: generate + run trials
#   ./run_experiment.sh v4 --domain ds        # Single domain only
#   ./run_experiment.sh v4 --generate-only    # Phase 1 only: generate candidates
#   ./run_experiment.sh v4 --run-only         # Phase 2 only: run agent trials
#   ./run_experiment.sh v4 --run-only --resume  # Resume incomplete trials
#
# Options:
#   --domain DOMAIN    Run single domain (ds, finance, hr, sde)
#   --generate-only    Phase 1 only: generate variant candidates
#   --run-only         Phase 2 only: run agent trials (variants must exist)
#   --resume           Skip variants with complete trials (use with --run-only)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$(dirname "$SCRIPT_DIR")")"

EXPERIMENT="${1:?Usage: $0 <experiment> [options]}"
shift

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Agent model for trials
# Options: gemini_3_flash, gemini_3_pro, opus_4_5, sonnet_4_5, gpt_5_2
MODEL="${MODEL:-gemini_3_flash}"

# Experiment settings
NUM_TRIALS="${NUM_TRIALS:-3}"           # Trials per variant (for pass@3)
MAX_LEVEL="${MAX_LEVEL:-2}"             # Segment removal depth (1 or 2)
MAX_VARIANTS_PER_LEVEL="${MAX_VARIANTS_PER_LEVEL:-10}"  # Max variants per level per task
MAX_WORKERS="${MAX_WORKERS:-4}"         # Concurrent LLM calls during candidate generation (--generate)
PARALLEL_VARIANTS="${PARALLEL_VARIANTS:-1}"  # Concurrent variants during agent trial execution (--run)

# Trajectory grounding
TRAJECTORY_DIR="${TRAJECTORY_DIR:-experiments/agentcompany/golden_trajectories}"

# Pre-filtering strategy (how to select top candidates)
# auto: grounding first (trajectory > checkpoint), then priority_score (recommended)
# priority_max/priority_mean: score by priority only
SELECTION_STRATEGY="${SELECTION_STRATEGY:-auto}"

# LHAW tasks by domain
declare -A DOMAIN_TASKS=(
    ["ds"]="ds_answer_numerical_data_question ds_fix_table_values_and_missing_answers ds_format_excel_sheets ds_predictive_modeling ds_visualize_data_in_pie_and_bar_chart"
    ["finance"]="finance_budget_variance finance_check_attendance_payroll finance_expense_validation"
    ["hr"]="hr_check_attendance_multiple_days hr_check_attendance_one_day hr_create_employee_manual hr_new_grad_job_description"
    ["sde"]="sde_create_sqlite_database"
)

# ─────────────────────────────────────────────────────────────────────────────
# Parse options
# ─────────────────────────────────────────────────────────────────────────────

SINGLE_DOMAIN=""
RUN_ONLY=""
GENERATE_ONLY=""
RESUME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)
            SINGLE_DOMAIN="$2"
            shift 2
            ;;
        --run-only)
            RUN_ONLY="1"
            shift
            ;;
        --generate-only)
            GENERATE_ONLY="1"
            shift
            ;;
        --resume)
            RESUME="1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine which domains to run
if [[ -n "$SINGLE_DOMAIN" ]]; then
    if [[ -z "${DOMAIN_TASKS[$SINGLE_DOMAIN]:-}" ]]; then
        echo "Unknown domain: $SINGLE_DOMAIN"
        echo "Valid domains: ds finance hr sde"
        exit 1
    fi
    DOMAINS=("$SINGLE_DOMAIN")
else
    DOMAINS=("ds" "finance" "hr" "sde")
fi

if [[ -n "$RUN_ONLY" && -n "$GENERATE_ONLY" ]]; then
    echo "Conflicting options: --run-only and --generate-only cannot be used together."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧪 LHAW Synthetic Pipeline"
echo "   Experiment: $EXPERIMENT"
echo "   Domains: ${DOMAINS[*]}"
if [[ -n "$GENERATE_ONLY" ]]; then
    echo "   Mode: Generate variants only (no agent trials)"
elif [[ -n "$RUN_ONLY" ]]; then
    echo "   Mode: Run trials only (variants exist)"
    echo "   Model: $MODEL | Trials: $NUM_TRIALS per variant"
    [[ -n "$RESUME" ]] && echo "   Resume: ON (skip complete variants)"
else
    echo "   Mode: Full (generate + run trials)"
    echo "   Model: $MODEL | Trials: $NUM_TRIALS per variant"
    [[ -n "$RESUME" ]] && echo "   Resume: ON (skip complete variants)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for domain in "${DOMAINS[@]}"; do
    tasks="${DOMAIN_TASKS[$domain]}"
    exp_name="${EXPERIMENT}_${domain}"
    
    echo ""
    echo "▶ Domain: $domain"
    echo "  Experiment: $exp_name"
    echo "  Tasks: $tasks"
    
    # Step 1: Generate variants (if not --run-only)
    if [[ -z "$RUN_ONLY" ]]; then
        if [[ ! -f "synthetic/outputs/$exp_name/underspec_candidates.csv" ]]; then
            echo ""
            echo "  📝 Generating variants (strategy=$SELECTION_STRATEGY)..."
            python scripts/run_tac_underspec.py --generate \
                --tasks $tasks \
                --experiment "$exp_name" \
                --max-level "$MAX_LEVEL" \
                --max-variants-per-level "$MAX_VARIANTS_PER_LEVEL" \
                --generate-workers "$MAX_WORKERS" \
                --selection-strategy "$SELECTION_STRATEGY" \
                --trajectory-dir "$TRAJECTORY_DIR"
        else
            echo "  ✓ Variants already exist"
        fi
    fi
    
    # Step 2: Run agent trials (if not --generate-only)
    if [[ -z "$GENERATE_ONLY" ]]; then
        echo ""
        echo "  🚀 Running agent trials ($MODEL, $NUM_TRIALS trials)..."
        
        python scripts/run_tac_underspec.py --run \
            --experiment "$exp_name" \
            --model "$MODEL" \
            --num-trials "$NUM_TRIALS" \
            --parallel-variants "$PARALLEL_VARIANTS" \
            ${RESUME:+--resume}
    fi
    
    echo "  ✅ $domain complete"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Experiments complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ─────────────────────────────────────────────────────────────────────────────
# Completion check
# ─────────────────────────────────────────────────────────────────────────────
if [[ -z "$GENERATE_ONLY" ]]; then
    echo ""
    echo "📋 Checking completion status..."
    
    # Check if all variants have complete trials
    INCOMPLETE=0
    for domain in "${DOMAINS[@]}"; do
        exp_name="${EXPERIMENT}_${domain}"
        # Count incomplete variants (< NUM_TRIALS evaluations in latest run)
        count=$(python -c "
import pandas as pd
from pathlib import Path
from collections import defaultdict

RUNS_DIR = Path('experiments/agentcompany/runs')
csv = Path('synthetic/outputs/${exp_name}/underspec_candidates.csv')
if not csv.exists():
    print(0)
    exit()

df = pd.read_csv(csv)
valid_vids = set(df['variant_id'].tolist())

# Find latest run per variant
runs_by_vid = defaultdict(list)
for run_dir in RUNS_DIR.iterdir():
    if not run_dir.is_dir() or '${exp_name}_' not in run_dir.name:
        continue
    for vid in valid_vids:
        if f'_{vid}_' in run_dir.name:
            eval_dir = run_dir / 'final_results' / 'deterministic'
            n_evals = len(list(eval_dir.rglob('evaluation_*.json'))) if eval_dir.exists() else 0
            runs_by_vid[vid].append((run_dir.stat().st_mtime, n_evals))
            break

# Count incomplete (skip empty runs with 0 evals)
incomplete = 0
for vid, runs in runs_by_vid.items():
    if runs:
        # Sort by mtime descending, find latest non-empty run
        sorted_runs = sorted(runs, key=lambda x: x[0], reverse=True)
        latest = None
        for run in sorted_runs:
            if run[1] > 0:  # n_evals > 0
                latest = run
                break
        if latest is None or latest[1] < ${NUM_TRIALS}:
            incomplete += 1
    else:
        incomplete += 1

# Also count variants with no runs at all
no_runs = len(valid_vids) - len(runs_by_vid)
print(incomplete + no_runs)
" || { echo "⚠️  Completion check failed for ${exp_name}" >&2; echo 0; })
        
        if [[ "$count" -gt 0 ]]; then
            echo "  ⚠️  $exp_name: $count incomplete variants"
            INCOMPLETE=$((INCOMPLETE + count))
        else
            echo "  ✅ $exp_name: all variants complete"
        fi
    done
    
    echo ""
    if [[ "$INCOMPLETE" -gt 0 ]]; then
        echo "⚠️  $INCOMPLETE total incomplete variants"
        echo "   Run with --resume to complete remaining trials:"
        echo "   ./scripts/synth_pipeline/run_experiment.sh $EXPERIMENT --run-only --resume"
        echo ""
        echo "   ❌ Do NOT run filter.sh until all variants are complete!"
    else
        echo "✅ All variants complete!"
        echo ""
        echo "📊 Next steps:"
        echo "   ./scripts/synth_pipeline/filter.sh $EXPERIMENT"
    fi
else
    echo ""
    echo "📊 Next steps:"
    echo "   ./scripts/synth_pipeline/run_experiment.sh $EXPERIMENT --run-only"
fi

