#!/usr/bin/env bash
set -euo pipefail

# Run TAC baselines across ALL priority models for LHAW benchmark
#
# Usage:
#   ./scripts/synth_pipeline/tac_full_baselines.sh              # Run all pending models
#   ./scripts/synth_pipeline/tac_full_baselines.sh --resume     # Resume (skip completed)
#   ./scripts/synth_pipeline/tac_full_baselines.sh --dry-run    # Show what would run
#   ./scripts/synth_pipeline/tac_full_baselines.sh --models opus_4_5 gpt_5_2  # Specific models

# Navigate to LHAW root (two levels up from this script: synth_pipeline -> scripts -> lhaw)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LHAW_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$LHAW_ROOT"

# All 33 AgentCompany tasks
ALL_TASKS=(
    admin_make_spreadsheet
    admin_mass_forms_filling
    admin_remove_pages
    admin_translate_sales_chat
    admin_watch_video
    ds_answer_numerical_data_question
    ds_coffee
    ds_find_meeting_spreadsheet
    ds_fix_table_values_and_missing_answers
    ds_format_excel_sheets
    ds_predictive_modeling
    ds_stock_analysis_slides
    ds_visualize_data_in_pie_and_bar_chart
    finance_budget_variance
    finance_check_attendance_payroll
    finance_expense_validation
    finance_invoice_matching
    finance_nonqualified_bill_ask_for_reimburse
    hr_check_attendance_multiple_days
    hr_check_attendance_multiple_days_department
    hr_check_attendance_one_day
    hr_create_employee_manual
    hr_new_grad_job_description
    hr_organize_talent_info
    hr_populate_salary_increase_memo
    hr_resume_categorization
    hr_salary_analysis
    ml_generate_gradcam
    ml_grade_exam
    research_answer_questions
    sde_copy_table_from_pdf_to_xlsx
    sde_create_sqlite_database
    sde_run_rising_wave_locally
)

# Model shortcuts — sourced from constants.py (single source of truth)
eval "$(python3 "$SCRIPT_DIR/../../constants.py" bash_models)"

# Default settings (sync with constants.py: DEFAULTS)
NUM_TRIALS=${NUM_TRIALS:-3}
MAX_ITERATIONS=${MAX_ITERATIONS:-100}   # agent iterations

# Parse arguments
DRY_RUN=false
RESUME=false
SELECTED_MODELS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        --models)
            shift
            while [[ $# -gt 0 ]] && [[ "${1:0:2}" != "--" ]]; do
                SELECTED_MODELS+=("$1")
                shift
            done
            ;;
        --num_trials)
            NUM_TRIALS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dry-run] [--resume] [--models m1 m2 ...] [--workers N]"
            exit 1
            ;;
    esac
done

# Use selected models or all models
if [[ ${#SELECTED_MODELS[@]} -gt 0 ]]; then
    MODEL_KEYS=("${SELECTED_MODELS[@]}")
else
    MODEL_KEYS=("${!MODELS[@]}")
fi

echo "========================================"
echo "TAC Full Baselines Run"
echo "========================================"
echo "Tasks:       ${#ALL_TASKS[@]}"
echo "Models:      ${#MODEL_KEYS[@]}"
echo "Trials: $NUM_TRIALS (pass@$NUM_TRIALS)"
echo "Dry run:     $DRY_RUN"
echo "Resume:      $RESUME"
echo "========================================"
echo ""

echo "Models to run:"
for key in "${MODEL_KEYS[@]}"; do
    if [[ -z "${MODELS[$key]:-}" ]]; then
        echo "  ❌ $key (unknown)"
    else
        echo "  • $key → ${MODELS[$key]}"
    fi
done
echo ""

TASK_STR="${ALL_TASKS[*]}"
RUNS_DIR="experiments/agentcompany/runs"
FAILED_MODELS=()

run_model() {
    local format_name="$1"
    local model="${MODELS[$format_name]}"
    local run_name="${format_name}_pass${NUM_TRIALS}"
    local run_dir="$RUNS_DIR/run_${run_name}"
    
    # Check if already complete (for --resume)
    # A run is complete only if pass_k_summary CSVs exist (written after all evals succeed).
    # Checking just final_results/ is insufficient — it's created before failure checks.
    if $RESUME; then
        for summary in "$RUNS_DIR"/run_${run_name}_*/final_results/summaries/pass_k_summary_*.csv; do
            if [[ -f "$summary" ]]; then
                echo "⏭️  SKIP $format_name (already complete: $(dirname "$(dirname "$summary")"))"
                return 0
            fi
        done
    fi
    
    echo ""
    echo "========================================"
    echo "🚀 Running: $format_name"
    echo "   Model:   $model"
    echo "   Run:     $run_name"
    echo "========================================"
    
    if $DRY_RUN; then
        echo "[DRY RUN] Would execute:"
        echo "python3 task_completion_agentcompany.py --format_name $run_name --backend_model $model --num_trials $NUM_TRIALS --max_iterations $MAX_ITERATIONS --task_names $TASK_STR"
    else
        echo "Executing..."
        if ! python3 task_completion_agentcompany.py \
            --format_name "$run_name" \
            --backend_model "$model" \
            --num_trials "$NUM_TRIALS" \
            --max_iterations "$MAX_ITERATIONS" \
            --task_names $TASK_STR; then
            echo "⚠️  $format_name finished with errors"
            FAILED_MODELS+=("$format_name")
            return 1
        fi
    fi
}

# Run each model
for format_name in "${MODEL_KEYS[@]}"; do
    if [[ -z "${MODELS[$format_name]:-}" ]]; then
        echo "❌ Skipping unknown model: $format_name"
        continue
    fi
    run_model "$format_name" || true
done

echo ""
echo "========================================"
if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo "❌ Completed with failures: ${FAILED_MODELS[*]}"
else
    echo "✅ All models complete!"
fi
echo "========================================"

# Export golden trajectories for synthetic pipeline
if ! $DRY_RUN && [[ ${#FAILED_MODELS[@]} -eq 0 ]]; then
    echo ""
    echo "Exporting golden trajectories..."
    python3 scripts/export_tac_golden_trajectories.py
fi

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    exit 1
fi

