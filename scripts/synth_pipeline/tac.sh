#!/usr/bin/env bash
set -euo pipefail

# Run TheAgentCompany (TAC) tasks with a single model
#
# Usage:
#   ./scripts/synth_pipeline/tac.sh opus_4_5                    # All 33 tasks with Opus 4.5
#   ./scripts/synth_pipeline/tac.sh gpt_5_2 --tasks ds_coffee   # Single task with GPT-5.2
#   ./scripts/synth_pipeline/tac.sh gpt_5_2 --num_trials 3       # Override trials
#
# Priority 14 models (for tac_full_baselines.sh):
#   Frontier: opus_4_5, sonnet_4_5, gpt_5_2, gpt_5_1, gpt_5, o3_pro, gemini_3_pro
#   Fast: haiku_4_5, gemini_3_flash
#   Open: kimi_k2, qwen3_235b, llama4_maverick, glm_4p5_air, nova_2_lite

# Navigate to LHAW root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LHAW_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$LHAW_ROOT"

# All AgentCompany tasks
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

show_help() {
    echo "Usage: $0 <model_shortcut> [--tasks task1 task2 ...] [--num_trials N]"
    echo ""
    echo "Model shortcuts:"
    for key in "${!MODELS[@]}"; do
        echo "  $key -> ${MODELS[$key]}"
    done
    echo ""
    echo "Examples:"
    echo "  $0 opus_4_5                           # All tasks with Opus 4.5"
    echo "  $0 gpt_5_2 --tasks ds_coffee          # Single task with GPT-5.2"
    echo "  $0 gpt_5_2 --num_trials 3              # Override trials"
    exit 0
}

# Handle help
if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]] || [[ -z "${1:-}" ]]; then
    show_help
fi

# Get model from shortcut
MODEL_KEY="${1:-}"
shift

if [[ -z "${MODELS[$MODEL_KEY]:-}" ]]; then
    echo "Error: Unknown model shortcut '$MODEL_KEY'"
    echo "Available: ${!MODELS[*]}"
    exit 1
fi

BACKEND_MODEL="${MODELS[$MODEL_KEY]}"
FORMAT_NAME="${MODEL_KEY}_pass${NUM_TRIALS}"

# Parse optional arguments
CUSTOM_TASKS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --tasks)
            shift
            while [[ $# -gt 0 ]] && [[ "${1:0:2}" != "--" ]]; do
                # Split on spaces to handle zsh passing $VAR as one string
                for _t in $1; do CUSTOM_TASKS+=("$_t"); done
                shift
            done
            ;;
        --num_trials)
            NUM_TRIALS="$2"
            FORMAT_NAME="${MODEL_KEY}_pass${NUM_TRIALS}"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Use custom tasks or all tasks
if [[ ${#CUSTOM_TASKS[@]} -gt 0 ]]; then
    TASKS_TO_RUN=("${CUSTOM_TASKS[@]}")
else
    TASKS_TO_RUN=("${ALL_TASKS[@]}")
fi

echo "========================================"
echo "TAC (TheAgentCompany) Task Run"
echo "========================================"
echo "Model:       $MODEL_KEY -> $BACKEND_MODEL"
echo "Format:      $FORMAT_NAME"
echo "Trials:      $NUM_TRIALS"
echo "Tasks:       ${#TASKS_TO_RUN[@]}"
echo "========================================"
echo ""

exec python3 task_completion_agentcompany.py \
    --format_name "$FORMAT_NAME" \
    --backend_model "$BACKEND_MODEL" \
    --num_trials "$NUM_TRIALS" \
    --max_iterations "$MAX_ITERATIONS" \
    --task_names "${TASKS_TO_RUN[@]}"
