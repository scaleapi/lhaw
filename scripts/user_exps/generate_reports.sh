#!/bin/bash
# Generate HTML reports for experiments
#
# Usage:
#   ./generate_reports.sh <type> <experiment>
#
# Types:
#   all          - Comprehensive 3-way analysis (RECOMMENDED)
#   ablation     - Agent prompting strategy ablation analysis
#   trajectories - Per-model trajectory viewers for debugging
#
# Examples:
#   ./generate_reports.sh all my_exp                         # Main analysis report
#   ./generate_reports.sh ablation my_exp                    # Ablation: agent prompting
#   ./generate_reports.sh trajectories my_exp                # Generate trajectory HTML
#   MODELS='gpt_5_2' ./generate_reports.sh ablation my_exp   # Ablation for single model

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$(dirname "$SCRIPT_DIR")")"

TYPE="${1:?Usage: $0 <all|ablation|trajectories> <experiment>}"
EXPERIMENT="${2:?Usage: $0 <type> <experiment>}"
NUM_TRIALS="${NUM_TRIALS:-3}"
MODELS="${MODELS:-}"
REPORTS_DIR="synthetic/outputs"

mkdir -p "$REPORTS_DIR"

# Auto-detect models with prompted data for this experiment.
detect_prompted_models() {
    runs_dir="experiments/agentcompany/runs"
    local detected=""
    prefix="run_underspecified_user_sim_${EXPERIMENT}_"
    for dir in "$runs_dir"/${prefix}*; do
        [[ -d "$dir" ]] || continue
        name="$(basename "$dir")"
        remainder="${name#${prefix}}"
        if [[ "$remainder" =~ ^(.+)_default_ ]]; then
            model="${BASH_REMATCH[1]}"
            if [[ ! " $detected " =~ " $model " ]]; then
                detected="$detected $model"
            fi
        fi
    done
    detected="${detected# }"
    echo "$detected"
}

case "$TYPE" in
    all)
        if [[ -z "$MODELS" ]]; then
            MODELS="$(detect_prompted_models)"
            if [[ -z "$MODELS" ]]; then
                echo "No prompted experiments found for ${EXPERIMENT}"
                echo "   Looking for: run_underspecified_user_sim_${EXPERIMENT}_*_default_*"
                exit 1
            fi
            echo "Auto-detected models with prompted data: $MODELS"
        fi
        echo "Comparing all conditions (${EXPERIMENT})"
        
        python scripts/compare_tac_conditions.py \
            --experiment "${EXPERIMENT}" \
            --models "$(echo "$MODELS" | tr ' ' ',')" \
            --min-evals "${NUM_TRIALS}"
        ;;
    
    ablation)
        echo "Generating ablation analysis: agent prompting strategies (${EXPERIMENT})"
        
        if [[ -z "$MODELS" ]]; then
            MODELS=""
            for dir in synthetic/outputs/ablation_agent_prompt_*_${EXPERIMENT}_*_react; do
                [[ -d "$dir" ]] || continue
                name="$(basename "$dir")"
                remainder="${name#ablation_agent_prompt_*_${EXPERIMENT}_}"
                model="${remainder%_react}"
                if [[ -n "$model" ]] && [[ ! " $MODELS " =~ " $model " ]]; then
                    MODELS="$MODELS $model"
                fi
            done
            MODELS="${MODELS# }"
            
            if [[ -z "$MODELS" ]]; then
                echo "No ablation experiments found for ${EXPERIMENT}"
                echo "   Looking for: ablation_agent_prompt_{baseline,usersim}_${EXPERIMENT}_*_react"
                echo ""
                echo "Run the ablation first:"
                echo "   ./scripts/user_exps/run_experiment.sh ablation_agent_prompt ${EXPERIMENT}"
                exit 1
            fi
            echo "Auto-detected models with ablation data: $MODELS"
        fi
        
        for model in $MODELS; do
            for cond in baseline usersim; do
                for strategy in none react reflexion plan_and_execute; do
                    exp="ablation_agent_prompt_${cond}_${EXPERIMENT}_${model}_${strategy}"
                    if [[ -d "synthetic/outputs/${exp}" ]]; then
                        echo "  Found: ${exp}"
                    fi
                done
            done
        done
        
        python scripts/compare_tac_conditions.py \
            --experiment "${EXPERIMENT}" \
            --models "$(echo "$MODELS" | tr ' ' ',')" \
            --min-evals "${NUM_TRIALS}" \
            --ablation agent_prompt
        ;;
    
    trajectories)
        if [[ -z "$MODELS" ]]; then
            MODELS="$(detect_prompted_models)"
            if [[ -z "$MODELS" ]]; then
                echo "No prompted experiments found for ${EXPERIMENT}"
                echo "   Looking for: run_underspecified_user_sim_${EXPERIMENT}_*_default_*"
                exit 1
            fi
            echo "Auto-detected models with prompted data: $MODELS"
        fi
        TRAJ_DIR="experiments/agentcompany/runs/reports"
        mkdir -p "$TRAJ_DIR"
        echo "Generating trajectory viewers (${EXPERIMENT}) [parallel]"
        for model in $MODELS; do
            echo "  ${model}: prompted, unprompted, baseline"
            (python scripts/view_tac_trajectory.py "*user_sim_${EXPERIMENT}_${model}_default*" --output "${TRAJ_DIR}/trajectories_${EXPERIMENT}_${model}_prompted.html" 2>/dev/null || true) &
            (python scripts/view_tac_trajectory.py "*user_sim_${EXPERIMENT}_${model}_none*" --output "${TRAJ_DIR}/trajectories_${EXPERIMENT}_${model}_none.html" 2>/dev/null || true) &
            (python scripts/view_tac_trajectory.py "*baseline_${EXPERIMENT}_${model}*" --output "${TRAJ_DIR}/trajectories_baseline_${EXPERIMENT}_${model}.html" 2>/dev/null || true) &
        done
        wait
        echo "Trajectory HTML files in ${TRAJ_DIR}/"
        ;;
        
    *)
        echo "Unknown type: $TYPE"
        echo ""
        echo "Usage: $0 <all|ablation|trajectories> <experiment>"
        echo ""
        echo "Types:"
        echo "  all          - Comprehensive 3-way analysis (RECOMMENDED)"
        echo "  ablation     - Agent prompting strategy ablation analysis"
        echo "  trajectories - Per-model trajectory viewers for debugging"
        echo ""
        echo "Examples:"
        echo "  $0 all my_exp                          # Generate main report"
        echo "  $0 ablation my_exp                     # Agent prompting ablation"
        echo "  $0 trajectories my_exp                 # Generate trajectory HTML"
        echo "  MODELS='gpt_5_2' $0 ablation my_exp    # Ablation for single model"
        exit 1
        ;;
esac

echo "Done!"
