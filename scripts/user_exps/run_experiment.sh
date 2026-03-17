#!/bin/bash
# Run user simulator experiments
#
# Usage:
#   ./run_experiment.sh <type> <experiment> [model]
#
# Types:
#   usersim      - With user simulator + prompting (default)
#   usersim_none - With user simulator, NO prompting
#   baseline     - No user simulator
#
# Examples:
#   ./run_experiment.sh usersim my_exp                    # All models
#   ./run_experiment.sh usersim my_exp sonnet_4_5         # Single model
#   ./run_experiment.sh usersim_none my_exp gpt_5_2       # No prompting, single model
#   ./run_experiment.sh baseline my_exp                   # Baseline, all models

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$(dirname "$SCRIPT_DIR")")"

TYPE="${1:?Usage: $0 <usersim|usersim_none|baseline> <experiment> [model]}"
EXPERIMENT="${2:?Usage: $0 <type> <experiment> [model]}"
SINGLE_MODEL="${3:-}"

MODELS="${SINGLE_MODEL:-sonnet_4_5 gpt_5_2 opus_4_5 gemini_3_flash gemini_3_pro}"
NUM_TRIALS="${NUM_TRIALS:-3}"
PARALLEL_VARIANTS="${PARALLEL_VARIANTS:-1}"

# Filtered JSON: override with FILTERED_JSON env var, or auto-detect from experiment
if [[ -z "${FILTERED_JSON:-}" ]]; then
    FILTERED_JSON="synthetic/outputs/benchmark_filtered/${EXPERIMENT}_filtered.json"
    if [[ ! -f "$FILTERED_JSON" ]]; then
        echo "ERROR: Filtered JSON not found: $FILTERED_JSON"
        echo "Run filter step first, or set FILTERED_JSON env var to point to your filtered samples."
        exit 1
    fi
fi

STRATEGY="${4:-}"

case "$TYPE" in
    usersim)
        PROMPT="default"
        USER_HELP="--include-user-help --clarification-prompt $PROMPT"
        ;;
    usersim_none)
        PROMPT="none"
        USER_HELP="--include-user-help --clarification-prompt $PROMPT"
        ;;
    baseline)
        USER_HELP=""
        ;;
    ablation_agent_prompt)
        echo "📊 Delegating to ablation script..."
        ablation_args=("$EXPERIMENT")
        [[ -n "$SINGLE_MODEL" ]] && ablation_args+=("$SINGLE_MODEL")
        [[ -n "$SINGLE_MODEL" && -n "$STRATEGY" ]] && ablation_args+=("$STRATEGY")
        ./scripts/user_exps/run_ablation_agent_prompt.sh "${ablation_args[@]}"
        exit $?
        ;;
    *)
        echo "❌ Unknown type: $TYPE"
        echo "Usage: $0 <usersim|usersim_none|baseline|ablation_agent_prompt> <experiment> [model] [strategy]"
        exit 1
        ;;
esac

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Type: $TYPE | Experiment: $EXPERIMENT | Models: $MODELS"
echo "📂 Samples: $(basename "$FILTERED_JSON")"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for model in $MODELS; do
    if [[ "$TYPE" == "baseline" ]]; then
        exp_name="baseline_${EXPERIMENT}_${model}"
    elif [[ "$TYPE" == "usersim" ]]; then
        exp_name="user_sim_${EXPERIMENT}_${model}_default"
    else
        exp_name="user_sim_${EXPERIMENT}_${model}_none"
    fi
    
    if [[ ! -d "synthetic/outputs/${exp_name}" ]] || [[ ! -f "synthetic/outputs/${exp_name}/underspec_candidates.csv" ]]; then
        echo "📥 Loading samples for: $exp_name"
        python scripts/run_tac_underspec.py \
            --load-samples "$FILTERED_JSON" \
            -e "$exp_name"
    fi
    
    echo ""
    echo "🚀 Running: $exp_name ($model)"
    
    python scripts/run_tac_underspec.py --run \
        -e "$exp_name" \
        $USER_HELP \
        --model "$model" \
        --num-trials "$NUM_TRIALS" \
        --parallel-variants "$PARALLEL_VARIANTS" \
        --resume
    
    echo "✅ Done: $exp_name"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All experiments complete!"
echo "   Next: ./scripts/user_exps/generate_reports.sh all $EXPERIMENT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

