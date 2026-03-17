#!/bin/bash
# Run agent prompt strategy ablation experiments
#
# Tests prompting strategies in TWO conditions:
#   1. baseline: No user simulator (agent must guess)
#   2. usersim:  With user simulator (agent can ask clarification questions)
#
# Strategies:
#   - none: Baseline (no strategy instructions)
#   - react: ReAct pattern (Thought -> Action -> Observation)
#   - reflexion: Self-correction (Act -> Self-Assess -> Decide)
#   - plan_and_execute: Upfront planning (Plan -> Execute -> Re-plan)
#
# Usage:
#   ./run_ablation_agent_prompt.sh <experiment> [model] [strategy]
#
# Examples:
#   ./run_ablation_agent_prompt.sh my_exp                         # All strategies, GPT-5.2, both conditions
#   ./run_ablation_agent_prompt.sh my_exp gpt_5_2 react           # Single strategy
#   ./run_ablation_agent_prompt.sh my_exp sonnet_4_5              # All strategies, Sonnet
#   STRATEGIES="react reflexion" ./run_ablation_agent_prompt.sh my_exp  # Subset
#   CONDITIONS="baseline" ./run_ablation_agent_prompt.sh my_exp   # Baseline only
#
# Environment variables:
#   NUM_TRIALS:         Trials per variant (default: 3)
#   PARALLEL_VARIANTS:  Concurrent variants (default: 1)
#   STRATEGIES:         Space-separated strategies (default: all)
#   CONDITIONS:         Space-separated conditions: "baseline usersim" (default: both)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$(dirname "$SCRIPT_DIR")")"

EXPERIMENT="${1:?Usage: $0 <experiment> [model] [strategy]}"
MODEL="${2:-gpt_5_2}"
SINGLE_STRATEGY="${3:-}"

# "none" is NOT included by default because existing experiments serve as the baseline:
#   - baseline_{exp}_{model} = ablation "none" for baseline condition
#   - user_sim_{exp}_{model}_default = ablation "none" for usersim condition
if [[ -n "$SINGLE_STRATEGY" ]]; then
    STRATEGIES="$SINGLE_STRATEGY"
else
    STRATEGIES="${STRATEGIES:-react reflexion plan_and_execute}"
fi

CONDITIONS="${CONDITIONS:-baseline usersim}"

NUM_TRIALS="${NUM_TRIALS:-3}"
PARALLEL_VARIANTS="${PARALLEL_VARIANTS:-1}"

FILTERED_JSON="synthetic/outputs/benchmark_filtered/${EXPERIMENT}_filtered.json"
if [[ ! -f "$FILTERED_JSON" ]]; then
    echo "Filtered samples not found: $FILTERED_JSON"
    echo "   Run: ./scripts/synth_pipeline/filter.sh $EXPERIMENT"
    exit 1
fi

echo "============================================================"
echo "Ablation: Agent Prompt Strategies"
echo "============================================================"
echo "Experiment: $EXPERIMENT | Model: $MODEL"
echo "Samples: $(basename "$FILTERED_JSON")"
echo "Strategies: $STRATEGIES"
echo "Conditions: $CONDITIONS"
echo "Config: $NUM_TRIALS trials, $PARALLEL_VARIANTS parallel variants"
echo "============================================================"

for condition in $CONDITIONS; do
    echo ""
    echo "== CONDITION: $condition =="
    
    for strategy in $STRATEGIES; do
        exp_name="ablation_agent_prompt_${condition}_${EXPERIMENT}_${MODEL}_${strategy}"
        
        echo ""
        echo "------------------------------------------------------------"
        echo "Strategy: $strategy | Condition: $condition"
        echo "------------------------------------------------------------"
        
        if [[ ! -d "synthetic/outputs/${exp_name}" ]] || [[ ! -f "synthetic/outputs/${exp_name}/underspec_candidates.csv" ]]; then
            echo "Loading samples for: $exp_name"
            python scripts/run_tac_underspec.py \
                --load-samples "$FILTERED_JSON" \
                -e "$exp_name"
        fi
        
        echo ""
        echo "Running: $exp_name ($MODEL, strategy=$strategy)"
        
        if [[ "$condition" == "baseline" ]]; then
            USER_HELP_FLAGS=""
        else
            USER_HELP_FLAGS="--include-user-help --clarification-prompt default"
        fi
        
        python scripts/run_tac_underspec.py --run \
            -e "$exp_name" \
            $USER_HELP_FLAGS \
            --agent-strategy "$strategy" \
            --model "$MODEL" \
            --num-trials "$NUM_TRIALS" \
            --parallel-variants "$PARALLEL_VARIANTS" \
            --resume
        
        echo "Done: $exp_name (strategy=$strategy, condition=$condition)"
    done
done

echo ""
echo "============================================================"
echo "Ablation complete!"
echo ""
echo "Experiments created:"
for condition in $CONDITIONS; do
    for strategy in $STRATEGIES; do
        echo "   - ablation_agent_prompt_${condition}_${EXPERIMENT}_${MODEL}_${strategy}"
    done
done
echo ""
echo "Next steps:"
echo "   1. Generate reports:"
echo "      MODELS=\"$MODEL\" ./scripts/user_exps/generate_reports.sh ablation $EXPERIMENT"
echo ""
echo "   2. Or analyze manually:"
echo "      python scripts/process_tac_usersim.py -e ablation_agent_prompt_usersim_${EXPERIMENT}_${MODEL}_react"
echo "      python scripts/process_tac_usersim.py -e ablation_agent_prompt_baseline_${EXPERIMENT}_${MODEL}_react"
echo "============================================================"
