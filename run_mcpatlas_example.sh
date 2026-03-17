#!/usr/bin/env bash
# =================================================================
# MCP-Atlas: Small reproducible end-to-end example
# =================================================================
#
# Mirrors run_tac_example.sh / run_swebench_example.sh for MCP-Atlas.
#
# Prerequisites:
#   - Docker running (8GB+ memory recommended)
#   - MCP Atlas repo cloned:
#       cd experiments/mcpatlas
#       git clone <mcp-atlas-repo> mcp-atlas
#       cd mcp-atlas && git submodule update --init --recursive
#   - MCP Atlas Docker image built (one-time):
#       cd experiments/mcpatlas/mcp-atlas && make build
#   - MCP Atlas services running in separate terminals:
#       Terminal 1: cd experiments/mcpatlas/mcp-atlas && make run-docker
#       Terminal 2: cd experiments/mcpatlas/mcp-atlas && make run-mcp-completion
#   - LLM_API_KEY set (or OPENAI_API_KEY fallback)
#   - source .env
#
# Structure:
#   Part 1 (Steps 1-5): Baselines + underspec WITHOUT ask_user
#     → Run with normal completion service
#   Part 2 (Steps 6-7): Underspec WITH ask_user + persona ablations
#     → Restart completion service with: USER_TOOL_ENABLED=True make run-mcp-completion
#   Part 3 (Steps 8-11): Analysis + charts
#     → No completion service needed, just post-processing
#
# Task selection:
#   BASELINE_MODELS controls which models run baselines. Tasks must pass
#   ALL baseline models (intersection) to be selected for underspec generation.
#   Default: 1 model for speed. Paper used all reference models.
#   To match the paper: set BASELINE_MODELS=("${MODELS[@]}")
#
# To reproduce exact paper results (Table 3, 6, 8):
#   MODELS=(opus_4_5 sonnet_4_5 gemini_3_pro gemini_3_flash gpt_5_2)
#   BASELINE_MODELS=("${MODELS[@]}")
#   LIMIT=100 (or remove for all tasks)
#
# Usage: bash run_mcpatlas_example.sh
# =================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ── Configuration ──
EXP="mcpatlas_med_exp"
LIMIT=15                  # Number of tasks to run
NUM_TRIALS=3             # Trials per task (for pass@3)
CONCURRENCY=5            # Parallel completions

# Short names are resolved by constants.py → full LiteLLM identifiers
MODELS=(gemini_3_flash sonnet_4_6 gpt_5_2)

# Models used for baseline task selection (intersection of passed tasks).
# Paper used all reference models; default uses first model for speed.
BASELINE_MODELS=("${MODELS[0]}")

PASSED_JSON="experiments/mcpatlas/runs/passed_task_ids_${EXP}.json"
UNDERSPEC_DIR="experiments/mcpatlas/underspec_output/${EXP}"
REPORT_DIR="experiments/mcpatlas/reports/${EXP}"

# Helper: find the most recent run directory matching a format_name prefix.
# Optional second arg is a grep -v filter (e.g. "_ask_" to exclude ask runs).
latest_run() {
  local results
  results=$(ls -td experiments/mcpatlas/runs/run_${1}_* 2>/dev/null)
  if [[ -n "${2:-}" ]]; then
    results=$(echo "$results" | grep -v "$2")
  fi
  echo "$results" | head -1
}

# ═══════════════════════════════════════════════════════════════
#  Part 1: Baselines + Underspec (no ask_user needed)
#  Run with: make run-mcp-completion (normal, no special env vars)
# ═══════════════════════════════════════════════════════════════

# ── 1. Run baselines on original tasks (pass@3 per baseline model) ──
# MCP-Atlas.csv is auto-downloaded from HuggingFace (ScaleAI/MCP-Atlas) if not present.
for model in "${BASELINE_MODELS[@]}"; do
  echo "Running baselines: $model"
  python task_completion_mcpatlas.py \
    --format_name "${EXP}_baseline_${model}" \
    --backend_model "$model" \
    --num_trials "$NUM_TRIALS" \
    --max_k "$NUM_TRIALS" \
    --limit "$LIMIT" \
    --concurrency "$CONCURRENCY"
done

# ── 2. Select tasks that passed ALL baseline models (intersection) ──
BASELINE_DIRS=()
for model in "${BASELINE_MODELS[@]}"; do
  for d in $(ls -td experiments/mcpatlas/runs/run_${EXP}_baseline_${model}_* 2>/dev/null); do
    BASELINE_DIRS+=("$d")
    break
  done
done
if [[ ${#BASELINE_DIRS[@]} -eq 0 ]]; then
  echo "ERROR: No baseline run directories found"
  exit 1
fi
python scripts/filter_passed_tasks.py "${BASELINE_DIRS[@]}" -o "$PASSED_JSON"

# ── 3. Generate underspec variants for passed tasks ──
python scripts/generate_mcpatlas_underspec.py \
  "$PASSED_JSON" \
  --output-dir "$UNDERSPEC_DIR" \
  --severity DELETE \
  --max-level 2

# ── 4. Run underspec trials WITHOUT ask_user (all models, pass@3) ──
for model in "${MODELS[@]}"; do
  python task_completion_mcpatlas.py \
    --format_name "${EXP}_underspec_${model}" \
    --backend_model "$model" \
    --num_trials "$NUM_TRIALS" \
    --max_k "$NUM_TRIALS" \
    --input_csv "$UNDERSPEC_DIR/underspec_prompts.csv" \
    --concurrency "$CONCURRENCY"
done

# ── 5. Run baselines for remaining models on passed tasks ──
# Step 1 baselines only ran for BASELINE_MODELS (for task selection).
# Charts need baselines for ALL models, so run the rest here.
TASK_IDS=$(python3 -c "import json; print(','.join(json.load(open('$PASSED_JSON'))))")
for model in "${MODELS[@]}"; do
  EXISTING=$(latest_run "${EXP}_baseline_${model}")
  if [[ -n "$EXISTING" ]]; then
    echo "Baseline already exists for $model: $(basename $EXISTING)"
    continue
  fi
  echo "Running baselines: $model"
  python task_completion_mcpatlas.py \
    --format_name "${EXP}_baseline_${model}" \
    --backend_model "$model" \
    --num_trials "$NUM_TRIALS" \
    --max_k "$NUM_TRIALS" \
    --task_ids "$TASK_IDS" \
    --concurrency "$CONCURRENCY"
done

# ═══════════════════════════════════════════════════════════════
#  Part 2: ask_user experiments
#  IMPORTANT: Restart the completion service BEFORE running this part:
#    Terminal 2: USER_TOOL_ENABLED=True make run-mcp-completion
# ═══════════════════════════════════════════════════════════════

# ── 6. User simulator experiments (all models, with ask_user) ──
for model in "${MODELS[@]}"; do
  USER_TOOL_ENABLED=True USE_SYSTEM_PROMPT_IN_COMPLETION=true \
    python task_completion_mcpatlas.py \
    --format_name "${EXP}_underspec_${model}_ask" \
    --backend_model "$model" \
    --num_trials "$NUM_TRIALS" \
    --max_k "$NUM_TRIALS" \
    --input_csv "$UNDERSPEC_DIR/underspec_prompts.csv" \
    --concurrency "$CONCURRENCY"
done

# ── 7. Persona ablation — Cost of Information (§4.3) ──
# Tests how user persona (perceived cost of asking) affects clarification.
#   supervisor         — user is sitting with you, eager to help
#   standard_assistant — user is available but working on other tasks
#   busy_executive     — user is very busy, minimize interruptions
ABLATION_MODEL="${MODELS[0]}"

for persona in supervisor standard_assistant busy_executive; do
  USER_TOOL_ENABLED=True \
  USE_SYSTEM_PROMPT_IN_COMPLETION=true \
  USER_TYPE=$persona \
    python task_completion_mcpatlas.py \
      --format_name "${EXP}_persona_${persona}_${ABLATION_MODEL}" \
      --backend_model "$ABLATION_MODEL" \
      --num_trials "$NUM_TRIALS" \
      --max_k "$NUM_TRIALS" \
      --input_csv "$UNDERSPEC_DIR/underspec_prompts.csv" \
      --concurrency "$CONCURRENCY"
done

# ═══════════════════════════════════════════════════════════════
#  Part 3: Analysis & Charts (no completion service needed)
# ═══════════════════════════════════════════════════════════════

# ── 8. Process underspec results (ambiguity classification) ──
FIRST_RUN=$(latest_run "${EXP}_underspec_${MODELS[0]}" "_ask_")
if [[ -n "$FIRST_RUN" ]]; then
  python experiments/mcpatlas/scripts/process_underspec_run.py \
    --dataset-dir "$(pwd)/$UNDERSPEC_DIR" \
    --run-dir "$(pwd)/$FIRST_RUN"
else
  echo "Skipping step 8: no underspec run found for ${MODELS[0]}"
fi

# ── 9. Per-model pass@3 comparison charts ──
mkdir -p "$REPORT_DIR"
for model in "${MODELS[@]}"; do
  BASELINE_RUN=$(latest_run "${EXP}_baseline_${model}")
  UNDERSPEC_RUN=$(latest_run "${EXP}_underspec_${model}" "_ask_")
  WITH_USER_RUN=$(latest_run "${EXP}_underspec_${model}_ask")

  if [[ -n "$BASELINE_RUN" && -n "$UNDERSPEC_RUN" && -n "$WITH_USER_RUN" ]]; then
    python experiments/mcpatlas/scripts/plot_pass3_from_runs.py \
      --baseline-run "$BASELINE_RUN" \
      --no-user-run "$UNDERSPEC_RUN" \
      --with-user-run "$WITH_USER_RUN" \
      --output-dir "$REPORT_DIR" \
      --model-name "$model"
  else
    echo "Skipping charts for $model (missing baseline/underspec/ask runs)"
  fi
done

# ── 10. Analyze ask_user questions + failure mode plots (§4.4) ──
for model in "${MODELS[@]}"; do
  WITH_USER_RUN=$(latest_run "${EXP}_underspec_${model}_ask")
  if [[ -n "$WITH_USER_RUN" ]]; then
    python experiments/mcpatlas/scripts/analyze_ask_user.py \
      --run-dir "$WITH_USER_RUN" \
      --output "$REPORT_DIR/ask_user_${model}.csv" \
      --judge-model gpt-4o

    python experiments/mcpatlas/scripts/plot_ask_user.py \
      --input "$REPORT_DIR/ask_user_${model}.csv" \
      --output-dir "$REPORT_DIR" \
      --prefix "$model"
  fi
done

# Persona analysis
for persona in supervisor standard_assistant busy_executive; do
  PERSONA_RUN=$(latest_run "${EXP}_persona_${persona}_${ABLATION_MODEL}")
  if [[ -n "$PERSONA_RUN" ]]; then
    python experiments/mcpatlas/scripts/analyze_ask_user.py \
      --run-dir "$PERSONA_RUN" \
      --output "$REPORT_DIR/ask_user_persona_${persona}_${ABLATION_MODEL}.csv" \
      --judge-model gpt-4o

    python experiments/mcpatlas/scripts/plot_ask_user.py \
      --input "$REPORT_DIR/ask_user_persona_${persona}_${ABLATION_MODEL}.csv" \
      --output-dir "$REPORT_DIR" \
      --prefix "persona_${persona}_${ABLATION_MODEL}"
  fi
done

# ── 11. Pass@3 breakdown by number of segments removed ──
ALL_UNDERSPEC_RUNS=()
for model in "${MODELS[@]}"; do
  RUN=$(latest_run "${EXP}_underspec_${model}" "_ask_")
  [[ -n "$RUN" ]] && ALL_UNDERSPEC_RUNS+=("$RUN")
done
if [[ ${#ALL_UNDERSPEC_RUNS[@]} -gt 0 ]]; then
  python experiments/mcpatlas/scripts/underspec_pass3_by_segments.py \
    "${ALL_UNDERSPEC_RUNS[@]}" \
    --base-path "$UNDERSPEC_DIR" \
    --output "$REPORT_DIR/pass3_by_segments.csv"
fi

# ── 12. Summary table (Paper Table 3) ──
python experiments/mcpatlas/scripts/compute_mcpatlas_summary.py \
  --exp-name "$EXP" \
  --models "${MODELS[@]}" \
  --output "$REPORT_DIR/summary.csv"

echo ""
echo "Done! Results are in experiments/mcpatlas/runs/"
echo "Underspec variants are in $UNDERSPEC_DIR"
echo "Reports (when generated) are in $REPORT_DIR"
