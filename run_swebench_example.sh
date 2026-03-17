#!/usr/bin/env bash
# =================================================================
# SWE-Bench Pro: Small reproducible end-to-end example
# =================================================================
#
# Mirrors run_tac_example.sh for the SWE-Bench Pro pipeline.
#
# Prerequisites:
#   - conda activate lhaw311 (Python 3.11+)
#   - python -m pip install -e swebenchpro/SWE-bench_Pro-os/SWE-agent
#   - modal token new (Modal auth for container orchestration)
#   - LLM_API_KEY and LLM_BASE_URL set (or OPENAI_API_KEY fallback)
#   - source .env
#
# Task selection:
#   BASELINE_MODELS controls which models run baselines. The paper required
#   |F2P| > 2 AND pass@3 > 0 for ALL reference models (417 → 100 tasks).
#   Default: 1 model for speed. To match the paper:
#     set BASELINE_MODELS=("${MODELS[@]}")
#
# To reproduce exact paper results (Table 3, 6, 7, 10):
#   LIMIT=75 (or remove --limit entirely for all F2P-eligible tasks)
#   TARGET_VARIANTS=200
#   MODELS=(gpt_5_2 sonnet_4_5 gemini_3_pro gemini_3_flash)
#   BASELINE_MODELS=("${MODELS[@]}")
#   See experiments/swebench/README.md for full CLI reference.
#
# Usage: bash run_swebench_example.sh
#   Uncomment steps as you progress through the pipeline.
# =================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ── Configuration ──
EXP="swebench_small_exp"
LIMIT=5                  # Number of base tasks (after F2P filter)
TARGET_VARIANTS=10       # Total underspec variants to generate
NUM_TRIALS=3             # Trials per variant (for pass@3)
DOCKERHUB_USER="jefzda"
TRAJ_DIR="experiments/swebench/golden_trajectories"

# Short names are resolved by constants.py -> full LiteLLM identifiers
# with litellm_proxy/ prefix auto-added when LLM_BASE_URL is set
MODELS=(gemini_3_flash sonnet_4_6 gpt_5_2)

# Models used for baseline task selection (intersection of passed tasks).
# Paper used all reference models; default uses first model for speed.
BASELINE_MODELS=("${MODELS[0]}")

# ═══════════════════════════════════════════════════════════════
#  Phase A: Baselines + Task Selection + Generate + Classify
# ═══════════════════════════════════════════════════════════════

# ── 1. Run baselines per baseline model (pass@k trials) ──
# --prepare-baselines creates instances.yaml from F2P-filtered original tasks
# and writes results to baseline_N/ directories (not exp_N/).
# Golden trajectories are auto-exported to $TRAJ_DIR.
# format_name is model-agnostic so the experiment directory name
# doesn't embed a specific model (sibling dirs do via --model-suffix).
BASELINE_DIRS=()
for model in "${BASELINE_MODELS[@]}"; do
  echo "Running baselines: $model"
  python task_completion_swebench.py --prepare-baselines --run \
    --format_name "$EXP" \
    --limit "$LIMIT" \
    --backend_model "$model" \
    --num_trials "$NUM_TRIALS" \
    --concurrency 5

  # Capture the directory just created (newest run_${EXP}_* dir)
  MODEL_DIR=$(ls -td experiments/swebench/runs/run_${EXP}_* 2>/dev/null | head -1)
  if [[ -z "$MODEL_DIR" ]]; then
    echo "ERROR: No baseline run directory found for $model"
    exit 1
  fi

  # Evaluate patches via Docker (produces eval_results/)
  python scripts/process_swebench_underspec.py \
    --exp-dir "$MODEL_DIR" --run-eval --eval-only \
    --dockerhub-username "$DOCKERHUB_USER"

  # Summarize as pass_k_per_task.csv (same format as MCP-Atlas)
  python scripts/summarize_swebench_baselines.py \
    --exp-dir "$MODEL_DIR" --num_trials "$NUM_TRIALS"

  BASELINE_DIRS+=("$MODEL_DIR")
done

# ── 1b. Select tasks that passed ALL baseline models (intersection) ──
PASSED_JSON="experiments/swebench/runs/passed_task_ids_${EXP}.json"
python scripts/filter_passed_tasks.py "${BASELINE_DIRS[@]}" -o "$PASSED_JSON"

# First baseline dir is the experiment root (model-agnostic name)
EXP_DIR="${BASELINE_DIRS[0]}"
echo "Experiment directory: $EXP_DIR"

# ── 2. Generate underspec variants for passed tasks ──
# Uses golden trajectories from step 1 for grounded segment extraction.
# --tasks-file restricts generation to tasks that passed all baselines.
# --runs-dir reuses Step 1's dir so baseline_N/ are already in place.
python task_completion_swebench.py --generate \
  --format_name "$EXP" \
  --runs-dir "$EXP_DIR" \
  --severity delete \
  --limit "$LIMIT" \
  --target-variants "$TARGET_VARIANTS" \
  --max-level 2 \
  --trajectory-dir "$TRAJ_DIR" \
  --tasks-file "$PASSED_JSON"

# ── 3. Run underspec trials (first model, baselines already in baseline_N/) ──
python task_completion_swebench.py --run \
  --exp-dir "$EXP_DIR" \
  --backend_model "${MODELS[0]}" \
  --num_trials "$NUM_TRIALS" \
  --skip-baseline \
  --concurrency 5

# ── 4. Evaluate patches + classify variants ──
python scripts/process_swebench_underspec.py \
  --exp-dir "$EXP_DIR" \
  --run-eval \
  --dockerhub-username "$DOCKERHUB_USER" \
  --judge

# ── 5. Filter to benchmark quotas (50/30/20 OC/divergent/benign) ──
python scripts/filter_swebench_samples.py \
  --input "$EXP_DIR/underspec_results.csv" \
  --max-total "$TARGET_VARIANTS"

# ── 6. Freeze filtered variants for multi-model comparison ──
python task_completion_swebench.py --freeze \
  --exp-dir "$EXP_DIR" \
  --filtered-csv "$EXP_DIR/underspec_results_filtered.csv"

# ═══════════════════════════════════════════════════════════════
#  Phase B: Multi-model comparison on frozen variants
# ═══════════════════════════════════════════════════════════════

# ── 7. Run remaining models on same frozen variants (includes baselines) ──
for i in $(seq 1 $((${#MODELS[@]} - 1))); do
  model="${MODELS[$i]}"

  python task_completion_swebench.py --run \
    --exp-dir "$EXP_DIR" \
    --model-suffix "$model" \
    --backend_model "$model" \
    --num_trials "$NUM_TRIALS" \
    --concurrency 5

  python scripts/process_swebench_underspec.py \
    --exp-dir "${EXP_DIR}_${model}" \
    --run-eval \
    --dockerhub-username "$DOCKERHUB_USER"
done

# ── 8. User simulator experiments (all models, with ask_user tool) ──
# Requires ask_user infrastructure in swebenchpro/SWE-bench_Pro-os/SWE-agent/
for i in $(seq 0 $((${#MODELS[@]} - 1))); do
  model="${MODELS[$i]}"
  suffix="${model}_ask"

  python task_completion_swebench.py --run \
    --exp-dir "$EXP_DIR" \
    --model-suffix "$suffix" \
    --backend_model "$model" \
    --ask-user \
    --skip-baseline \
    --num_trials "$NUM_TRIALS" \
    --concurrency 5

  python scripts/process_swebench_underspec.py \
    --exp-dir "${EXP_DIR}_${suffix}" \
    --run-eval \
    --dockerhub-username "$DOCKERHUB_USER"
done

# ═══════════════════════════════════════════════════════════════
#  Paper Tables (Table 3, 6, 7, 10)
# ═══════════════════════════════════════════════════════════════

# ── 9. Generate paper tables ──
python scripts/compute_swebench_metrics.py --exp-dir "$EXP_DIR"

# ── 10. (Optional) Export benchmark JSON for distribution ──
# python scripts/export_swebench_dataset.py \
#   --input "$EXP_DIR/underspec_results_filtered.csv" \
#   --exp-dir "$EXP_DIR"
