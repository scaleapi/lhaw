#!/bin/bash
# LHAW mini reproduction — 4 tasks, 3 models, end-to-end
#
# Copy-paste steps individually (each depends on the previous completing).
# Run from the LHAW root: src/models/research/lhaw/
#
# Task selection:
#   TAC uses a fixed task list (TASKS). The paper selected 13 tasks with
#   avg checkpoint accuracy >= 50% across reference models (see SELECTED_TASKS
#   in constants.py). This example uses 4 of those for speed.
#
# To reproduce exact paper results (Table 3, 6, 7, 9, 11):
#   Use all 13 SELECTED_TASKS from constants.py (or all 33 AgentCompany tasks).
#   MODELS=(opus_4_5 sonnet_4_5 gemini_3_pro gemini_3_flash gpt_5_2)
#   See README.md "Reproduce Paper Results (TAC)" section for the full commands
#   using tac_full_baselines.sh and run_experiment.sh.
#
# Steps 1–5 are the core pipeline (run once).
# Steps 6–8 run user simulator analysis (per model).

set -euo pipefail

TASKS=(finance_budget_variance hr_new_grad_job_description sde_create_sqlite_database ds_visualize_data_in_pie_and_bar_chart)
EXP="delete_small_exp"
MODELS=(gemini_3_flash sonnet_4_6 gemini_3_1_flash_lite)

# ── 1. Baselines (4 tasks, all models, pass@3) ──
for model in "${MODELS[@]}"; do
  ./scripts/synth_pipeline/tac.sh "$model" --tasks "${TASKS[@]}"
done

# ── 1b. Export golden trajectories from baseline runs ──
python scripts/export_tac_golden_trajectories.py --tasks "${TASKS[@]}"

# ── 2. Generate underspec variants (3 per level, up to 2-segment) ──
python scripts/run_tac_underspec.py --generate \
  --tasks "${TASKS[@]}" \
  --experiment "$EXP" \
  --max-variants-per-level 3 --max-level 2 \
  --trajectory-dir experiments/agentcompany/golden_trajectories

# ── 3. Run agent trials on generated variants (pass@3, first model) ──
python scripts/run_tac_underspec.py --run \
  --experiment "$EXP" \
  --model "${MODELS[0]}" \
  --num-trials 3

# ── 4. Process results + ambiguity classification ──
python scripts/process_tac_underspec.py -e "$EXP" --judge

# ── 5. Filter to benchmark format ──
python scripts/filter_tac_samples.py \
  --experiment "$EXP" --max-total 20

# ── 6. User simulator experiments (all models) ──
# run_experiment.sh handles naming:
#   baseline_<exp>_<model>
#   user_sim_<exp>_<model>_default
#   user_sim_<exp>_<model>_none
for model in "${MODELS[@]}"; do
  ./scripts/user_exps/run_experiment.sh baseline "$EXP" "$model"
  ./scripts/user_exps/run_experiment.sh usersim "$EXP" "$model"
done

# ── 7. Generate reports (auto-detects all models with data) ──
./scripts/user_exps/generate_reports.sh all "$EXP"
./scripts/user_exps/generate_reports.sh trajectories "$EXP"

# ── 8. (Optional) Drill into a single model's results ──
# python scripts/process_tac_usersim.py -e user_sim_"$EXP"_gemini_3_flash_default
# python scripts/process_tac_usersim.py -e user_sim_"$EXP"_gemini_3_flash_default --html

# ── 9. (Optional) Agent prompting strategy ablation ──
# Tests react / reflexion / plan_and_execute strategies.
# Uses the same filtered samples from Step 5.
#
# Minimal validation (1 model, 1 strategy, usersim only):
#   CONDITIONS="usersim" ./scripts/user_exps/run_experiment.sh ablation_agent_prompt "$EXP" gemini_3_flash react
#
# Single model, all strategies, both conditions:
#   ./scripts/user_exps/run_experiment.sh ablation_agent_prompt "$EXP" gemini_3_flash
#
# Generate ablation report:
#   MODELS='gemini_3_flash' ./scripts/user_exps/generate_reports.sh ablation "$EXP"
