# LHAW - Long-Horizon Augmented Workflows

Benchmark for evaluating LLM agents on **strategic clarification in underspecified workflows**.

**Core Question**: Can agents recognize when critical information is missing and ask the right clarifying questions before acting?

| Resource | Link |
|----------|------|
| **Paper** | [LHAW: Controllable Underspecification for Long-Horizon Tasks](https://arxiv.org/abs/2602.10525) |
| **Dataset** | [ScaleAI/lhaw on Hugging Face](https://huggingface.co/datasets/ScaleAI/lhaw) (285 variants, CC BY 4.0) |
| **Blog** | [Introducing LHAW](https://labs.scale.com/blog/LHAW) |

> **Note:** The TAC experiment infrastructure in this repo was adopted from [scaleapi/mrt](https://github.com/scaleapi/mrt) and the original codebase [TheAgentCompany/TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany).

---

## Overview

LHAW is a **dataset-agnostic synthetic pipeline** that transforms well-specified tasks into controllably underspecified variants by systematically removing information across four dimensions — Goals, Constraints, Inputs, and Context — at configurable severity levels.

**285 benchmark variants** across three domains:

| Domain | Source Benchmark | Tasks | Variants | Description |
|--------|-----------------|-------|----------|-------------|
| **TAC** | [TheAgentCompany](https://github.com/TheAgentCompany/TheAgentCompany) | 13 | 85 | Enterprise workflows (DS, Finance, HR, SDE) |
| **SWE-Bench Pro** | [SWE-Bench Pro](https://arxiv.org/abs/2509.16941) | 75 | 100 | Real-world GitHub issue repair |
| **MCP-Atlas** | [MCP-Atlas](https://arxiv.org/abs/2602.00933) | 100 | 100 | Tool-use across MCP server integrations |

### Pipeline

The end-to-end pipeline has four stages, each producing outputs consumed by the next:

1. **Baselines** — Run pass@k on original, well-specified tasks to establish success rates and golden trajectories.
2. **Underspec Variants** — Extract removable segments from task prompts, generate underspecified variants (delete/vaguify/genericize), run agent trials, and classify each variant as outcome-critical, divergent, or benign based on terminal state divergence.
3. **Filter Benchmark** — Select top candidates targeting a domain-specific distribution (TAC: 40/30/30, SWE-Bench: 50/30/20 outcome-critical / divergent / benign) to produce the final benchmark JSON.
4. **User Simulator** — Run clarification experiments where an LLM-powered user answers agent questions via MCP. Compare baseline (no `ask_user` tool) vs. user-sim (with tool) to measure the value of clarification.

> See [`synthetic/README.md`](synthetic/README.md) for pipeline data structures and programmatic usage.

---

## Setup

```bash
cd lhaw
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

cp .env.example .env   # edit with your credentials
source .env            # export LLM_API_KEY, LLM_BASE_URL, LLM_EVAL_MODEL
```

All scripts use centralized defaults from `constants.py`. Override via env vars (e.g. `NUM_TRIALS=3 PARALLEL_VARIANTS=2`).

**Run tests:**

```bash
python -m pytest tests -q
```

**Docker requirements:** Each agent trial spawns 2 Docker containers (OpenHands orchestrator + sandbox runtime). With `PARALLEL_VARIANTS=1` (default), only 2 containers run at a time. Increase `PARALLEL_VARIANTS` only if you have sufficient memory (each pair needs ~4-8 GB).

**Model shortcuts** (defined in `constants.py`):

| Provider | Shortcuts |
|----------|-----------|
| Anthropic | `opus_4_6`, `opus_4_5`, `sonnet_4_6`, `sonnet_4_5`, `sonnet_4`, `haiku_4_5` |
| OpenAI | `gpt_5_2`, `gpt_5_1`, `gpt_5`, `o3_pro`, `o3`, `gpt_4_1_mini` |
| Google | `gemini_3_pro`, `gemini_3_flash`, `gemini_3_1_pro`, `gemini_3_1_flash_lite` |
| Other | `kimi_k2`, `qwen3_235b`, `llama4_maverick`, `glm_4p5_air`, `nova_2_lite` |

---

## Reproduce Paper Results

Each domain has a self-contained example script for quick iteration and a detailed sub-README for full reproduction. The pipeline stages are the same across domains — only the agent backend and evaluation differ.

### TAC (TheAgentCompany)

| Report | Paper Table | Script |
|--------|-------------|--------|
| Overall performance summary | Table 3 | `generate_reports.sh all` |
| Pass@3 by information dimension | Table 6 | `generate_reports.sh all` |
| Avg checkpoint progress by dimension | Table 7 | `generate_reports.sh all` |
| Pass@3 by ambiguity class | Table 9 | `generate_reports.sh all` |
| Agentic prompting ablation | Table 11 | `generate_reports.sh ablation` |

**Quick start:** `run_tac_example.sh` (4 tasks, 3 models). Edit variables at top to customize.

**Full reproduction** (all 33 tasks, all models, pass@3):

```bash
# 1. Baselines + golden trajectories
./scripts/synth_pipeline/tac_full_baselines.sh

# 2. Generate + run underspec variants
./scripts/synth_pipeline/run_experiment.sh lhaw_tac

# 3. Process + filter benchmark (40/30/30)
python scripts/process_tac_underspec.py -e "lhaw_tac_*" --judge
./scripts/synth_pipeline/filter.sh lhaw_tac --max 85

# 4. User simulator experiments
./scripts/user_exps/run_experiment.sh baseline lhaw_tac
./scripts/user_exps/run_experiment.sh usersim lhaw_tac

# 5. Generate reports (Tables 3, 6, 7, 9, 11)
./scripts/user_exps/generate_reports.sh all lhaw_tac
./scripts/user_exps/generate_reports.sh ablation lhaw_tac
```

All stages support `--resume` to skip completed work and continue from where you left off. See `run_tac_example.sh` for the step-by-step flow with comments.

### SWE-Bench Pro

Requires Python 3.11+, SWE-agent (fork), and Modal for container orchestration.

| Report | Paper Table | Script |
|--------|-------------|--------|
| Overall performance summary | Table 3 | `compute_swebench_metrics.py` |
| Pass@3 by information dimension | Table 6 | `compute_swebench_metrics.py` |
| Avg checkpoint progress by dimension | Table 7 | `compute_swebench_metrics.py` |
| Pass@3 by ambiguity class | Table 10 | `compute_swebench_metrics.py` |

**Quick start:** `run_swebench_example.sh` (5 tasks, 3 models, 10 variants). Edit `DOCKERHUB_USER` at top of script.

**Full details:** [`experiments/swebench/README.md`](experiments/swebench/README.md) — setup, CLI reference, multi-model experiments, output structure, and troubleshooting.

### MCP-Atlas

Requires Docker and the MCP-Atlas repo with running MCP servers.

| Report | Paper Table | Script |
|--------|-------------|--------|
| Overall performance summary | Table 3 | `task_completion_mcpatlas.py` + `plot_pass3_from_runs.py` |
| Pass@3 by segments removed | Table 2 | `underspec_pass3_by_segments.py` |
| Pass@3 by information dimension | Table 6 | `plot_pass3_from_runs.py --mapping-csv` |
| Pass@3 by ambiguity class | Table 8 | `plot_pass3_from_runs.py --mapping-csv` |
| Ask-user failure modes | Table 5 | `analyze_ask_user.py` + `plot_ask_user.py` |
| User persona ablation | Table 4 | `analyze_ask_user.py` (per persona) |

**Quick start:** `run_mcpatlas_example.sh` (15 tasks, 3 models, default LIMIT=15). Edit `BASELINE_MODELS` to control task selection.

**Service note:** The script runs in two phases. Baselines and underspec-without-ask run with the normal completion service. Before ask_user experiments, stop the service and restart with `USER_TOOL_ENABLED=True make run-mcp-completion`. See script comments for details.

**Full details:** [`experiments/mcpatlas/README.md`](experiments/mcpatlas/README.md) — setup (API keys, data imports, MCP server list), CLI reference, paper ablations, and troubleshooting.

---

## Using Pre-computed Variants from HuggingFace

The [ScaleAI/lhaw](https://huggingface.co/datasets/ScaleAI/lhaw) dataset contains all 285 benchmark variants used in the paper. Using it lets you **skip the generation phase** (segment extraction, variant generation, empirical validation) and go straight to running baselines + evaluation with the existing run scripts on the validated underspecified variants.

```bash
# Load variants for any benchmark:
python scripts/load_hf_dataset.py --dataset MCP-Atlas --output-dir experiments/mcpatlas/underspec_output/hf_variants
python scripts/load_hf_dataset.py --dataset TheAgentCompany --output-dir experiments/agentcompany/hf_variants
python scripts/load_hf_dataset.py --dataset "SWE-Bench Pro" --output-dir experiments/swebench/hf_variants
```

Then modify the run script variables to point at the loaded data and skip generation:

| Benchmark | Run script | What to change | Steps to skip |
|-----------|-----------|----------------|---------------|
| **MCP-Atlas** | `run_mcpatlas_example.sh` | Set `UNDERSPEC_DIR`, `BASELINE_MODELS`, use `--task_ids` | Steps 2-3, 5 |
| **TAC** | `run_tac_example.sh` | Set `TASKS` from HF, set `FILTERED_JSON` | Steps 1b-5 |
| **SWE-Bench Pro** | `run_swebench_example.sh` | Copy YAMLs into `EXP_DIR`, drop `--skip-baseline` | Steps 1-2 |

**MCP-Atlas:** Step 4 feeds underspec prompts via `--input_csv "$UNDERSPEC_DIR/underspec_prompts.csv"`. Set `BASELINE_MODELS=("${MODELS[@]}")` so all model baselines run in step 1 (step 5 depends on `PASSED_JSON` from the skipped step 2, so skip it too):

```bash
# In run_mcpatlas_example.sh, change:
UNDERSPEC_DIR="experiments/mcpatlas/underspec_output/hf_variants"
BASELINE_MODELS=("${MODELS[@]}")
# In step 1, replace --limit "$LIMIT" with:
TASK_IDS=$(python3 -c "import pandas as pd; print(','.join(pd.read_csv('$UNDERSPEC_DIR/underspec_prompts.csv')['TASK'].unique()))")
# --task_ids "$TASK_IDS"
# Skip steps 2-3, 5, 8, and 11. Continue from step 4.
# (Steps 8 and 11 require json/ from the generation phase.)
```

**TAC:** Step 6 (`run_experiment.sh`) loads underspec prompts via `--load-samples "$FILTERED_JSON"`, so pointing `FILTERED_JSON` at the HF benchmark hooks them in. Extract the task list from it for step 1 baselines:

```bash
# Before running run_tac_example.sh:
HF_DIR="experiments/agentcompany/hf_variants"
TASKS=($(python3 -c "import json; print(' '.join(sorted(set(v['task'] for v in json.load(open('$HF_DIR/benchmark.json'))))))"))
# Use TASKS in step 1, then for step 6:
export FILTERED_JSON="$HF_DIR/benchmark.json"
```

**SWE-Bench Pro:** Step 3 (`--run --exp-dir`) reads `instances.yaml` for both baselines and underspec trials. Since steps 1-2 are skipped, **remove `--skip-baseline`** from step 3 so baselines run against `original_instances.yaml`:

```bash
EXP_DIR="experiments/swebench/runs/run_hf_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EXP_DIR"
cp experiments/swebench/hf_variants/{instances,original_instances}.yaml "$EXP_DIR/"
cp experiments/swebench/hf_variants/underspec_candidates.csv "$EXP_DIR/"
# In step 3, remove --skip-baseline. Continue steps 3 → 4 → 5 → 9.
```

> **Note:** To generate new variants with different parameters (severity, tasks, segment count), run the full generation phase using the run scripts instead.

---

## Taxonomy Dimensions

Segments are classified by 4 dimensions (see `synthetic/configs/taxonomy.yaml`):

| Dimension | Description | Subdimensions |
|-----------|-------------|---------------|
| **GOAL** | What to produce | objective, identifier, format |
| **CONSTRAINT** | How to do it | procedure, criteria, deadline |
| **INPUT** | From where | location, tool/API, reference |
| **CONTEXT** | Background info | domain knowledge, history |

Each segment has:
- **criticality**: 0.0 / 0.5 / 1.0 (how important for success)
- **guessability**: 0.0 / 0.5 / 1.0 (how likely to guess correctly)
- **priority_score**: `criticality × (1 - guessability)` (higher = better for removal)

---

## Metrics

**Task Performance:**

| Metric | Formula | Meaning |
|--------|---------|---------|
| **pass@k** | 1 - C(n-c,k)/C(n,k) | P(>=1 success in k trials), unbiased estimator |
| **pass^k** | C(c,k)/C(n,k) | P(k consecutive successes) |
| **Ckpt%** | avg(score/total) per trial | Average checkpoint progress (partial completion) |

Where n=trials, c=successes. Uses the unbiased combinatorial estimator from [HumanEval (Chen et al., 2021)](https://arxiv.org/abs/2107.03374).

**Clarification Behavior (Table 3):**

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Ask%** | trials_with_ask_user / total_trials | Fraction of trials invoking the user tool |
| **Avg/Traj** | total_questions / trials_with_questions | Mean questions per trajectory (among those that asked) |
| **Gain/Q** | delta_pass@3 / total_questions | Performance gain per question asked |

**Ambiguity Classification:**

| Class | Definition | Oracle Decision |
|-------|------------|-----------------|
| **outcome-critical** | 0/N success + divergent terminal states | CLARIFY |
| **divergent** | Some success (1-2/3) + variable outcomes | PROCEED |
| **benign** | N/N success despite missing info | PROCEED |
| **new_task** | 0/N + 1 state (LLM judged as different task) | FILTERED OUT |

---

## Ablations

### Agent Prompting Strategies (Table 11)

| Strategy | Description | File |
|----------|-------------|------|
| `none` | Baseline — no strategy instructions | — |
| `react` | Thought -> Action -> Observation cycle | `experiments/agentcompany/openhands/agent_strategies/react.md` |
| `reflexion` | Act -> Self-Assess -> Decide loop | `experiments/agentcompany/openhands/agent_strategies/reflexion.md` |
| `plan_and_execute` | Plan -> Execute -> Re-plan | `experiments/agentcompany/openhands/agent_strategies/plan_and_execute.md` |

### Severity Levels (Table 12)

| Severity | Effect | Example |
|----------|--------|---------|
| `delete` | Remove entirely | "Save to `report.xlsx`" -> _(gone)_ |
| `vaguify` | Replace with vague placeholder | "Save to `report.xlsx`" -> "Save to the file" |
| `genericize` | Replace with generic value | "Save to `report.xlsx`" -> "Save to `output.xlsx`" |

### Segment Extraction Grounding

| Flag | Grounding |
|------|-----------|
| _(default)_ | Prompt + trajectory + checkpoints (recommended) |
| `--no-trajectory` | Prompt + checkpoints |
| `--no-checkpoints` | Prompt + trajectory |
| `--prompt-only` | Prompt only |

---

## File Structure

```
lhaw/
├── README.md
├── run_tac_example.sh                 # TAC mini reproduction (4 tasks, 3 models)
├── run_swebench_example.sh            # SWE-Bench Pro mini reproduction (5 tasks, 3 models)
├── run_mcpatlas_example.sh            # MCP-Atlas mini reproduction (15 tasks, 3 models)
├── constants.py                       # Centralized defaults + model registry
├── task_completion_agentcompany.py    # TAC orchestrator
├── task_completion_mcpatlas.py        # MCP-Atlas orchestrator
├── scripts/
│   ├── synth_pipeline/                # TAC-specific shell orchestration
│   │   ├── run_experiment.sh          # Generate + run underspec trials
│   │   ├── filter.sh                  # Process + filter benchmark
│   │   ├── tac.sh                     # Run TAC baseline (single model)
│   │   └── tac_full_baselines.sh      # Run TAC baseline (all models)
│   ├── user_exps/                     # TAC-specific user simulator scripts
│   │   ├── run_experiment.sh          # User simulator experiments
│   │   ├── run_ablation_agent_prompt.sh
│   │   └── generate_reports.sh        # Reports (all / ablation / trajectories)
│   ├── run_tac_underspec.py           # TAC underspec runner (generate + run trials)
│   ├── process_tac_underspec.py       # TAC results processing + LLM judge
│   ├── process_tac_usersim.py         # TAC user simulator metrics
│   ├── compare_tac_conditions.py      # TAC cross-condition comparison
│   ├── filter_tac_samples.py           # TAC benchmark filtering (40/30/30)
│   ├── filter_passed_tasks.py          # Cross-benchmark baseline task selection
│   ├── summarize_swebench_baselines.py # SWE-Bench baseline pass@k summary
│   ├── filter_swebench_samples.py     # SWE-Bench quota filter
│   ├── process_swebench_underspec.py  # SWE-Bench eval + classify
│   ├── export_swebench_dataset.py     # SWE-Bench benchmark export
│   ├── compute_swebench_metrics.py    # SWE-Bench ICML tables
│   ├── compute_phase_b_results.py     # SWE-Bench cross-model comparison
│   ├── generate_mcpatlas_underspec.py # MCP-Atlas underspec variant generation
│   ├── load_hf_dataset.py            # Load pre-computed variants from HuggingFace
│   ├── view_tac_trajectory.py         # TAC trajectory HTML viewer
│   └── export_tac_golden_trajectories.py  # TAC golden trajectory export
├── task_completion_swebench.py        # SWE-Bench orchestrator
├── synthetic/                         # Pipeline internals (see synthetic/README.md)
│   └── adapters/
│       ├── tac.py                     # TAC adapter
│       ├── swebench.py               # SWE-Bench adapter
│       └── mcpatlas.py               # MCP-Atlas adapter
├── evaluation/                        # Scoring, pass@k, tac_eval.py (checkpoint grader CLI)
├── task_pairs_agentcompany/           # 33 TAC task definitions
├── swebenchpro/                       # SWE-Bench Pro data (git submodule)
├── experiments/
│   ├── agentcompany/                  # TAC: OpenHands agent + MCP user sim + runs/
│   ├── swebench/                      # SWE-Bench: README.md + runs/
│   └── mcpatlas/                      # MCP-Atlas: README.md + configs/ + scripts/ + runs/
└── tests/                             # pytest suite
```

---

## Citation

```bibtex
@misc{pu2026lhawcontrollableunderspecificationlonghorizon,
      title={LHAW: Controllable Underspecification for Long-Horizon Tasks},
      author={George Pu and Michael S. Lee and Udari Madhushani Sehwag and David J. Lee and Bryan Zhu and Yash Maurya and Mohit Raghavendra and Yuan Xue and Samuel Marc Denton},
      year={2026},
      eprint={2602.10525},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.10525},
}
```
