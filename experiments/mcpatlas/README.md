# MCP Atlas Evaluation

Evaluation infrastructure for [MCP Atlas](https://github.com/scaleapi/mcp-atlas) - a benchmark for evaluating AI models' tool-use capabilities across 36 MCP servers with 500 tasks.

## Quick Start

### 1. Clone MCP Atlas Repository

```bash
cd lhaw/experiments/mcpatlas
# TODO: switch to scaleapi/mcp-atlas-lhaw once the proper fork is made
git clone git@github.com:sam-scale/mcp-atlas-lhaw-private.git mcp-atlas
cd mcp-atlas
git submodule update --init --recursive

# (Optional) Add upstream for tracking differences from original
git remote add upstream https://github.com/scaleapi/mcp-atlas.git
```

### 2. Setup Environment

Copy the environment template and add your API keys:

```bash
cp mcp-atlas/env.template mcp-atlas/.env
# Edit mcp-atlas/.env with your API keys
```

**Important:** Values in `.env` must NOT be quoted. Use:
```bash
AIRTABLE_API_KEY=pat2ue5qp1dPF5mmG...
```
Not:
```bash
AIRTABLE_API_KEY="pat2ue5qp1dPF5mmG..."  # Wrong! Quotes break JSON config
```

Required API keys depend on which tasks you want to run:
- **No setup needed (20 servers)**: calculator, wikipedia, filesystem, git, fetch, arxiv, etc.
- **API keys only (11 servers)**: GitHub, Google Maps, Brave Search, etc.
- **API keys + data imports (5 servers)**: Airtable, Notion, MongoDB, Slack, Google Calendar

### 3. Build & Run Services

```bash
# One-time: build the Docker image
cd experiments/mcpatlas/mcp-atlas && make build

# Terminal 1: Start MCP servers
cd experiments/mcpatlas/mcp-atlas && make run-docker

# Terminal 2: Start completion service
cd experiments/mcpatlas/mcp-atlas && make run-mcp-completion
```

**Two-phase service workflow:** The `ask_user` tool is registered at service startup
based on the `USER_TOOL_ENABLED` environment variable. This means:

1. **Baselines + underspec without ask_user** — start the service normally (above)
2. **Underspec with ask_user + persona ablations** — stop the service (Ctrl-C), then
   restart with: `USER_TOOL_ENABLED=True make run-mcp-completion`

Setting `USER_TOOL_ENABLED=True` only on the orchestrator command is **not sufficient**
— the service must be restarted with it for the tool to be available.

```bash
# Terminal 3: Run evaluation
python task_completion_mcpatlas.py \
    --format_name full_baseline \
    --backend_model opus_4_5 \
    --num_trials 3 \
    --max_k 3 \
    --concurrency 10
```

The dataset (`MCP-Atlas.csv`) is automatically downloaded from [HuggingFace](https://huggingface.co/datasets/ScaleAI/MCP-Atlas) on first run.

### CLI Options

| Argument | Description |
|----------|-------------|
| `--format_name` | Experiment name (required) |
| `--backend_model` | LLM model identifier (required) |
| `--input_csv` | Input CSV (default: `experiments/mcpatlas/MCP-Atlas.csv`) |
| `--task_ids` | Comma-separated task IDs to run |
| `--limit` | Limit number of tasks |
| `--start_index` | Start index (0-based) |
| `--end_index` | End index (exclusive) |
| `--num_trials` | Number of runs for pass@k (default: 3) |
| `--max_k` | Max k for pass@k metrics |
| `--concurrency` | Parallel completions (default: 10) |
| `--skip_scoring` | Skip LLM-as-judge scoring step |
| `--evaluator_model` | LLM model for evaluation scoring |

### Tool Response Caching (Optional)

<!-- TODO: remove internal Scale infra references (sgpml-cache, CodeArtifact) before OSS release -->

Enable Redis-based caching for MCP tool calls to reduce API costs and improve latency:

```bash
# Set up Scale PyPI access (required for sgpml-cache dependency)
AWS_PROFILE=ml-worker python ~/src/models/scripts_py3/scale_scripts/exe/maybe_refresh_codeartifact.py --export .codeartifact-pip-conf
export UV_INDEX=$(pip config get global.index-url)

# Run with caching enabled
TOOL_CACHE_ENABLED=True python task_completion_mcpatlas.py \
    --format_name baseline_v1 \
    --backend_model sonnet_4_5
```

| Variable | Description | Default |
|----------|-------------|---------|
| `TOOL_CACHE_ENABLED` | Enable tool response caching | `False` |
| `TOOL_CACHE_REDIS_URL` | Redis connection URL | *(must be set)* |
| `TOOL_CACHE_NAMESPACE` | Redis key namespace for isolation | `mcp_eval` |
| `TOOL_CACHE_TTL_DAYS` | Cache TTL in days | `365` |

Cache keys are deterministic (args serialized with `sort_keys=True`). Error responses are NOT cached. `ask_user` calls are never cached.


## Output Structure

```
experiments/mcpatlas/runs/
└── run_{format}_{model}_{timestamp}/
    ├── completions_*_{exp}_{ts}.csv       # Raw completions per experiment
    ├── scores_*_{exp}_{ts}/scored_*.csv   # Scores per experiment
    ├── results_{exp}.html                 # HTML visualization per experiment
    ├── evaluation_{format}_{model}.json   # Aggregated evaluation results
    ├── pass_k_per_task.csv                # pass@k / pass^k per task + AGGREGATE
    └── final_results/
        └── <task_id>/result.json          # Per-task results in our format
```

## How It Works

This orchestrator wraps the official MCP Atlas scripts:

1. **Docker Setup**: Starts `agent-environment` container (port 1984) for MCP servers and `mcp_eval` service (port 3000) for completions
2. **Completions**: Calls `mcp_completion_script.py` which runs agentic loops with tool calls
3. **Scoring**: Calls `mcp_evals_scores.py` which uses LLM-as-judge to evaluate completions against ground truth claims
4. **Conversion**: Converts MCP Atlas CSV output to our pass@k compatible format

## MCP Servers

### No Setup Required (20 servers)
calculator, wikipedia, filesystem, git, weather, fetch, whois, ddg-search, arxiv, pubmed, open-library, mcp-code-executor, and more.

### API Keys Required (11 servers)
exa, brave-search, oxylabs, github, google-maps, lara-translate, stripe, shopify, sendgrid, and more.

### API Keys + Data Import Required (5 servers)

These servers require accounts on external services, data imports, and API keys. The data exports are included in the cloned repo at `mcp-atlas/data_exports/`.

| Server | Setup Steps |
|--------|-------------|
| **Airtable** | Copy base from shared link in `airtable_database_online_link.txt`, set `AIRTABLE_API_KEY` |
| **Google Calendar** | Unzip `calendar_mcp_eval_export.zip`, import .ics file, set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN` |
| **Notion** | Unzip `notion_mcp_eval_export.zip`, import via Settings > Import, set `NOTION_TOKEN` |
| **MongoDB** | Unzip `mongo_dump_video_game_store.zip`, restore with `mongorestore`, set `MONGODB_CONNECTION_STRING` |
| **Slack** | Unzip `slack_mcp_eval_export.zip`, import to workspace, set `SLACK_MCP_XOXC_TOKEN`, `SLACK_MCP_XOXD_TOKEN` |

See [MCP Atlas data_exports README](https://github.com/scaleapi/mcp-atlas/tree/main/data_exports) for detailed instructions.

**Note:** Most tasks (~400/500) don't require these data imports. Use the default server filtering to run only tasks with available servers.

## Underspecified Variant Generation

Generate underspecified prompts from MCP Atlas tasks for strategic clarification testing.

### Quick Start

```bash
cd lhaw
source .venv/bin/activate && source .env

# Generate for tasks that passed baselines (JSON list of task IDs)
python scripts/generate_mcpatlas_underspec.py \
    passed_task_ids.json \
    --output-dir experiments/mcpatlas/underspec_output \
    --severity DELETE --max-level 2
```

### Options

| Argument | Description |
|----------|-------------|
| `--severity` | DELETE (default), VAGUIFY, or GENERICIZE |
| `--top-k` | Limit segments extracted (default: all) |
| `--variant-top-k` | Variants per task in CSV (default: 1) |
| `--max-level` | 1=single segment, 2=pairs |

### Output

```
underspec_output/
├── underspec_prompts.csv   # PROMPT = underspecified, + original_prompt, removed_value
├── original_prompts.csv    # PROMPT = original (for reference)
├── summary.json            # Run metadata
└── json/                   # Full segment/variant details per task
```

### Programmatic Usage

```python
from synthetic.pipeline import SyntheticPipeline, Severity

pipeline = SyntheticPipeline()
result = pipeline.process_mcpatlas_task("689f4d693e212e8ef3390720", severity=Severity.DELETE)

print(f"Segments: {len(result.segments)}")
for v in result.variants:
    print(v.underspecified_prompt)
```

---

## Full Underspecification Experiment Workflow

End-to-end pipeline for running underspecification experiments with the ask_user tool.

### Step 1: Get Baselines

Run baseline experiments on the full dataset:
```bash
python task_completion_mcpatlas.py \
    --format_name full_baseline \
    --backend_model anthropic/claude-opus-4-5-20251101 \
    --num_trials 3 \
    --max_k 3 \
    --concurrency 10
```

### Step 2: Prepare Underspec Dataset

Run the underspecification pipeline to create modified prompts:
```bash
python scripts/prepare_underspec_dataset.py configs/intersection_delete_2segments.yaml
```

### Step 3: Filter with Gemini

Re-run on Gemini to filter which underspec tasks are actually ambiguous:
```bash
python task_completion_mcpatlas.py \
    --format_name full_underspec_2_seg \
    --backend_model gemini_3_pro \
    --num_trials 3 \
    --max_k 3 \
    --concurrency 20 \
    --input_csv <underspec_output_dir>/underspec_prompts.csv
```

### Step 4: Split Results

Categorize results into: benign underspec, new task, outcome critical underspec:
```bash
uv run python scripts/process_underspec_run.py \
    --dataset-dir experiments/mcpatlas/underspec_datasets/<dataset_dir> \
    --run-dir <run_dir>
```

### Step 5: Sample Dataset

Create a sub-sampled CSV for focused evaluation:
```bash
python scripts/sample_from_filtered.py \
    --input <filtered_underspec_with_baseline.csv> \
    --output <sampled_filtered_underspec_with_baseline.csv>
```

### Step 6: Run Pass@3 Experiments

Re-run pass@3 for benchmarked models with and without ask_user tool:
```bash
# Without ask_user (run with normal completion service)
python task_completion_mcpatlas.py \
    --format_name sampled_underspecified \
    --backend_model anthropic/claude-opus-4-5-20251101 \
    --num_trials 3 \
    --max_k 3 \
    --concurrency 3 \
    --input_csv <sampled_filtered_underspec_with_baseline.csv>
```

> **Stop & restart the completion service** before running with ask_user:
> In Terminal 2, press Ctrl-C, then restart with `USER_TOOL_ENABLED=True make run-mcp-completion`

```bash
# With ask_user
USER_TOOL_ENABLED=True USE_SYSTEM_PROMPT_IN_COMPLETION=true \
  python task_completion_mcpatlas.py \
    --format_name sampled_underspecified_with_user \
    --backend_model anthropic/claude-opus-4-5-20251101 \
    --num_trials 3 \
    --max_k 3 \
    --concurrency 3 \
    --input_csv <sampled_filtered_underspec_with_baseline.csv>
```

### Step 7: Analyze Base Models

Generate pass@3 comparison plots for base benchmarked models:
```bash
python scripts/plot_pass3_comparison.py \
    --dataset-dir experiments/mcpatlas/underspec_datasets/<dataset_dir> \
    --config scripts/runs_config.yaml
```

### Step 8: Analyze Other Models

Generate comparison plots for additional models:
```bash
python scripts/plot_pass3_from_runs.py \
    --baseline-run <baseline_run_dir> \
    --no-user-run <underspec_run_dir> \
    --with-user-run <underspec_with_user_run_dir> \
    --output-dir <output_dir> \
    --model-name "Model Name" \
    --mapping-csv <sampled_filtered_underspec_with_baseline.csv>
```

### Step 9: Judge Ask-User Questions

Evaluate the quality of ask_user questions using LLM judging:
```bash
python scripts/analyze_ask_user.py \
    --run-dir <underspec_with_user_run_dir> \
    --output ask_user_results.csv \
    --judge-model gpt-4o
```

### Step 10: Analyze Ask-User Results

Generate analysis plots for ask_user question quality:
```bash
python scripts/plot_ask_user.py \
    --input <ask_user_results.csv> \
    --output-dir <output_dir> \
    --filter-passed false  # or true for passed-only
```

---

## Paper Results & Ablations

Instructions for reproducing results and ablation studies from the [LHAW paper](https://arxiv.org/abs/2602.10525).

### §4.2 Value of Information (Tables 2, 3, 6–8)

The main MCP-Atlas results are produced by the full workflow (Steps 1-11 above) or `run_mcpatlas_example.sh`. Below maps each paper table to the script that generates it:

| Paper Table | Content | Script / Step |
|-------------|---------|---------------|
| Table 2 | Pass@3 by number of segments removed | `underspec_pass3_by_segments.py` (Step 11) |
| Table 3 | Overall task success, user behavior stats, Gain/Q | `compute_mcpatlas_summary.py` (Step 12) |
| Table 6 | Pass@3 by information dimension | `plot_pass3_from_runs.py --mapping-csv` (uses dimension column from underspec CSV) |
| Table 7 | Avg checkpoint progress by dimension | Same as Table 6 (checkpoint data from scored CSVs) |
| Table 8 | Pass@3 by ambiguity class (MCP-Atlas) | `plot_pass3_from_runs.py --mapping-csv` → `pass3_by_ambiguity.png` (Step 9) |

**Producing Table 2 (segments removed breakdown):**
```bash
python experiments/mcpatlas/scripts/underspec_pass3_by_segments.py \
    <underspec_run_dir_1> <underspec_run_dir_2> ... \
    --base-path <underspec_output_dir> \
    --output pass3_by_segments.csv
```

**Producing Table 8 (ambiguity class breakdown):**
```bash
python experiments/mcpatlas/scripts/plot_pass3_from_runs.py \
    --baseline-run <baseline_run> \
    --no-user-run <underspec_run> \
    --with-user-run <underspec_ask_run> \
    --output-dir reports/ \
    --model-name "opus_4_5" \
    --mapping-csv <underspec_prompts.csv>
```
This produces both `pass3_comparison.png` (Table 3 bars) and `pass3_by_ambiguity.png` (Table 8 bars).

**Producing Tables 6-7 (dimension breakdown):**

The dimension information is in the underspec CSV (`information_dimension` column). Use `underspec_stats.py` to inspect the distribution:
```bash
python experiments/mcpatlas/scripts/underspec_stats.py \
    --input <underspec_output_dir>/filtered_underspec.csv
```

### §4.3 Cost of Information (Table 4)

Tests how user persona (perceived cost of asking) affects clarification behavior. Three personas append different system prompt suffixes:

| Persona | `USER_TYPE` value | Behavior |
|---------|-------------------|----------|
| Supervisor | `supervisor` | User is sitting with you, waiting to help. Do not guess. |
| Standard Assistant | `standard_assistant` | User is available but working on other tasks. |
| Busy Executive | `busy_executive` | User is very busy. Only interrupt for factual failures. |

Run the same underspec tasks with each persona:
```bash
for persona in supervisor standard_assistant busy_executive; do
  USER_TOOL_ENABLED=True \
  USE_SYSTEM_PROMPT_IN_COMPLETION=true \
  USER_TYPE=$persona \
    python task_completion_mcpatlas.py \
      --format_name "underspec_${persona}" \
      --backend_model sonnet_4_5 \
      --num_trials 3 --max_k 3 \
      --input_csv <sampled_underspec.csv> \
      --concurrency 10
done
```

**Prerequisites:**
- The completion service must be running with `USER_TOOL_ENABLED=True make run-mcp-completion` (stop and restart if it was started without this)

**Required env vars on the orchestrator command** (all three):
- `USER_TOOL_ENABLED=True` — enables the ask_user tool
- `USE_SYSTEM_PROMPT_IN_COMPLETION=true` — injects the system prompt (which includes persona text)
- `USER_TYPE` — selects the persona variant

Then compare Gain%/Q across personas using the same analysis pipeline (Steps 9-10).

### §4.4 Failure Modes of Clarifying Questions (Table 5)

Uses an LLM judge to classify ask_user questions into 7 failure categories with 3 sub-labels each (see paper Appendix I.1 for the full taxonomy).

```bash
# Run judging on a with-user experiment
python scripts/analyze_ask_user.py \
    --run-dir <underspec_with_user_run_dir> \
    --output ask_user_analysis.csv \
    --judge-model gpt-4o

# Visualize failure mode breakdown (Figure 5)
python scripts/plot_ask_user.py \
    --input ask_user_analysis.csv \
    --output-dir plots/

# Filter to only failed trials (Table 17)
python scripts/plot_ask_user.py \
    --input ask_user_analysis.csv \
    --output-dir plots/ \
    --filter-passed false
```

**Failure categories**: Question Quality, Question Targeting, Information Integration, Over-Clarification, Under-Clarification, Timing & Strategy, Response Misinterpretation.

---

## Analysis Utilities

Additional scripts in `experiments/mcpatlas/scripts/` for ad-hoc exploration and reporting:

| Script | Purpose |
|--------|---------|
| `underspec_viewer.py` | Interactive HTML viewer for exploring underspec variants (prompts, segments, trial traces) |
| `underspec_stats.py` | Print statistics (dimension, severity, segments) grouped by ambiguity class |
| `sample_from_filtered.py` | Sample from filtered variants with diversity constraints (class balance, task spread) |
| `generate_categorization_report.py` | HTML report showing tasks grouped by category with judge reasoning |
| `aggregate_passed_tasks.py` | Aggregate passed task IDs across model runs (union/intersection) |
| `categorize_underspec_tasks.py` | LLM-based categorization (overspecified/new_task/underspecified) — alternative to `process_underspec_run.py` |
| `compute_mcpatlas_summary.py` | Consolidated Table 3 summary: baseline/underspec/+user pass@3, Ask%, Gain/Q |
| `plot_pass3_comparison.py` | Config-file-driven pass@3 charts — alternative to `plot_pass3_from_runs.py` |
| `prepare_underspec_dataset.py` | Config-driven dataset preparation pipeline (uses `configs/*.yaml`) |
| `run_ask_user_analysis.sh` | Batch wrapper for `analyze_ask_user.py` + `plot_ask_user.py` across multiple runs |

---

## Troubleshooting

### Docker containers won't start

Check Docker is running and has sufficient memory (8GB+ recommended):
```bash
docker info
```

### Services not ready

Check container logs:
```bash
docker logs mcpatlas-agent-env
```

### API errors

Verify your .env file has the required API keys for the servers your tasks use.

