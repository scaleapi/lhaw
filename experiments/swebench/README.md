# SWE-bench Pro Pipeline

End-to-end pipeline for generating, running, evaluating, and classifying underspecified SWE-bench Pro variants.

## Setup

Follow the environment setup in the [root README](../../README.md#setup), then activate and install SWE-bench-specific dependencies:

```bash
source .venv/bin/activate

# SWE-bench Pro + SWE-agent (submodules)
git submodule sync
git submodule update --init swebenchpro/SWE-bench_Pro-os
cd swebenchpro/SWE-bench_Pro-os && git submodule update --init SWE-agent

# Switch SWE-agent to ask_user fork branch
cd SWE-agent
git fetch origin
git checkout -b lhaw/ask-user-tool origin/lhaw/ask-user-tool
cd ../../..

# Install SWE-agent
uv pip install -e swebenchpro/SWE-bench_Pro-os/SWE-agent

# Modal auth (for container deployment)
modal token new
```

Source `.env` before every session:
```bash
set -a && source .env && set +a
```

## Pipeline Stages

### Stage 1: Baselines + golden trajectories

Run baseline SWE-agent on original tasks and export `.traj` files for grounded segment extraction.
Mirrors TAC step 1 (`tac.sh` + `export_tac_golden_trajectories.py`).
Results are written to `baseline_N/` directories (not `exp_N/`) so they coexist with Stage 3's underspec trials in the same directory — no copying needed.

See `run_swebench_example.sh` step 1 for the full commands.

**Produces:**
- `baseline_1/`, `baseline_2/`, `baseline_3/` — baseline trial results (preds.json, trajectories)
- `experiments/swebench/golden_trajectories/<instance_id>/*.traj` — golden trajectories for Stage 2

### Stage 2: Generate variants

Runs the synthetic pipeline: F2P filter, segment extraction, uniform sampling, variant generation.
Uses golden trajectories from Stage 1 for grounded extraction (degrades gracefully without them).
Use `--runs-dir <exp_dir>` to write into Stage 1's directory so everything shares one exp_dir.

```bash
python task_completion_swebench.py --generate \
    --format_name trial_v1 \
    --runs-dir <exp_dir> \
    --severity delete \
    --target-variants 200 \
    --trajectory-dir experiments/swebench/golden_trajectories
```

**Adds to `<exp_dir>/`:**
- `instances.yaml` — SWE-agent instance definitions for all variants
- `original_instances.yaml` — original task instances for baselines
- `underspec_candidates.csv` — variant metadata (dimensions, segments, criticality)
- `config.json` — run configuration (overwrites Stage 1's)

### Stage 3: Run SWE-agent trials

Runs SWE-agent on each variant across N trials for pass@k evaluation.
Since Stage 1 wrote baselines to `baseline_N/` in the same directory, use `--skip-baseline`.

```bash
python task_completion_swebench.py --run \
    --exp-dir <exp_dir> \
    --backend_model gpt_5_2 \
    --num_trials 3 \
    --skip-baseline \
    --concurrency 10
```

**Produces:** `exp_1/`, `exp_2/`, `exp_3/` (underspec results) alongside the existing `baseline_1/`, `baseline_2/`, `baseline_3/` (from Stage 1), each containing per-instance SWE-agent trajectories and `preds.json`.

### Stage 4: Evaluate predictions

Runs SWE-bench Pro Docker evaluation on all patches. Handles both variant (`exp_N/`) and baseline (`baseline_N/`) predictions.

```bash
# Evaluate only (no classification)
python scripts/process_swebench_underspec.py \
    --exp-dir experiments/swebench/runs/run_trial_v1_<timestamp> \
    --run-eval --eval-only \
    --dockerhub-username jefzda

# Evaluate + classify in one step
python scripts/process_swebench_underspec.py \
    --exp-dir experiments/swebench/runs/run_trial_v1_<timestamp> \
    --run-eval --dockerhub-username jefzda --judge
```

**Produces:** `exp_N/eval_results/` and `baseline_N/eval_results/` directories with per-instance `*_output.json` files.

### Stage 5: Classify variants

Computes F2P terminal states across trials and classifies each variant.

```bash
python scripts/process_swebench_underspec.py \
    --exp-dir experiments/swebench/runs/run_trial_v1_<timestamp> \
    --judge  # use LLM judge for new_task detection
```

**Produces:** `underspec_results.csv` with per-variant classification (outcome-critical, divergent, benign, new_task).

### Stage 6: Filter to quotas

Selects a balanced subset matching target quotas (default: 50% OC / 30% divergent / 20% benign).

```bash
python scripts/filter_swebench_samples.py \
    --input experiments/swebench/runs/run_trial_v1_<timestamp>/underspec_results.csv \
    --max-total 100
```

**Produces:** `underspec_results_filtered.csv`.

### Stage 7: Export benchmark JSON

Exports the final benchmark dataset with terminal states.

```bash
python scripts/export_swebench_dataset.py \
    --input experiments/swebench/runs/run_trial_v1_<timestamp>/underspec_results_filtered.csv \
    --exp-dir experiments/swebench/runs/run_trial_v1_<timestamp> \
    --swe-bench-csv swebenchpro/SWE-bench_Pro-os/swe_bench_pro_full.csv
```

**Produces:** `lhaw_benchmark_swebench.json`.

## Multi-Model Experiments

### Phase A: Build the benchmark (single model)

Run the full pipeline (stages 1-5) with one model to establish ground-truth labels:

```bash
# Generate + run
python task_completion_swebench.py --generate --run --format_name exp_v1 --severity delete --target-variants 200 --num_trials 3

# Process + filter
python scripts/process_swebench_underspec.py --exp-dir <exp_dir> --run-eval --dockerhub-username jefzda --judge
python scripts/filter_swebench_samples.py --input <exp_dir>/underspec_results.csv --max-total 100
```

### Freeze variants

Lock the filtered set for cross-model comparison:

```bash
python task_completion_swebench.py --freeze \
    --exp-dir <exp_dir> \
    --filtered-csv <exp_dir>/underspec_results_filtered.csv
```

**Produces:** `frozen_instances.yaml` (exactly 100 variants).

### Phase B: Evaluate other models

Run each model on the frozen variants:

```bash
# Regular mode (short names resolved via constants.py)
python task_completion_swebench.py --run \
    --exp-dir <exp_dir> \
    --model-suffix sonnet_4_5 \
    --backend_model sonnet_4_5 \
    --num_trials 3

# Ask-user mode (with clarification tool, baselines already exist)
python task_completion_swebench.py --run \
    --exp-dir <exp_dir> \
    --model-suffix sonnet_4_5_ask \
    --backend_model sonnet_4_5 \
    --ask-user \
    --skip-baseline \
    --num_trials 3
```

Each `--model-suffix` creates a sibling directory: `<exp_dir>_<suffix>/`.

### Evaluate + process per model

```bash
python scripts/process_swebench_underspec.py \
    --exp-dir <exp_dir>_sonnet_4_5 \
    --run-eval --dockerhub-username jefzda
```

### Compute metrics

```bash
# ICML tables (all models, dimension breakdowns, LaTeX)
# Models are auto-detected from sibling directories (<exp_dir>_<suffix>/)
python scripts/compute_swebench_metrics.py --exp-dir <exp_dir>

# Phase B cross-model comparison
python scripts/compute_phase_b_results.py --exp-dir <exp_dir>
```

Both scripts default to `experiments/swebench/runs/run_exp_v1_20260208_165126` if `--exp-dir` is omitted.

## CLI Reference

### `task_completion_swebench.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--generate` | - | Stage 2: generate variants |
| `--run` | - | Stage 3: run SWE-agent trials |
| `--prepare-baselines` | - | Stage 1: run baselines + export golden trajectories |
| `--exp-dir` | - | Experiment directory (for `--run`) |
| `--format_name` | - | Experiment name (required for `--generate`) |
| `--tasks-file` | - | JSON file with task list |
| `--instance-id` | - | Single instance ID to run |
| `--limit` | - | Limit number of tasks |
| `--backend_model` | `gpt_5_2` | Agent model — short name (e.g. `gpt_5_2`, `sonnet_4_6`) or full LiteLLM identifier |
| `--num-trials` | 3 | Number of trials for pass@k |
| `--concurrency` | 10 | Parallel SWE-agent workers |
| `--generate-concurrency` | 10 | Parallel LLM workers for `--generate` |
| `--startup-timeout` | 600 | Container startup timeout (seconds) |
| `--runtime-timeout` | 900 | Container runtime timeout (seconds) |
| `--dockerhub-username` | `jefzda` | DockerHub username for SWE-bench Pro images |
| `--runs-dir` | auto | Override output directory |
| `--trajectory-dir` | `experiments/swebench/golden_trajectories` | Trajectory dir for grounded extraction |
| `--severity` | `delete` | Removal strategy: delete/vaguify/genericize |
| `--f2p-threshold` | 2 | Minimum F2P tests for task selection (i.e. F2P > 2) |
| `--target-variants` | 200 | Target number of variants |
| `--max-level` | 2 | Max segments to remove (1=single, 2=pairs) |
| `--top-k-per-level` | all | Limit combinations per segment level |
| `--ask-user` | false | Enable ask_user clarification tool |
| `--reasoning-effort` | - | Thinking level: low/medium/high |
| `--dry-run` | false | Generate files only, don't run SWE-agent |
| `--skip-baseline` | false | Skip baseline runs |
| `--model-suffix` | - | Per-model directory tag |
| `--freeze` | false | Create frozen_instances.yaml from filtered CSV |
| `--filtered-csv` | - | Path to filtered CSV (for `--freeze`) |

### `scripts/process_swebench_underspec.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp-dir` | **required** | Experiment directory with exp_N/ dirs |
| `--output`, `-o` | `{exp_dir}/underspec_results.csv` | Output CSV path |
| `--run-eval` | false | Run Docker evaluation before processing |
| `--dockerhub-username` | - | DockerHub username (required with `--run-eval`) |
| `--num-workers` | 50 | Parallel eval workers |
| `--eval-only` | false | Only run evaluation, skip classification |
| `--judge` | false | Use LLM judge for new_task classification |
| `--judge-model` | `gemini/gemini-3-flash-preview` | LLM model for judging |

### `scripts/filter_swebench_samples.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | **required** | Input CSV from process step |
| `--output`, `-o` | `{input}_filtered.csv` | Output filtered CSV |
| `--max-total` | 100 | Maximum total samples |
| `--max-per-task` | 8 | Max samples per original task (OC bypasses) |
| `--quotas` | `50,30,20` | OC,divergent,benign quotas |
| `--max-per-instance` | - | Max samples per SWE-bench instance |
| `--max-per-dimension` | - | Max samples per underspec dimension |
| `--include-new-task` | false | Include new_task class |
| `--critical-only` | false | Only keep outcome-critical |
| `--percentage-caps` | false | Use percentage caps instead of exact counts |
| `--analyze` | false | Analyze distribution without filtering |
| `--seed` | - | Random seed |

### `scripts/export_swebench_dataset.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--input`, `-i` | **required** | Input CSV/JSON from filter step |
| `--output`, `-o` | `lhaw_benchmark_swebench.json` | Output JSON path |
| `--pre-trial` | false | Export from candidates (no classification) |
| `--exp-dir` | - | Experiment dir (for terminal states) |
| `--swe-bench-csv` | - | Path to swe_bench_pro_full.csv |

### `scripts/compute_swebench_metrics.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp-dir` | *(required)* | Base experiment directory |

### `scripts/compute_phase_b_results.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--exp-dir` | *(required)* | Base experiment directory |

## Output Structure

```
experiments/swebench/runs/run_<name>_<timestamp>/
├── config.json                   # Run configuration
├── instances.yaml                # Variant instance definitions
├── underspec_candidates.csv      # Variant metadata (from --generate)
├── frozen_instances.yaml         # Locked variants (from --freeze)
├── underspec_results.csv         # Classification results
├── underspec_results_filtered.csv # Filtered subset
├── icml_metrics.json             # Raw metrics output
├── exp_1/                        # Trial 1
│   ├── <instance_id>/            # Per-instance SWE-agent output
│   │   └── *.traj               # Trajectory file
│   ├── preds.json               # All predictions
│   ├── patches_for_eval.json    # Eval input
│   └── eval_results/            # Docker eval output
│       └── <original_id>/
│           └── trial1__V_<suffix>_output.json
├── exp_2/                        # Trial 2
├── exp_3/                        # Trial 3
├── baseline_1/                   # Baseline trial 1
│   ├── preds.json
│   └── eval_results/
│       └── <instance_id>/
│           └── baseline1_output.json
├── baseline_2/
└── baseline_3/
```

Model-suffix directories are siblings:
```
runs/
├── run_exp_v1_20260208_165126/           # GPT-5.2 (main)
├── run_exp_v1_20260208_165126_sonnet_4_5/
├── run_exp_v1_20260208_165126_sonnet_4_5_ask/
├── run_exp_v1_20260208_165126_gemini_3_flash/
└── ...
```

## Troubleshooting

### Container timeouts

Some instances (notably qutebrowser, tutanota) routinely timeout in Modal Docker. Increase limits:
```bash
--startup-timeout 600 --runtime-timeout 1800
```

If an eval hangs beyond 30 minutes, it's safe to kill and re-run (existing results are preserved).

### Modal authentication

```bash
modal token new
```

If Modal tokens expire mid-run, the eval will fail with auth errors. Re-authenticate and re-run with `--run-eval` (already-evaluated instances are skipped).

### SIGPIPE errors

SWE-agent occasionally logs SIGPIPE errors when the container process exits. These are harmless and don't affect results.

### Null byte corruption

The LLM occasionally corrupts `\xa0` (non-breaking space) into `\x00` (null byte) during variant generation. The pipeline sanitizes these automatically. If you see null byte errors in downstream tools, re-run `--generate`.

### Missing `sweagent` command

```bash
python -m pip install -e swebenchpro/SWE-bench_Pro-os/SWE-agent
```

### preds.json overwrite bug

SWE-agent's `run-batch` overwrites `preds.json` on rerun (known bug). If you need to reconstruct, build from individual `.pred` files:
```python
import json
from pathlib import Path

preds = {}
for pred_file in Path("exp_1").glob("*/*.pred"):
    with open(pred_file) as f:
        data = json.loads(f.read())  # .pred files contain JSON text
    preds[data["instance_id"]] = data
```

## Notes

- **`compute_swebench_metrics.py` auto-detects models** from sibling directories of `--exp-dir`. No manual model list editing is needed.
- **`compute_phase_b_results.py` overlaps with `compute_swebench_metrics.py`**. The phase-B script is a lighter quick-summary tool; the metrics script is the authoritative source for paper tables. Both read from the same experiment directory.
- **Subset runs work correctly**. The pipeline is fully data-driven through `instances.yaml` — using `--limit N` in steps 1-2 properly scopes all downstream steps to that subset.
