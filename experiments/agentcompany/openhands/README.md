# Running Agent Trials with OpenHands

This module runs multiple TAC task trials in parallel using the [OpenHands](https://github.com/All-Hands-AI/OpenHands) framework. Two modes are supported:

* **Baseline**: CodeActAgent without additional tools
* **With MCP Tools**: Agent gets access to custom tools (e.g., `ask_user` for clarification)

## Prerequisites

1. **Python 3.10+** with required packages (`python -m pip install -r requirements.txt`)
2. **Docker** — OpenHands runs agents inside Docker containers
3. **OpenHands container images** — Pulled automatically from `docker.openhands.dev`

## Installation

1. Clone the repository and navigate to this directory:
   ```bash
   cd experiments/agentcompany/openhands
   ```
2. Install Python dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Set up environment variables (see [Environment Setup](#environment-setup) below)

## Code Structure

- `main.py`: Entry point — parses CLI args and orchestrates parallel execution
- `utils.py`: Workspace preparation, result formatting, Docker management
- `task.py`: Core task execution logic and OpenHands container orchestration
- `task_completion_agentcompany.py`: Deterministic evaluation of agent trajectories
- `mcp_tools/`: Custom MCP tools that extend agent capabilities — see [mcp_tools/README.md](mcp_tools/README.md)

## Setup Guide

### Environment Setup

Create a `.env` file with the following variables:

```bash
# Required
LLM_BASE_URL=<your-litellm-or-api-base-url>
LLM_API_KEY=<your-api-key>

# OpenHands version (default: 1.1)
OPENHANDS_VERSION=1.1
```

### Task Data

Tasks are loaded from `task_pairs_agentcompany/{task_name}/`. Each task directory contains:
- `task.json` — Task definition (prompt, checkpoints, etc.)
- `evaluator.py` — Deterministic evaluation script

## Usage

### Via Shell Script

```bash
bash run.sh --tasks-file tasks.txt --model anthropic/claude-sonnet-4-20250514 \
  --max-workers 4 --max-iterations 100 --max-budget-per-task 10
```

### Via Python

```bash
python main.py --tasks-file tasks.txt --model openai/gpt-4o \
  --max-workers 4 --max-iterations 100 --max-budget-per-task 10
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gpt-4o` | Model identifier |
| `--tasks-file` | required | File with task IDs, one per line |
| `--max-workers` | `4` | Parallel task limit |
| `--max-iterations` | `100` | Max agent iterations per task |
| `--max-budget-per-task` | `10.0` | Max cost in USD per task |
| `--cleanup-containers` | `both` | When to clean Docker containers: `none`, `before`, `after`, `both` |
| `--safe-mode` | off | Disables container cleanup (recommended for concurrent runs) |

Run with `--help` for the complete list.

## Output Structure

```
runs/run_{format_name}_{YYYYMMDD_HHMMSS}_{uuid8}/
├── metadata.json
├── tasks.txt
├── local_results/{task}/openhands_{model}/{date_sessionId}/
│   ├── workspace/
│   └── state/sessions/{uuid}/events/{0..N}.json
├── final_results/
│   ├── {task}/session_{i+1}_{uuid}.json
│   ├── deterministic/{task}/{model}/evaluation_{i}.json
│   └── summaries/pass_k_summary_{format_name}_{model}.csv
```

## Cost Management

1. Set `--max-budget-per-task` to a reasonable limit
2. Start with a small subset of tasks to gauge token usage
3. Use cheaper models for testing before expensive ones

## Troubleshooting

| Issue | Check |
|-------|-------|
| API access | Verify `LLM_BASE_URL` and `LLM_API_KEY` in `.env` |
| Docker errors | Ensure Docker is running; check for port conflicts |
| Task not found | Verify `task_pairs_agentcompany/{task_name}/task.json` exists |
| MCP tool missing | Check that `config.toml` has `enable_mcp = true` |
