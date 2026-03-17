# User Simulator Experiment Scripts

Using filtered (aka benchmark-ready) samples, we have two scripts: `run_experiment.sh` and `generate_reports.sh`.

## Quick Start

```bash
# Navigate to LHAW root, and run experiments
./scripts/user_exps/run_experiment.sh usersim v5         # With prompting
./scripts/user_exps/run_experiment.sh usersim_none v5    # Without prompting
./scripts/user_exps/run_experiment.sh baseline v4        # No user simulator

# Generate analysis
./scripts/user_exps/generate_reports.sh all v5
```

## Experiment Types

| Type | ask_user Tool | Agent Prompted |
|------|---------------|----------------|
| `usersim` | ✅ Available | ✅ Yes |
| `usersim_none` | ✅ Available | ❌ No |
| `baseline` | ❌ Not available | N/A |

## Options

```bash
# Single model
./scripts/user_exps/run_experiment.sh usersim v5 sonnet_4_5

# Override settings
NUM_TRIALS=5 ./scripts/user_exps/run_experiment.sh baseline v4
MODELS="sonnet_4_5 gpt_5_2" ./scripts/user_exps/generate_reports.sh all v5

# Debug trajectories
./scripts/user_exps/generate_reports.sh trajectories v5
```

## Notes

- Samples auto-loaded if needed
- `--resume` skips completed variants
- Sequential by default (`PARALLEL_VARIANTS=2` or higher to speed up if you have memory)
