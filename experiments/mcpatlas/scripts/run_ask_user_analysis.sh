#!/bin/bash
# Batch analyze ask_user questions across multiple runs.
# Usage: Edit BASE and RUNS below, then run:
#   bash experiments/mcpatlas/scripts/run_ask_user_analysis.sh
set -e

# Point to your underspec dataset directory
BASE="${1:?Usage: $0 <underspec_dataset_dir>}"
JUDGE="anthropic/claude-sonnet-4-5-20250929"

# List the run directories to process (names only, not full paths)
RUNS=(
    # Example: "run_underspec_gpt_5_2_ask_20260128_044134"
    # Add your run directory names here
)

if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "No runs configured. Edit RUNS in this script or pass run dirs as args."
    echo "Usage: $0 <underspec_dataset_dir> [run_dir_names...]"
    exit 1
fi

cd "$(dirname "$0")"

for RUN in "${RUNS[@]}"; do
    echo "=== Processing $RUN ==="

    python analyze_ask_user.py \
        --run-dir "$BASE/$RUN" \
        --output "$BASE/$RUN/ask_user.csv" \
        --judge-model "$JUDGE"

    python plot_ask_user.py \
        --input "$BASE/$RUN/ask_user.csv" \
        --output-dir "$BASE/$RUN/ask_user/"

    echo "=== Done with $RUN ==="
    echo
done

echo "All runs complete!"
