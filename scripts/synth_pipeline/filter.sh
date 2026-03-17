#!/bin/bash
# Create benchmark from experiment results (process + filter)
#
# Usage:
#   ./filter.sh <experiment> [--max N] [--analyze]
#
# Examples:
#   ./filter.sh v4                  # Process + filter
#   ./filter.sh v4 --max 150        # Filter to 150 samples
#   ./filter.sh v4 --analyze        # Just analyze distribution (no output)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$(dirname "$SCRIPT_DIR")")"

EXPERIMENT="${1:?Usage: $0 <experiment> [--max N] [--analyze]}"
shift

# Defaults
MAX_SAMPLES="${MAX_SAMPLES:-100}"
ANALYZE_ONLY=""

# Parse options
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max) MAX_SAMPLES="$2"; shift 2 ;;
        --analyze) ANALYZE_ONLY="1"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

DOMAINS=("ds" "finance" "hr" "sde")
SYNTH_OUTPUTS="synthetic/outputs"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 LHAW Benchmark Filtering"
echo "   Experiment: $EXPERIMENT"
echo "   Target: $MAX_SAMPLES samples (40/30/30 split)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check that at least one domain exists
FOUND_ANY=""
for domain in "${DOMAINS[@]}"; do
    if [[ -d "$SYNTH_OUTPUTS/${EXPERIMENT}_${domain}" ]]; then
        FOUND_ANY="1"
        break
    fi
done

if [[ -z "$FOUND_ANY" ]]; then
    echo "❌ No experiments found for $EXPERIMENT"
    echo "   Expected: $SYNTH_OUTPUTS/${EXPERIMENT}_{ds,finance,hr,sde}/"
    exit 1
fi

echo ""
if [[ -n "$ANALYZE_ONLY" ]]; then
    echo "📊 Analyzing distribution..."
    python scripts/filter_tac_samples.py \
        --experiment "$EXPERIMENT" \
        --analyze
else
    echo "🔍 Filtering to $MAX_SAMPLES samples..."
    mkdir -p "$SYNTH_OUTPUTS/benchmark_filtered"
    OUTPUT="$SYNTH_OUTPUTS/benchmark_filtered/${EXPERIMENT}_filtered.csv"
    
    python scripts/filter_tac_samples.py \
        --experiment "$EXPERIMENT" \
        --max-per-task 10 \
        --max-total "$MAX_SAMPLES" \
        --output "$OUTPUT"
    
    # Derive JSON path from CSV path
    JSON_OUTPUT="${OUTPUT%.csv}.json"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Benchmark created!"
    echo "   CSV:  $OUTPUT"
    echo "   JSON: $JSON_OUTPUT"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

