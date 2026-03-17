"""
Sample from filtered_underspec.csv with class proportions and diversity maximization.

Joins with underspec_prompts.csv to include TRAJECTORY, prompts, and other fields.

Usage:
    python sample_from_filtered.py --input /path/to/filtered_underspec.csv --output sampled.csv
    python sample_from_filtered.py --input /path/to/filtered_underspec.csv --output sampled.csv --total 100
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from utils import safe_csv_read, safe_csv_write


# Default class proportions
DEFAULT_PROPORTIONS = {
    "outcome-critical": 0.50,
    "divergent": 0.30,
    "benign": 0.20,
}

# Columns to join from underspec_prompts.csv
PROMPT_COLUMNS = [
    "PROMPT",  # The underspecified prompt (what model sees)
    "original_prompt",
    "TRAJECTORY",
    "ENABLED_TOOLS",
    "GTFA_CLAIMS",
]


def find_prompts_file(input_path: Path) -> Optional[Path]:
    """Find underspec_prompts.csv relative to input file."""
    # Try underspec_generation subdirectory
    prompts_path = input_path.parent / "underspec_generation" / "underspec_prompts.csv"
    if prompts_path.exists():
        return prompts_path

    # Try same directory
    prompts_path = input_path.parent / "underspec_prompts.csv"
    if prompts_path.exists():
        return prompts_path

    return None


def load_prompts_lookup(prompts_path: Path) -> Dict[str, Dict[str, str]]:
    """Load prompts CSV and build lookup by variant_id."""
    rows = safe_csv_read(prompts_path)
    lookup = {}
    for row in rows:
        task = row.get("TASK", "")
        variant_id = row.get("variant_id", "")
        if task and variant_id:
            key = f"{task}::{variant_id}"
            lookup[key] = {col: row.get(col, "") for col in PROMPT_COLUMNS}
    return lookup


def join_with_prompts(rows: List[Dict], prompts_lookup: Dict[str, Dict[str, str]]) -> List[Dict]:
    """Join rows with prompt data."""
    joined = []
    for row in rows:
        variant_id = row.get("variant_id", "")
        new_row = dict(row)

        # Add prompt columns
        prompt_data = prompts_lookup.get(variant_id, {})
        for col in PROMPT_COLUMNS:
            new_row[col] = prompt_data.get(col, "")

        joined.append(new_row)
    return joined


def parse_json_or_string(value: str) -> List[str]:
    """Parse a value that may be a JSON array or a plain string."""
    if not value or value.strip() == "":
        return []

    value = value.strip()
    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass

    return [value]


def get_num_segments(variant_local_id: str) -> int:
    """Extract number of segments removed from variant_local_id."""
    segment_matches = re.findall(r"S\d+", variant_local_id)
    return len(segment_matches)


def compute_diversity_score(
    row: Dict,
    selected_tasks: Counter,
    selected_dims: Counter,
    selected_segs: Counter,
) -> float:
    """
    Compute a diversity score for a row based on how much it adds to diversity.
    Higher score = more diverse (adds underrepresented values).
    """
    task_id = row.get("task", "")
    dimensions = parse_json_or_string(row.get("dimension", ""))
    num_segs = get_num_segments(row.get("variant_local_id", ""))

    score = 0.0

    # Prefer tasks we haven't seen (or seen less)
    task_count = selected_tasks.get(task_id, 0)
    if task_count == 0:
        score += 10.0  # Strong bonus for new task
    else:
        score -= task_count * 2.0  # Penalty for repeated tasks

    # Prefer underrepresented dimensions
    for dim in dimensions:
        dim_count = selected_dims.get(dim, 0)
        total_dims = sum(selected_dims.values()) or 1
        # Bonus inversely proportional to current representation
        score += 1.0 / (dim_count + 1)

    # Prefer underrepresented segment counts
    seg_count = selected_segs.get(num_segs, 0)
    total_segs = sum(selected_segs.values()) or 1
    score += 2.0 / (seg_count + 1)

    return score


def select_diverse_sample(
    rows: List[Dict],
    n: int,
    selected_tasks: Counter,
    selected_dims: Counter,
    selected_segs: Counter,
    seed: int = 42,
) -> List[Dict]:
    """
    Select n rows from the given list, maximizing diversity.
    Uses greedy selection with diversity scoring.
    """
    if n >= len(rows):
        return rows

    random.seed(seed)
    selected = []
    remaining = list(rows)

    # Shuffle to break ties randomly
    random.shuffle(remaining)

    for _ in range(n):
        if not remaining:
            break

        # Score all remaining rows
        scored = []
        for row in remaining:
            score = compute_diversity_score(
                row, selected_tasks, selected_dims, selected_segs
            )
            scored.append((score, row))

        # Select the highest scoring row
        scored.sort(key=lambda x: x[0], reverse=True)
        best_row = scored[0][1]

        # Update counters
        task_id = best_row.get("task", "")
        dimensions = parse_json_or_string(best_row.get("dimension", ""))
        num_segs = get_num_segments(best_row.get("variant_local_id", ""))

        selected_tasks[task_id] += 1
        for dim in dimensions:
            selected_dims[dim] += 1
        selected_segs[num_segs] += 1

        selected.append(best_row)
        remaining.remove(best_row)

    return selected


def sample_with_proportions(
    rows: List[Dict],
    total: int,
    proportions: Dict[str, float],
    seed: int = 42,
) -> List[Dict]:
    """
    Sample rows with specified class proportions and diversity maximization.
    """
    # Group rows by class
    by_class: Dict[str, List[Dict]] = {}
    for row in rows:
        cls = row.get("ambiguity_class", "")
        if cls in proportions:
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(row)

    # Calculate target counts per class
    targets = {}
    remaining_total = total
    for cls, prop in sorted(proportions.items(), key=lambda x: x[1], reverse=True):
        target = int(total * prop)
        # Cap at available rows
        available = len(by_class.get(cls, []))
        target = min(target, available)
        targets[cls] = target
        remaining_total -= target

    # Distribute any remaining slots to classes with available rows
    for cls in proportions:
        if remaining_total <= 0:
            break
        available = len(by_class.get(cls, []))
        can_add = available - targets[cls]
        if can_add > 0:
            add = min(can_add, remaining_total)
            targets[cls] += add
            remaining_total -= add

    print(f"Target counts per class: {targets}")
    available_str = {k: len(v) for k, v in by_class.items()}
    print(f"Available per class: {available_str}")

    # Select diverse samples from each class
    # Use shared counters across classes to maximize global diversity
    selected_tasks: Counter = Counter()
    selected_dims: Counter = Counter()
    selected_segs: Counter = Counter()

    all_selected = []

    # Process classes in order of proportion (largest first)
    for cls in sorted(proportions.keys(), key=lambda c: proportions[c], reverse=True):
        n = targets.get(cls, 0)
        class_rows = by_class.get(cls, [])

        if n > 0 and class_rows:
            selected = select_diverse_sample(
                class_rows, n, selected_tasks, selected_dims, selected_segs, seed
            )
            all_selected.extend(selected)
            print(f"  Selected {len(selected)} from {cls}")

    return all_selected


def print_sample_stats(rows: List[Dict]) -> None:
    """Print statistics about the sampled data."""
    print(f"\n=== Sample Statistics ===")
    print(f"Total sampled: {len(rows)}")

    # Class distribution
    class_counts = Counter(r.get("ambiguity_class", "") for r in rows)
    print(f"\nBy class:")
    for cls, count in sorted(class_counts.items()):
        pct = 100 * count / len(rows) if rows else 0
        print(f"  {cls}: {count} ({pct:.1f}%)")

    # Task diversity
    unique_tasks = set(r.get("task", "") for r in rows)
    print(f"\nUnique tasks: {len(unique_tasks)}")

    # Dimension distribution
    dim_counts: Counter = Counter()
    for r in rows:
        for dim in parse_json_or_string(r.get("dimension", "")):
            dim_counts[dim] += 1
    print(f"\nBy dimension:")
    for dim, count in sorted(dim_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {dim}: {count}")

    # Segment count distribution
    seg_counts: Counter = Counter()
    for r in rows:
        num_segs = get_num_segments(r.get("variant_local_id", ""))
        seg_counts[num_segs] += 1
    print(f"\nBy segments removed:")
    for seg, count in sorted(seg_counts.items()):
        print(f"  {seg}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample from filtered_underspec.csv with class proportions and diversity"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to filtered_underspec.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output CSV file path",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=100,
        help="Total number of samples (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--outcome-critical",
        type=float,
        default=0.50,
        help="Proportion of outcome-critical (default: 0.50)",
    )
    parser.add_argument(
        "--divergent",
        type=float,
        default=0.30,
        help="Proportion of divergent (default: 0.30)",
    )
    parser.add_argument(
        "--benign",
        type=float,
        default=0.20,
        help="Proportion of benign (default: 0.20)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    # Build proportions dict
    proportions = {
        "outcome-critical": args.outcome_critical,
        "divergent": args.divergent,
        "benign": args.benign,
    }

    # Validate proportions sum to ~1.0
    total_prop = sum(proportions.values())
    if abs(total_prop - 1.0) > 0.01:
        print(f"Warning: Proportions sum to {total_prop:.2f}, not 1.0")

    print(f"Loading data from: {input_path}")
    rows = safe_csv_read(input_path)
    print(f"  Loaded {len(rows)} rows")

    # Load prompts for joining
    prompts_path = find_prompts_file(input_path)
    prompts_lookup = {}
    if prompts_path:
        print(f"Loading prompts from: {prompts_path}")
        prompts_lookup = load_prompts_lookup(prompts_path)
        print(f"  Loaded {len(prompts_lookup)} prompt entries")
    else:
        print("Warning: underspec_prompts.csv not found, prompts/TRAJECTORY will not be included")

    print(f"\nSampling {args.total} rows with proportions: {proportions}")
    sampled = sample_with_proportions(rows, args.total, proportions, args.seed)

    # Join with prompts data
    if prompts_lookup:
        sampled = join_with_prompts(sampled, prompts_lookup)
        print(f"Joined with prompt data (added: {', '.join(PROMPT_COLUMNS)})")

    print_sample_stats(sampled)

    # Write output
    if sampled:
        fieldnames = list(sampled[0].keys())
        safe_csv_write(output_path, sampled, fieldnames)
        print(f"\nSaved to: {output_path}")
    else:
        print("Error: No rows sampled")


if __name__ == "__main__":
    main()
