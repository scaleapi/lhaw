"""
Print statistics by ambiguity class from filtered_underspec.csv.

Usage:
    python underspec_stats.py --input /path/to/filtered_underspec.csv
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from utils import safe_csv_read


def parse_json_or_string(value: str) -> List[str]:
    """
    Parse a value that may be a JSON array or a plain string.

    Args:
        value: String like 'goal' or '["goal", "goal"]'

    Returns:
        List of values (single-element list for plain strings)
    """
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


def parse_variant_id(variant_local_id: str) -> Dict[str, Any]:
    """
    Parse variant_local_id to extract segment count and severity.

    Args:
        variant_local_id: String like 'V_S1_S2_delete' or 'V_S1_delete'

    Returns:
        Dict with 'segments' (int) and 'severity' (str)
    """
    # Find all segment patterns like S1, S2, etc.
    segment_matches = re.findall(r"S\d+", variant_local_id)
    segments = len(segment_matches)

    # Extract severity (last part after final underscore)
    parts = variant_local_id.split("_")
    severity = parts[-1] if parts else "unknown"

    return {"segments": segments, "severity": severity}


def compute_stats(rows: List[Dict]) -> Dict[str, Dict[str, Counter]]:
    """
    Compute statistics grouped by ambiguity class.

    Args:
        rows: List of row dicts from CSV

    Returns:
        Dict mapping ambiguity_class to stat counters
    """
    # Group rows by ambiguity class
    by_class: Dict[str, List[Dict]] = {}
    for row in rows:
        cls = row.get("ambiguity_class", "unknown")
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(row)

    # Compute stats for each class
    stats = {}
    for cls, class_rows in by_class.items():
        dimensions: Counter = Counter()
        subdimensions: Counter = Counter()
        segments: Counter = Counter()
        severities: Counter = Counter()

        for row in class_rows:
            # Parse and count dimensions
            for dim in parse_json_or_string(row.get("dimension", "")):
                dimensions[dim] += 1

            # Parse and count subdimensions
            for subdim in parse_json_or_string(row.get("subdimension", "")):
                subdimensions[subdim] += 1

            # Parse variant_id for segments and severity
            variant_local_id = row.get("variant_local_id", "")
            parsed = parse_variant_id(variant_local_id)
            segments[parsed["segments"]] += 1
            severities[parsed["severity"]] += 1

        stats[cls] = {
            "count": len(class_rows),
            "dimensions": dimensions,
            "subdimensions": subdimensions,
            "segments": segments,
            "severities": severities,
        }

    return stats


def print_stats(stats: Dict[str, Dict], total: int) -> None:
    """
    Print formatted statistics.

    Args:
        stats: Statistics dict from compute_stats
        total: Total record count
    """
    print(f"=== Total Records: {total} ===\n")

    # Print in specified order
    class_order = ["benign", "new_task", "outcome-critical", "divergent"]

    for cls in class_order:
        if cls not in stats:
            continue

        class_stats = stats[cls]
        count = class_stats["count"]
        print(f"=== {cls.replace('_', ' ').title()} ({count} tasks) ===")

        # Dimensions
        dims = dict(class_stats["dimensions"].most_common())
        print(f"Dimensions: {dims}")

        # Subdimensions
        subdims = dict(class_stats["subdimensions"].most_common())
        print(f"Subdimensions: {subdims}")

        # Segments removed
        segs = dict(sorted(class_stats["segments"].items()))
        print(f"Segments Removed: {segs}")

        # Severity
        sevs = dict(class_stats["severities"].most_common())
        print(f"Severity: {sevs}")

        print()

    # Print any unexpected classes
    for cls in stats:
        if cls not in class_order:
            class_stats = stats[cls]
            count = class_stats["count"]
            print(f"=== {cls} ({count} tasks) ===")
            dims = dict(class_stats["dimensions"].most_common())
            print(f"Dimensions: {dims}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Print statistics by ambiguity class from filtered_underspec.csv"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to filtered_underspec.csv",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    rows = safe_csv_read(input_path)
    stats = compute_stats(rows)
    print_stats(stats, len(rows))


if __name__ == "__main__":
    main()
