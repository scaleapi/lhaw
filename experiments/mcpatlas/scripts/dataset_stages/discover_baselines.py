"""Auto-discover baseline run directories from inference data."""

import re
from pathlib import Path
from typing import Dict


def discover_baseline_runs(base_dir: Path, model_patterns: Dict[str, str]) -> Dict[str, Path]:
    """
    Auto-discover baseline run directories using glob patterns.

    Args:
        base_dir: Base directory to search in
        model_patterns: Dict mapping model names to glob patterns

    Returns:
        Dict mapping model names to their most recent run directory paths

    Raises:
        ValueError: If no runs found for a model or if run directory is invalid

    Strategy:
        1. Glob match each pattern in base_dir
        2. Extract timestamps from directory names
        3. Return most recent match per model
        4. Validate: combined/pass_k_per_task.csv must exist
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    discovered = {}

    for model, pattern in model_patterns.items():
        matches = list(base_dir.glob(pattern))

        if not matches:
            raise ValueError(f"No runs found for {model} with pattern '{pattern}' in {base_dir}")

        # Extract timestamps and sort by most recent
        def extract_timestamp(path: Path) -> str:
            """Extract YYYYMMDD_HHMMSS timestamp from directory name."""
            match = re.search(r"(\d{8}_\d{6})", path.name)
            return match.group(1) if match else ""

        matches_with_ts = [(p, extract_timestamp(p)) for p in matches]
        matches_with_ts.sort(key=lambda x: x[1], reverse=True)

        # Take most recent match
        most_recent = matches_with_ts[0][0]

        # Validate that this is a proper run directory
        pass_k_file = most_recent / "combined" / "pass_k_per_task.csv"
        if not pass_k_file.exists():
            raise ValueError(
                f"Invalid run directory for {model}: {most_recent}\n"
                f"Expected file not found: {pass_k_file}"
            )

        discovered[model] = most_recent

    return discovered
