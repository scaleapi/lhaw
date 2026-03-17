"""
MCP Atlas Adapter
=================
Convert MCP Atlas tasks to UnifiedTask format for the synthetic pipeline.

Data Source: experiments/mcpatlas/MCP-Atlas.csv
    - TASK: Task ID (hex string)
    - PROMPT: Task description
    - ENABLED_TOOLS: JSON list of available tools
    - GTFA_CLAIMS: JSON list of ground truth assertions
    - TRAJECTORY: JSON list of tool call sequence

Category inference uses the most common tool prefix from ENABLED_TOOLS.
"""

import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from datasets import load_dataset

from ..core import UnifiedTask
from .base import BaseTaskAdapter

logger = logging.getLogger(__name__)

# Increase CSV field size limit for large trajectory fields
csv.field_size_limit(10 * 1024 * 1024)

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
LOCAL_CSV_PATH = REPO_ROOT / "experiments" / "mcpatlas" / "MCP-Atlas.csv"
HF_DATASET_ID = "ScaleAI/MCP-Atlas"
TRAJECTORY_CACHE_DIR = REPO_ROOT / "experiments" / "mcpatlas" / "trajectories"


class MCPAtlasAdapter(BaseTaskAdapter):
    """
    Adapter for MCP Atlas tasks.

    Converts MCP Atlas CSV format to UnifiedTask for the synthetic pipeline.
    Supports optional task ID filtering for working with subsets.

    Usage:
        adapter = MCPAtlasAdapter()  # All tasks
        adapter = MCPAtlasAdapter(task_ids=["68a398aa...", "689bd255..."])  # Filtered

        tasks = adapter.list_tasks()
        task = adapter.load_task("68a398aa2f58036e8d45eddf")
    """

    DOMAIN = "mcpatlas"

    def __init__(self, task_ids: Optional[List[str]] = None):
        """
        Initialize the adapter.

        Args:
            task_ids: Optional filter - only load these task IDs.
                      If None, all tasks are available.
        """
        self.csv_path = self._ensure_dataset()
        self.task_ids_filter: Optional[Set[str]] = set(task_ids) if task_ids else None
        self._data: Optional[Dict[str, Dict[str, Any]]] = None

    def _ensure_dataset(self) -> Path:
        """
        Get local path, downloading from HuggingFace if needed.

        Returns:
            Path to the local CSV file.
        """
        if LOCAL_CSV_PATH.exists():
            return LOCAL_CSV_PATH

        logger.info(f"Local CSV not found, downloading from HuggingFace: {HF_DATASET_ID}")
        LOCAL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

        try:
            ds = load_dataset(HF_DATASET_ID, split="train")
            ds.to_csv(str(LOCAL_CSV_PATH))
            logger.info(f"Downloaded {len(ds)} tasks to: {LOCAL_CSV_PATH}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download MCP-Atlas from HuggingFace ({HF_DATASET_ID}): {e}"
            ) from e

        return LOCAL_CSV_PATH

    def _load_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Lazy load and cache CSV data.

        Returns:
            Dict mapping task_id to row data.
        """
        if self._data is not None:
            return self._data

        self._data = {}
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id = row.get("TASK", "")
                if not task_id:
                    continue

                # Apply filter if set
                if self.task_ids_filter and task_id not in self.task_ids_filter:
                    continue

                self._data[task_id] = row

        logger.info(f"Loaded {len(self._data)} MCP Atlas tasks")
        return self._data

    def _parse_json_field(self, value: str, default: Any = None) -> Any:
        """Safely parse a JSON field from CSV."""
        if not value:
            return default if default is not None else []
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # Try parsing as Python literal (some fields use single quotes)
            try:
                import ast

                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                logger.warning(f"Failed to parse JSON: {value[:100]}...")
                return default if default is not None else []

    def _infer_category(self, enabled_tools: List[str]) -> str:
        """
        Infer task category from the most common tool prefix.

        Args:
            enabled_tools: List of tool names (e.g., ["github_search_repos", "fetch_fetch"])

        Returns:
            Most common prefix (e.g., "github") or "unknown"
        """
        if not enabled_tools:
            return "unknown"

        # Extract prefixes (part before first underscore)
        prefixes = []
        for tool in enabled_tools:
            if isinstance(tool, str) and "_" in tool:
                prefix = tool.split("_")[0]
                prefixes.append(prefix)

        if not prefixes:
            return "unknown"

        # Return most common prefix
        return Counter(prefixes).most_common(1)[0][0]

    def load_task(self, task_id: str, **kwargs) -> UnifiedTask:
        """
        Load an MCP Atlas task into unified format.

        Args:
            task_id: Task identifier (hex string like "68a398aa2f58036e8d45eddf")
            **kwargs: Additional options (ignored)

        Returns:
            UnifiedTask with prompt, checkpoints, category, trajectory_path, etc.

        Raises:
            ValueError: If task_id not found
        """
        data = self._load_data()

        if task_id not in data:
            raise ValueError(f"Task '{task_id}' not found in MCP Atlas dataset")

        row = data[task_id]

        # Parse fields
        prompt = row.get("PROMPT", "")
        enabled_tools = self._parse_json_field(row.get("ENABLED_TOOLS", ""), [])
        gtfa_claims = self._parse_json_field(row.get("GTFA_CLAIMS", ""), [])
        trajectory = self._parse_json_field(row.get("TRAJECTORY", ""), [])

        # Infer category from tools
        category = self._infer_category(enabled_tools)

        # Convert GTFA_CLAIMS to checkpoint strings
        checkpoints = []
        for claim in gtfa_claims:
            if isinstance(claim, str):
                checkpoints.append(claim)
            elif isinstance(claim, dict):
                # Handle structured claims (extract text/assertion field)
                claim_text = claim.get("claim") or claim.get("assertion") or str(claim)
                checkpoints.append(claim_text)

        # Cache trajectory to file for pipeline compatibility
        trajectory_path = None
        if trajectory:
            trajectory_path = self._cache_trajectory(task_id, trajectory)

        return UnifiedTask(
            id=task_id,
            domain=self.DOMAIN,
            category=category,
            prompt=prompt,
            checkpoints=checkpoints,
            trajectory_path=trajectory_path,
        )

    def _cache_trajectory(self, task_id: str, trajectory: List[Dict[str, Any]]) -> str:
        """
        Cache trajectory to a JSON file for pipeline compatibility.

        Invalidates the cache entry if the source CSV is newer than the
        cached file (e.g. after a re-download from HuggingFace).

        Args:
            task_id: Task identifier
            trajectory: List of tool call dictionaries

        Returns:
            Path to the cached trajectory file
        """
        TRAJECTORY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = TRAJECTORY_CACHE_DIR / f"{task_id}.json"

        csv_mtime = self.csv_path.stat().st_mtime if self.csv_path.exists() else 0
        cache_stale = not cache_path.exists() or cache_path.stat().st_mtime < csv_mtime

        if cache_stale:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(trajectory, f, indent=2)
            logger.debug(f"Cached trajectory to {cache_path}")

        return str(cache_path)

    def list_tasks(self) -> List[str]:
        """
        List all available task IDs.

        Returns:
            Sorted list of task IDs (respects task_ids filter if set)
        """
        data = self._load_data()
        return sorted(data.keys())

    def get_checkpoints(self, task_id: str) -> List[str]:
        """
        Get GTFA_CLAIMS as checkpoints for a task.

        Args:
            task_id: Task identifier

        Returns:
            List of checkpoint/assertion strings
        """
        task = self.load_task(task_id)
        return task.checkpoints

    def get_enabled_tools(self, task_id: str) -> List[str]:
        """
        Get the list of enabled tools for a task.

        Args:
            task_id: Task identifier

        Returns:
            List of tool names
        """
        data = self._load_data()
        if task_id not in data:
            raise ValueError(f"Task '{task_id}' not found")

        return self._parse_json_field(data[task_id].get("ENABLED_TOOLS", ""), [])

    def get_trajectory(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get the ground truth trajectory for a task.

        Args:
            task_id: Task identifier

        Returns:
            List of tool call dictionaries
        """
        data = self._load_data()
        if task_id not in data:
            raise ValueError(f"Task '{task_id}' not found")

        return self._parse_json_field(data[task_id].get("TRAJECTORY", ""), [])

    def get_raw_row(self, task_id: str) -> Dict[str, Any]:
        """
        Get the raw CSV row for a task (for debugging).

        Args:
            task_id: Task identifier

        Returns:
            Raw row dictionary from CSV
        """
        data = self._load_data()
        if task_id not in data:
            raise ValueError(f"Task '{task_id}' not found")

        return data[task_id]
