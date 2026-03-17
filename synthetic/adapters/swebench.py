"""
SWE-bench Pro Adapter
=====================
Convert SWE-bench Pro tasks to UnifiedTask format.

Loads from HuggingFace: ScaleAI/SWE-bench_Pro
Supports optional local trajectory data for grounded segment extraction.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core import UnifiedTask
from .base import BaseTaskAdapter

logger = logging.getLogger(__name__)


class SWEBenchProAdapter(BaseTaskAdapter):
    """Adapter for SWE-bench Pro tasks."""

    def __init__(self, trajectory_dir: str):
        """
        Args:
            trajectory_dir: Path to directory containing trajectory folders.
                           Each subfolder should be named by instance_id and contain a .traj file.
                           Required for grounded segment extraction.
        """
        self._dataset = None
        self._index: Optional[Dict[str, int]] = None
        self._trajectory_dir = Path(trajectory_dir)

    @property
    def dataset(self):
        """Lazy load dataset from HuggingFace."""
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset("ScaleAI/SWE-bench_Pro", split="test")
            # Build index for fast lookup
            self._index = {row["instance_id"]: i for i, row in enumerate(self._dataset)}
        return self._dataset

    def _get_row(self, instance_id: str) -> Dict:
        """Get a row by instance_id."""
        _ = self.dataset  # Ensure loaded
        if instance_id not in self._index:
            raise ValueError(f"Instance not found: {instance_id}")
        return self._dataset[self._index[instance_id]]

    DOMAIN = "swebench"

    @staticmethod
    def format_prompt(row: Dict) -> str:
        """Format problem statement with requirements and interface."""
        problem_statement = row.get("problem_statement", "")
        requirements = row.get("requirements", "")
        interface = row.get("interface", "")

        prompt = problem_statement
        if requirements:
            prompt += f"\n\nRequirements:\n{requirements}"
        if interface:
            prompt += f"\n\nNew interfaces introduced:\n{interface}"

        return prompt

    def get_trajectory_path(self, instance_id: str) -> Optional[str]:
        """Get trajectory file path for an instance.

        Warns if trajectory is not found since grounded extraction depends on it.
        """
        traj_folder = self._trajectory_dir / instance_id
        if not traj_folder.exists():
            logger.warning(f"Trajectory folder not found for {instance_id}: {traj_folder}")
            return None

        # Find .traj file (sorted for determinism when multiple exist)
        traj_files = sorted(traj_folder.glob("*.traj"))
        if traj_files:
            return str(traj_files[0])
        logger.warning(f"No .traj file in {traj_folder}")
        return None

    def load_trajectory_summary(self, instance_id: str, max_chars: int = 50000) -> Optional[str]:
        """
        Load trajectory as summary for grounding.

        Returns a formatted summary of agent actions/observations.
        """
        traj_path = self.get_trajectory_path(instance_id)
        if not traj_path:
            return None

        try:
            with open(traj_path) as f:
                data = json.load(f)
        except Exception:
            return None

        trajectory = data.get("trajectory", [])
        if not trajectory:
            return None

        # Format trajectory steps
        lines = [f"Total steps: {len(trajectory)}\n"]
        for i, step in enumerate(trajectory):
            action = step.get("action", "")[:200]
            observation = step.get("observation", "")[:300]
            lines.append(f"Step {i+1}:")
            lines.append(f"  Action: {action}")
            if observation:
                lines.append(f"  Observation: {observation[:200]}...")
            lines.append("")

        content = "\n".join(lines)
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... [truncated]"
        return content

    def load_task(self, instance_id: str) -> UnifiedTask:
        """
        Load a SWE-bench Pro task into unified format.

        Args:
            instance_id: Instance ID (e.g., "django__django-12345")

        Returns:
            UnifiedTask with formatted prompt and optional trajectory path
        """
        row = self._get_row(instance_id)
        prompt = self.format_prompt(row)

        # Extract repo as category
        repo = row.get("repo", "")

        # Get trajectory path if available
        traj_path = self.get_trajectory_path(instance_id)

        return UnifiedTask(
            id=instance_id,
            domain="swebench",
            category=repo,
            prompt=prompt,
            trajectory_path=traj_path,
        )

    def list_tasks(self) -> List[str]:
        """List all available SWE-bench Pro instance IDs."""
        return [row["instance_id"] for row in self.dataset]

    def get_raw_task(self, instance_id: str) -> Dict:
        """Get raw task data for additional metadata access."""
        return dict(self._get_row(instance_id))

    def load_trajectory_data(self, instance_id: str) -> List[Dict[str, Any]]:
        """
        Load raw trajectory data for grounded extraction.

        Returns list of {action, observation, thought} dicts.
        """
        traj_path = self.get_trajectory_path(instance_id)
        if not traj_path:
            return []

        try:
            with open(traj_path) as f:
                data = json.load(f)
            return data.get("trajectory", [])
        except Exception as e:
            logger.warning(f"Failed to load trajectory for {instance_id}: {e}")
            return []

    def get_test_info(self, instance_id: str) -> Dict[str, Any]:
        """
        Get test-related info for grounded extraction.

        Returns:
            {
                "fail_to_pass": [...],
                "test_patch": "...",
                "pass_to_pass": [...],
            }
        """
        row = self._get_row(instance_id)
        return {
            "fail_to_pass": row.get("fail_to_pass", []),
            "test_patch": row.get("test_patch", ""),
            "pass_to_pass": row.get("pass_to_pass", []),
        }

    def list_f2p_filtered_tasks(self, threshold: int = 2) -> List[str]:
        """Return instance_ids with F2P test count > threshold.

        This filters to tasks that have enough fail-to-pass tests to be
        meaningful for underspecification evaluation.
        """
        filtered = []
        for row in self.dataset:
            f2p_raw = row.get("fail_to_pass", "[]")
            try:
                # HF stores as Python repr with single quotes; ast.literal_eval handles both
                f2p_tests = ast.literal_eval(f2p_raw) if isinstance(f2p_raw, str) else f2p_raw
            except (ValueError, SyntaxError, TypeError):
                f2p_tests = []
            if len(f2p_tests) > threshold:
                filtered.append(row["instance_id"])
        return filtered
