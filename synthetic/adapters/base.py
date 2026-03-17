"""
Base Task Adapter
=================
Abstract base class for benchmark adapters.

All adapters must implement:
- load_task(task_id) -> UnifiedTask
- list_tasks() -> List[str]
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from ..core import UnifiedTask


class BaseTaskAdapter(ABC):
    """Abstract base class for benchmark task adapters."""

    # Override in subclass
    DOMAIN: str = "unknown"
    # Override in subclass or set S3_BUCKET env var for trajectory fallback
    S3_BUCKET: str = os.environ.get("S3_BUCKET", "")
    S3_TRAJECTORY_PREFIX: str = "research/lhaw/{domain}/gt_trajectory"

    @abstractmethod
    def load_task(self, task_id: str, **kwargs) -> UnifiedTask:
        """Load a task into unified format."""
        pass

    @abstractmethod
    def list_tasks(self) -> List[str]:
        """List all available task IDs."""
        pass

    def get_trajectory_path(
        self, task_id: str, trajectory_dir: Optional[str] = None
    ) -> Optional[str]:
        """Get trajectory path, checking local directory first, then S3.

        Args:
            task_id: Task name
            trajectory_dir: Local directory containing {task_id}.json files
                           (produced by export_tac_golden_trajectories.py)

        Returns:
            Local path if file exists, S3 URL otherwise, or None if neither.
        """
        if trajectory_dir:
            local_path = Path(trajectory_dir) / f"{task_id}.json"
            if local_path.exists():
                return str(local_path)

        return self.get_trajectory_url(task_id)

    def get_trajectory_url(self, task_id: str) -> Optional[str]:
        """Get S3 URL for golden trajectory. Returns None if S3_BUCKET is not set."""
        if not self.S3_BUCKET:
            return None
        prefix = self.S3_TRAJECTORY_PREFIX.format(domain=self.DOMAIN)
        return f"s3://{self.S3_BUCKET}/{prefix}/{task_id}.json"

    def get_checkpoints(self, task_id: str) -> List[str]:
        """Get checkpoints for a task. Override to provide checkpoints."""
        return []
