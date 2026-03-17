"""
Domain Adapters
===============
Convert domain-specific task formats to UnifiedTask.

Base Class:
    BaseTaskAdapter - Abstract base with:
        - load_task(task_id) → UnifiedTask
        - list_tasks() → List[str]
        - get_trajectory_url(task_id) → str
        - get_checkpoints(task_id) → List[str]

Adapters:
    TACAdapter       - TheAgentCompany tasks
    MCPAtlasAdapter  - MCP Atlas tasks (tool-use)

Usage:
    from synthetic.adapters.tac import TACAdapter
    from synthetic.adapters.mcpatlas import MCPAtlasAdapter
"""

# Export base class only - import adapters directly to control dependency loading
from .base import BaseTaskAdapter

__all__ = ["BaseTaskAdapter"]
