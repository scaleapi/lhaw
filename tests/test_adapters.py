"""Tests for benchmark adapters (synthetic/adapters/)."""

from pathlib import Path

import pytest
from synthetic.adapters.base import BaseTaskAdapter
from synthetic.adapters.mcpatlas import LOCAL_CSV_PATH, MCPAtlasAdapter
from synthetic.adapters.swebench import SWEBenchProAdapter
from synthetic.adapters.tac import TACAdapter

REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# TAC Adapter (always runs — task_pairs_agentcompany is checked in)
# ---------------------------------------------------------------------------

TAC_DIR = REPO_ROOT / "task_pairs_agentcompany"


class TestTACAdapter:
    @pytest.fixture(autouse=True)
    def _adapter(self):
        self.adapter = TACAdapter(task_pairs_dir=str(TAC_DIR))
        self.tasks = self.adapter.list_tasks()

    def test_loads_tasks(self):
        assert len(self.tasks) > 0

    def test_domain(self):
        assert self.adapter.DOMAIN == "tac"

    def test_load_single_task(self):
        task = self.adapter.load_task(self.tasks[0])
        assert task.domain == "tac"
        assert len(task.prompt) > 0

    def test_category_from_task_name(self):
        task = self.adapter.load_task(self.tasks[0])
        assert task.category == self.tasks[0].split("_")[0]

    def test_checkpoints(self):
        cps = self.adapter.get_checkpoints(self.tasks[0])
        assert isinstance(cps, list)

    def test_to_dict(self):
        d = self.adapter.load_task(self.tasks[0]).to_dict()
        assert d["domain"] == "tac"
        assert "prompt" in d

    def test_is_base_adapter(self):
        assert isinstance(self.adapter, BaseTaskAdapter)


# ---------------------------------------------------------------------------
# SWE-Bench Pro Adapter
# ---------------------------------------------------------------------------

SWEBENCH_TRAJ_DIR = REPO_ROOT / "experiments" / "swebench" / "golden_trajectories"

needs_swebench = pytest.mark.skipif(
    not SWEBENCH_TRAJ_DIR.exists(),
    reason="SWE-Bench Pro requires golden_trajectories/ dir",
)


@needs_swebench
class TestSWEBenchProAdapter:
    @pytest.fixture(autouse=True)
    def _adapter(self):
        self.adapter = SWEBenchProAdapter(trajectory_dir=str(SWEBENCH_TRAJ_DIR))
        self.tasks = self.adapter.list_tasks()

    def test_loads_tasks(self):
        assert len(self.tasks) > 0

    def test_domain(self):
        assert self.adapter.DOMAIN == "swebench"

    def test_load_single_task(self):
        task = self.adapter.load_task(self.tasks[0])
        assert task.domain == "swebench"
        assert len(task.prompt) > 0

    def test_category_is_repo(self):
        task = self.adapter.load_task(self.tasks[0])
        assert "/" in task.category or "__" in task.category

    def test_format_prompt_roundtrip(self):
        raw = self.adapter.get_raw_task(self.tasks[0])
        prompt = SWEBenchProAdapter.format_prompt(raw)
        assert raw["problem_statement"] in prompt

    def test_to_dict(self):
        d = self.adapter.load_task(self.tasks[0]).to_dict()
        assert d["domain"] == "swebench"
        assert "prompt" in d

    def test_is_base_adapter(self):
        assert isinstance(self.adapter, BaseTaskAdapter)


# ---------------------------------------------------------------------------
# MCP-Atlas Adapter
# ---------------------------------------------------------------------------

needs_mcpatlas = pytest.mark.skipif(
    not LOCAL_CSV_PATH.exists(),
    reason="MCP-Atlas.csv not present",
)


@needs_mcpatlas
class TestMCPAtlasAdapter:
    @pytest.fixture(autouse=True)
    def _adapter(self):
        self.adapter = MCPAtlasAdapter()
        self.tasks = self.adapter.list_tasks()

    def test_loads_tasks(self):
        assert len(self.tasks) > 0

    def test_domain(self):
        assert self.adapter.DOMAIN == "mcpatlas"

    def test_load_single_task(self):
        task = self.adapter.load_task(self.tasks[0])
        assert task.domain == "mcpatlas"
        assert len(task.prompt) > 0
        assert len(task.checkpoints) > 0

    def test_category_inference(self):
        categories = {self.adapter.load_task(tid).category for tid in self.tasks[:10]}
        assert len(categories) > 0

    def test_task_id_filter(self):
        sample_id = self.tasks[0]
        filtered = MCPAtlasAdapter(task_ids=[sample_id])
        assert filtered.list_tasks() == [sample_id]

    def test_get_enabled_tools(self):
        tools = self.adapter.get_enabled_tools(self.tasks[0])
        assert isinstance(tools, list)

    def test_get_trajectory(self):
        trajectory = self.adapter.get_trajectory(self.tasks[0])
        assert isinstance(trajectory, list)

    def test_to_dict(self):
        d = self.adapter.load_task(self.tasks[0]).to_dict()
        assert d["domain"] == "mcpatlas"
        assert "prompt" in d

    def test_is_base_adapter(self):
        assert isinstance(self.adapter, BaseTaskAdapter)


# ---------------------------------------------------------------------------
# SWE-Bench format_prompt (unit test — no data dependency)
# ---------------------------------------------------------------------------


class TestSWEBenchFormatPrompt:
    """Tests for SWEBenchProAdapter.format_prompt (static method, no dataset needed)."""

    def test_full_prompt(self):
        row = {
            "problem_statement": "Fix the bug in module X",
            "requirements": "Must handle edge case Y",
            "interface": "def foo(bar: int) -> str",
        }
        prompt = SWEBenchProAdapter.format_prompt(row)
        assert "Fix the bug in module X" in prompt
        assert "Requirements:" in prompt
        assert "Must handle edge case Y" in prompt
        assert "New interfaces introduced:" in prompt
        assert "def foo(bar: int) -> str" in prompt

    def test_prompt_no_requirements(self):
        row = {"problem_statement": "Fix bug", "requirements": "", "interface": ""}
        prompt = SWEBenchProAdapter.format_prompt(row)
        assert prompt == "Fix bug"
        assert "Requirements:" not in prompt

    def test_prompt_empty(self):
        prompt = SWEBenchProAdapter.format_prompt({})
        assert prompt == ""
