"""Tests for scripts/ — filtering, task extraction, experiment resolution, base64 contract, classification, and reporting."""

import base64
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# ── extract_task_from_variant_id ────────────────────────────────────────


class TestExtractTaskFromVariantId:
    def test_basic(self):
        from scripts.filter_tac_samples import extract_task_from_variant_id

        assert (
            extract_task_from_variant_id("finance_budget_variance_V1_goal")
            == "finance_budget_variance"
        )

    def test_prefix_collision_resolved(self):
        from scripts.filter_tac_samples import extract_task_from_variant_id

        assert (
            extract_task_from_variant_id("hr_check_attendance_multiple_days_V3_input+constraint")
            == "hr_check_attendance_multiple_days"
        )

    def test_one_day_vs_multiple_days(self):
        from scripts.filter_tac_samples import extract_task_from_variant_id

        assert (
            extract_task_from_variant_id("hr_check_attendance_one_day_V1_goal")
            == "hr_check_attendance_one_day"
        )
        assert (
            extract_task_from_variant_id("hr_check_attendance_multiple_days_V2_input")
            == "hr_check_attendance_multiple_days"
        )

    def test_no_match(self):
        from scripts.filter_tac_samples import extract_task_from_variant_id

        assert extract_task_from_variant_id("unknown_task_V1_goal") is None

    def test_ds_tasks(self):
        from scripts.filter_tac_samples import extract_task_from_variant_id

        assert (
            extract_task_from_variant_id("ds_visualize_data_in_pie_and_bar_chart_V4_input+goal")
            == "ds_visualize_data_in_pie_and_bar_chart"
        )


# ── _resolve_exp_dirs ───────────────────────────────────────────────────


class TestResolveExpDirs:
    def test_exact_match(self):
        from scripts.filter_tac_samples import _resolve_exp_dirs

        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = Path(tmp) / "my_exp"
            exp_dir.mkdir()
            (exp_dir / "underspec_candidates.csv").touch()
            with patch("scripts.filter_tac_samples.PATHS", {"synthetic_outputs": tmp}):
                dirs, name = _resolve_exp_dirs("my_exp")
                assert dirs == ["my_exp"] and name == "my_exp"

    def test_domain_expansion(self):
        from scripts.filter_tac_samples import _resolve_exp_dirs

        with tempfile.TemporaryDirectory() as tmp:
            for domain in ["ds", "finance"]:
                d = Path(tmp) / f"lhaw_tac_{domain}"
                d.mkdir()
                (d / "underspec_candidates.csv").touch()
            with patch("scripts.filter_tac_samples.PATHS", {"synthetic_outputs": tmp}):
                dirs, name = _resolve_exp_dirs("lhaw_tac")
                assert set(dirs) == {"lhaw_tac_ds", "lhaw_tac_finance"}
                assert name == "lhaw_tac"

    def test_fallback_when_nothing_found(self):
        from scripts.filter_tac_samples import _resolve_exp_dirs

        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.filter_tac_samples.PATHS", {"synthetic_outputs": tmp}):
                dirs, name = _resolve_exp_dirs("nonexistent")
                assert dirs == ["nonexistent"] and name == "nonexistent"


# ── Primary task base64 encoding/decoding contract ──────────────────────


class TestPrimaryTaskB64Contract:
    """Validates the encoding contract between task.py (encoder) and mcp_server.py (decoder)."""

    @staticmethod
    def _encode(task_def: dict) -> str:
        return base64.b64encode(json.dumps(task_def).encode("utf-8")).decode("ascii")

    @staticmethod
    def _decode(encoded: str) -> dict:
        return json.loads(base64.b64decode(encoded).decode("utf-8"))

    def test_roundtrip(self):
        task_def = {
            "primary_task": "Create a budget report",
            "underspecified_task": "Create a report",
            "removed_segments": [
                {"value": "budget", "dimension": "goal"},
                {"value": "Q4", "dimension": "context"},
            ],
        }
        assert self._decode(self._encode(task_def)) == task_def

    def test_encoding_is_toml_safe(self):
        encoded = self._encode({"primary_task": 'She said "hello" \\ backslash'})
        assert "\\" not in encoded and '"' not in encoded


# ── extract_json_from_response (LLM judge JSON parsing) ────────────────


class TestExtractJsonFromResponse:
    def _extract(self, content):
        from scripts.process_tac_underspec import extract_json_from_response

        return extract_json_from_response(content)

    def test_raw_json(self):
        assert self._extract('{"task_completed": true}') == {"task_completed": True}

    def test_json_in_code_block(self):
        content = '```json\n{"task_completed": false, "reasoning": "nope"}\n```'
        result = self._extract(content)
        assert result == {"task_completed": False, "reasoning": "nope"}

    def test_json_in_bare_code_block(self):
        content = '```\n{"key": "value"}\n```'
        assert self._extract(content) == {"key": "value"}

    def test_json_embedded_in_text(self):
        content = 'Here is my analysis:\n{"task_completed": true, "reasoning": "done"}\nEnd.'
        result = self._extract(content)
        assert result is not None
        assert result["task_completed"] is True

    def test_garbage_returns_none(self):
        assert self._extract("no json here at all") is None

    def test_empty_string(self):
        assert self._extract("") is None

    def test_nested_json(self):
        content = '{"outer": {"inner": 42}}'
        result = self._extract(content)
        assert result == {"outer": {"inner": 42}}

    def test_whitespace_padded(self):
        content = '   \n  {"key": "val"}  \n  '
        assert self._extract(content) == {"key": "val"}


# ── get_new_task_judge_prompt (empty response handling) ─────────────────


class TestGetNewTaskJudgePrompt:
    def _prompt(self, underspec_prompt, responses):
        from scripts.process_tac_underspec import get_new_task_judge_prompt

        return get_new_task_judge_prompt(underspec_prompt, responses)

    def test_empty_responses_replaced(self):
        prompt = self._prompt("Do the task", ["", None, "  "])
        assert "[No response recorded]" in prompt
        assert "Trial 1:" in prompt and "Trial 3:" in prompt

    def test_normal_responses_preserved(self):
        prompt = self._prompt("Do X", ["I did X", "X is done"])
        assert "I did X" in prompt
        assert "X is done" in prompt
        assert "[No response recorded]" not in prompt

    def test_all_empty_responses(self):
        prompt = self._prompt("Task", [])
        assert "[No response recorded]" in prompt

    def test_mixed_responses(self):
        prompt = self._prompt("Task", ["good result", "", "another result"])
        assert "good result" in prompt
        assert "[No response recorded]" in prompt
        assert "another result" in prompt


# ── resolve_experiments pattern matching ────────────────────────────────


class TestResolveExperiments:
    def test_single(self):
        from scripts.process_tac_underspec import resolve_experiments

        assert resolve_experiments("v1_ds") == ["v1_ds"]

    def test_comma_separated(self):
        from scripts.process_tac_underspec import resolve_experiments

        result = resolve_experiments("v1_ds, v1_finance, v1_hr")
        assert result == ["v1_ds", "v1_finance", "v1_hr"]

    def test_glob_with_matching_dirs(self):
        from scripts.process_tac_underspec import resolve_experiments

        with tempfile.TemporaryDirectory() as tmp:
            for name in ["lhaw_tac_ds", "lhaw_tac_finance", "other_exp"]:
                d = Path(tmp) / name
                d.mkdir()
                (d / "underspec_candidates.csv").touch()
            with patch("scripts.process_tac_underspec.SYNTH_OUTPUTS", Path(tmp)):
                result = resolve_experiments("lhaw_tac_*")
                assert "lhaw_tac_ds" in result
                assert "lhaw_tac_finance" in result
                assert "other_exp" not in result

    def test_glob_no_matches(self):
        from scripts.process_tac_underspec import resolve_experiments

        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.process_tac_underspec.SYNTH_OUTPUTS", Path(tmp)):
                assert resolve_experiments("nonexistent_*") == []


# ── Model boundary matching (compare_tac_conditions.py bug fix) ────────


class TestModelBoundaryMatching:
    """Validates that model matching uses boundary tokens to avoid substring collisions."""

    @staticmethod
    def _matches(model: str, run_dir_name: str) -> bool:
        return f"_{model}_" in f"_{run_dir_name}_"

    def test_exact_match(self):
        assert self._matches("gemini_3_flash", "run_gemini_3_flash_baseline")

    def test_no_substring_collision(self):
        assert not self._matches("gemini_3_flash", "run_gemini_3_1_flash_lite_baseline")

    def test_longest_first_prevents_ambiguity(self):
        models = ["gemini_3_flash", "gemini_3_1_flash_lite"]
        run_dir = "run_gemini_3_1_flash_lite_baseline"
        matched = None
        for model in sorted(models, key=len, reverse=True):
            if self._matches(model, run_dir):
                matched = model
                break
        assert matched == "gemini_3_1_flash_lite"


# ── Eval-file trial index extraction ───────────────────────────────────


class TestEvalFileTrialIndex:
    """Validates the index extraction from evaluation_N.json filenames."""

    @staticmethod
    def _extract_index(filename: str) -> int:
        stem = Path(filename).stem
        suffix = stem.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else -1

    def test_standard(self):
        assert self._extract_index("evaluation_0.json") == 0
        assert self._extract_index("evaluation_2.json") == 2
        assert self._extract_index("evaluation_15.json") == 15

    def test_gap_in_sequence(self):
        files = ["evaluation_0.json", "evaluation_2.json", "evaluation_4.json"]
        indices = [self._extract_index(f) for f in files]
        assert indices == [0, 2, 4]

    def test_malformed_name(self):
        assert self._extract_index("evaluation_abc.json") == -1

    def test_response_alignment_with_gaps(self):
        all_responses = ["resp_0", "resp_1", "resp_2", "resp_3", "resp_4"]
        eval_files = ["evaluation_0.json", "evaluation_2.json", "evaluation_4.json"]
        aligned = []
        for f in eval_files:
            idx = self._extract_index(f)
            aligned.append(all_responses[idx] if idx < len(all_responses) else "")
        assert aligned == ["resp_0", "resp_2", "resp_4"]


# ── Ambiguity classification rules ────────────────────────────────────


class TestAmbiguityClassification:
    """Tests the rule-based classification logic from process_tac_underspec.py build_summary_df."""

    @staticmethod
    def _classify(n: int, c: int, n_unique_states: int, judge_model: str = None) -> str:
        """Mirrors the classification logic in build_summary_df."""
        all_failed = c == 0
        all_succeeded = c == n
        some_succeeded = 0 < c < n

        if all_failed:
            if n_unique_states == 1:
                if judge_model:
                    return "new_task_or_outcome-critical"
                else:
                    return "outcome-critical"
            else:
                return "outcome-critical"
        elif all_succeeded:
            return "benign"
        elif some_succeeded:
            return "divergent"
        else:
            return "unknown"

    def test_all_fail_divergent_states_is_outcome_critical(self):
        assert self._classify(n=3, c=0, n_unique_states=3) == "outcome-critical"

    def test_all_fail_single_state_no_judge_is_outcome_critical(self):
        assert self._classify(n=3, c=0, n_unique_states=1) == "outcome-critical"

    def test_all_fail_single_state_with_judge_needs_llm(self):
        result = self._classify(n=3, c=0, n_unique_states=1, judge_model="gemini")
        assert result == "new_task_or_outcome-critical"

    def test_all_succeed_is_benign(self):
        assert self._classify(n=3, c=3, n_unique_states=1) == "benign"

    def test_partial_success_is_divergent(self):
        assert self._classify(n=3, c=1, n_unique_states=2) == "divergent"
        assert self._classify(n=3, c=2, n_unique_states=2) == "divergent"

    def test_edge_case_n_equals_1_success(self):
        assert self._classify(n=1, c=1, n_unique_states=1) == "benign"

    def test_edge_case_n_equals_1_failure(self):
        assert self._classify(n=1, c=0, n_unique_states=1) == "outcome-critical"


# ── Per-class baseline scoping (compare_tac_conditions.py) ─────────────


class TestPerClassBaselineScoping:
    """Verifies that the by-class table computes baselines from tasks in each class, not globally."""

    @staticmethod
    def _build_data():
        """Build mock data mimicking compare_tac_conditions structure."""
        baseline_df = pd.DataFrame(
            [
                {
                    "task": "task_A",
                    "model": "m1",
                    "pass_at_3": 0.5,
                    "ambiguity_class": "outcome-critical",
                    "has_run": True,
                    "n_trials": 3,
                    "avg_pct": 50,
                },
                {
                    "task": "task_B",
                    "model": "m1",
                    "pass_at_3": 0.8,
                    "ambiguity_class": "divergent",
                    "has_run": True,
                    "n_trials": 3,
                    "avg_pct": 80,
                },
                {
                    "task": "task_A",
                    "model": "m1",
                    "pass_at_3": 0.3,
                    "ambiguity_class": "benign",
                    "has_run": True,
                    "n_trials": 3,
                    "avg_pct": 30,
                },
            ]
        )
        original_baseline = {
            "task_A": {"m1": {"pass_at_3": 1.0, "avg_ckpt_acc": 0.9}},
            "task_B": {"m1": {"pass_at_3": 0.5, "avg_ckpt_acc": 0.6}},
        }
        return baseline_df, original_baseline

    def test_outcome_critical_uses_task_A_only(self):
        bl_df, orig = self._build_data()
        cls_tasks = set(bl_df.loc[bl_df["ambiguity_class"] == "outcome-critical", "task"].unique())
        assert cls_tasks == {"task_A"}
        vals = [orig[t]["m1"]["pass_at_3"] for t in cls_tasks if t in orig]
        assert vals == [1.0]

    def test_divergent_uses_task_B_only(self):
        bl_df, orig = self._build_data()
        cls_tasks = set(bl_df.loc[bl_df["ambiguity_class"] == "divergent", "task"].unique())
        assert cls_tasks == {"task_B"}
        vals = [orig[t]["m1"]["pass_at_3"] for t in cls_tasks if t in orig]
        assert vals == [0.5]

    def test_benign_uses_task_A_only(self):
        bl_df, orig = self._build_data()
        cls_tasks = set(bl_df.loc[bl_df["ambiguity_class"] == "benign", "task"].unique())
        assert cls_tasks == {"task_A"}
        vals = [orig[t]["m1"]["pass_at_3"] for t in cls_tasks if t in orig]
        assert vals == [1.0]

    def test_global_average_differs_from_class_scoped(self):
        _, orig = self._build_data()
        global_avg = sum(v["m1"]["pass_at_3"] for v in orig.values()) / len(orig)
        assert global_avg == 0.75
        assert orig["task_A"]["m1"]["pass_at_3"] != global_avg
        assert orig["task_B"]["m1"]["pass_at_3"] != global_avg


# ── Gain/Q computation ─────────────────────────────────────────────────


class TestGainPerQ:
    """Tests the Gain/Q metric from compare_tac_conditions.py."""

    @staticmethod
    def _gain_per_q(gain_pct: float, total_questions: float) -> float:
        if total_questions <= 0:
            return 0.0
        return gain_pct / total_questions

    def test_basic(self):
        assert abs(self._gain_per_q(10.0, 5) - 2.0) < 1e-10

    def test_zero_questions_returns_zero(self):
        assert self._gain_per_q(10.0, 0) == 0.0

    def test_negative_gain(self):
        assert self._gain_per_q(-5.0, 10) < 0

    def test_no_gain(self):
        assert self._gain_per_q(0.0, 10) == 0.0
