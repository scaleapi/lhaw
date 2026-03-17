"""Tests for evaluation/ — pass@k, pass^k, scoring, aggregation, grader, and TAC eval helpers."""

import json
import math
import os
import random
import tempfile

import pytest
from evaluation.common import grader
from evaluation.pass_k_metrics import (
    TaskTrialResults,
    TrialResult,
    aggregate_pass_metrics,
    compute_task_metrics,
    load_evaluation_results,
    pass_at_k,
    pass_hat_k,
)
from evaluation.scoring import (
    Checkpoint,
    Result,
    bonus_for_completing_any,
    bonus_for_completing_final,
)

# ── pass@k ──────────────────────────────────────────────────────────────


class TestPassAtK:

    def test_basic_formula(self):
        result = pass_at_k(10, 5, 3)
        expected = 1 - math.comb(5, 3) / math.comb(10, 3)
        assert abs(result - expected) < 1e-10

    def test_all_succeed(self):
        assert pass_at_k(10, 10, 3) == 1.0

    def test_none_succeed(self):
        assert pass_at_k(10, 0, 3) == 0.0

    def test_enough_successes_for_guarantee(self):
        assert pass_at_k(10, 9, 3) == 1.0
        assert pass_at_k(5, 4, 3) == 1.0

    def test_single_trial_success(self):
        assert pass_at_k(1, 1, 1) == 1.0

    def test_single_trial_failure(self):
        assert pass_at_k(1, 0, 1) == 0.0

    def test_k_equals_n(self):
        assert pass_at_k(3, 3, 3) == 1.0
        assert pass_at_k(3, 0, 3) == 0.0
        assert pass_at_k(3, 2, 3) == 1.0

    def test_k_equals_1_is_empirical_rate(self):
        assert abs(pass_at_k(10, 5, 1) - 0.5) < 1e-10
        assert abs(pass_at_k(10, 7, 1) - 0.7) < 1e-10
        assert abs(pass_at_k(100, 33, 1) - 0.33) < 1e-10

    def test_invalid_k_zero(self):
        with pytest.raises(ValueError):
            pass_at_k(10, 5, 0)

    def test_invalid_k_negative(self):
        with pytest.raises(ValueError):
            pass_at_k(10, 5, -1)

    def test_fallback_when_n_lt_k(self):
        result = pass_at_k(2, 1, 3)
        expected = 1.0 - (1.0 - 0.5) ** 3
        assert abs(result - expected) < 1e-10

    def test_n_equals_zero(self):
        result = pass_at_k(0, 0, 3)
        assert result == 0.0

    def test_monotone_increasing_with_k(self):
        values = [pass_at_k(10, 3, k) for k in range(1, 8)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-10

    def test_monotone_increasing_with_c(self):
        values = [pass_at_k(10, c, 3) for c in range(11)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1] + 1e-10

    def test_paper_values(self):
        assert pass_at_k(3, 2, 3) == 1.0
        assert pass_at_k(3, 1, 3) == 1.0
        assert pass_at_k(3, 0, 3) == 0.0

    def test_always_between_0_and_1(self):
        random.seed(42)
        for _ in range(100):
            n = random.randint(1, 50)
            c = random.randint(0, n)
            k = random.randint(1, min(n, 10))
            result = pass_at_k(n, c, k)
            assert 0.0 <= result <= 1.0, f"pass_at_k({n}, {c}, {k}) = {result}"


# ── pass^k ──────────────────────────────────────────────────────────────


class TestPassHatK:

    def test_basic_formula(self):
        result = pass_hat_k(10, 5, 3)
        expected = math.comb(5, 3) / math.comb(10, 3)
        assert abs(result - expected) < 1e-10

    def test_all_succeed(self):
        assert pass_hat_k(10, 10, 3) == 1.0

    def test_none_succeed(self):
        assert pass_hat_k(10, 0, 3) == 0.0

    def test_not_enough_successes(self):
        assert pass_hat_k(10, 2, 3) == 0.0

    def test_single_trial(self):
        assert pass_hat_k(1, 1, 1) == 1.0
        assert pass_hat_k(1, 0, 1) == 0.0

    def test_k_equals_1_is_empirical_rate(self):
        assert abs(pass_hat_k(10, 5, 1) - 0.5) < 1e-10

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            pass_hat_k(10, 5, 0)

    def test_monotone_decreasing_with_k(self):
        values = [pass_hat_k(10, 5, k) for k in range(1, 6)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1] - 1e-10

    def test_n_equals_zero(self):
        assert pass_hat_k(0, 0, 3) == 0.0

    def test_pass_hat_always_lte_pass_at(self):
        random.seed(42)
        for _ in range(100):
            n = random.randint(1, 50)
            c = random.randint(0, n)
            k = random.randint(1, min(n, 10))
            hat = pass_hat_k(n, c, k)
            at = pass_at_k(n, c, k)
            assert hat <= at + 1e-10, f"pass^{k}({n},{c})={hat} > pass@{k}={at}"

    def test_specific_known_values(self):
        assert abs(pass_hat_k(10, 5, 3) - 1 / 12) < 1e-10
        assert pass_hat_k(3, 3, 3) == 1.0
        assert pass_hat_k(10, 2, 3) == 0.0


# ── compute_task_metrics / aggregate ────────────────────────────────────


class TestComputeTaskMetrics:
    def test_basic(self):
        trials = [
            TrialResult(success=True, score=3, total=3),
            TrialResult(success=True, score=3, total=3),
            TrialResult(success=False, score=1, total=3),
        ]
        results = TaskTrialResults(task_name="test_task", trials=trials)
        metrics = compute_task_metrics(results, k=3)
        assert metrics["task"] == "test_task"
        assert metrics["n_trials"] == 3
        assert metrics["n_success"] == 2
        assert abs(metrics["pass_1"] - 2 / 3) < 1e-10
        assert "pass_3" in metrics
        assert "pass_at_3" in metrics

    def test_empty(self):
        results = TaskTrialResults(task_name="empty")
        metrics = compute_task_metrics(results, k=3)
        assert metrics["n_trials"] == 0
        assert metrics["pass_1"] == 0.0
        assert metrics["avg_score"] == 0.0
        assert metrics["max_score"] == 0.0

    def test_all_pass(self):
        trials = [TrialResult(success=True, score=1, total=1) for _ in range(5)]
        results = TaskTrialResults(task_name="all_pass", trials=trials)
        metrics = compute_task_metrics(results, k=3)
        assert metrics["pass_at_3"] == 1.0
        assert metrics["pass_3"] == 1.0
        assert metrics["avg_percent"] == 100.0

    def test_partial_scores(self):
        trials = [
            TrialResult(success=False, score=2, total=5),
            TrialResult(success=False, score=3, total=5),
            TrialResult(success=True, score=5, total=5),
        ]
        results = TaskTrialResults(task_name="partial", trials=trials)
        metrics = compute_task_metrics(results, k=3)
        assert abs(metrics["avg_score"] - (2 + 3 + 5) / 3) < 1e-10
        assert metrics["max_score"] == 5.0
        assert abs(metrics["avg_percent"] - (40 + 60 + 100) / 3) < 1e-10

    def test_zero_total_checkpoint(self):
        trials = [TrialResult(success=False, score=0, total=0)]
        results = TaskTrialResults(task_name="zero_total", trials=trials)
        metrics = compute_task_metrics(results, k=1)
        assert metrics["avg_percent"] == 0.0


class TestAggregatePassMetrics:
    def test_basic(self):
        task1 = TaskTrialResults("t1", [TrialResult(True, 1, 1)] * 3)
        task2 = TaskTrialResults("t2", [TrialResult(False, 0, 1)] * 3)
        agg = aggregate_pass_metrics([task1, task2], k=3)
        assert agg["n_tasks"] == 2
        assert abs(agg["avg_pass_1"] - 0.5) < 1e-10
        assert len(agg["task_details"]) == 2

    def test_empty(self):
        agg = aggregate_pass_metrics([], k=3)
        assert agg["n_tasks"] == 0

    def test_single_task(self):
        task = TaskTrialResults("t1", [TrialResult(True, 1, 1)] * 3)
        agg = aggregate_pass_metrics([task], k=3)
        assert agg["avg_pass_1"] == 1.0
        assert agg["avg_pass_3"] == 1.0
        assert agg["avg_pass_at_3"] == 1.0


# ── Checkpoint scoring ──────────────────────────────────────────────────


class TestCheckpoint:
    def test_valid(self):
        cp = Checkpoint(total=5, result=3)
        assert cp.total == 5 and cp.result == 3

    def test_zero(self):
        Checkpoint(total=0, result=0)

    def test_result_exceeds_total(self):
        with pytest.raises(ValueError):
            Checkpoint(total=3, result=5)

    def test_negative_total(self):
        with pytest.raises(ValueError):
            Checkpoint(total=-1, result=0)

    def test_negative_result(self):
        with pytest.raises(ValueError):
            Checkpoint(total=5, result=-1)

    def test_equal_total_and_result(self):
        cp = Checkpoint(total=10, result=10)
        assert cp.result == cp.total


class TestResult:
    def test_default_scoring(self):
        r = Result(checkpoints=[Checkpoint(3, 2), Checkpoint(5, 4)])
        assert r.final_score == {"total": 8, "result": 6}

    def test_from_dict(self):
        data = {"checkpoints": [{"total": 3, "result": 2}, {"total": 5, "result": 5}]}
        r = Result.from_dict(data)
        assert r.final_score == {"total": 8, "result": 7}

    def test_to_dict_roundtrip(self):
        r = Result(checkpoints=[Checkpoint(3, 3), Checkpoint(2, 1)])
        d = r.to_dict()
        assert d["checkpoints"] == [{"total": 3, "result": 3}, {"total": 2, "result": 1}]
        assert d["final_score"] == {"total": 5, "result": 4}

    def test_from_dict_missing_checkpoints(self):
        with pytest.raises(KeyError):
            Result.from_dict({})

    def test_from_dict_bad_type(self):
        with pytest.raises(TypeError):
            Result.from_dict("not a dict")

    def test_custom_scoring_strategy(self):
        custom = lambda cps: {"total": len(cps), "result": sum(1 for cp in cps if cp.result > 0)}
        r = Result(
            checkpoints=[Checkpoint(5, 0), Checkpoint(3, 2), Checkpoint(4, 4)],
            scoring_strategy=custom,
        )
        assert r.final_score == {"total": 3, "result": 2}

    def test_empty_checkpoints(self):
        r = Result(checkpoints=[])
        assert r.final_score == {"total": 0, "result": 0}


class TestBonusForCompletingFinal:
    def test_final_complete(self):
        cps = [Checkpoint(5, 2), Checkpoint(3, 1), Checkpoint(4, 4)]
        assert bonus_for_completing_final(cps) == {"total": 12, "result": 12}

    def test_final_incomplete(self):
        cps = [Checkpoint(5, 5), Checkpoint(3, 1), Checkpoint(4, 2)]
        assert bonus_for_completing_final(cps) == {"total": 12, "result": 8}

    def test_empty(self):
        assert bonus_for_completing_final([]) == {"total": 0, "result": 0}

    def test_single_checkpoint_full(self):
        assert bonus_for_completing_final([Checkpoint(5, 5)]) == {"total": 5, "result": 5}

    def test_single_checkpoint_partial(self):
        assert bonus_for_completing_final([Checkpoint(5, 3)]) == {"total": 5, "result": 3}


class TestBonusForCompletingAny:
    def test_later_checkpoint_complete(self):
        cps = [Checkpoint(5, 2), Checkpoint(3, 3)]
        assert bonus_for_completing_any(cps) == {"total": 8, "result": 8}

    def test_none_complete(self):
        cps = [Checkpoint(5, 2), Checkpoint(3, 1)]
        assert bonus_for_completing_any(cps) == {"total": 8, "result": 3}

    def test_first_complete(self):
        cps = [Checkpoint(5, 5), Checkpoint(3, 1)]
        assert bonus_for_completing_any(cps) == {"total": 8, "result": 6}

    def test_empty(self):
        assert bonus_for_completing_any([]) == {"total": 0, "result": 0}

    def test_all_partial(self):
        cps = [Checkpoint(5, 2), Checkpoint(3, 1), Checkpoint(4, 2)]
        assert bonus_for_completing_any(cps) == {"total": 12, "result": 5}

    def test_only_middle_complete(self):
        cps = [Checkpoint(5, 2), Checkpoint(3, 3), Checkpoint(4, 1)]
        assert bonus_for_completing_any(cps) == {"total": 12, "result": 9}


# ── compute_task_metrics with varying k ────────────────────────────────


class TestComputeTaskMetricsVaryingK:
    def test_k_equals_1(self):
        trials = [TrialResult(True, 1, 1)] * 2 + [TrialResult(False, 0, 1)]
        r = TaskTrialResults("t", trials)
        m = compute_task_metrics(r, k=1)
        assert "pass_1" in m and "pass_at_1" in m
        assert abs(m["pass_1"] - 2 / 3) < 1e-10
        assert abs(m["pass_at_1"] - 2 / 3) < 1e-10

    def test_k_equals_5(self):
        trials = [TrialResult(True, 1, 1)] * 7 + [TrialResult(False, 0, 1)] * 3
        r = TaskTrialResults("t", trials)
        m = compute_task_metrics(r, k=5)
        assert "pass_5" in m and "pass_at_5" in m
        assert 0.0 <= m["pass_5"] <= 1.0
        assert 0.0 <= m["pass_at_5"] <= 1.0
        assert m["pass_at_5"] >= m["pass_5"]

    def test_non_uniform_totals(self):
        trials = [
            TrialResult(success=False, score=3, total=5),
            TrialResult(success=True, score=4, total=4),
            TrialResult(success=False, score=1, total=3),
        ]
        r = TaskTrialResults("t", trials)
        m = compute_task_metrics(r, k=3)
        assert abs(m["avg_score"] - (3 + 4 + 1) / 3) < 1e-10
        expected_pct = (60 + 100 + 100 / 3) / 3
        assert abs(m["avg_percent"] - expected_pct) < 1e-10


# ── load_evaluation_results ────────────────────────────────────────────


class TestLoadEvaluationResults:
    def test_loads_from_json_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "finance_budget_variance"
            model = "test-model"
            det_dir = os.path.join(tmp, "deterministic", task, model)
            os.makedirs(det_dir)
            for i, (score, total) in enumerate([(3, 5), (5, 5), (2, 5)]):
                with open(os.path.join(det_dir, f"evaluation_{i}.json"), "w") as f:
                    json.dump({"final_score": {"result": score, "total": total}}, f)
            results = load_evaluation_results(tmp, [task], model)
            assert len(results) == 1
            assert results[0].task_name == task
            assert results[0].n == 3
            assert results[0].c == 1  # only 5/5 is success

    def test_missing_task_dir_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = load_evaluation_results(tmp, ["nonexistent_task"], "model")
            assert results == []

    def test_malformed_json_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            task = "ds_fix_table_values_and_missing_answers"
            model = "test-model"
            det_dir = os.path.join(tmp, "deterministic", task, model)
            os.makedirs(det_dir)
            with open(os.path.join(det_dir, "evaluation_0.json"), "w") as f:
                f.write("NOT VALID JSON")
            with open(os.path.join(det_dir, "evaluation_1.json"), "w") as f:
                json.dump({"final_score": {"result": 2, "total": 2}}, f)
            results = load_evaluation_results(tmp, [task], model)
            assert len(results) == 1
            assert results[0].n == 1  # only the valid file


# ── grader decorator ──────────────────────────────────────────────────


class TestGraderDecorator:
    def test_normal_return(self):
        @grader
        def check():
            return (True, 5)

        assert check() == (True, 5)

    def test_none_becomes_false(self):
        @grader
        def check():
            return None

        assert check() is False

    def test_exception_becomes_false(self):
        @grader
        def check():
            raise RuntimeError("crash")

        assert check() is False

    def test_scalar_false_passthrough(self):
        @grader
        def check():
            return False

        assert check() is False

    def test_preserves_function_name(self):
        @grader
        def grade_checkpoint1():
            return True

        assert grade_checkpoint1.__name__ == "grade_checkpoint1"
