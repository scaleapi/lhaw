"""Tests for synthetic/ — pipeline logic: deduplication, scoring, sorting, clean_prompt."""

from synthetic.pipeline import SegmentExtractor, VariantGenerator, clean_prompt

from synthetic.core import Dimension, Segment

# ── Segment: mutable default isolation (dataclass pitfall) ──────────────


class TestSegmentMutableDefaults:
    def test_mutable_defaults_isolation(self):
        s1 = Segment(id="S1", dimension=Dimension.GOAL)
        s2 = Segment(id="S2", dimension=Dimension.GOAL)
        s1.checkpoint_refs.append("CP1")
        s1.metadata["key"] = "val"
        assert s2.checkpoint_refs == []
        assert s2.metadata == {}


# ── Priority score computation ──────────────────────────────────────────


class TestPriorityScore:
    def test_formula(self):
        seg = Segment(id="S1", dimension=Dimension.GOAL, criticality=0.8, guessability=0.3)
        seg.priority_score = seg.criticality * (1 - seg.guessability)
        assert abs(seg.priority_score - 0.56) < 1e-10

    def test_high_criticality_low_guessability(self):
        seg = Segment(id="S1", dimension=Dimension.INPUT, criticality=1.0, guessability=0.0)
        seg.priority_score = seg.criticality * (1 - seg.guessability)
        assert seg.priority_score == 1.0

    def test_low_criticality_high_guessability(self):
        seg = Segment(id="S1", dimension=Dimension.CONTEXT, criticality=0.0, guessability=1.0)
        seg.priority_score = seg.criticality * (1 - seg.guessability)
        assert seg.priority_score == 0.0

    def test_equal_criticality_guessability(self):
        seg = Segment(id="S1", dimension=Dimension.GOAL, criticality=0.5, guessability=0.5)
        seg.priority_score = seg.criticality * (1 - seg.guessability)
        assert abs(seg.priority_score - 0.25) < 1e-10


# ── Segment sorting by priority ─────────────────────────────────────────


class TestSegmentSorting:
    def _seg(self, id: str, priority: float) -> Segment:
        return Segment(id=id, dimension=Dimension.GOAL, priority_score=priority)

    def test_descending_sort(self):
        segs = [self._seg("S3", 0.2), self._seg("S1", 0.9), self._seg("S2", 0.5)]
        sorted_segs = sorted(segs, key=lambda s: s.priority_score, reverse=True)
        assert [s.id for s in sorted_segs] == ["S1", "S2", "S3"]


# ── _score_combos ───────────────────────────────────────────────────────


class TestScoreCombos:
    def _gen(self) -> VariantGenerator:
        return VariantGenerator.__new__(VariantGenerator)

    def _seg(
        self, id: str, priority: float, used_in_traj: bool = False, checkpoint_refs: list = None
    ) -> Segment:
        return Segment(
            id=id,
            dimension=Dimension.GOAL,
            priority_score=priority,
            is_used_in_trajectory=used_in_traj,
            checkpoint_refs=checkpoint_refs or [],
        )

    def test_auto_strategy_grounding_dominates(self):
        grounded = self._seg("S1", 0.1, used_in_traj=True, checkpoint_refs=["CP1"])
        ungrounded = self._seg("S2", 0.9, used_in_traj=False)
        scored = self._gen()._score_combos([(grounded,), (ungrounded,)], "auto")
        scores = {s[0][0].id: s[1] for s in scored}
        assert scores["S1"] > scores["S2"]

    def test_auto_strategy_trajectory_beats_checkpoint_only(self):
        traj_only = self._seg("S1", 0.1, used_in_traj=True)
        ckpt_only = self._seg("S2", 0.9, used_in_traj=False, checkpoint_refs=["CP1"])
        scored = self._gen()._score_combos([(traj_only,), (ckpt_only,)], "auto")
        scores = {s[0][0].id: s[1] for s in scored}
        assert scores["S1"] > scores["S2"]

    def test_auto_strategy_values(self):
        seg = self._seg("S1", 0.7, used_in_traj=True, checkpoint_refs=["CP1", "CP2"])
        scored = self._gen()._score_combos([(seg,)], "auto")
        expected = 1 * 1000 + 1 * 100 + 0.7
        assert abs(scored[0][1] - expected) < 1e-10

    def test_priority_mean_strategy(self):
        s1 = self._seg("S1", 0.8)
        s2 = self._seg("S2", 0.3)
        scored = self._gen()._score_combos([(s1, s2)], "priority_mean")
        assert abs(scored[0][1] - 0.55) < 1e-10

    def test_priority_max_strategy(self):
        s1 = self._seg("S1", 0.8)
        s2 = self._seg("S2", 0.3)
        scored = self._gen()._score_combos([(s1, s2)], "priority_max")
        assert abs(scored[0][1] - 0.8) < 1e-10

    def test_multi_combo_sorting(self):
        s1 = self._seg("S1", 0.9, used_in_traj=True)
        s2 = self._seg("S2", 0.5)
        s3 = self._seg("S3", 0.3)
        combos = [(s1,), (s2,), (s3,), (s1, s2)]
        scored = self._gen()._score_combos(combos, "auto")
        scored.sort(key=lambda x: x[1], reverse=True)
        top_combo_ids = {s.id for s in scored[0][0]}
        assert "S1" in top_combo_ids


# ── _format_segment ────────────────────────────────────────────────────


class TestFormatSegment:
    def _gen(self) -> VariantGenerator:
        return VariantGenerator.__new__(VariantGenerator)

    def test_basic_format(self):
        seg = Segment(id="S1", dimension=Dimension.INPUT, value="data.csv")
        assert self._gen()._format_segment(seg) == 'S1 [INPUT] "data.csv"'

    def test_grounded_with_checkpoint(self):
        seg = Segment(
            id="S2",
            dimension=Dimension.GOAL,
            value="report.xlsx",
            is_used_in_trajectory=True,
            first_use_pct=0.15,
            checkpoint_refs=["CP1"],
        )
        result = self._gen()._format_segment(seg)
        assert result == 'S2 [GOAL] "report.xlsx" (used at 15%, affects CP1)'

    def test_grounded_without_pct(self):
        seg = Segment(
            id="S3",
            dimension=Dimension.CONSTRAINT,
            value="round up",
            is_used_in_trajectory=True,
            first_use_pct=None,
        )
        assert self._gen()._format_segment(seg) == 'S3 [CONSTRAINT] "round up" (used in trajectory)'

    def test_multiple_checkpoints(self):
        seg = Segment(
            id="S5",
            dimension=Dimension.GOAL,
            value="summary",
            is_used_in_trajectory=True,
            first_use_pct=0.5,
            checkpoint_refs=["CP1", "CP2", "CP3"],
        )
        result = self._gen()._format_segment(seg)
        assert result == 'S5 [GOAL] "summary" (used at 50%, affects CP1, CP2, CP3)'

    def test_no_grounding(self):
        seg = Segment(id="S6", dimension=Dimension.CONTEXT, value="2024 fiscal year")
        assert self._gen()._format_segment(seg) == 'S6 [CONTEXT] "2024 fiscal year"'


# ── clean_prompt (all 5 rules) ─────────────────────────────────────────


class TestCleanPrompt:
    # -- Double commas (list item removal) --
    def test_double_comma_cleaned(self):
        result = clean_prompt("save file, , and close", "save file, report.csv, and close")
        assert result == "save file, and close"

    def test_double_comma_preserved_if_in_original(self):
        assert ", ," in clean_prompt("a, , b", "a, , b")

    # -- Double spaces (word removal) --
    def test_double_spaces_cleaned(self):
        result = clean_prompt("save  the file", "save report.csv the file")
        assert result == "save the file"

    # -- "and," artifact --
    def test_and_comma_cleaned(self):
        result = clean_prompt("open and, close", "open and the_value, close")
        assert "and," not in result

    # -- Orphaned comma at paragraph start --
    def test_newline_comma_uppercase(self):
        result = clean_prompt(
            "first paragraph\n\n, second part", "first paragraph\n\nthe info second part"
        )
        assert result == "first paragraph\nSecond part"

    # -- Duplicate list markers --
    def test_duplicate_list_markers(self):
        result = clean_prompt("items:\n-\n- first", "items:\n- removed\n- first")
        assert result == "items:\n- first"

    # -- Empty bullet lines --
    def test_empty_bullet_cleaned(self):
        result = clean_prompt("items:\n- \n- second", "items:\n- removed\n- second")
        assert result == "items:\n- second"

    def test_empty_star_bullet_cleaned(self):
        result = clean_prompt("items:\n* \n* second", "items:\n* removed\n* second")
        assert result == "items:\n* second"

    # -- Excessive blank lines --
    def test_triple_blank_lines_collapsed(self):
        result = clean_prompt("first\n\n\n\nsecond", "first\n\nremoved\n\nsecond")
        assert result == "first\n\nsecond"

    def test_double_blank_preserved(self):
        prompt = "first\n\nsecond"
        assert clean_prompt(prompt, prompt) == prompt

    # -- Safety: patterns in original are preserved --
    def test_preserves_clean_prompt(self):
        original = "Create a CSV report with sales data"
        assert clean_prompt(original, original) == original

    # -- Trailing whitespace --
    def test_trailing_whitespace_stripped(self):
        for line in clean_prompt("line one   \nline two  ", None).split("\n"):
            assert line == line.rstrip()

    # -- No original → all rules apply --
    def test_no_original_applies_all(self):
        result = clean_prompt("fix  the, , spacing", None)
        assert "  " not in result and ", ," not in result

    # -- Edge cases --
    def test_multiline_preserved(self):
        prompt = "line1\nline2\nline3"
        assert clean_prompt(prompt, prompt) == prompt


# ── _deduplicate_segments ───────────────────────────────────────────────


class TestDeduplicateSegments:
    def _seg(self, id: str, dim: Dimension, value: str) -> Segment:
        return Segment(id=id, dimension=dim, value=value)

    def _ext(self) -> SegmentExtractor:
        return SegmentExtractor.__new__(SegmentExtractor)

    def test_no_duplicates(self):
        segs = [
            self._seg("S1", Dimension.INPUT, "data.csv"),
            self._seg("S2", Dimension.GOAL, "report.xlsx"),
        ]
        assert len(self._ext()._deduplicate_segments(segs)) == 2

    def test_substring_same_dimension(self):
        segs = [
            self._seg("S1", Dimension.INPUT, "data.csv"),
            self._seg("S2", Dimension.INPUT, "data"),
        ]
        result = self._ext()._deduplicate_segments(segs)
        assert len(result) == 1 and result[0].value == "data.csv"

    def test_substring_different_dimension_preserved(self):
        segs = [
            self._seg("S1", Dimension.INPUT, "report.csv"),
            self._seg("S2", Dimension.GOAL, "report"),
        ]
        assert len(self._ext()._deduplicate_segments(segs)) == 2

    def test_empty(self):
        assert self._ext()._deduplicate_segments([]) == []

    def test_empty_value_skipped(self):
        segs = [self._seg("S1", Dimension.INPUT, ""), self._seg("S2", Dimension.INPUT, "data.csv")]
        result = self._ext()._deduplicate_segments(segs)
        assert len(result) == 1 and result[0].value == "data.csv"

    def test_case_insensitive_dedup(self):
        segs = [
            self._seg("S1", Dimension.INPUT, "Data.CSV"),
            self._seg("S2", Dimension.INPUT, "data.csv"),
        ]
        assert len(self._ext()._deduplicate_segments(segs)) == 1

    def test_exact_duplicate_same_dimension(self):
        segs = [
            self._seg("S1", Dimension.GOAL, "output.xlsx"),
            self._seg("S2", Dimension.GOAL, "output.xlsx"),
        ]
        assert len(self._ext()._deduplicate_segments(segs)) == 1
