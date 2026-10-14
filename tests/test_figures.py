"""The data shaping behind each report figure, tested without plotting.

Every figure in the report is produced from results.csv by one of these
functions. The arithmetic is what can be wrong -- a lifelong curve that pools
across episode lengths tells the opposite story from one that does not
(6.67) -- so the shaping is separated from the drawing and pinned here.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.make_figures import (
    episodes_from_rows,
    lifelong_pooled,
    lifelong_fixed_length,
    paired_halves,
    by_modality,
    failure_taxonomy,
)


def _row(ep, si, success=False, modality="category", category="mirror", **kw):
    r = {"episode_id": str(ep), "subtask_index": str(si), "success": str(success),
         "modality": modality, "category": category, "spl": "0.0",
         "ever_matched": "False", "reason": "timeout"}
    r.update({k: str(v) for k, v in kw.items()})
    return r


class TestEpisodesFromRows:
    def test_a_new_episode_starts_at_subtask_zero(self):
        rows = [_row(0, 0), _row(0, 1), _row(1, 0), _row(1, 1), _row(1, 2)]

        eps = episodes_from_rows(rows)

        assert [len(e) for e in eps] == [2, 3]

    def test_repeated_episode_ids_do_not_merge_episodes(self):
        """GOAT-Bench numbers episodes per scene, so ids repeat (6.46).
        Splitting on subtask_index == 0 is what makes this correct."""
        rows = [_row(0, 0), _row(0, 1), _row(0, 0), _row(0, 1)]

        assert [len(e) for e in episodes_from_rows(rows)] == [2, 2]

    def test_empty(self):
        assert episodes_from_rows([]) == []


class TestLifelongCurves:
    def test_pooled_counts_every_episode_at_every_index_it_has(self):
        rows = [_row(0, 0, True), _row(0, 1, False),
                _row(1, 0, False), _row(1, 1, True), _row(1, 2, True)]

        idx, sr, n = lifelong_pooled(rows)

        assert idx == [0, 1, 2]
        assert n == [2, 2, 1]
        assert sr == pytest.approx([0.5, 0.5, 1.0])

    def test_fixed_length_uses_only_episodes_of_that_length(self):
        """The control for 6.67: composition cannot change across the x axis."""
        rows = [_row(0, 0, True), _row(0, 1, True),
                _row(1, 0, False), _row(1, 1, False), _row(1, 2, False)]

        idx, sr, n = lifelong_fixed_length(rows, length=2)

        assert idx == [0, 1]
        assert n == [1, 1]
        assert sr == pytest.approx([1.0, 1.0])

    def test_fixed_length_with_no_such_episode_is_empty(self):
        rows = [_row(0, 0), _row(0, 1)]

        assert lifelong_fixed_length(rows, length=7) == ([], [], [])


class TestPairedHalves:
    def test_a_second_half_improvement_is_counted(self):
        rows = [_row(0, 0, False), _row(0, 1, False),
                _row(0, 2, True), _row(0, 3, True)]

        better, worse, tied = paired_halves(rows)

        assert (better, worse, tied) == (1, 0, 0)

    def test_an_odd_length_episode_splits_with_the_extra_in_the_second_half(self):
        rows = [_row(0, 0, False), _row(0, 1, True), _row(0, 2, True)]

        better, worse, tied = paired_halves(rows)

        assert (better, worse, tied) == (1, 0, 0)

    def test_equal_halves_are_tied_not_assigned(self):
        rows = [_row(0, 0, True), _row(0, 1, True)]

        assert paired_halves(rows) == (0, 0, 1)


class TestByModality:
    def test_rates_per_modality(self):
        rows = [_row(0, 0, True, modality="image"),
                _row(0, 1, False, modality="image"),
                _row(0, 2, True, modality="language")]

        out = by_modality(rows)

        assert out["image"]["n"] == 2
        assert out["image"]["sr"] == pytest.approx(0.5)
        assert out["language"]["sr"] == pytest.approx(1.0)

    def test_a_missing_modality_is_absent_rather_than_zero(self):
        rows = [_row(0, 0, True, modality="image")]

        assert "language" not in by_modality(rows)


class TestFailureTaxonomy:
    def test_counts_by_reason(self):
        fails = [{"reason": "timeout"}, {"reason": "timeout"},
                 {"reason": "gave_up"}]

        out = failure_taxonomy(fails)

        assert out == [("timeout", 2), ("gave_up", 1)]

    def test_ordered_most_common_first(self):
        fails = [{"reason": "a"}, {"reason": "b"}, {"reason": "b"}]

        assert failure_taxonomy(fails)[0] == ("b", 2)
