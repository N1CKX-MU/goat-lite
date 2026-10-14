"""Pairing two eval arms, and the identity check that makes it meaningful.

results.csv has no scene_id, and GOAT-Bench numbers episodes per scene, so
218 dev subtasks share only 86 distinct (episode_id, subtask_index) pairs
(6.46). Arms can only be paired POSITIONALLY, and that is only sound if the
metadata agrees row by row. Done by hand four times in one session; the
hand-written version is where a silent mispairing would come from.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.compare_arms import ArmMismatch, pair_rows, sign_test


def _row(ep, si, cat="mirror", mod="category", success="False", **kw):
    r = {"episode_id": ep, "subtask_index": str(si), "category": cat,
         "modality": mod, "success": success, "ever_matched": "False",
         "detector_in_view": "False"}
    r.update(kw)
    return r


class TestSignTest:
    def test_no_disagreement_is_p_one(self):
        assert sign_test(0, 0) == pytest.approx(1.0)

    def test_a_single_one_way_flip_is_not_evidence(self):
        assert sign_test(1, 0) == pytest.approx(1.0)

    def test_six_one_way_flips_clear_the_conventional_bar(self):
        """The threshold quoted all through Section 7.10."""
        assert sign_test(5, 0) == pytest.approx(0.0625)
        assert sign_test(6, 0) == pytest.approx(0.03125)
        assert sign_test(6, 0) < 0.05

    def test_it_is_two_sided(self):
        assert sign_test(6, 0) == sign_test(0, 6)

    def test_balanced_disagreement_is_no_evidence(self):
        assert sign_test(5, 5) == pytest.approx(1.0)

    def test_the_measured_week10_arms_do_not_clear_it(self):
        """language 0.24 vs 0.28 was 4/0; the v2 detector on 218 was 7/4."""
        assert sign_test(4, 0) == pytest.approx(0.125)
        assert sign_test(7, 4) > 0.5


class TestPairRows:
    def test_pairs_on_the_common_prefix(self):
        a = [_row("e0", 0), _row("e0", 1), _row("e0", 2)]
        b = [_row("e0", 0), _row("e0", 1)]

        pa, pb = pair_rows(a, b)
        assert len(pa) == len(pb) == 2

    def test_repeated_episode_ids_pair_positionally(self):
        """Two scenes both numbering their episodes from 0 is the normal case,
        not an error -- matching on the id would cross-pair them."""
        a = [_row("0", 0, cat="mirror"), _row("0", 0, cat="mirror")]
        b = [_row("0", 0, cat="mirror"), _row("0", 0, cat="mirror")]

        pa, pb = pair_rows(a, b)
        assert len(pa) == 2

    def test_a_metadata_disagreement_raises_rather_than_silently_pairing(self):
        a = [_row("e0", 0, cat="mirror"), _row("e0", 1, cat="plant")]
        b = [_row("e0", 0, cat="mirror"), _row("e0", 1, cat="piano")]

        with pytest.raises(ArmMismatch) as exc:
            pair_rows(a, b)
        assert "row 1" in str(exc.value)

    def test_the_mismatch_names_both_sides(self):
        a = [_row("e0", 0, mod="language")]
        b = [_row("e0", 0, mod="image")]

        with pytest.raises(ArmMismatch) as exc:
            pair_rows(a, b)
        assert "language" in str(exc.value) and "image" in str(exc.value)

    def test_an_empty_arm_pairs_to_nothing(self):
        assert pair_rows([_row("e0", 0)], []) == ([], [])
