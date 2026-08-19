"""Language matching must respect the goal's category.

A GOAT-Bench language goal describes an instance OF a known category. Scoring
the description against every remembered instance regardless of class let a
'window glass' node win a 'microwave' goal on a CLIP similarity of 0.257 --
with no microwave in memory at all -- after which the agent navigated to the
window and stopped 12.5 m from the real target.
"""

import numpy as np
import pytest

from src.matching.goal_matcher import GoalMatcher
from src.sim.env import GoalSpec


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class _Node:
    def __init__(self, cls_name, embed, xyz=(0.0, 0.0, 0.0), conf=0.5):
        self.cls_name = cls_name
        self.cls_id = 0
        self.clip_embed = _unit(embed)
        self.world_xyz = np.array(xyz, dtype=np.float64)
        self.confidence = conf


class _DB:
    def __init__(self, nodes):
        self._nodes = nodes

    def all_nodes(self):
        return list(self._nodes)

    def query_by_class_name(self, name):
        return [n for n in self._nodes if n.cls_name == name]


class _Encoder:
    """Text encodes to a fixed vector; nodes are constructed relative to it."""
    def __init__(self, text_vec):
        self._t = _unit(text_vec)

    def encode_text(self, texts):
        return np.stack([self._t for _ in texts])


def _matcher(text_vec, threshold=0.24):
    return GoalMatcher(_Encoder(text_vec), language_threshold=threshold)


class TestLanguageRespectsCategory:
    def test_does_not_match_a_different_class(self):
        """The regression: a decoy of the wrong class must never be returned."""
        # decoy is highly similar to the text, but is not a microwave
        decoy = _Node("window glass", [1.0, 0.0, 0.0], xyz=(9.0, 0.0, 9.0))
        db = _DB([decoy])
        goal = GoalSpec(modality="language",
                        value="the microwave above the oven",
                        category="microwave")
        node, score = _matcher([1.0, 0.0, 0.0]).match(goal, db)
        assert node is None, "matched an instance of the wrong category"

    def test_matches_the_right_class_even_if_less_similar(self):
        decoy = _Node("window glass", [1.0, 0.0, 0.0])
        target = _Node("microwave", [0.7, 0.7, 0.0], xyz=(1.0, 0.0, 1.0))
        db = _DB([decoy, target])
        goal = GoalSpec(modality="language", value="the microwave",
                        category="microwave")
        node, score = _matcher([1.0, 0.0, 0.0]).match(goal, db)
        assert node is target

    def test_picks_best_among_same_class(self):
        near = _Node("mirror", [0.6, 0.8, 0.0], xyz=(1.0, 0.0, 0.0))
        best = _Node("mirror", [1.0, 0.0, 0.0], xyz=(5.0, 0.0, 5.0))
        db = _DB([near, best])
        goal = GoalSpec(modality="language",
                        value="the mirror next to the sink", category="mirror")
        node, score = _matcher([1.0, 0.0, 0.0]).match(goal, db)
        assert node is best
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_threshold_still_applies_within_the_class(self):
        weak = _Node("mirror", [0.0, 1.0, 0.0])
        db = _DB([weak])
        goal = GoalSpec(modality="language", value="the mirror",
                        category="mirror")
        node, score = _matcher([1.0, 0.0, 0.0]).match(goal, db)
        assert node is None, "similarity below threshold should not match"

    def test_no_category_falls_back_to_searching_all(self):
        """Older callers that build a GoalSpec without a category still work."""
        n = _Node("mirror", [1.0, 0.0, 0.0])
        db = _DB([n])
        goal = GoalSpec(modality="language", value="the mirror")  # no category
        node, _ = _matcher([1.0, 0.0, 0.0]).match(goal, db)
        assert node is n

    def test_empty_memory_returns_none(self):
        goal = GoalSpec(modality="language", value="the mirror",
                        category="mirror")
        node, score = _matcher([1.0, 0.0, 0.0]).match(goal, _DB([]))
        assert node is None
        assert score == 0.0
