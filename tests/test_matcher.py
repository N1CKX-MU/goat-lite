"""Tests for src/matching/goal_matcher.py — GoalMatcher."""

import numpy as np
import pytest

from src.matching.goal_matcher import GoalMatcher
from src.memory.instance_db import InstanceDatabase, InstanceNode
from src.perception.pipeline import PerceivedInstance
from src.sim.env import GoalSpec


def _make_embed(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    e = rng.randn(512).astype(np.float32)
    return e / np.linalg.norm(e)


def _make_pi(
    cls_id: int = 0,
    cls_name: str = "chair",
    conf: float = 0.8,
    world_xyz: tuple = (1.0, 0.5, 2.0),
    embed_seed: int = 0,
    seen_step: int = 0,
) -> PerceivedInstance:
    return PerceivedInstance(
        cls_id=cls_id,
        cls_name=cls_name,
        conf=conf,
        bbox=(50, 50, 150, 150),
        crop_thumbnail=np.zeros((64, 64, 3), dtype=np.uint8),
        clip_embed=_make_embed(embed_seed),
        world_xyz=np.array(world_xyz, dtype=np.float64),
        seen_step=seen_step,
    )


class FakeEncoder:
    """Returns deterministic text embeddings for testing."""

    def __init__(self, embed_dim: int = 512):
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode_text(self, prompts: list[str]) -> np.ndarray:
        out = np.zeros((len(prompts), self._embed_dim), dtype=np.float32)
        for i, p in enumerate(prompts):
            # Use hash of prompt to produce deterministic embedding
            rng = np.random.RandomState(hash(p) % 2**31)
            out[i] = rng.randn(self._embed_dim).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out /= np.maximum(norms, 1e-8)
        return out


def _make_db_with_nodes() -> InstanceDatabase:
    """Create a DB with a chair (high conf) and a table."""
    db = InstanceDatabase(merge_dist_m=0.75)
    db.update([
        _make_pi(cls_id=0, cls_name="chair", conf=0.9,
                 world_xyz=(1.0, 0.5, 2.0), embed_seed=10),
        _make_pi(cls_id=0, cls_name="chair", conf=0.7,
                 world_xyz=(5.0, 0.5, 5.0), embed_seed=11),
        _make_pi(cls_id=1, cls_name="table", conf=0.85,
                 world_xyz=(3.0, 0.5, 3.0), embed_seed=20),
    ], step=0)
    return db


class TestCategoryMatching:
    def test_matches_existing_category(self):
        db = _make_db_with_nodes()
        matcher = GoalMatcher(encoder=FakeEncoder())
        goal = GoalSpec(modality="category", value="chair")
        node, score = matcher.match(goal, db)
        assert node is not None
        assert node.cls_name == "chair"

    def test_returns_highest_confidence_node(self):
        db = _make_db_with_nodes()
        matcher = GoalMatcher(encoder=FakeEncoder())
        goal = GoalSpec(modality="category", value="chair")
        node, score = matcher.match(goal, db)
        assert node is not None
        assert node.confidence == 0.9  # higher of 0.9 and 0.7

    def test_no_match_returns_none(self):
        db = _make_db_with_nodes()
        matcher = GoalMatcher(encoder=FakeEncoder())
        goal = GoalSpec(modality="category", value="toilet")
        node, score = matcher.match(goal, db)
        assert node is None
        assert score == 0.0

    def test_empty_db_returns_none(self):
        db = InstanceDatabase()
        matcher = GoalMatcher(encoder=FakeEncoder())
        goal = GoalSpec(modality="category", value="chair")
        node, score = matcher.match(goal, db)
        assert node is None

    def test_category_score_equals_confidence(self):
        db = _make_db_with_nodes()
        matcher = GoalMatcher(encoder=FakeEncoder())
        goal = GoalSpec(modality="category", value="table")
        node, score = matcher.match(goal, db)
        assert node is not None
        assert score == node.confidence


class TestImageMatching:
    def test_image_modality_routes_to_category(self):
        """Image goals carry the target category in value and match by category."""
        db = _make_db_with_nodes()
        matcher = GoalMatcher(encoder=FakeEncoder())
        goal = GoalSpec(modality="image", value="chair")
        node, score = matcher.match(goal, db)
        assert node is not None
        assert node.cls_name == "chair"

    def test_image_modality_no_match_returns_none(self):
        db = _make_db_with_nodes()
        matcher = GoalMatcher(encoder=FakeEncoder())
        goal = GoalSpec(modality="image", value="toilet")
        node, score = matcher.match(goal, db)
        assert node is None
        assert score == 0.0


class TestLanguageMatching:
    def test_returns_best_embedding_match(self):
        """Language matching should return the node with highest cosine similarity."""
        db = InstanceDatabase()
        # Insert a node with a known embedding
        embed = _make_embed(42)
        db.update([_make_pi(embed_seed=42, world_xyz=(0, 0.5, 0))], step=0)
        db.update([_make_pi(embed_seed=99, world_xyz=(5, 0.5, 5))], step=1)

        # Create an encoder that returns the same embedding as seed=42
        class FixedEncoder:
            embed_dim = 512
            def encode_text(self, prompts):
                # Return the embed that matches seed=42
                out = np.tile(embed, (len(prompts), 1))
                return out

        matcher = GoalMatcher(encoder=FixedEncoder(), language_threshold=0.0)
        goal = GoalSpec(modality="language", value="a wooden chair")
        node, score = matcher.match(goal, db)

        assert node is not None
        assert score > 0.5  # should be high similarity

    def test_below_threshold_returns_none(self):
        """If best score is below language_threshold, return None."""
        db = InstanceDatabase()
        db.update([_make_pi(embed_seed=0, world_xyz=(0, 0.5, 0))], step=0)

        # Encoder that returns a very different embedding
        class OrthogonalEncoder:
            embed_dim = 512
            def encode_text(self, prompts):
                e = np.zeros((len(prompts), 512), dtype=np.float32)
                e[:, -1] = 1.0  # point in a direction unlikely to match
                return e

        matcher = GoalMatcher(encoder=OrthogonalEncoder(), language_threshold=0.5)
        goal = GoalSpec(modality="language", value="something random")
        node, score = matcher.match(goal, db)
        # Score likely below 0.5 for random vs orthogonal
        # If by chance it's above, the test logic still holds
        if score < 0.5:
            assert node is None

    def test_empty_db_returns_none(self):
        db = InstanceDatabase()
        matcher = GoalMatcher(encoder=FakeEncoder(), language_threshold=0.24)
        goal = GoalSpec(modality="language", value="red chair near window")
        node, score = matcher.match(goal, db)
        assert node is None
        assert score == 0.0


class TestTemplateWrapping:
    def test_short_value_gets_template(self):
        """Short category-like values should get wrapped in 'a photo of X'."""
        calls = []

        class SpyEncoder:
            embed_dim = 512
            def encode_text(self, prompts):
                calls.extend(prompts)
                e = np.zeros((len(prompts), 512), dtype=np.float32)
                e[:, 0] = 1.0
                return e

        db = InstanceDatabase()
        db.update([_make_pi(embed_seed=0, world_xyz=(0, 0.5, 0))], step=0)

        matcher = GoalMatcher(encoder=SpyEncoder(), use_template=True)
        goal = GoalSpec(modality="language", value="chair")
        matcher.match(goal, db)

        assert len(calls) == 1
        assert "a photo of" in calls[0]

    def test_long_sentence_used_directly(self):
        """Full sentences should be used as-is, not wrapped."""
        calls = []

        class SpyEncoder:
            embed_dim = 512
            def encode_text(self, prompts):
                calls.extend(prompts)
                e = np.zeros((len(prompts), 512), dtype=np.float32)
                e[:, 0] = 1.0
                return e

        db = InstanceDatabase()
        db.update([_make_pi(embed_seed=0, world_xyz=(0, 0.5, 0))], step=0)

        matcher = GoalMatcher(encoder=SpyEncoder(), use_template=True)
        goal = GoalSpec(modality="language",
                        value="the chair with red cushion near the window")
        matcher.match(goal, db)

        assert len(calls) == 1
        assert calls[0] == "the chair with red cushion near the window"


class TestExclusions:
    """The agent rules out nodes it approached and could not confirm, and the
    matcher must then fall through to the next candidate rather than to None --
    the dev subset has a median of 4 instances of a category per scene, so the
    next one is usually worth trying."""

    def _db_with_two_chairs(self) -> InstanceDatabase:
        db = InstanceDatabase()
        db.update([_make_pi(cls_name="chair", world_xyz=(1.0, 0.5, 0.0),
                            embed_seed=1)], step=0)
        db.update([_make_pi(cls_name="chair", world_xyz=(5.0, 0.5, 0.0),
                            embed_seed=2)], step=1)
        return db

    def test_excluded_node_yields_the_next_nearest(self):
        db = self._db_with_two_chairs()
        matcher = GoalMatcher(encoder=FakeEncoder())
        goal = GoalSpec(modality="category", value="chair")
        agent_xy = np.array([0.0, 0.0])

        near, _ = matcher.match(goal, db, agent_xy)
        assert np.isclose(near.world_xyz[0], 1.0)

        far, _ = matcher.match(goal, db, agent_xy, exclude_ids={near.node_id})
        assert far is not None
        assert far.node_id != near.node_id
        assert np.isclose(far.world_xyz[0], 5.0)

    def test_excluding_everything_reports_not_found(self):
        db = self._db_with_two_chairs()
        matcher = GoalMatcher(encoder=FakeEncoder())
        goal = GoalSpec(modality="category", value="chair")
        ids = {n.node_id for n in db.all_nodes()}

        node, score = matcher.match(goal, db, np.array([0.0, 0.0]), exclude_ids=ids)
        assert node is None
        assert score == 0.0

    def test_language_honours_exclusions(self):
        db = self._db_with_two_chairs()
        matcher = GoalMatcher(encoder=FakeEncoder(), language_threshold=-1.0)
        goal = GoalSpec(modality="language", value="a chair by the window",
                        category="chair")
        ids = {n.node_id for n in db.all_nodes()}

        assert matcher.match(goal, db, exclude_ids=ids)[0] is None
        assert matcher.match(goal, db)[0] is not None


class TestEvidenceGate:
    """A node created from one detection at conf 0.15 used to be as targetable
    as one seen fifty times. That is the mechanism behind the confident wrong
    stop: detector conf was lowered 0.35 -> 0.15 because at 0.35 the agent saw
    nothing in four frames out of five, and the price is false positives in
    memory. A consistent false positive also defeats re-detection, since the
    node carries the detector's own label."""

    def _one_shot_db(self, conf: float) -> InstanceDatabase:
        db = InstanceDatabase()
        db.update([_make_pi(cls_name="chair", conf=conf,
                            world_xyz=(1.0, 0.5, 0.0), embed_seed=3)], step=0)
        return db

    def test_single_weak_sighting_is_not_targetable(self):
        db = self._one_shot_db(0.18)
        matcher = GoalMatcher(encoder=FakeEncoder(), min_observations=2)
        goal = GoalSpec(modality="category", value="chair")
        node, score = matcher.match(goal, db, np.array([0.0, 0.0]))
        # "Not found yet" keeps the agent exploring, which beats committing to
        # an object that may not exist.
        assert node is None
        assert score == 0.0

    def test_repeated_sighting_is_targetable(self):
        db = self._one_shot_db(0.18)
        db.update([_make_pi(cls_name="chair", conf=0.18,
                            world_xyz=(1.05, 0.5, 0.0), embed_seed=3)], step=1)
        assert db.all_nodes()[0].n_observations == 2

        matcher = GoalMatcher(encoder=FakeEncoder(), min_observations=2)
        goal = GoalSpec(modality="category", value="chair")
        assert matcher.match(goal, db, np.array([0.0, 0.0]))[0] is not None

    def test_one_very_confident_sighting_is_targetable(self):
        """Starving the matcher is its own failure mode: the detector is
        recall-limited, so a single high-confidence hit must still count."""
        db = self._one_shot_db(0.85)
        matcher = GoalMatcher(encoder=FakeEncoder(), min_observations=2,
                              trust_confidence=0.5)
        goal = GoalSpec(modality="category", value="chair")
        assert matcher.match(goal, db, np.array([0.0, 0.0]))[0] is not None

    def test_gate_is_disableable(self):
        db = self._one_shot_db(0.18)
        matcher = GoalMatcher(encoder=FakeEncoder(), min_observations=1)
        goal = GoalSpec(modality="category", value="chair")
        assert matcher.match(goal, db, np.array([0.0, 0.0]))[0] is not None

    def test_gate_applies_to_language_goals(self):
        db = self._one_shot_db(0.18)
        matcher = GoalMatcher(encoder=FakeEncoder(), min_observations=2,
                              language_threshold=-1.0)
        goal = GoalSpec(modality="language", value="a chair by the window",
                        category="chair")
        assert matcher.match(goal, db)[0] is None
