"""Tests for src/memory/instance_db.py — InstanceDatabase + InstanceNode."""

import numpy as np
import pytest

from src.memory.instance_db import InstanceDatabase, InstanceNode
from src.perception.pipeline import PerceivedInstance


def _make_embed(seed: int = 0) -> np.ndarray:
    """Create a deterministic L2-normalized 512-dim embedding."""
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


class TestInstanceNode:
    def test_dataclass_fields(self):
        node = InstanceNode(
            node_id=0,
            cls_id=1,
            cls_name="chair",
            world_xyz=np.array([1.0, 0.5, 2.0]),
            clip_embed=_make_embed(0),
            confidence=0.9,
            first_seen_step=0,
            last_seen_step=5,
            n_observations=3,
            best_thumbnail=np.zeros((64, 64, 3), dtype=np.uint8),
        )
        assert node.node_id == 0
        assert node.clip_embed.shape == (512,)
        assert node.world_xyz.shape == (3,)
        assert node.n_observations == 3


class TestInstanceDatabaseBasic:
    def test_empty_database(self):
        db = InstanceDatabase()
        assert db.all_nodes() == []

    def test_single_detection_creates_node(self):
        db = InstanceDatabase()
        pi = _make_pi()
        db.update([pi], step=0)
        nodes = db.all_nodes()
        assert len(nodes) == 1
        assert nodes[0].cls_name == "chair"
        assert nodes[0].n_observations == 1
        assert nodes[0].first_seen_step == 0
        assert nodes[0].last_seen_step == 0

    def test_node_id_increments(self):
        db = InstanceDatabase()
        pi1 = _make_pi(world_xyz=(0.0, 0.5, 0.0))
        pi2 = _make_pi(world_xyz=(10.0, 0.5, 10.0))  # far away → new node
        db.update([pi1], step=0)
        db.update([pi2], step=1)
        nodes = db.all_nodes()
        assert len(nodes) == 2
        ids = {n.node_id for n in nodes}
        assert len(ids) == 2  # unique IDs

    def test_query_by_class(self):
        db = InstanceDatabase()
        db.update([_make_pi(cls_id=0, cls_name="chair", world_xyz=(0, 0.5, 0))], step=0)
        db.update([_make_pi(cls_id=1, cls_name="table", world_xyz=(5, 0.5, 5))], step=1)

        chairs = db.query_by_class(0)
        tables = db.query_by_class(1)
        assert len(chairs) == 1
        assert chairs[0].cls_name == "chair"
        assert len(tables) == 1
        assert tables[0].cls_name == "table"
        assert db.query_by_class(99) == []

    def test_query_by_embedding(self):
        db = InstanceDatabase()
        embed0 = _make_embed(0)
        db.update([_make_pi(embed_seed=0, world_xyz=(0, 0.5, 0))], step=0)
        db.update([_make_pi(embed_seed=42, world_xyz=(5, 0.5, 5))], step=1)

        results = db.query_by_embedding(embed0, top_k=2)
        assert len(results) == 2
        # First result should be the closest match
        best_node, best_score = results[0]
        assert best_score > results[1][1]  # best score > second score


class TestMerging:
    def test_three_close_detections_merge_to_one(self):
        """3 detections of same class at slightly different xy → 1 node, n_obs=3."""
        db = InstanceDatabase(merge_dist_m=0.75)
        pi1 = _make_pi(world_xyz=(1.0, 0.5, 2.0), seen_step=0)
        pi2 = _make_pi(world_xyz=(1.1, 0.5, 2.1), seen_step=1)
        pi3 = _make_pi(world_xyz=(0.9, 0.5, 1.9), seen_step=2)
        db.update([pi1], step=0)
        db.update([pi2], step=1)
        db.update([pi3], step=2)

        nodes = db.all_nodes()
        assert len(nodes) == 1
        assert nodes[0].n_observations == 3
        assert nodes[0].first_seen_step == 0
        assert nodes[0].last_seen_step == 2

    def test_two_far_apart_create_two_nodes(self):
        """2 chairs 2m apart → 2 nodes."""
        db = InstanceDatabase(merge_dist_m=0.75)
        pi1 = _make_pi(world_xyz=(0.0, 0.5, 0.0), seen_step=0)
        pi2 = _make_pi(world_xyz=(2.0, 0.5, 2.0), seen_step=1)
        db.update([pi1], step=0)
        db.update([pi2], step=1)

        nodes = db.all_nodes()
        assert len(nodes) == 2

    def test_close_but_dissimilar_embedding_creates_two(self):
        """Chair at 60cm away with very different embedding → 2 nodes."""
        db = InstanceDatabase(merge_dist_m=0.75, merge_embed_sim=0.85)
        # Use very different embeddings
        pi1 = _make_pi(world_xyz=(1.0, 0.5, 2.0), embed_seed=0, seen_step=0)
        pi2 = _make_pi(world_xyz=(1.5, 0.5, 2.3), embed_seed=999, seen_step=1)

        # Verify embeddings are actually dissimilar
        sim = np.dot(pi1.clip_embed, pi2.clip_embed)
        assert sim < 0.85, f"Embeddings too similar ({sim:.3f}), test is invalid"

        db.update([pi1], step=0)
        db.update([pi2], step=1)

        nodes = db.all_nodes()
        assert len(nodes) == 2

    def test_different_class_never_merges(self):
        """A chair and a table at the same location → 2 nodes."""
        db = InstanceDatabase(merge_dist_m=0.75)
        pi1 = _make_pi(cls_id=0, cls_name="chair", world_xyz=(1.0, 0.5, 2.0))
        pi2 = _make_pi(cls_id=1, cls_name="table", world_xyz=(1.0, 0.5, 2.0))
        db.update([pi1, pi2], step=0)

        nodes = db.all_nodes()
        assert len(nodes) == 2

    def test_merge_updates_position(self):
        """Merged node position should be running mean of observations."""
        db = InstanceDatabase(merge_dist_m=0.75)
        pi1 = _make_pi(world_xyz=(1.0, 0.5, 2.0), seen_step=0)
        pi2 = _make_pi(world_xyz=(1.2, 0.5, 2.2), seen_step=1)
        db.update([pi1], step=0)
        db.update([pi2], step=1)

        node = db.all_nodes()[0]
        # Running mean of (1.0, 0.5, 2.0) and (1.2, 0.5, 2.2)
        np.testing.assert_allclose(node.world_xyz, [1.1, 0.5, 2.1], atol=0.05)

    def test_merge_updates_embedding(self):
        """Merged node embedding should be re-normalized running mean."""
        db = InstanceDatabase(merge_dist_m=0.75)
        pi1 = _make_pi(embed_seed=0, world_xyz=(1.0, 0.5, 2.0), seen_step=0)
        pi2 = _make_pi(embed_seed=0, world_xyz=(1.1, 0.5, 2.1), seen_step=1)
        db.update([pi1], step=0)
        db.update([pi2], step=1)

        node = db.all_nodes()[0]
        # Embedding should still be L2-normalized
        assert abs(np.linalg.norm(node.clip_embed) - 1.0) < 1e-5

    def test_merge_keeps_best_thumbnail(self):
        """Thumbnail should come from the highest-confidence observation."""
        db = InstanceDatabase(merge_dist_m=0.75)
        thumb_low = np.full((64, 64, 3), 50, dtype=np.uint8)
        thumb_high = np.full((64, 64, 3), 200, dtype=np.uint8)

        pi1 = _make_pi(conf=0.6, world_xyz=(1.0, 0.5, 2.0), seen_step=0)
        pi1.crop_thumbnail = thumb_low
        pi2 = _make_pi(conf=0.95, world_xyz=(1.1, 0.5, 2.1), seen_step=1)
        pi2.crop_thumbnail = thumb_high

        db.update([pi1], step=0)
        db.update([pi2], step=1)

        node = db.all_nodes()[0]
        assert node.confidence == 0.95
        assert np.all(node.best_thumbnail == thumb_high)

    def test_merge_confidence_is_running_max(self):
        """Confidence should be the max across all merged observations."""
        db = InstanceDatabase(merge_dist_m=0.75)
        pi1 = _make_pi(conf=0.7, world_xyz=(1.0, 0.5, 2.0), seen_step=0)
        pi2 = _make_pi(conf=0.9, world_xyz=(1.1, 0.5, 2.1), seen_step=1)
        pi3 = _make_pi(conf=0.6, world_xyz=(1.05, 0.5, 2.05), seen_step=2)
        db.update([pi1], step=0)
        db.update([pi2], step=1)
        db.update([pi3], step=2)

        node = db.all_nodes()[0]
        assert node.confidence == 0.9

    def test_multiple_detections_same_step(self):
        """Multiple detections in one update call."""
        db = InstanceDatabase(merge_dist_m=0.75)
        pi1 = _make_pi(cls_id=0, cls_name="chair", world_xyz=(0, 0.5, 0))
        pi2 = _make_pi(cls_id=1, cls_name="table", world_xyz=(5, 0.5, 5))
        pi3 = _make_pi(cls_id=0, cls_name="chair", world_xyz=(0.1, 0.5, 0.1))

        db.update([pi1, pi2, pi3], step=0)
        nodes = db.all_nodes()
        # pi1 and pi3 should merge (same class, close), pi2 is separate
        assert len(nodes) == 2


class TestMemorySize:
    def test_memory_under_budget(self):
        """DB with 100 nodes should be well under 20 MB."""
        import sys
        db = InstanceDatabase()
        for i in range(100):
            pi = _make_pi(
                world_xyz=(float(i), 0.5, float(i)),
                embed_seed=i,
                seen_step=i,
            )
            db.update([pi], step=i)

        nodes = db.all_nodes()
        # Rough size estimate: 512*4 bytes per embed + 64*64*3 per thumb + overhead
        embed_bytes = sum(n.clip_embed.nbytes for n in nodes)
        thumb_bytes = sum(n.best_thumbnail.nbytes for n in nodes)
        total_bytes = embed_bytes + thumb_bytes
        total_mb = total_bytes / (1024 * 1024)
        assert total_mb < 20, f"Memory {total_mb:.1f} MB exceeds 20 MB budget"
