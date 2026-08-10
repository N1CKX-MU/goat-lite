"""Tests for src/mapping/occupancy.py — SemanticMap."""

import numpy as np
import pytest

from src.mapping.occupancy import SemanticMap
from src.mapping.transforms import make_intrinsics


def _identity_pose():
    return np.eye(4, dtype=np.float64)


def _make_K():
    return make_intrinsics(90.0, 256, 256)


class TestSemanticMapInit:
    def test_grid_dimensions(self):
        sm = SemanticMap(size_m=10.0, resolution_m=0.1)
        occ = sm.get_occupancy()
        expected = int(10.0 / 0.1)
        assert occ.shape == (expected, expected)

    def test_initial_state_all_unknown(self):
        sm = SemanticMap(size_m=10.0, resolution_m=0.1)
        occ = sm.get_occupancy()
        assert np.all(occ == -1)

    def test_explored_initially_false(self):
        sm = SemanticMap(size_m=10.0, resolution_m=0.1)
        assert not np.any(sm.get_explored())


class TestWorldGridConversion:
    def test_center_of_map(self):
        sm = SemanticMap(size_m=10.0, resolution_m=0.1)
        # Origin is at (-size/2, -size/2), so world (0,0) should map to center
        row, col = sm.world_to_grid(np.array([0.0, 0.0]))
        grid_size = int(10.0 / 0.1)
        assert abs(row - grid_size // 2) <= 1
        assert abs(col - grid_size // 2) <= 1

    def test_roundtrip(self):
        sm = SemanticMap(size_m=20.0, resolution_m=0.05)
        xy = np.array([3.0, -2.0])
        ij = sm.world_to_grid(xy)
        xy_back = sm.grid_to_world(ij)
        np.testing.assert_allclose(xy_back, xy, atol=0.05)


class TestUpdateFromDepth:
    def _make_uniform_depth(self, h=256, w=256, val=3.0):
        return np.full((h, w), val, dtype=np.float32)

    def test_marks_cells_occupied(self):
        """Depth update should mark some cells as occupied."""
        sm = SemanticMap(size_m=24.0, resolution_m=0.05)
        K = _make_K()
        pose = _identity_pose()
        # Place camera at (0, 1.0, 0) — 1m height so points land in valid height range
        pose[1, 3] = 1.0
        depth = self._make_uniform_depth(val=3.0)
        sm.update_from_depth(depth, pose, K)

        occ = sm.get_occupancy()
        assert np.any(occ == 1), "Should have some occupied cells"

    def test_marks_cells_free(self):
        """Cells along rays to occupied cells should be marked free."""
        sm = SemanticMap(size_m=24.0, resolution_m=0.05)
        K = _make_K()
        pose = _identity_pose()
        pose[1, 3] = 1.0
        depth = self._make_uniform_depth(val=3.0)
        sm.update_from_depth(depth, pose, K)

        occ = sm.get_occupancy()
        assert np.any(occ == 0), "Should have some free cells"

    def test_marks_explored(self):
        """Updated cells should be marked explored."""
        sm = SemanticMap(size_m=24.0, resolution_m=0.05)
        K = _make_K()
        pose = _identity_pose()
        pose[1, 3] = 1.0
        depth = self._make_uniform_depth(val=3.0)
        sm.update_from_depth(depth, pose, K)

        explored = sm.get_explored()
        assert np.any(explored), "Should have some explored cells"

    def test_height_filter(self):
        """Points at floor level (y close to camera y) or ceiling should be filtered."""
        sm = SemanticMap(size_m=24.0, resolution_m=0.05)
        K = _make_K()
        # Camera at y=1.0, floor at y=0. Points landing at y<0.1 or y>1.5 filtered.
        # With identity rotation and depth=1.0, most points land near y=1.0
        # which is within the valid range relative to floor.
        pose = _identity_pose()
        pose[1, 3] = 1.0  # camera at 1m height

        # Uniform depth of 1m — with 90 deg FOV the point cloud spans
        # a reasonable range. Some points will be in valid height range.
        depth = self._make_uniform_depth(val=1.0)
        sm.update_from_depth(depth, pose, K)

        occ = sm.get_occupancy()
        # Should have at least some occupied cells from valid-height points
        n_occupied = np.sum(occ == 1)
        n_total_pixels = 256 * 256
        # Not ALL pixels should produce occupied cells (height filter removes some)
        assert n_occupied < n_total_pixels

    def test_multiple_updates_accumulate(self):
        """Two updates from different poses should explore more area."""
        sm = SemanticMap(size_m=24.0, resolution_m=0.05)
        K = _make_K()
        depth = self._make_uniform_depth(val=2.0)

        pose1 = _identity_pose()
        pose1[1, 3] = 1.0
        sm.update_from_depth(depth, pose1, K)
        explored_after_1 = np.sum(sm.get_explored())

        pose2 = _identity_pose()
        pose2[1, 3] = 1.0
        pose2[0, 3] = 3.0  # shifted 3m in X
        sm.update_from_depth(depth, pose2, K)
        explored_after_2 = np.sum(sm.get_explored())

        assert explored_after_2 > explored_after_1

    def test_out_of_bounds_ignored(self):
        """Points projecting outside the grid should not crash."""
        sm = SemanticMap(size_m=4.0, resolution_m=0.05)  # small map
        K = _make_K()
        pose = _identity_pose()
        pose[1, 3] = 1.0
        depth = self._make_uniform_depth(val=10.0)  # far points → outside 4m map
        # Should not raise
        sm.update_from_depth(depth, pose, K)


class TestUpdateFromDetections:
    def test_increments_class_counts(self):
        sm = SemanticMap(size_m=24.0, resolution_m=0.05, num_classes=37)
        from src.perception.pipeline import PerceivedInstance
        pi = PerceivedInstance(
            cls_id=5,
            cls_name="chair",
            conf=0.9,
            bbox=(50, 50, 150, 150),
            crop_thumbnail=np.zeros((64, 64, 3), dtype=np.uint8),
            clip_embed=np.zeros(512, dtype=np.float32),
            world_xyz=np.array([1.0, 0.5, 2.0]),
            seen_step=0,
        )
        sm.update_from_detections([pi])
        counts = sm.get_class_counts()
        # Class 5 should have nonzero counts somewhere
        assert np.any(counts[:, :, 5] > 0)

    def test_gaussian_kernel_spreads(self):
        """Detection should affect a small neighborhood, not just one cell."""
        sm = SemanticMap(size_m=24.0, resolution_m=0.05, num_classes=37)
        from src.perception.pipeline import PerceivedInstance
        pi = PerceivedInstance(
            cls_id=3,
            cls_name="table",
            conf=0.8,
            bbox=(50, 50, 150, 150),
            crop_thumbnail=np.zeros((64, 64, 3), dtype=np.uint8),
            clip_embed=np.zeros(512, dtype=np.float32),
            world_xyz=np.array([0.0, 0.5, 0.0]),
            seen_step=0,
        )
        sm.update_from_detections([pi])
        counts = sm.get_class_counts()
        nonzero = np.sum(counts[:, :, 3] > 0)
        assert nonzero > 1, "Should spread to more than 1 cell"

    def test_multiple_detections(self):
        sm = SemanticMap(size_m=24.0, resolution_m=0.05, num_classes=37)
        from src.perception.pipeline import PerceivedInstance
        pi1 = PerceivedInstance(
            cls_id=5, cls_name="chair", conf=0.9,
            bbox=(50, 50, 150, 150),
            crop_thumbnail=np.zeros((64, 64, 3), dtype=np.uint8),
            clip_embed=np.zeros(512, dtype=np.float32),
            world_xyz=np.array([1.0, 0.5, 2.0]),
            seen_step=0,
        )
        pi2 = PerceivedInstance(
            cls_id=10, cls_name="couch", conf=0.7,
            bbox=(50, 50, 150, 150),
            crop_thumbnail=np.zeros((64, 64, 3), dtype=np.uint8),
            clip_embed=np.zeros(512, dtype=np.float32),
            world_xyz=np.array([-2.0, 0.5, 1.0]),
            seen_step=0,
        )
        sm.update_from_detections([pi1, pi2])
        counts = sm.get_class_counts()
        assert np.any(counts[:, :, 5] > 0)
        assert np.any(counts[:, :, 10] > 0)


class TestOutOfVocabularyDetections:
    """The map has one channel per GOAT class. A detector with a different
    vocabulary (the stock COCO yolov8n fallback emits ids up to 79) must not
    index past the end of _class_counts and kill the episode."""

    @staticmethod
    def _inst(cls_id):
        from src.perception.pipeline import PerceivedInstance
        return PerceivedInstance(
            cls_id=cls_id,
            cls_name=f"cls{cls_id}",
            conf=0.9,
            bbox=(50, 50, 150, 150),
            crop_thumbnail=np.zeros((64, 64, 3), dtype=np.uint8),
            clip_embed=np.zeros(512, dtype=np.float32),
            world_xyz=np.array([1.0, 0.5, 2.0]),
            seen_step=0,
        )

    def test_class_id_beyond_num_classes_does_not_raise(self):
        sm = SemanticMap(size_m=24.0, resolution_m=0.05, num_classes=37)
        sm.update_from_detections([self._inst(69)])  # a COCO id
        assert 69 in sm.dropped_cls_ids

    def test_negative_class_id_dropped(self):
        sm = SemanticMap(size_m=24.0, resolution_m=0.05, num_classes=37)
        sm.update_from_detections([self._inst(-1)])
        assert -1 in sm.dropped_cls_ids

    def test_valid_detections_still_recorded_alongside_invalid(self):
        sm = SemanticMap(size_m=24.0, resolution_m=0.05, num_classes=37)
        sm.update_from_detections([self._inst(69), self._inst(5)])
        counts = sm.get_class_counts()
        assert np.any(counts[:, :, 5] > 0), "valid detection must survive"
        assert sm.dropped_cls_ids == {69}

    def test_boundary_id_is_valid(self):
        sm = SemanticMap(size_m=24.0, resolution_m=0.05, num_classes=37)
        sm.update_from_detections([self._inst(36)])  # last valid channel
        assert sm.dropped_cls_ids == set()
        assert np.any(sm.get_class_counts()[:, :, 36] > 0)

    def test_no_drops_reported_for_clean_vocabulary(self):
        sm = SemanticMap(size_m=24.0, resolution_m=0.05, num_classes=37)
        sm.update_from_detections([self._inst(i) for i in range(0, 36, 7)])
        assert sm.dropped_cls_ids == set()
