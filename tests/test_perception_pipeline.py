"""Tests for src/perception/pipeline.py — PerceptionPipeline + PerceivedInstance."""

from __future__ import annotations

import numpy as np
import pytest

from src.perception.detector import Detection
from src.perception.pipeline import PerceivedInstance, PerceptionPipeline


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

class FakeDetector:
    """Returns canned detections."""

    def __init__(self, detections: list[Detection] | None = None):
        self._detections = detections or []

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        return self._detections


class FakeEncoder:
    """Returns deterministic embeddings based on crop mean intensity."""

    def __init__(self, embed_dim: int = 512):
        self._embed_dim = embed_dim

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def encode_image(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, self._embed_dim), dtype=np.float32)
        out = np.zeros((len(crops), self._embed_dim), dtype=np.float32)
        for i, crop in enumerate(crops):
            out[i, 0] = float(crop.mean()) / 255.0
            out[i, 1] = 1.0
        # L2-normalize
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        out /= norms
        return out


def _make_obs(
    h: int = 256, w: int = 256, depth_val: float = 2.0
) -> "Observation":
    """Create a fake Observation with uniform RGB and depth."""
    from src.sim.env import Observation, GoalSpec

    rgb = np.full((h, w, 3), 128, dtype=np.uint8)
    depth = np.full((h, w), depth_val, dtype=np.float32)
    pose = np.eye(4, dtype=np.float64)
    return Observation(
        rgb=rgb,
        depth=depth,
        pose=pose,
        compass=0.0,
        gps=np.array([0.0, 0.0]),
        current_goal=GoalSpec(modality="category", value="chair"),
    )


def _make_intrinsics(h: int = 256, w: int = 256, hfov: float = 90.0) -> np.ndarray:
    from src.mapping.transforms import make_intrinsics
    return make_intrinsics(hfov, w, h)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPerceivedInstance:
    def test_dataclass_fields(self):
        embed = np.zeros(512, dtype=np.float32)
        pi = PerceivedInstance(
            cls_id=0,
            cls_name="chair",
            conf=0.9,
            bbox=(10, 20, 50, 60),
            crop_thumbnail=np.zeros((64, 64, 3), dtype=np.uint8),
            clip_embed=embed,
            world_xyz=np.array([1.0, 0.5, 2.0]),
            seen_step=0,
        )
        assert pi.cls_name == "chair"
        assert pi.clip_embed.shape == (512,)
        assert pi.world_xyz.shape == (3,)
        assert pi.crop_thumbnail.shape == (64, 64, 3)


class TestPerceptionPipeline:
    def _make_pipeline(
        self, detections: list[Detection] | None = None, min_crop_size: int = 32
    ) -> PerceptionPipeline:
        detector = FakeDetector(detections or [])
        encoder = FakeEncoder(embed_dim=512)
        intrinsics = _make_intrinsics()
        return PerceptionPipeline(
            detector=detector,
            encoder=encoder,
            intrinsics=intrinsics,
            min_crop_size=min_crop_size,
        )

    def test_no_detections_returns_empty(self):
        pipe = self._make_pipeline(detections=[])
        obs = _make_obs()
        result = pipe.process(obs, step=0)
        assert result == []

    def test_single_detection_returns_one_instance(self):
        det = Detection(cls_id=0, cls_name="chair", conf=0.8,
                        bbox=(50, 50, 150, 150))
        pipe = self._make_pipeline(detections=[det])
        obs = _make_obs(depth_val=3.0)
        result = pipe.process(obs, step=5)

        assert len(result) == 1
        pi = result[0]
        assert pi.cls_id == 0
        assert pi.cls_name == "chair"
        assert pi.conf == 0.8
        assert pi.bbox == (50, 50, 150, 150)
        assert pi.seen_step == 5
        assert pi.clip_embed.shape == (512,)
        # Embedding should be L2-normalized
        assert abs(np.linalg.norm(pi.clip_embed) - 1.0) < 1e-5
        # world_xyz should be a 3-vector
        assert pi.world_xyz.shape == (3,)

    def test_small_crop_filtered(self):
        """Detections smaller than min_crop_size should be skipped."""
        small_det = Detection(cls_id=1, cls_name="cup", conf=0.6,
                              bbox=(100, 100, 110, 110))  # 10x10 crop
        pipe = self._make_pipeline(detections=[small_det], min_crop_size=32)
        obs = _make_obs()
        result = pipe.process(obs, step=0)
        assert result == []

    def test_large_enough_crop_not_filtered(self):
        det = Detection(cls_id=1, cls_name="cup", conf=0.6,
                        bbox=(100, 100, 140, 140))  # 40x40 crop
        pipe = self._make_pipeline(detections=[det], min_crop_size=32)
        obs = _make_obs()
        result = pipe.process(obs, step=0)
        assert len(result) == 1

    def test_multiple_detections(self):
        dets = [
            Detection(cls_id=0, cls_name="chair", conf=0.9,
                      bbox=(10, 10, 80, 80)),
            Detection(cls_id=2, cls_name="table", conf=0.7,
                      bbox=(120, 50, 200, 150)),
        ]
        pipe = self._make_pipeline(detections=dets)
        obs = _make_obs()
        result = pipe.process(obs, step=3)
        assert len(result) == 2
        assert result[0].cls_name == "chair"
        assert result[1].cls_name == "table"

    def test_thumbnail_is_64x64(self):
        det = Detection(cls_id=0, cls_name="chair", conf=0.8,
                        bbox=(20, 20, 200, 200))
        pipe = self._make_pipeline(detections=[det])
        obs = _make_obs()
        result = pipe.process(obs, step=0)
        assert result[0].crop_thumbnail.shape == (64, 64, 3)

    def test_invalid_depth_skips_detection(self):
        """If depth at bbox center is invalid, detection should be skipped."""
        det = Detection(cls_id=0, cls_name="chair", conf=0.8,
                        bbox=(50, 50, 150, 150))
        pipe = self._make_pipeline(detections=[det])
        obs = _make_obs(depth_val=0.0)  # invalid depth
        result = pipe.process(obs, step=0)
        assert result == []

    def test_world_xyz_uses_depth_and_pose(self):
        """With identity pose and known depth, world_xyz should be predictable."""
        det = Detection(cls_id=0, cls_name="chair", conf=0.9,
                        bbox=(96, 96, 160, 160))  # center = (128, 128), 64x64
        pipe = self._make_pipeline(detections=[det])
        obs = _make_obs(depth_val=5.0)
        result = pipe.process(obs, step=0)
        assert len(result) == 1
        xyz = result[0].world_xyz
        # Center pixel with identity pose: x_cam ~ 0, y_cam ~ 0, z_cam = -5.
        # Habitat cameras look along -Z, so an object 5 m ahead is at z = -5.
        # Previously asserted +5, matching the old OpenCV-style back-projection
        # that mirrored points behind the camera once the pose was applied.
        assert abs(xyz[2] - (-5.0)) < 0.5  # z ~ -depth

    def test_bbox_clipped_to_image(self):
        """Bbox that extends beyond image should be clipped."""
        det = Detection(cls_id=0, cls_name="chair", conf=0.8,
                        bbox=(200, 200, 300, 300))  # extends past 256x256
        pipe = self._make_pipeline(detections=[det])
        obs = _make_obs()
        result = pipe.process(obs, step=0)
        # Should still produce a result (crop clipped to image bounds)
        assert len(result) == 1

    def test_step_propagated(self):
        det = Detection(cls_id=0, cls_name="chair", conf=0.8,
                        bbox=(50, 50, 150, 150))
        pipe = self._make_pipeline(detections=[det])
        obs = _make_obs()
        result = pipe.process(obs, step=42)
        assert result[0].seen_step == 42

    def test_clip_embed_normalized(self):
        dets = [
            Detection(cls_id=0, cls_name="chair", conf=0.9,
                      bbox=(10, 10, 100, 100)),
            Detection(cls_id=1, cls_name="table", conf=0.7,
                      bbox=(120, 50, 220, 150)),
        ]
        pipe = self._make_pipeline(detections=dets)
        obs = _make_obs()
        result = pipe.process(obs, step=0)
        for pi in result:
            norm = np.linalg.norm(pi.clip_embed)
            assert abs(norm - 1.0) < 1e-5, f"Embedding not normalized: {norm}"
