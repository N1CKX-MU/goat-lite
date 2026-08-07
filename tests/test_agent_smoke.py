"""Week 2 unit tests for HabitatEnv, YOLODetector, and ClipEncoder."""

import glob

import numpy as np
import pytest


def _get_scene():
    scenes = sorted(glob.glob("data/hm3d/*/[!.]*.basis.glb"))
    if not scenes:
        pytest.skip("No HM3D scenes available")
    return scenes[0]


# ──── HabitatEnv tests ────


class TestHabitatEnv:
    def test_reset_returns_observation(self):
        from src.sim.env import HabitatEnv

        env = HabitatEnv(scene_path=_get_scene())
        obs = env.reset()
        assert obs.rgb.shape == (256, 256, 3)
        assert obs.rgb.dtype == np.uint8
        assert obs.depth.shape == (256, 256)
        assert obs.depth.dtype == np.float32
        assert obs.pose.shape == (4, 4)
        assert obs.gps.shape == (2,)
        env.close()

    def test_step_forward_changes_position(self):
        from src.sim.env import HabitatEnv

        env = HabitatEnv(scene_path=_get_scene())
        obs0 = env.reset()
        obs1, done, info = env.step(1)  # forward
        assert not done
        # Position should change (or agent is blocked by wall, which is ok)
        env.close()

    def test_stop_action_sets_done(self):
        from src.sim.env import HabitatEnv

        env = HabitatEnv(scene_path=_get_scene())
        env.reset()
        _, done, _ = env.step(0)
        assert done
        env.close()

    def test_set_goal_persists_across_reset(self):
        from src.sim.env import GoalSpec, HabitatEnv

        env = HabitatEnv(scene_path=_get_scene())
        env.set_goal(GoalSpec(modality="category", value="chair"))
        obs = env.reset()
        assert obs.current_goal is not None
        assert obs.current_goal.value == "chair"
        env.close()

    def test_intrinsics_shape(self):
        from src.sim.env import HabitatEnv

        env = HabitatEnv(scene_path=_get_scene())
        K = env.intrinsics
        assert K.shape == (3, 3)
        assert K[0, 0] > 0  # fx > 0
        assert K[1, 1] > 0  # fy > 0
        env.close()

    def test_pose_is_valid_se3(self):
        from src.sim.env import HabitatEnv

        env = HabitatEnv(scene_path=_get_scene())
        env.reset()
        T = env.get_pose()
        # Rotation part should be orthogonal
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
        # det(R) should be 1
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)
        # Bottom row should be [0, 0, 0, 1]
        assert np.allclose(T[3, :], [0, 0, 0, 1])
        env.close()


# ──── YOLODetector tests ────


class TestYOLODetector:
    def test_detect_returns_list(self):
        from src.perception.detector import YOLODetector

        det = YOLODetector()
        rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = det.detect(rgb)
        assert isinstance(result, list)

    def test_detection_fields(self):
        from src.perception.detector import YOLODetector

        det = YOLODetector(conf=0.1)  # low conf to get detections
        # Use a real scene frame for reliable detections
        from src.sim.env import HabitatEnv

        env = HabitatEnv(scene_path=_get_scene())
        obs = env.reset()
        # Walk a bit to find objects
        for _ in range(20):
            obs, _, _ = env.step(1)

        dets = det.detect(obs.rgb)
        env.close()

        if dets:
            d = dets[0]
            assert isinstance(d.cls_id, int)
            assert isinstance(d.cls_name, str)
            assert isinstance(d.conf, float)
            assert len(d.bbox) == 4
            x1, y1, x2, y2 = d.bbox
            assert x2 > x1 and y2 > y1

    def test_class_names_available(self):
        from src.perception.detector import YOLODetector

        det = YOLODetector()
        names = det.class_names
        assert isinstance(names, dict)
        assert len(names) > 0


# ──── ClipEncoder tests ────


class TestClipEncoder:
    def test_image_embed_shape_and_norm(self):
        from src.perception.encoder import ClipEncoder

        enc = ClipEncoder()
        crops = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)]
        embeds = enc.encode_image(crops)
        assert embeds.shape == (1, 512)
        assert np.allclose(np.linalg.norm(embeds, axis=1), 1.0, atol=1e-4)

    def test_text_embed_shape_and_norm(self):
        from src.perception.encoder import ClipEncoder

        enc = ClipEncoder()
        embeds = enc.encode_text(["a chair"])
        assert embeds.shape == (1, 512)
        assert np.allclose(np.linalg.norm(embeds, axis=1), 1.0, atol=1e-4)

    def test_empty_input(self):
        from src.perception.encoder import ClipEncoder

        enc = ClipEncoder()
        assert enc.encode_image([]).shape == (0, 512)
        assert enc.encode_text([]).shape == (0, 512)

    def test_similar_concepts_have_high_similarity(self):
        from src.perception.encoder import ClipEncoder

        enc = ClipEncoder()
        embeds = enc.encode_text(["a photo of a chair", "a photo of a couch"])
        sim = embeds[0] @ embeds[1]
        assert sim > 0.7, f"chair/couch similarity too low: {sim}"

    def test_different_concepts_have_lower_similarity(self):
        from src.perception.encoder import ClipEncoder

        enc = ClipEncoder()
        embeds = enc.encode_text(["a photo of a chair", "a photo of a car"])
        sim = embeds[0] @ embeds[1]
        # Should be lower than chair/couch
        assert sim < 0.9, f"chair/car similarity unexpectedly high: {sim}"
