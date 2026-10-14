"""Intrinsics must track the sensor resolution, not a baked-in 256.

Measured 2026-10-14 (6.60): the detector scores mAP50 0.132 on held-out scenes
at 512 px input and 0.087 at 256 px -- a 34% loss, larger than the entire gain
from retraining it. The agent's RGB sensor is 256x256, so raising it is the
single largest perception lever available, and that is only safe if every
geometric consumer derives its intrinsics from the actual frame size.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mapping.transforms import depth_to_pointcloud_camera, make_intrinsics


class TestIntrinsicsScaleWithResolution:
    def test_focal_length_doubles_when_the_frame_doubles(self):
        k256 = make_intrinsics(90.0, 256, 256)
        k512 = make_intrinsics(90.0, 512, 512)

        assert k512[0, 0] == 2.0 * k256[0, 0]
        assert k512[1, 1] == 2.0 * k256[1, 1]

    def test_principal_point_stays_at_the_frame_centre(self):
        for n in (128, 256, 512, 640):
            k = make_intrinsics(90.0, n, n)
            assert k[0, 2] == n / 2.0
            assert k[1, 2] == n / 2.0

    def test_the_same_world_ray_survives_a_resolution_change(self):
        """A pixel at the same *fractional* position must back-project to the
        same direction at any resolution, or doubling the sensor silently
        rescales the map."""
        depth = 3.0
        rays = []
        for n in (256, 512):
            k = make_intrinsics(90.0, n, n)
            d = np.full((n, n), depth, dtype=np.float32)
            pts = depth_to_pointcloud_camera(d, k)
            # the pixel one quarter across and one quarter down
            p = pts.reshape(n, n, 3)[n // 4, n // 4]
            rays.append(p / np.linalg.norm(p))

        np.testing.assert_allclose(rays[0], rays[1], atol=1e-6)

    def test_a_90_degree_hfov_puts_the_frame_edge_at_45_degrees(self):
        """Independent of resolution: tan(hfov/2) = (width/2) / fx."""
        for n in (256, 512):
            k = make_intrinsics(90.0, n, n)
            fx, cx = k[0, 0], k[0, 2]
            assert np.isclose(cx / fx, np.tan(np.deg2rad(45.0)))


def _result(**kw):
    from src.eval.runner import SubtaskResult

    return SubtaskResult(
        episode_id="e0", subtask_index=0, category="mirror",
        modality="category", distance_to_goal=0.5, steps_taken=10,
        actual_path_length=2.0, shortest_path_length=1.5, reason="ok", **kw,
    )


class TestRunConfigIsRecorded:
    """Three sweep grids were run today whose output directories were
    indistinguishable from their contents alone; the arms had to be told apart
    by directory name and paired positionally. summary.json now carries the
    config that produced it."""

    def test_summary_carries_the_config_it_was_given(self, tmp_path):
        from src.eval.report import save_results
        import json

        r = _result(success=True, spl=0.5)
        save_results([r], str(tmp_path), config={"resolution": 512, "seed": 42})

        summary = json.load(open(tmp_path / "summary.json"))
        assert summary["config"]["resolution"] == 512
        assert summary["config"]["seed"] == 42

    def test_summary_is_still_written_without_a_config(self, tmp_path):
        from src.eval.report import save_results
        import json

        r = _result(success=False, spl=0.0)
        save_results([r], str(tmp_path))

        summary = json.load(open(tmp_path / "summary.json"))
        assert summary["total_subtasks"] == 1
        assert "config" not in summary
