"""Unit tests for coordinate transforms (Week 3).

Habitat world frame: Y-up, agent default faces -Z.
Map grid frame: 2D XZ plane (world X -> grid col, world Z -> grid row).
"""

import numpy as np
import pytest

from src.mapping.transforms import (
    depth_to_pointcloud_camera,
    pointcloud_camera_to_world,
    bbox_center_depth_to_world_xyz,
    world_to_grid,
    grid_to_world,
    make_intrinsics,
)


# ──── Intrinsics helper ────

class TestMakeIntrinsics:
    def test_90_degree_hfov_256(self):
        K = make_intrinsics(hfov_deg=90.0, width=256, height=256)
        assert K.shape == (3, 3)
        # For 90° hfov: fx = w / (2 * tan(45°)) = 256 / 2 = 128
        assert np.isclose(K[0, 0], 128.0, atol=0.1)
        assert np.isclose(K[1, 1], 128.0, atol=0.1)
        assert np.isclose(K[0, 2], 128.0, atol=0.1)
        assert np.isclose(K[1, 2], 128.0, atol=0.1)


# ──── Depth to point cloud ────

class TestDepthToPointcloud:
    def test_center_pixel_points_forward(self):
        """Center pixel at 1m depth should be at (0, 0, 1) in camera frame
        (camera looks along +Z in OpenGL/Habitat convention with flipped axes)."""
        K = make_intrinsics(90.0, 256, 256)
        depth = np.zeros((256, 256), dtype=np.float32)
        depth[128, 128] = 1.0  # 1 meter at center pixel
        pc = depth_to_pointcloud_camera(depth, K)
        # Should have exactly one valid point (others are zero depth)
        valid = pc[pc[:, 2] > 0]
        assert len(valid) >= 1
        # Center pixel -> roughly (0, 0, 1) in camera frame
        pt = valid[0]
        assert np.isclose(pt[2], 1.0, atol=0.05)
        assert abs(pt[0]) < 0.05
        assert abs(pt[1]) < 0.05

    def test_output_shape(self):
        K = make_intrinsics(90.0, 256, 256)
        depth = np.ones((256, 256), dtype=np.float32)
        pc = depth_to_pointcloud_camera(depth, K)
        assert pc.shape == (256 * 256, 3)

    def test_zero_depth_excluded(self):
        K = make_intrinsics(90.0, 256, 256)
        depth = np.zeros((256, 256), dtype=np.float32)
        pc = depth_to_pointcloud_camera(depth, K)
        valid = pc[pc[:, 2] > 0]
        assert len(valid) == 0


# ──── Camera to world ────

class TestPointcloudCameraToWorld:
    def test_identity_pose(self):
        """With identity pose, camera frame == world frame."""
        pts_cam = np.array([[1.0, 2.0, 3.0]])
        pose = np.eye(4)
        pts_world = pointcloud_camera_to_world(pts_cam, pose)
        np.testing.assert_allclose(pts_world, pts_cam, atol=1e-6)

    def test_translation_only(self):
        """Pose with translation shifts points."""
        pts_cam = np.array([[0.0, 0.0, 0.0]])
        pose = np.eye(4)
        pose[:3, 3] = [10.0, 20.0, 30.0]
        pts_world = pointcloud_camera_to_world(pts_cam, pose)
        np.testing.assert_allclose(pts_world, [[10.0, 20.0, 30.0]], atol=1e-6)

    def test_rotation_90_around_y(self):
        """90° rotation around Y should swap X and Z."""
        pts_cam = np.array([[1.0, 0.0, 0.0]])
        pose = np.eye(4)
        # 90° around Y: x->z, z->-x
        pose[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
        pts_world = pointcloud_camera_to_world(pts_cam, pose)
        np.testing.assert_allclose(pts_world, [[0.0, 0.0, 1.0]], atol=1e-6)


# ──── Bbox center + depth to world xyz ────

class TestBboxCenterDepthToWorld:
    def test_center_pixel_at_1m_identity_pose(self):
        K = make_intrinsics(90.0, 256, 256)
        pose = np.eye(4)
        depth_map = np.zeros((256, 256), dtype=np.float32)
        depth_map[128, 128] = 2.0
        bbox = (64, 64, 192, 192)  # center at (128, 128)
        xyz = bbox_center_depth_to_world_xyz(bbox, depth_map, K, pose)
        assert xyz is not None
        assert np.isclose(xyz[2], 2.0, atol=0.1)

    def test_returns_none_for_zero_depth(self):
        K = make_intrinsics(90.0, 256, 256)
        pose = np.eye(4)
        depth_map = np.zeros((256, 256), dtype=np.float32)
        bbox = (64, 64, 192, 192)
        xyz = bbox_center_depth_to_world_xyz(bbox, depth_map, K, pose)
        assert xyz is None


# ──── World <-> grid ────

class TestWorldGridConversion:
    def test_roundtrip(self):
        """world -> grid -> world should be close to original."""
        origin = np.array([0.0, 0.0])
        resolution = 0.05
        xy = np.array([1.5, -2.3])
        ij = world_to_grid(xy, origin, resolution)
        xy_back = grid_to_world(ij, origin, resolution)
        np.testing.assert_allclose(xy_back, xy, atol=resolution)

    def test_origin_maps_to_center(self):
        origin = np.array([0.0, 0.0])
        resolution = 0.05
        ij = world_to_grid(np.array([0.0, 0.0]), origin, resolution)
        xy = grid_to_world(ij, origin, resolution)
        np.testing.assert_allclose(xy, [0.0, 0.0], atol=resolution)

    def test_positive_offset(self):
        origin = np.array([-5.0, -5.0])
        resolution = 0.05
        ij = world_to_grid(np.array([0.0, 0.0]), origin, resolution)
        # 5m / 0.05 = 100 cells from origin
        assert ij[0] == 100
        assert ij[1] == 100
