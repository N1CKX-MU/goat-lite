"""Coordinate transform utilities.

Habitat world frame: Y-up, agent default faces -Z.
Camera frame: OpenGL convention (X-right, Y-up, Z-out-of-screen / -Z is forward).
Map grid frame: 2D, world X -> col, world Z -> row.
"""

from __future__ import annotations

import numpy as np


def make_intrinsics(hfov_deg: float, width: int, height: int) -> np.ndarray:
    """Build a 3x3 camera intrinsic matrix from horizontal FOV and resolution."""
    hfov_rad = np.deg2rad(hfov_deg)
    fx = width / (2.0 * np.tan(hfov_rad / 2.0))
    fy = fx  # square pixels
    cx = width / 2.0
    cy = height / 2.0
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1],
    ], dtype=np.float32)


def depth_to_pointcloud_camera(
    depth: np.ndarray, K: np.ndarray, min_depth: float = 0.01, max_depth: float = 10.0
) -> np.ndarray:
    """Convert a depth image to a point cloud in camera frame.

    Args:
        depth: [H, W] float32, meters.
        K: [3, 3] intrinsic matrix.
        min_depth: ignore pixels below this depth.
        max_depth: ignore pixels above this depth.

    Returns:
        [H*W, 3] point cloud in camera frame (X-right, Y-down, Z-forward).
        Invalid points have (0, 0, 0).
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel grid
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    # Back-project: Habitat uses OpenGL camera convention
    # In OpenGL: X-right, Y-up, Z-out (negative Z is forward)
    # But depth is positive distance along optical axis
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack and reshape
    pc = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Zero out invalid depths
    valid = (depth.ravel() >= min_depth) & (depth.ravel() <= max_depth)
    pc[~valid] = 0.0

    return pc


def pointcloud_camera_to_world(
    pts_camera: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    """Transform points from camera frame to world frame.

    Args:
        pts_camera: [N, 3] points in camera frame.
        pose: [4, 4] SE(3) camera-to-world transform.

    Returns:
        [N, 3] points in world frame.
    """
    R = pose[:3, :3]
    t = pose[:3, 3]
    return (R @ pts_camera.T).T + t


def bbox_center_depth_to_world_xyz(
    bbox: tuple[int, int, int, int],
    depth_map: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray,
    patch_size: int = 5,
) -> np.ndarray | None:
    """Back-project bbox center to a 3D world coordinate using depth.

    Takes the median depth in a small patch around the bbox center
    for robustness against noisy depth.

    Args:
        bbox: (x1, y1, x2, y2) in pixel coords.
        depth_map: [H, W] float32, meters.
        K: [3, 3] intrinsic matrix.
        pose: [4, 4] SE(3) camera-to-world.
        patch_size: size of patch to sample depth from.

    Returns:
        [3] world xyz, or None if depth is invalid.
    """
    x1, y1, x2, y2 = bbox
    cx_px = (x1 + x2) // 2
    cy_px = (y1 + y2) // 2

    h, w = depth_map.shape
    half = patch_size // 2
    r0 = max(0, cy_px - half)
    r1 = min(h, cy_px + half + 1)
    c0 = max(0, cx_px - half)
    c1 = min(w, cx_px + half + 1)

    patch = depth_map[r0:r1, c0:c1]
    valid = patch[(patch > 0.01) & (patch < 10.0)]
    if len(valid) == 0:
        return None

    d = float(np.median(valid))

    # Back-project single pixel
    fx, fy = K[0, 0], K[1, 1]
    ox, oy = K[0, 2], K[1, 2]
    x_cam = (cx_px - ox) * d / fx
    y_cam = (cy_px - oy) * d / fy
    z_cam = d

    pt_cam = np.array([[x_cam, y_cam, z_cam]])
    pt_world = pointcloud_camera_to_world(pt_cam, pose)
    return pt_world[0]


def world_to_grid(
    xy_world: np.ndarray,
    origin: np.ndarray,
    resolution: float,
) -> tuple[int, int]:
    """Convert world XZ coordinates to grid row, col.

    Args:
        xy_world: [2] array of (world_x, world_z).
        origin: [2] array of (min_x, min_z) — bottom-left corner of the map.
        resolution: meters per grid cell.

    Returns:
        (row, col) integer grid indices.
    """
    offset = xy_world - origin
    col = int(np.round(offset[0] / resolution))
    row = int(np.round(offset[1] / resolution))
    return (row, col)


def grid_to_world(
    ij: tuple[int, int],
    origin: np.ndarray,
    resolution: float,
) -> np.ndarray:
    """Convert grid (row, col) to world XZ coordinates.

    Args:
        ij: (row, col) grid indices.
        origin: [2] array of (min_x, min_z).
        resolution: meters per grid cell.

    Returns:
        [2] array of (world_x, world_z).
    """
    row, col = ij
    x = origin[0] + col * resolution
    z = origin[1] + row * resolution
    return np.array([x, z], dtype=np.float64)
