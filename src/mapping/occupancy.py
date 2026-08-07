"""2D semantic occupancy map built from depth + detections.

The map lives on the XZ ground plane (Habitat Y is up).
"""

from __future__ import annotations

import numpy as np
from skimage.draw import line as skimage_line

from src.mapping.transforms import (
    depth_to_pointcloud_camera,
    pointcloud_camera_to_world,
    world_to_grid,
    grid_to_world,
)


class SemanticMap:
    def __init__(
        self,
        size_m: float = 24.0,
        resolution_m: float = 0.05,
        num_classes: int = 37,
        floor_height: float = 0.0,
        min_height: float = 0.1,
        max_height: float = 1.5,
    ):
        self._size_m = size_m
        self._resolution = resolution_m
        self._num_classes = num_classes
        self._floor_height = floor_height
        self._min_height = min_height
        self._max_height = max_height

        self._grid_size = int(size_m / resolution_m)
        # Origin is at (-size/2, -size/2) in world XZ
        self._origin = np.array([-size_m / 2.0, -size_m / 2.0])

        # Grids
        self._occupancy = np.full(
            (self._grid_size, self._grid_size), -1, dtype=np.int8
        )
        self._explored = np.zeros(
            (self._grid_size, self._grid_size), dtype=bool
        )
        self._class_counts = np.zeros(
            (self._grid_size, self._grid_size, num_classes), dtype=np.int16
        )

    # ── Public accessors ──

    def get_occupancy(self) -> np.ndarray:
        return self._occupancy

    def get_explored(self) -> np.ndarray:
        return self._explored

    def get_class_counts(self) -> np.ndarray:
        return self._class_counts

    def world_to_grid(self, xy: np.ndarray) -> tuple[int, int]:
        return world_to_grid(xy, self._origin, self._resolution)

    def grid_to_world(self, ij: tuple[int, int]) -> np.ndarray:
        return grid_to_world(ij, self._origin, self._resolution)

    # ── Depth update ──

    def update_from_depth(
        self,
        depth: np.ndarray,
        pose: np.ndarray,
        intrinsics: np.ndarray,
        downsample: int = 4,
    ) -> None:
        """Project depth into occupancy grid.

        1. Back-project depth to point cloud in camera frame.
        2. Transform to world frame.
        3. Filter by height (keep only points at min_height..max_height above floor).
        4. Mark occupied cells. Mark free cells along rays (Bresenham).
        """
        pc_cam = depth_to_pointcloud_camera(depth, intrinsics)
        pc_world = pointcloud_camera_to_world(pc_cam, pose)

        # Downsample for speed
        pc_world = pc_world[::downsample]

        # Filter out zero points (invalid depth)
        valid = np.any(pc_world != 0.0, axis=1)
        pc_world = pc_world[valid]

        if len(pc_world) == 0:
            return

        # Height filter: keep points within [min_height, max_height] above floor
        heights = pc_world[:, 1]  # Y is up in Habitat
        height_mask = (
            (heights >= self._floor_height + self._min_height)
            & (heights <= self._floor_height + self._max_height)
        )
        pc_valid = pc_world[height_mask]

        if len(pc_valid) == 0:
            return

        gs = self._grid_size

        # Vectorized: convert all points to grid coords at once
        xz = pc_valid[:, [0, 2]]  # [N, 2]
        offsets = xz - self._origin
        cols = np.round(offsets[:, 0] / self._resolution).astype(np.int32)
        rows = np.round(offsets[:, 1] / self._resolution).astype(np.int32)

        # Filter to in-bounds points
        in_bounds = (rows >= 0) & (rows < gs) & (cols >= 0) & (cols < gs)
        rows_valid = rows[in_bounds]
        cols_valid = cols[in_bounds]

        # Mark occupied
        self._occupancy[rows_valid, cols_valid] = 1
        self._explored[rows_valid, cols_valid] = True

        # Ray clearing: mark free cells between camera and each occupied cell
        cam_world_xz = np.array([pose[0, 3], pose[2, 3]])
        cam_offset = cam_world_xz - self._origin
        cam_col = int(np.round(cam_offset[0] / self._resolution))
        cam_row = int(np.round(cam_offset[1] / self._resolution))
        cam_row = max(0, min(gs - 1, cam_row))
        cam_col = max(0, min(gs - 1, cam_col))

        # Subsample points for ray clearing (most rays overlap heavily)
        unique_cells = np.unique(
            np.column_stack([rows_valid, cols_valid]), axis=0
        )
        for r, c in unique_cells:
            rr, rc = skimage_line(cam_row, cam_col, int(r), int(c))
            # Mark all but last cell (the occupied endpoint) as free
            if len(rr) > 1:
                ray_r = rr[:-1]
                ray_c = rc[:-1]
                mask = (ray_r >= 0) & (ray_r < gs) & (ray_c >= 0) & (ray_c < gs)
                ray_r = ray_r[mask]
                ray_c = rc[:-1][mask]
                # Only mark as free if not already occupied
                not_occ = self._occupancy[ray_r, ray_c] != 1
                self._occupancy[ray_r[not_occ], ray_c[not_occ]] = 0
                self._explored[ray_r, ray_c] = True

    # ── Detection update ──

    def update_from_detections(
        self,
        instances: list,
        kernel_size: int = 3,
    ) -> None:
        """Increment class_counts at projected grid cells with a small Gaussian kernel."""
        if not instances:
            return

        gs = self._grid_size
        half = kernel_size // 2

        # Build a small Gaussian-ish kernel (1s with center weighted more)
        kernel = np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ], dtype=np.int16)

        for inst in instances:
            xz = np.array([inst.world_xyz[0], inst.world_xyz[2]])
            r, c = self.world_to_grid(xz)

            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    ri = r + dr
                    ci = c + dc
                    if 0 <= ri < gs and 0 <= ci < gs:
                        w = kernel[dr + half, dc + half]
                        self._class_counts[ri, ci, inst.cls_id] += w
