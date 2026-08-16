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


def map_extent_for_scene(
    bounds_lo, bounds_hi, margin_m: float = 2.0, min_size_m: float = 12.0
) -> tuple[float, tuple[float, float]]:
    """Grid size and origin covering a scene's navigable extent.

    Returns ``(size_m, origin_xz)`` for SemanticMap. Square, because the grid is
    square, and padded so the agent can map slightly past the navmesh edge.

    Sizing the map to the scene matters: over the 36 GOAT-Bench val scenes the
    default 24 m window centred on the world origin fails to contain 18 of them.
    """
    lo = np.asarray(bounds_lo, dtype=np.float64)
    hi = np.asarray(bounds_hi, dtype=np.float64)
    ext_x = float(hi[0] - lo[0])
    ext_z = float(hi[2] - lo[2])
    size = max(ext_x, ext_z) + 2.0 * margin_m
    size = max(size, min_size_m)
    cx = float(lo[0] + hi[0]) / 2.0
    cz = float(lo[2] + hi[2]) / 2.0
    return size, (cx - size / 2.0, cz - size / 2.0)


class SemanticMap:
    def __init__(
        self,
        size_m: float = 24.0,
        resolution_m: float = 0.05,
        num_classes: int = 37,
        floor_height: float = 0.0,
        min_height: float = 0.1,
        max_height: float = 1.5,
        camera_height: float | None = None,
        min_points_per_cell: int = 3,
        l_occ: int = 3,
        l_free: int = 1,
        clamp: int = 12,
        occ_threshold: int = 3,
        free_threshold: int = -2,
        origin: tuple[float, float] | None = None,
    ):
        self._size_m = size_m
        self._resolution = resolution_m
        self._num_classes = num_classes
        self._floor_height = floor_height
        # When set, the floor is derived per-update as (camera Y - camera_height)
        # instead of using the fixed floor_height. HM3D floors sit at arbitrary
        # world Y (-0.54 in the first val scene, not 0), so a fixed absolute band
        # samples the wrong slab of the scene -- typically the ceiling, which
        # then marks the entire floor plan occupied.
        self._camera_height = camera_height
        self._min_height = min_height
        self._max_height = max_height

        self._grid_size = int(size_m / resolution_m)
        # World XZ of grid cell (0, 0). Defaults to a window centred on the
        # WORLD origin, which is wrong for HM3D: measured over the 36 val
        # scenes, 18 do not fit inside a 24 m window placed there and 4 are
        # larger than 24 m outright (00894 spans 33.8 m). Anything outside stays
        # permanently unknown, so the agent cannot plan through regions of half
        # the val set. Callers should pass an origin derived from
        # HabitatEnv.get_scene_bounds().
        self._origin = (
            np.asarray(origin, dtype=np.float64)
            if origin is not None
            else np.array([-size_m / 2.0, -size_m / 2.0])
        )

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
        # Out-of-vocabulary class ids seen in update_from_detections().
        self._dropped_cls_ids: set[int] = set()

        # Evidence grid behind _occupancy. The old code did
        # ``_occupancy[r, c] = 1`` from a SINGLE depth pixel and never undid it,
        # while ray-clearing refused to touch anything already marked occupied.
        # Every projection error was therefore permanent, the grid filled with
        # phantom obstacles over an episode, and A* ran out of routes: measured
        # against the navmesh, only 37% of cells the map called free were
        # actually navigable and 45% of cells it called occupied were.
        #
        # Two mechanisms replace that:
        #   min_points_per_cell -- within one frame a real surface projects many
        #     pixels into a 5 cm cell; a stray sample projects one or two.
        #   clamped log-odds -- occupancy accumulates across frames and free
        #     evidence can outvote it, so a mistake is recoverable. The clamp is
        #     what makes recovery possible: without it a cell hit often enough
        #     could never be cleared again.
        self._min_points_per_cell = min_points_per_cell
        self._l_occ = l_occ
        self._l_free = l_free
        self._clamp = clamp
        self._occ_threshold = occ_threshold
        self._free_threshold = free_threshold
        self._log_odds = np.zeros(
            (self._grid_size, self._grid_size), dtype=np.int16
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
        if self._camera_height is not None:
            # pose translation is the camera; the floor is a fixed drop below it.
            floor = float(pose[1, 3]) - self._camera_height
        else:
            floor = self._floor_height
        height_mask = (
            (heights >= floor + self._min_height)
            & (heights <= floor + self._max_height)
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
        if rows_valid.size == 0:
            return

        # How many points landed in each cell this frame? A real surface fills a
        # 5 cm cell with many pixels; a bad sample contributes one or two, and
        # those should not create an obstacle.
        flat = rows_valid.astype(np.int64) * gs + cols_valid.astype(np.int64)
        per_cell = np.bincount(flat, minlength=gs * gs)
        hit_flat = np.nonzero(per_cell >= self._min_points_per_cell)[0]
        if hit_flat.size == 0:
            return
        hit_rows, hit_cols = np.divmod(hit_flat, gs)

        # Positive evidence at the surface.
        self._log_odds[hit_rows, hit_cols] += self._l_occ
        self._explored[hit_rows, hit_cols] = True

        # Ray clearing: negative evidence between the camera and each surface.
        cam_world_xz = np.array([pose[0, 3], pose[2, 3]])
        cam_offset = cam_world_xz - self._origin
        cam_col = int(np.round(cam_offset[0] / self._resolution))
        cam_row = int(np.round(cam_offset[1] / self._resolution))
        cam_row = max(0, min(gs - 1, cam_row))
        cam_col = max(0, min(gs - 1, cam_col))

        for r, c in zip(hit_rows, hit_cols):
            rr, rc = skimage_line(cam_row, cam_col, int(r), int(c))
            if len(rr) <= 1:
                continue
            # Everything before the endpoint was seen through, so it is free.
            ray_r, ray_c = rr[:-1], rc[:-1]
            mask = (ray_r >= 0) & (ray_r < gs) & (ray_c >= 0) & (ray_c < gs)
            ray_r, ray_c = ray_r[mask], ray_c[mask]
            # Unlike the old version this is NOT gated on "not already
            # occupied" -- that gate is precisely what made bad marks permanent.
            self._log_odds[ray_r, ray_c] -= self._l_free
            self._explored[ray_r, ray_c] = True

        # Clamp so no cell becomes unrecoverable in either direction.
        np.clip(self._log_odds, -self._clamp, self._clamp, out=self._log_odds)

        # Derive the tri-state grid the planner consumes.
        self._occupancy[:] = -1
        self._occupancy[self._log_odds >= self._occ_threshold] = 1
        self._occupancy[self._log_odds <= self._free_threshold] = 0

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
            # The map only has a channel per GOAT class. A detector with a
            # different vocabulary (e.g. the stock COCO yolov8n fallback, which
            # emits ids up to 79) would index out of bounds and crash the whole
            # episode, so drop what the map cannot represent.
            cls_id = int(inst.cls_id)
            if not 0 <= cls_id < self._num_classes:
                self._dropped_cls_ids.add(cls_id)
                continue

            xz = np.array([inst.world_xyz[0], inst.world_xyz[2]])
            r, c = self.world_to_grid(xz)

            for dr in range(-half, half + 1):
                for dc in range(-half, half + 1):
                    ri = r + dr
                    ci = c + dc
                    if 0 <= ri < gs and 0 <= ci < gs:
                        w = kernel[dr + half, dc + half]
                        self._class_counts[ri, ci, cls_id] += w

    def mark_occupied(self, row: int, col: int, weight: int | None = None) -> None:
        """Force strong occupancy evidence into one cell.

        Used when habitat refuses a motion: whatever is there is solid, whatever
        the depth sensor suggested. Applied at the clamp so a single collision
        outvotes accumulated free evidence immediately.
        """
        gs = self._grid_size
        if not (0 <= row < gs and 0 <= col < gs):
            return
        self._log_odds[row, col] = self._clamp if weight is None else weight
        self._explored[row, col] = True
        self._occupancy[row, col] = 1

    @property
    def dropped_cls_ids(self) -> set[int]:
        """Class ids seen but not representable in this map.

        Non-empty almost always means the detector's vocabulary does not match
        ``num_classes`` -- typically the finetuned GOAT weights are missing and
        the detector silently fell back to COCO yolov8n.
        """
        return self._dropped_cls_ids
