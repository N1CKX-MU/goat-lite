"""Frontier detection for exploration.

A frontier is a connected boundary between free and unknown space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class Frontier:
    centroid_ij: tuple[int, int]  # (row, col) centroid of the frontier
    size: int                     # number of frontier cells
    cells: np.ndarray             # [N, 2] array of (row, col) coordinates


def find_frontiers(
    occupancy: np.ndarray,
    explored: np.ndarray,
    min_size: int = 20,
) -> list[Frontier]:
    """Find frontiers between free and unknown space.

    A frontier cell is a free cell that has at least one unknown neighbor.

    Args:
        occupancy: [H, W] int8, values in {-1=unknown, 0=free, 1=occupied}.
        explored: [H, W] bool.
        min_size: minimum number of cells for a frontier to be kept.

    Returns:
        List of Frontier objects, sorted by size descending.
    """
    free_mask = occupancy == 0

    # Dilate the unknown mask by one pixel to find free cells adjacent to unknown
    unknown_mask = occupancy == -1
    kernel = np.ones((3, 3), dtype=bool)
    unknown_neighbor = ndimage.binary_dilation(unknown_mask, structure=kernel)

    # Frontier = free AND adjacent to unknown
    frontier_mask = free_mask & unknown_neighbor

    if not np.any(frontier_mask):
        return []

    # Label connected components
    labeled, n_labels = ndimage.label(frontier_mask)

    frontiers = []
    for label_id in range(1, n_labels + 1):
        cells = np.argwhere(labeled == label_id)
        size = len(cells)
        if size < min_size:
            continue
        centroid = cells.mean(axis=0).astype(int)
        frontiers.append(Frontier(
            centroid_ij=(int(centroid[0]), int(centroid[1])),
            size=size,
            cells=cells,
        ))

    # Sort by size descending
    frontiers.sort(key=lambda f: f.size, reverse=True)
    return frontiers
