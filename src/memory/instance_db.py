"""Instance memory database — stores and merges object detections over time."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.perception.pipeline import PerceivedInstance


@dataclass
class InstanceNode:
    node_id: int
    cls_id: int
    cls_name: str
    world_xyz: np.ndarray       # [3], running mean
    clip_embed: np.ndarray      # [512], running mean, re-normalized
    confidence: float           # running max
    first_seen_step: int
    last_seen_step: int
    n_observations: int
    best_thumbnail: np.ndarray  # 64x64x3 uint8


class InstanceDatabase:
    def __init__(
        self,
        merge_dist_m: float = 0.75,
        merge_embed_sim: float = 0.85,
    ):
        self._merge_dist = merge_dist_m
        self._merge_sim = merge_embed_sim
        self._nodes: list[InstanceNode] = []
        self._next_id: int = 0

    def all_nodes(self) -> list[InstanceNode]:
        return list(self._nodes)

    def query_by_class(self, cls_id: int) -> list[InstanceNode]:
        return [n for n in self._nodes if n.cls_id == cls_id]

    def query_by_class_name(self, cls_name: str) -> list[InstanceNode]:
        return [n for n in self._nodes if n.cls_name == cls_name]

    def query_by_embedding(
        self, embed: np.ndarray, top_k: int = 5
    ) -> list[tuple[InstanceNode, float]]:
        if not self._nodes:
            return []
        scored = []
        for node in self._nodes:
            sim = float(np.dot(embed, node.clip_embed))
            scored.append((node, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def update(self, perceived: list[PerceivedInstance], step: int) -> None:
        for pi in perceived:
            self._integrate(pi, step)

    def _integrate(self, pi: PerceivedInstance, step: int) -> None:
        # Find candidate nodes: same class and within merge distance (XZ)
        candidates = []
        for node in self._nodes:
            if node.cls_id != pi.cls_id:
                continue
            dist = np.linalg.norm(node.world_xyz[[0, 2]] - pi.world_xyz[[0, 2]])
            if dist <= self._merge_dist:
                candidates.append((node, dist))

        if not candidates:
            self._create_node(pi, step)
            return

        # Sort by distance, merge with closest
        candidates.sort(key=lambda x: x[1])
        best_node, _ = candidates[0]

        # Check embedding similarity — if dissimilar, don't merge
        sim = float(np.dot(best_node.clip_embed, pi.clip_embed))
        if sim < self._merge_sim:
            self._create_node(pi, step)
            return

        self._merge_into(best_node, pi, step)

    def _create_node(self, pi: PerceivedInstance, step: int) -> None:
        node = InstanceNode(
            node_id=self._next_id,
            cls_id=pi.cls_id,
            cls_name=pi.cls_name,
            world_xyz=pi.world_xyz.copy(),
            clip_embed=pi.clip_embed.copy(),
            confidence=pi.conf,
            first_seen_step=step,
            last_seen_step=step,
            n_observations=1,
            best_thumbnail=pi.crop_thumbnail.copy(),
        )
        self._nodes.append(node)
        self._next_id += 1

    def _merge_into(
        self, node: InstanceNode, pi: PerceivedInstance, step: int
    ) -> None:
        n = node.n_observations
        # Running mean for position
        node.world_xyz = (node.world_xyz * n + pi.world_xyz) / (n + 1)
        # Running mean for embedding, then re-normalize
        new_embed = (node.clip_embed * n + pi.clip_embed) / (n + 1)
        norm = np.linalg.norm(new_embed)
        if norm > 1e-8:
            new_embed /= norm
        node.clip_embed = new_embed
        # Running max for confidence + best thumbnail
        if pi.conf > node.confidence:
            node.confidence = pi.conf
            node.best_thumbnail = pi.crop_thumbnail.copy()
        node.last_seen_step = step
        node.n_observations = n + 1
