"""Goal matching — find the best instance in memory for a given goal."""

from __future__ import annotations

import numpy as np

from src.memory.instance_db import InstanceDatabase, InstanceNode
from src.sim.env import GoalSpec


class GoalMatcher:
    def __init__(
        self,
        encoder,
        category_threshold: float = 0.0,
        language_threshold: float = 0.24,
        use_template: bool = True,
    ):
        self._encoder = encoder
        self._category_threshold = category_threshold
        self._language_threshold = language_threshold
        self._use_template = use_template

    def match(
        self, goal: GoalSpec, db: InstanceDatabase
    ) -> tuple[InstanceNode | None, float]:
        if goal.modality == "category":
            return self._match_category(goal, db)
        elif goal.modality == "language":
            return self._match_language(goal, db)
        else:
            return None, 0.0

    def _match_category(
        self, goal: GoalSpec, db: InstanceDatabase
    ) -> tuple[InstanceNode | None, float]:
        candidates = db.query_by_class_name(goal.value)
        if not candidates:
            return None, 0.0
        # Return highest confidence node
        best = max(candidates, key=lambda n: n.confidence)
        return best, best.confidence

    def _match_language(
        self, goal: GoalSpec, db: InstanceDatabase
    ) -> tuple[InstanceNode | None, float]:
        nodes = db.all_nodes()
        if not nodes:
            return None, 0.0

        # Prepare text prompt
        text = goal.value
        if self._use_template and len(text.split()) <= 3:
            text = f"a photo of {text}"

        text_embed = self._encoder.encode_text([text])[0]

        # Score against all node embeddings
        best_node = None
        best_score = -1.0
        for node in nodes:
            sim = float(np.dot(text_embed, node.clip_embed))
            if sim > best_score:
                best_score = sim
                best_node = node

        if best_score < self._language_threshold:
            return None, best_score

        return best_node, best_score
