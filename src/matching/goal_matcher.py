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
        self, goal: GoalSpec, db: InstanceDatabase, agent_xy: np.ndarray | None = None
    ) -> tuple[InstanceNode | None, float]:
        """Best instance in memory for this goal.

        ``agent_xy`` is the agent's world XZ. It only affects category/image
        goals, where any instance of the category satisfies the task and the
        nearest one is therefore the right target.
        """
        if goal.modality == "category":
            return self._match_category(goal, db, agent_xy)
        elif goal.modality == "language":
            return self._match_language(goal, db)
        elif goal.modality == "image":
            # Image goals carry the target category in `value`; with the
            # finetuned detector we localize the goal by that category.
            return self._match_category(goal, db, agent_xy)
        else:
            return None, 0.0

    def _match_category(
        self,
        goal: GoalSpec,
        db: InstanceDatabase,
        agent_xy: np.ndarray | None = None,
    ) -> tuple[InstanceNode | None, float]:
        candidates = db.query_by_class_name(goal.value)
        if not candidates:
            return None, 0.0
        if agent_xy is None:
            # No pose available (e.g. unit tests): fall back to confidence.
            best = max(candidates, key=lambda n: n.confidence)
            return best, best.confidence
        # Any instance of the category counts as success, so go to the CLOSEST
        # one. Picking the most confident instead made the agent walk past the
        # object it was standing next to (ending 0.98 m away, inside the success
        # radius) to chase a better-scored one across the scene, and time out.
        best = min(
            candidates,
            key=lambda n: float(
                np.linalg.norm(np.asarray(n.world_xyz)[[0, 2]] - agent_xy)
            ),
        )
        return best, best.confidence

    def _match_language(
        self, goal: GoalSpec, db: InstanceDatabase
    ) -> tuple[InstanceNode | None, float]:
        # A GOAT-Bench language goal always describes an instance OF a known
        # category ("the mirror next to the sink" is a mirror), and that
        # category travels on the GoalSpec. Restrict candidates to it.
        #
        # Scoring the description against every remembered instance instead
        # lets a barely-above-threshold CLIP similarity win outright: observed
        # a "window glass" node matched at 0.257 for a microwave goal, with
        # ZERO microwave nodes in memory, after which the agent navigated to
        # the window and confidently stopped 12.5 m from the real target.
        # With no same-class candidate the honest answer is "not found yet" --
        # returning None keeps the agent exploring instead.
        if goal.category:
            nodes = db.query_by_class_name(goal.category)
        else:
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
