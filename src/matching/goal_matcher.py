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
        min_observations: int = 2,
        trust_confidence: float = 0.5,
        image_threshold: float = -1.0,
    ):
        """
        Args:
            min_observations: how many times an instance must have been seen
                before it may be targeted. Detector ``conf`` is 0.15, because
                at 0.35 the agent saw nothing in four frames out of five, and
                the price of that recall is false positives in memory. A node
                built from one weak detection should not be a navigation goal:
                the agent navigates to it, and because the node carries the
                detector's own label the same false detection confirms it on
                arrival. Set to 1 to disable the gate.
            trust_confidence: a single sighting this confident is targetable
                regardless of ``min_observations``. Starving the matcher is its
                own failure mode on a recall-limited detector, and probing true
                goal instances produced confidences from 0.22 to 0.93.
            image_threshold: minimum image-goal similarity for a node to be
                accepted at all. See ``_match_image``.
        """
        self._encoder = encoder
        self._category_threshold = category_threshold
        self._language_threshold = language_threshold
        self._use_template = use_template
        self._min_observations = min_observations
        self._trust_confidence = trust_confidence
        # Image-goal similarity below this means "none of these look like the
        # goal". The default -1.0 is the minimum a cosine can take, so it ranks
        # by appearance and never rejects: a strict improvement on ranking by
        # distance that cannot regress the previous behaviour. 0.0 would NOT be
        # that neutral value -- two unrelated CLIP embeddings score around zero
        # and can land either side of it, so a threshold of 0.0 silently
        # discards about half of the single-candidate cases. The rejecting
        # value is a measurement not yet made.
        self._image_threshold = image_threshold

    def match(
        self,
        goal: GoalSpec,
        db: InstanceDatabase,
        agent_xy: np.ndarray | None = None,
        exclude_ids: set[int] | None = None,
    ) -> tuple[InstanceNode | None, float]:
        """Best instance in memory for this goal.

        ``agent_xy`` is the agent's world XZ. It only affects category/image
        goals, where any instance of the category satisfies the task and the
        nearest one is therefore the right target.

        ``exclude_ids`` are node ids the caller has already ruled out -- the
        agent passes the nodes it approached and could not confirm. Filtering
        here rather than in the agent matters: it lets the second-best instance
        win, which for a category goal with several instances in the scene is
        the difference between trying the next one and giving up.
        """
        if goal.modality == "category":
            return self._match_category(goal, db, agent_xy, exclude_ids)
        elif goal.modality == "language":
            return self._match_language(goal, db, exclude_ids)
        elif goal.modality == "image":
            return self._match_image(goal, db, agent_xy, exclude_ids)
        else:
            return None, 0.0

    def _eligible(self, nodes, exclude_ids):
        """Nodes this matcher is willing to hand back as a navigation target."""
        out = []
        for n in nodes:
            if exclude_ids and n.node_id in exclude_ids:
                continue
            if (n.n_observations < self._min_observations
                    and n.confidence < self._trust_confidence):
                continue
            out.append(n)
        return out

    def _match_category(
        self,
        goal: GoalSpec,
        db: InstanceDatabase,
        agent_xy: np.ndarray | None = None,
        exclude_ids: set[int] | None = None,
    ) -> tuple[InstanceNode | None, float]:
        candidates = self._eligible(db.query_by_class_name(goal.value), exclude_ids)
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

    def _match_image(
        self,
        goal: GoalSpec,
        db: InstanceDatabase,
        agent_xy: np.ndarray | None = None,
        exclude_ids: set[int] | None = None,
    ) -> tuple[InstanceNode | None, float]:
        """Pick the remembered instance that looks like the goal picture.

        An image subtask names ONE instance and only that one is credited, but
        this used to route straight to _match_category, i.e. the *nearest*
        instance of the class, ignoring the goal image entirely. Measured over
        the dev subset, 61% of image subtasks have more than one instance of
        the goal category in the scene (median 2, max 47), so the expected
        chance of picking the referenced one was 0.58 -- barely better than a
        coin toss on two-thirds of them, before any other limit applied.

        Both sides of the comparison are CLIP over detector crops (see
        PerceptionPipeline.embed_goal_image), so similarity is a plain dot
        product on L2-normalised vectors.

        Falls back to nearest-of-category when there is no goal embedding,
        which is the previous behaviour and the honest answer when the detector
        could not find the goal class in its own reference picture.
        """
        if goal.goal_embedding is None:
            return self._match_category(goal, db, agent_xy, exclude_ids)

        candidates = self._eligible(
            db.query_by_class_name(goal.value), exclude_ids
        )
        if not candidates:
            return None, 0.0

        best, best_score = None, -1.0
        for node in candidates:
            sim = float(np.dot(goal.goal_embedding, node.clip_embed))
            if sim > best_score:
                best, best_score = node, sim

        if best_score < self._image_threshold:
            # Nothing in memory resembles the goal picture. "Not found yet"
            # keeps the agent exploring rather than committing to whichever
            # instance of the class happens to be nearest.
            return None, best_score
        return best, best_score

    def _match_language(
        self,
        goal: GoalSpec,
        db: InstanceDatabase,
        exclude_ids: set[int] | None = None,
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
        nodes = self._eligible(nodes, exclude_ids)
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
