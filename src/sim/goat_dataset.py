"""GOAT-Bench dataset loader — parse episodes and goals from JSON files."""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GoalInfo:
    """Ground-truth info about a goal instance."""
    object_category: str
    object_id: str
    position: np.ndarray  # [3] world position
    lang_desc: str = ""
    image_goals: list[dict] = field(default_factory=list)
    view_points: list[dict] = field(default_factory=list)


@dataclass
class SubtaskSpec:
    """One subtask within an episode."""
    category: str
    modality: str  # "object" (category), "description" (language), "image"
    goal_key: str | None  # object_id for looking up in goals, None for category-only
    goal_info: GoalInfo | None = None
    subtask_index: int = 0
    # Every instance that counts as reaching this goal. For language/image
    # subtasks that is the one referenced instance; for category ("object")
    # subtasks GOAT-Bench accepts ANY instance of the category, so this holds
    # all of them. Evaluation must score against this list, not goal_info --
    # category subtasks legitimately have goal_key None and hence no goal_info.
    goal_candidates: list[GoalInfo] = field(default_factory=list)


@dataclass
class EpisodeSpec:
    """One full episode with start pose and ordered subtasks."""
    episode_id: str
    scene_id: str
    scene_dataset_config: str
    start_position: np.ndarray  # [3]
    # [4] quaternion in habitat's JSON order: (x, y, z, w).
    # NOT the (w, x, y, z) order numpy-quaternion's from_float_array expects --
    # convert with quaternion_from_coeffs() below, never from_float_array().
    start_rotation: np.ndarray
    subtasks: list[SubtaskSpec] = field(default_factory=list)


def quaternion_from_coeffs(coeffs) -> "np.quaternion":
    """Build a quaternion from habitat's JSON coefficient order (x, y, z, w).

    numpy-quaternion's ``from_float_array`` reads (w, x, y, z), so passing a
    habitat rotation straight to it silently reinterprets the components. For a
    typical yaw-only episode start that turns into a ~180 degree roll: the agent
    spawns upside down, the camera faces the floor, every depth point projects
    below the occupancy height band, the map stays empty, no frontiers are found
    and the agent gives up without moving. The forward vector survives the
    mix-up unchanged, which is what makes it so hard to spot.
    """
    x, y, z, w = (float(c) for c in coeffs)
    return np.quaternion(w, x, y, z)


def _parse_goal(g: dict) -> GoalInfo:
    """Build a GoalInfo from one raw GOAT-Bench goal entry."""
    return GoalInfo(
        object_category=g["object_category"],
        object_id=g["object_id"],
        position=np.array(g["position"], dtype=np.float64),
        lang_desc=g.get("lang_desc", ""),
        image_goals=g.get("image_goals", []),
        view_points=g.get("view_points", []),
    )


def _normalize_modality(raw: str) -> str:
    """Map GOAT-Bench modality names to our internal names."""
    mapping = {
        "object": "category",
        "description": "language",
        "image": "image",
    }
    return mapping.get(raw, raw)


def load_scene_episodes(
    content_path: str,
    scene_base: str,
) -> list[EpisodeSpec]:
    """Load all episodes from a per-scene GOAT-Bench content file.

    Args:
        content_path: path to the .json.gz content file.
        scene_base: e.g. "HY1NcmCgn3n.basis.glb" for goal key lookup.

    Returns:
        List of EpisodeSpec objects.
    """
    with gzip.open(content_path, "rt") as f:
        data = json.load(f)

    raw_episodes = data["episodes"]
    goals_dict = data.get("goals", {})

    episodes = []
    for raw_ep in raw_episodes:
        subtasks = []
        for i, task in enumerate(raw_ep["tasks"]):
            category = task[0]
            modality_raw = task[1]
            goal_key = task[2] if len(task) > 2 else None

            # Every annotated instance of this category in this scene.
            full_key = f"{scene_base}_{category}"
            candidates = [_parse_goal(g) for g in goals_dict.get(full_key, [])]

            goal_info = None
            if goal_key is not None:
                # language / image: one specific instance is the target.
                goal_info = next(
                    (c for c in candidates if c.object_id == goal_key), None
                )
                goal_candidates = [goal_info] if goal_info is not None else []
            else:
                # category ("object"): reaching any instance counts.
                goal_candidates = candidates

            subtasks.append(SubtaskSpec(
                category=category,
                modality=_normalize_modality(modality_raw),
                goal_key=goal_key,
                goal_info=goal_info,
                subtask_index=i,
                goal_candidates=goal_candidates,
            ))

        episodes.append(EpisodeSpec(
            episode_id=str(raw_ep["episode_id"]),
            scene_id=raw_ep["scene_id"],
            scene_dataset_config=raw_ep.get("scene_dataset_config", ""),
            start_position=np.array(raw_ep["start_position"], dtype=np.float64),
            start_rotation=np.array(raw_ep["start_rotation"], dtype=np.float64),
            subtasks=subtasks,
        ))

    return episodes


def load_val_unseen(
    dataset_root: str = "data/datasets/goat_bench/hm3d/v1/val_unseen",
) -> list[EpisodeSpec]:
    """Load all val_unseen episodes across all scenes.

    Args:
        dataset_root: path to the val_unseen directory.

    Returns:
        List of all EpisodeSpec objects, sorted by episode_id.
    """
    content_dir = os.path.join(dataset_root, "content")
    all_episodes = []

    for fname in sorted(os.listdir(content_dir)):
        if not fname.endswith(".json.gz"):
            continue
        scene_hash = fname.replace(".json.gz", "")
        content_path = os.path.join(content_dir, fname)

        # Find the full scene base name (e.g. "HY1NcmCgn3n.basis.glb")
        # from the first episode's scene_id
        with gzip.open(content_path, "rt") as f:
            data = json.load(f)
        if not data["episodes"]:
            continue
        scene_id = data["episodes"][0]["scene_id"]
        scene_base = scene_id.split("/")[-1]  # e.g. HY1NcmCgn3n.basis.glb

        episodes = load_scene_episodes(content_path, scene_base)
        all_episodes.extend(episodes)

    all_episodes.sort(key=lambda e: int(e.episode_id) if e.episode_id.isdigit() else e.episode_id)
    return all_episodes


def sample_dev_subset(
    episodes: list[EpisodeSpec],
    n: int = 30,
    seed: int = 42,
) -> list[EpisodeSpec]:
    """Deterministically sample a dev subset of episodes.

    Args:
        episodes: full list of episodes.
        n: number to sample.
        seed: random seed for reproducibility.

    Returns:
        Sampled subset.
    """
    rng = np.random.RandomState(seed)
    n = min(n, len(episodes))
    indices = rng.choice(len(episodes), size=n, replace=False)
    indices.sort()
    return [episodes[i] for i in indices]
