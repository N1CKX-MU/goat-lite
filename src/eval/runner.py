"""Evaluation runner — run agent on GOAT-Bench episodes and collect results."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import habitat_sim
import numpy as np
import quaternion as qt

from src.sim.env import HabitatEnv, GoalSpec, Observation
from src.sim.goat_dataset import EpisodeSpec, SubtaskSpec
from src.eval.metrics import subtask_success, compute_spl


@dataclass
class SubtaskResult:
    """Result of running one subtask."""
    episode_id: str
    subtask_index: int
    category: str
    modality: str
    success: bool
    spl: float
    distance_to_goal: float
    steps_taken: int
    actual_path_length: float
    shortest_path_length: float
    reason: str  # "success", "timeout", "wrong_stop", "no_plan"
    wall_time_s: float = 0.0


def _resolve_scene_path(scene_id: str, hm3d_root: str = "data/hm3d") -> str:
    """Convert GOAT-Bench scene_id to a local file path.

    scene_id format: "hm3d/val//00894-HY1NcmCgn3n/HY1NcmCgn3n.basis.glb"
    We need: "data/hm3d/00894-HY1NcmCgn3n/HY1NcmCgn3n.basis.glb"
    """
    parts = scene_id.split("/")
    # Find the scene directory part (e.g. "00894-HY1NcmCgn3n")
    scene_parts = [p for p in parts if p and p != "hm3d" and p != "val"]
    return f"{hm3d_root}/{'/'.join(scene_parts)}"


def _geodesic_distance(sim: habitat_sim.Simulator, start: np.ndarray, goal: np.ndarray) -> float:
    """Compute geodesic distance between two points using the navmesh."""
    path = habitat_sim.ShortestPath()
    path.requested_start = start.tolist()
    path.requested_end = goal.tolist()
    found = sim.pathfinder.find_path(path)
    if found:
        return float(path.geodesic_distance)
    return float("inf")


def _make_goal_spec(subtask: SubtaskSpec) -> GoalSpec:
    """Create a GoalSpec from a SubtaskSpec."""
    if subtask.modality == "category":
        return GoalSpec(
            modality="category",
            value=subtask.category,
            subtask_index=subtask.subtask_index,
        )
    elif subtask.modality == "language":
        desc = subtask.goal_info.lang_desc if subtask.goal_info else subtask.category
        return GoalSpec(
            modality="language",
            value=desc,
            subtask_index=subtask.subtask_index,
        )
    elif subtask.modality == "image":
        # Image goals are localized by their target category (the finetuned
        # detector emits GOAT categories directly). The matcher routes the
        # "image" modality through category matching on this value.
        return GoalSpec(
            modality="image",
            value=subtask.category,
            subtask_index=subtask.subtask_index,
        )
    else:
        return GoalSpec(
            modality="category",
            value=subtask.category,
            subtask_index=subtask.subtask_index,
        )


def run_episode(
    env: HabitatEnv,
    agent,
    episode: EpisodeSpec,
    max_steps_per_subtask: int = 500,
) -> list[SubtaskResult]:
    """Run a full multi-subtask episode.

    Args:
        env: HabitatEnv instance (already loaded with the right scene).
        agent: GoatAgent instance.
        episode: episode specification with subtasks.
        max_steps_per_subtask: hard step limit per subtask.

    Returns:
        List of SubtaskResult, one per subtask.
    """
    results = []

    # Set agent to episode start position
    sim = env._sim
    agent_hab = env._agent
    state = habitat_sim.AgentState()
    state.position = episode.start_position.tolist()
    state.rotation = qt.from_float_array(episode.start_rotation)
    agent_hab.set_state(state)

    for i, subtask in enumerate(episode.subtasks):
        t0 = time.perf_counter()

        # Set goal
        goal_spec = _make_goal_spec(subtask)
        env.set_goal(goal_spec)

        # Reset agent (keep memory from previous subtasks)
        obs = env._make_obs()
        keep_memory = (i > 0)
        agent.reset(obs, keep_memory=keep_memory)

        # Compute geodesic shortest path for SPL
        if subtask.goal_info is not None:
            goal_pos = subtask.goal_info.position
            agent_pos_start = env.get_gps()
            start_3d = np.array([
                agent_pos_start[0],
                float(episode.start_position[1]),
                agent_pos_start[1],
            ])
            shortest_path = _geodesic_distance(sim, start_3d, goal_pos)
        else:
            goal_pos = None
            shortest_path = float("inf")

        # Run the agent
        actual_path = 0.0
        prev_gps = env.get_gps().copy()
        steps = 0
        done = False

        for step in range(max_steps_per_subtask):
            obs = env._make_obs()
            action = agent.act(obs)
            steps += 1

            if action == 0:
                done = True
                break

            obs, env_done, info = env.step(action)

            # Accumulate path length
            curr_gps = env.get_gps()
            actual_path += float(np.linalg.norm(curr_gps - prev_gps))
            prev_gps = curr_gps.copy()

            if env_done:
                done = True
                break

        wall_time = time.perf_counter() - t0

        # Evaluate success
        final_pos_gps = env.get_gps()
        final_pos_3d = np.array([
            final_pos_gps[0],
            float(env._agent.get_state().position[1]),
            final_pos_gps[1],
        ])

        if goal_pos is not None:
            # Check if goal is in view by running perception on final frame
            final_obs = env._make_obs()
            detections = agent.perception.process(final_obs, step=steps)
            goal_in_view = any(
                hasattr(d, 'cls_name') and d.cls_name == subtask.category
                for d in detections
            )

            success = subtask_success(
                agent_pos=final_pos_3d,
                goal_pos=goal_pos,
                goal_in_view=goal_in_view,
            )

            dx = final_pos_3d[0] - goal_pos[0]
            dz = final_pos_3d[2] - goal_pos[2]
            dist_to_goal = float(np.sqrt(dx * dx + dz * dz))
        else:
            # No goal info — can't evaluate properly
            success = False
            goal_in_view = False
            dist_to_goal = float("inf")

        spl = compute_spl(success, shortest_path, actual_path)

        # Determine failure reason
        if success:
            reason = "success"
        elif steps >= max_steps_per_subtask:
            reason = "timeout"
        elif not goal_in_view and dist_to_goal <= 1.0:
            reason = "wrong_stop"
        else:
            reason = "no_plan"

        results.append(SubtaskResult(
            episode_id=episode.episode_id,
            subtask_index=i,
            category=subtask.category,
            modality=subtask.modality,
            success=success,
            spl=spl,
            distance_to_goal=dist_to_goal,
            steps_taken=steps,
            actual_path_length=actual_path,
            shortest_path_length=shortest_path,
            reason=reason,
            wall_time_s=wall_time,
        ))

    return results
