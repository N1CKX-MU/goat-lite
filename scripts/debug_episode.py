#!/usr/bin/env python3
"""Run one subtask and record what the agent believed at every step.

Diagnostic tool. Writes an MP4 (and optionally PNG frames) showing the
occupancy map, the planned path, the agent's real track, its remembered
instances, and the ground-truth goal it can never see -- side by side with the
first-person view and the live FSM state.

Usage:
    python scripts/debug_episode.py --episode 1 --subtask 2
    python scripts/debug_episode.py --episode 1 --subtask 2 --steps 200 --frames
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import habitat_sim

from src.sim.env import HabitatEnv
from src.sim.goat_dataset import (
    load_val_unseen,
    sample_dev_subset,
    quaternion_from_coeffs,
)
from src.eval.runner import _resolve_scene_path, _make_goal_spec
from src.utils.viz import render_topdown, draw_detections, compose_frame, legend_lines
from scripts.run_dev_eval import build_agent


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualise one subtask")
    ap.add_argument("--episode", type=int, default=1,
                    help="index into the deterministic dev subset")
    ap.add_argument("--subtask", type=int, default=0)
    ap.add_argument("--warmup-subtasks", type=int, default=None,
                    help="run the preceding subtasks first so memory/map match "
                         "the real eval (defaults to --subtask)")
    ap.add_argument("--warmup-steps", type=int, default=300)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-episodes", type=int, default=2,
                    help="dev-subset size; must match the run being debugged")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/debug")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--frames", action="store_true", help="also write PNGs")
    ap.add_argument("--dataset-root",
                    default="data/datasets/goat_bench/hm3d/v1/val_unseen")
    ap.add_argument("--hm3d-root", default="data/hm3d")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    eps = load_val_unseen(args.dataset_root)
    dev = sample_dev_subset(eps, n=args.n_episodes, seed=args.seed)
    ep = dev[args.episode]
    print(f"episode {args.episode}: {ep.episode_id}  scene={ep.scene_id}")
    print(f"subtasks: {[(s.modality, s.category) for s in ep.subtasks]}")

    env = HabitatEnv(scene_path=_resolve_scene_path(ep.scene_id, args.hm3d_root),
                     seed=args.seed)
    state = habitat_sim.AgentState()
    state.position = ep.start_position.tolist()
    state.rotation = quaternion_from_coeffs(ep.start_rotation)
    env._agent.set_state(state)

    agent = build_agent(env.intrinsics, scene_bounds=env.get_scene_bounds())

    # Replay the earlier subtasks so map and memory match the real run -- the
    # interesting failures only appear once memory is populated.
    warm = args.warmup_subtasks if args.warmup_subtasks is not None else args.subtask
    for i in range(min(warm, len(ep.subtasks))):
        env.set_goal(_make_goal_spec(ep.subtasks[i]))
        obs = env._make_obs()
        agent.reset(obs, keep_memory=(i > 0))
        for _ in range(args.warmup_steps):
            obs = env._make_obs()
            a = agent.act(obs)
            if a == 0:
                break
            env.step(a)
        print(f"  warmed subtask {i} ({ep.subtasks[i].category})")

    sub = ep.subtasks[args.subtask]
    goals = [np.asarray(c.position) for c in sub.goal_candidates]
    env.set_goal(_make_goal_spec(sub))
    obs = env._make_obs()
    agent.reset(obs, keep_memory=(args.subtask > 0))

    print(f"\ntracing subtask {args.subtask}: {sub.modality}/{sub.category}, "
          f"{len(goals)} valid instance(s)")

    writer = None
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(
            os.path.join(args.out, f"ep{args.episode}_sub{args.subtask}.mp4"),
            fps=args.fps, macro_block_size=None,
        )
    except Exception as e:  # pragma: no cover - video is a convenience
        print(f"  (no video writer: {e}; writing PNGs instead)")
        args.frames = True

    import cv2

    track = []
    stalled = 0
    prev_xy = env.get_gps().copy()
    closest, closest_step, closest_in_view = float("inf"), -1, False

    for step in range(args.steps):
        obs = env._make_obs()
        xy = env.get_gps()
        track.append(xy.copy())

        detections = agent.perception.process(obs, step=step)
        action = agent.act(obs)

        d_goal = min((float(np.hypot(xy[0] - g[0], xy[1] - g[2])) for g in goals),
                     default=float("nan"))
        d_node = (agent._distance_to_node(obs, agent._matched_node)
                  if agent._matched_node is not None else float("nan"))

        # Re-run the match purely to report its score. A wrong match is only
        # diagnosable alongside what the matcher scored and what it rejected.
        _, match_score = agent.matcher.match(obs.current_goal, agent.memory, xy)
        goal_in_view = agent._goal_in_view(detections)
        seen = sorted({d.cls_name for d in detections})
        seen_classes = ",".join(seen)[:26] if seen else "-"
        if d_goal < closest:
            closest, closest_step, closest_in_view = d_goal, step, goal_in_view
        candidates = agent.memory.query_by_class_name(sub.category)
        matched_err = float("nan")
        if agent._matched_node is not None and goals:
            mp = np.asarray(agent._matched_node.world_xyz)
            matched_err = min(
                float(np.hypot(mp[0] - g[0], mp[2] - g[2])) for g in goals
            )

        top = render_topdown(
            agent.semantic_map, xy, float(obs.compass),
            path=agent._current_path,
            track=track,
            nodes=agent.memory.all_nodes(),
            matched_node=agent._matched_node,
            goal_positions=goals,
            goal_category=sub.category,
            success_radius=agent.success_distance,
        )
        fpv = draw_detections(obs.rgb, detections)

        lines = [
            f"step   {step}/{args.steps}",
            f"state  {str(agent.state).split('.')[-1]}",
            f"action {({0:'STOP',1:'FWD',2:'LEFT',3:'RIGHT'}).get(action, action)}",
            f"goal   {sub.modality}/{sub.category}",
            "",
            f"dist to TRUE goal  {d_goal:.2f} m",
            f"dist to node       {d_node:.2f} m",
            f"success radius     {agent.success_distance:.2f} m",
            "",
            f"match score        {match_score:.3f}",
            f"MATCH ERROR        {matched_err:.2f} m",
            f"same-class nodes   {len(candidates)}",
            "",
            # The two predicates that gate STOP. Being within the radius is not
            # enough: VERIFYING must fire (on distance to the NODE, not the true
            # goal) and the goal class must be detected in the current frame.
            f"goal in view       {goal_in_view}",
            f"detected classes   {seen_classes}",
            f"verify steps       {agent._verify_steps}",
            "",
            f"path len   {len(agent._current_path) if agent._current_path else 0}",
            f"path idx   {agent._path_idx}",
            f"memory     {len(agent.memory.all_nodes())} nodes",
            f"detections {len(detections)}",
            f"collided   {getattr(obs, 'collided', False)}",
            f"stalled    {stalled} steps",
            "",
        ] + legend_lines()

        frame = compose_frame(top, fpv, lines)
        if writer is not None:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if args.frames:
            cv2.imwrite(os.path.join(args.out, f"f{step:04d}.png"), frame)

        if action == 0:
            print(f"  agent called STOP at step {step}, {d_goal:.2f} m from goal")
            break
        env.step(action)

        moved = float(np.linalg.norm(env.get_gps() - prev_xy))
        stalled = stalled + 1 if moved < 1e-3 else 0
        prev_xy = env.get_gps().copy()

    if writer is not None:
        writer.close()

    final = env.get_gps()
    d = min((float(np.hypot(final[0] - g[0], final[1] - g[2])) for g in goals),
            default=float("nan"))
    print(f"\nfinished {d:.2f} m from the nearest valid instance "
          f"(success needs <= {agent.success_distance} m and the goal in view)")
    print(f"CLOSEST approach: {closest:.2f} m at step {closest_step}, "
          f"goal_in_view={closest_in_view}")
    if closest <= agent.success_distance:
        print("  -> it WAS inside the success radius; failure is the stop "
              "condition, not navigation")
    print(f"wrote {args.out}/")
    env.close()


if __name__ == "__main__":
    main()
