"""Midterm demo: perception + memory + matching pipeline on a scene.

Usage:
    python scripts/midterm_demo.py --scene data/hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb

Steps:
    1. Load a scene.
    2. Random-walk the agent for 300 steps, running perception + memory each step.
    3. Print the instance DB summary.
    4. Run GoalMatcher on 5 hand-crafted goals and print results.
    5. Save a top-down plot.
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.sim.env import HabitatEnv, GoalSpec
from src.perception.detector import YOLODetector
from src.perception.encoder import ClipEncoder
from src.perception.pipeline import PerceptionPipeline
from src.mapping.occupancy import SemanticMap
from src.memory.instance_db import InstanceDatabase
from src.matching.goal_matcher import GoalMatcher


DEMO_GOALS = [
    GoalSpec(modality="category", value="chair"),
    GoalSpec(modality="category", value="couch"),
    GoalSpec(modality="category", value="bed"),
    GoalSpec(modality="language", value="a wooden chair"),
    GoalSpec(modality="language", value="the table near the window"),
]


def run_demo(scene_path: str, steps: int = 300, output: str = "outputs/midterm_demo.png"):
    print(f"Loading scene: {scene_path}")
    env = HabitatEnv(scene_path, resolution=(256, 256))

    print("Loading perception models...")
    detector = YOLODetector(device="cuda", fp16=True, imgsz=256)
    encoder = ClipEncoder(device="cuda", fp16=True)
    pipeline = PerceptionPipeline(
        detector=detector,
        encoder=encoder,
        intrinsics=env.intrinsics,
    )

    sem_map = SemanticMap(size_m=24.0, resolution_m=0.05)
    db = InstanceDatabase(merge_dist_m=0.75, merge_embed_sim=0.85)
    matcher = GoalMatcher(encoder=encoder, language_threshold=0.24)

    obs = env.reset()
    trajectory = [obs.gps.copy()]
    actions = [1, 1, 1, 2, 1, 1, 3, 1]  # biased random walk

    print(f"\nRunning {steps} steps...")
    for step in range(steps):
        # Perception
        instances = pipeline.process(obs, step=step)
        db.update(instances, step=step)

        # Map update every 2 steps
        if step % 2 == 0:
            sem_map.update_from_depth(obs.depth, obs.pose, env.intrinsics)

        if instances:
            sem_map.update_from_detections(instances)

        # Random walk
        action = actions[step % len(actions)]
        obs, done, info = env.step(action)
        trajectory.append(obs.gps.copy())

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}/{steps} | Nodes: {len(db.all_nodes())}")

    # ── Print DB summary ──
    nodes = db.all_nodes()
    print(f"\n{'='*60}")
    print(f"Instance DB: {len(nodes)} nodes")
    print(f"{'='*60}")
    for n in nodes:
        print(f"  [{n.node_id:3d}] {n.cls_name:15s} conf={n.confidence:.2f} "
              f"obs={n.n_observations:3d} pos=({n.world_xyz[0]:+.1f}, {n.world_xyz[2]:+.1f})")

    # ── Goal matching ──
    print(f"\n{'='*60}")
    print("Goal Matching Results")
    print(f"{'='*60}")
    matched_nodes = []
    for goal in DEMO_GOALS:
        node, score = matcher.match(goal, db)
        status = f"MATCH (node {node.node_id}, score={score:.3f})" if node else "NO MATCH"
        print(f"  [{goal.modality:8s}] \"{goal.value}\" → {status}")
        if node:
            matched_nodes.append(node)

    # ── Plot ──
    _save_plot(sem_map, trajectory, nodes, matched_nodes, output)
    env.close()
    print(f"\nPlot saved to {output}")


def _save_plot(sem_map, trajectory, nodes, matched_nodes, output):
    import os
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    occ = sem_map.get_occupancy()

    # Plot occupancy: unknown=gray, free=white, occupied=black
    display = np.ones((*occ.shape, 3), dtype=np.float32) * 0.7  # gray
    display[occ == 0] = [1.0, 1.0, 1.0]  # free = white
    display[occ == 1] = [0.0, 0.0, 0.0]  # occupied = black

    ax.imshow(display, origin="lower")

    # Plot trajectory
    traj = np.array(trajectory)
    traj_grid = np.array([sem_map.world_to_grid(p) for p in traj])
    ax.plot(traj_grid[:, 1], traj_grid[:, 0], "b-", alpha=0.4, linewidth=1)
    ax.plot(traj_grid[0, 1], traj_grid[0, 0], "go", markersize=8, label="Start")
    ax.plot(traj_grid[-1, 1], traj_grid[-1, 0], "rs", markersize=8, label="End")

    # Plot all instance nodes
    matched_ids = {n.node_id for n in matched_nodes}
    for n in nodes:
        xz = np.array([n.world_xyz[0], n.world_xyz[2]])
        r, c = sem_map.world_to_grid(xz)
        if n.node_id in matched_ids:
            ax.plot(c, r, "r*", markersize=14)
            ax.annotate(f"{n.cls_name}", (c, r), fontsize=7,
                        color="red", fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")
        else:
            ax.plot(c, r, "c^", markersize=6)
            ax.annotate(f"{n.cls_name}", (c, r), fontsize=6,
                        color="cyan", alpha=0.8,
                        xytext=(5, 5), textcoords="offset points")

    ax.set_title(f"Midterm Demo — {len(nodes)} instances, {len(traj)} steps")
    ax.legend(loc="upper right")
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="GOAT-Lite midterm demo")
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to HM3D .glb scene file")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=str, default="outputs/midterm_demo.png")
    args = parser.parse_args()
    run_demo(args.scene, args.steps, args.output)


if __name__ == "__main__":
    main()
