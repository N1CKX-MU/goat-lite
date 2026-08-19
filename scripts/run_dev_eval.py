#!/usr/bin/env python3
"""Run GOAT-Lite dev evaluation on a 30-episode subset of val_unseen.

Usage:
    python scripts/run_dev_eval.py [--n-episodes 30] [--output-dir outputs/dev_eval]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sim.env import HabitatEnv
from src.sim.goat_dataset import load_val_unseen, sample_dev_subset
from src.perception.detector import YOLODetector
from src.perception.encoder import ClipEncoder
from src.perception.pipeline import PerceptionPipeline
from src.mapping.occupancy import SemanticMap, map_extent_for_scene
from src.memory.instance_db import InstanceDatabase
from src.matching.goal_matcher import GoalMatcher
from src.agent.goat_agent import GoatAgent
from src.eval.runner import run_episode, _resolve_scene_path
from src.eval.report import save_results


def build_agent(
    intrinsics: np.ndarray,
    device: str = "cuda",
    camera_height: float = 1.41,
    scene_bounds: tuple = None,
) -> GoatAgent:
    """Build the full agent with all subsystems.

    ``scene_bounds`` is ``(lo, hi)`` from ``HabitatEnv.get_scene_bounds()``. When
    given, the occupancy grid is sized and centred on the scene; without it the
    map falls back to a 24 m window on the world origin, which does not contain
    18 of the 36 val scenes.
    """
    detector = YOLODetector(device=device)
    encoder = ClipEncoder(device=device)
    perception = PerceptionPipeline(detector, encoder, intrinsics)
    memory = InstanceDatabase()
    matcher = GoalMatcher(encoder)
    # camera_height lets the map derive the floor from the camera pose instead
    # of assuming floor Y == 0, which no HM3D scene satisfies.
    if scene_bounds is not None:
        size_m, origin = map_extent_for_scene(scene_bounds[0], scene_bounds[1])
        semantic_map = SemanticMap(
            size_m=size_m, origin=origin, camera_height=camera_height
        )
    else:
        semantic_map = SemanticMap(camera_height=camera_height)

    return GoatAgent(
        perception=perception,
        memory=memory,
        matcher=matcher,
        semantic_map=semantic_map,
        intrinsics=intrinsics,
        max_steps=500,
        success_distance=1.0,
    )


def main():
    parser = argparse.ArgumentParser(description="GOAT-Lite dev evaluation")
    parser.add_argument("--n-episodes", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hm3d-root", type=str, default="data/hm3d")
    parser.add_argument("--dataset-root", type=str,
                        default="data/datasets/goat_bench/hm3d/v1/val_unseen")
    args = parser.parse_args()

    # Output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/dev_eval_{timestamp}"

    print(f"Loading val_unseen episodes from {args.dataset_root}...")
    all_episodes = load_val_unseen(args.dataset_root)
    print(f"  Total: {len(all_episodes)} episodes")

    dev_episodes = sample_dev_subset(all_episodes, n=args.n_episodes, seed=args.seed)
    total_subtasks = sum(len(ep.subtasks) for ep in dev_episodes)
    print(f"  Dev subset: {len(dev_episodes)} episodes, {total_subtasks} subtasks")

    # Group episodes by scene for efficiency (avoid reloading scenes)
    by_scene: dict[str, list] = {}
    for ep in dev_episodes:
        by_scene.setdefault(ep.scene_id, []).append(ep)
    print(f"  Scenes to load: {len(by_scene)}")

    all_results = []
    scene_count = 0

    for scene_id, scene_episodes in by_scene.items():
        scene_count += 1
        scene_path = _resolve_scene_path(scene_id, args.hm3d_root)
        print(f"\n[{scene_count}/{len(by_scene)}] Loading scene: {scene_path}")

        try:
            env = HabitatEnv(scene_path=scene_path, seed=args.seed)
        except Exception as e:
            print(f"  ERROR loading scene: {e}")
            continue

        intrinsics = env.intrinsics
        agent = build_agent(
            intrinsics, device=args.device, scene_bounds=env.get_scene_bounds()
        )

        for ep in scene_episodes:
            print(f"  Episode {ep.episode_id}: {len(ep.subtasks)} subtasks...", end=" ")
            t0 = time.perf_counter()

            try:
                results = run_episode(env, agent, ep)
                all_results.extend(results)

                successes = sum(1 for r in results if r.success)
                elapsed = time.perf_counter() - t0
                print(f"{successes}/{len(results)} success, {elapsed:.1f}s")
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

        # A detector whose vocabulary does not match the map's num_classes is a
        # silent correctness problem, not a crash, now that the map drops
        # unrepresentable ids. Surface it loudly instead.
        dropped = getattr(agent.semantic_map, "dropped_cls_ids", set())
        if dropped:
            print(f"  WARNING: dropped {len(dropped)} out-of-vocabulary class ids "
                  f"from the semantic map, e.g. {sorted(dropped)[:8]}")
            print("           The detector's classes do not match the GOAT "
                  "vocabulary - is checkpoints/yolo_goat.pt present?")

        env.close()

        # Checkpoint after every scene. habitat's GL context has died mid-run
        # under WSL (EGL_NOT_INITIALIZED on a later scene load), and that is a
        # native abort -- no Python exception, no chance to save on the way out.
        # A full dev eval is hours, so losing all of it to a flaky driver on the
        # last scene is not acceptable. Saving per scene makes any crash cost at
        # most one scene.
        if all_results:
            save_results(all_results, args.output_dir)
            print(f"  [checkpoint] {len(all_results)} subtasks saved to "
                  f"{args.output_dir}")

    # Save results
    if all_results:
        save_results(all_results, args.output_dir)
    else:
        print("\nNo results collected!")

    print("\nDone.")


if __name__ == "__main__":
    main()
