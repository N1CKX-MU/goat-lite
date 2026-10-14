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
from src.perception.encoder import ClipEncoder, validate_encoder
from src.perception.pipeline import PerceptionPipeline
from src.mapping.occupancy import SemanticMap, map_extent_for_scene
from src.memory.instance_db import InstanceDatabase
from src.matching.goal_matcher import GoalMatcher
from src.agent.goat_agent import GoatAgent
from src.planning.exploration import FRONTIER_STRATEGIES
from src.eval.runner import run_episode, _resolve_scene_path
from src.eval.report import save_results, load_results, resumable_prefix


def build_agent(
    intrinsics: np.ndarray,
    device: str = "cuda",
    camera_height: float = 1.41,
    scene_bounds: tuple = None,
    min_observations: int = 2,
    language_threshold: float = 0.24,
    image_threshold: float = -1.0,
    merge_dist_m: float = 0.75,
    merge_embed_sim: float = 0.85,
    success_distance: float = 1.0,
    commit_distance_frac: float = None,
    planner: str = "astar",
    frontier_strategy: str = "nearest-largest",
    encoder_model: str = "ViT-B-32",
    encoder_pretrained: str = "laion2b_s34b_b79k",
) -> GoatAgent:
    """Build the full agent with all subsystems.

    ``scene_bounds`` is ``(lo, hi)`` from ``HabitatEnv.get_scene_bounds()``. When
    given, the occupancy grid is sized and centred on the scene; without it the
    map falls back to a 24 m window on the world origin, which does not contain
    18 of the 36 val scenes.

    ``min_observations`` is the evidence a remembered instance needs before the
    matcher will target it. It is a sweep axis, so it is threaded through rather
    than left at the GoalMatcher default.

    NOTE: this does not read configs/agent_default.yaml. The defaults here are
    the effective configuration at eval time.
    """
    detector = YOLODetector(device=device)
    validate_encoder(encoder_model, encoder_pretrained)
    encoder = ClipEncoder(model_name=encoder_model,
                          pretrained=encoder_pretrained, device=device)
    perception = PerceptionPipeline(detector, encoder, intrinsics)
    memory = InstanceDatabase(merge_dist_m=merge_dist_m,
                             merge_embed_sim=merge_embed_sim)
    matcher = GoalMatcher(
        encoder,
        min_observations=min_observations,
        language_threshold=language_threshold,
        image_threshold=image_threshold,
    )
    # camera_height lets the map derive the floor from the camera pose instead
    # of assuming floor Y == 0, which no HM3D scene satisfies.
    if scene_bounds is not None:
        size_m, origin = map_extent_for_scene(scene_bounds[0], scene_bounds[1])
        semantic_map = SemanticMap(
            size_m=size_m, origin=origin, camera_height=camera_height
        )
    else:
        semantic_map = SemanticMap(camera_height=camera_height)

    kwargs = {}
    if commit_distance_frac is not None:
        kwargs["commit_distance_frac"] = commit_distance_frac
    return GoatAgent(
        perception=perception,
        memory=memory,
        matcher=matcher,
        semantic_map=semantic_map,
        intrinsics=intrinsics,
        max_steps=500,
        success_distance=success_distance,
        planner=planner,
        frontier_strategy=frontier_strategy,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description="GOAT-Lite dev evaluation")
    parser.add_argument("--n-episodes", type=int, default=30,
                        help="size of the deterministic dev subset; 0 or a "
                             "negative value means the WHOLE split, which is "
                             "the final run reserved by PLAN.md Section 8")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hm3d-root", type=str, default="data/hm3d")
    # The detector is trained at 512 px and loses 34% of its mAP50 when the
    # input is 256 (6.60) -- more than the whole v2 retraining gained. All
    # intrinsics derive from the frame size, so this is safe to change; it
    # costs 4x the pixels per step.
    parser.add_argument("--resolution", type=int, default=256,
                        help="square RGB/depth sensor resolution (default 256)")
    parser.add_argument("--dataset-root", type=str,
                        default="data/datasets/goat_bench/hm3d/v1/val_unseen")
    parser.add_argument("--resume", action="store_true",
                        help="skip scenes already complete in the output "
                             "directory's results.csv and keep their rows")
    parser.add_argument("--min-observations", type=int, default=2,
                        help="times an instance must be seen before the "
                             "matcher will target it (1 disables the gate)")
    # Sweep axes (PLAN.md Section 7.10). Defaults mirror the code, so omitting
    # them all reproduces the configuration in configs/agent_default.yaml.
    parser.add_argument("--language-threshold", type=float, default=0.24)
    parser.add_argument("--image-threshold", type=float, default=-1.0)
    parser.add_argument("--merge-dist", type=float, default=0.75)
    parser.add_argument("--merge-embed-sim", type=float, default=0.85)
    parser.add_argument("--success-distance", type=float, default=1.0)
    parser.add_argument("--commit-distance-frac", type=float, default=None,
                        help="fraction of the success distance at which the "
                             "agent commits to a stop (default: the code's 0.8)")
    parser.add_argument("--planner", type=str, default="astar",
                        choices=["astar", "fmm"],
                        help="PLAN 7.10 planner axis")
    parser.add_argument("--frontier-strategy", type=str,
                        default="nearest-largest",
                        choices=list(FRONTIER_STRATEGIES),
                        help="PLAN 7.10 exploration axis")
    parser.add_argument("--encoder-model", type=str, default="ViT-B-32",
                        help="open_clip architecture; validated before any "
                             "scene loads")
    parser.add_argument("--encoder-pretrained", type=str,
                        default="laion2b_s34b_b79k")
    parser.add_argument("--no-memory", action="store_true",
                        help="memory-off arm of the ablation: clear instance "
                             "memory and the map before every subtask instead "
                             "of carrying them across an episode")
    args = parser.parse_args()

    # Output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_nomem" if args.no_memory else ""
        args.output_dir = f"outputs/dev_eval_{timestamp}{suffix}"

    print(f"Loading val_unseen episodes from {args.dataset_root}...")
    all_episodes = load_val_unseen(args.dataset_root)
    print(f"  Total: {len(all_episodes)} episodes")

    if args.n_episodes and args.n_episodes > 0:
        dev_episodes = sample_dev_subset(
            all_episodes, n=args.n_episodes, seed=args.seed)
        label = "Dev subset"
    else:
        # The full split. PLAN.md Section 8 reserves this for exactly one run
        # on the locked configuration, so it is never the default.
        dev_episodes = all_episodes
        label = "FULL SPLIT"
    total_subtasks = sum(len(ep.subtasks) for ep in dev_episodes)
    print(f"  {label}: {len(dev_episodes)} episodes, {total_subtasks} subtasks")

    # Group episodes by scene for efficiency (avoid reloading scenes)
    by_scene: dict[str, list] = {}
    for ep in dev_episodes:
        by_scene.setdefault(ep.scene_id, []).append(ep)
    print(f"  Scenes to load: {len(by_scene)}")

    all_results = []
    scene_count = 0
    scenes_done = 0

    # Recorded into summary.json so an output directory says what produced it.
    # vars(args) rather than a hand-listed dict: the hand-listed version is the
    # one that goes stale when a flag is added, which is the same failure the
    # results.csv fieldnames had.
    run_config = {k: v for k, v in sorted(vars(args).items())
                  if k not in ("output_dir", "resume")}

    if args.resume:
        # Resume at SCENE granularity, never mid-episode: memory and the map
        # persist across the subtasks of an episode, so replaying half an
        # episode would start it from state it never had.
        #
        # The prefix is matched POSITIONALLY. (episode_id, subtask_index) is not
        # unique across a run -- GOAT-Bench numbers episodes per scene, and over
        # this subset 218 subtasks share only 86 such pairs -- so matching on it
        # silently skips scenes that merely share ids with earlier ones.
        prior = load_results(args.output_dir)
        scene_keys = [
            [(ep.episode_id, i) for ep in eps for i in range(len(ep.subtasks))]
            for eps in by_scene.values()
        ]
        all_results, scenes_done = resumable_prefix(prior, scene_keys)
        if prior:
            print(f"  Resuming: {len(all_results)} subtasks over "
                  f"{scenes_done} complete scenes "
                  f"({len(prior) - len(all_results)} rows discarded as "
                  f"unverifiable)")

    for scene_id, scene_episodes in by_scene.items():
        scene_count += 1
        if scene_count <= scenes_done:
            print(f"\n[{scene_count}/{len(by_scene)}] Skipping (already done): "
                  f"{scene_id.split('/')[-2]}")
            continue

        scene_path = _resolve_scene_path(scene_id, args.hm3d_root)
        print(f"\n[{scene_count}/{len(by_scene)}] Loading scene: {scene_path}")

        try:
            env = HabitatEnv(
                scene_path=scene_path,
                seed=args.seed,
                resolution=(args.resolution, args.resolution),
            )
        except Exception as e:
            print(f"  ERROR loading scene: {e}")
            continue

        intrinsics = env.intrinsics
        agent = build_agent(
            intrinsics, device=args.device, scene_bounds=env.get_scene_bounds(),
            min_observations=args.min_observations,
            language_threshold=args.language_threshold,
            image_threshold=args.image_threshold,
            merge_dist_m=args.merge_dist,
            merge_embed_sim=args.merge_embed_sim,
            success_distance=args.success_distance,
            commit_distance_frac=args.commit_distance_frac,
            planner=args.planner,
            frontier_strategy=args.frontier_strategy,
            encoder_model=args.encoder_model,
            encoder_pretrained=args.encoder_pretrained,
        )

        for ep in scene_episodes:
            print(f"  Episode {ep.episode_id}: {len(ep.subtasks)} subtasks...", end=" ")
            t0 = time.perf_counter()

            try:
                results = run_episode(
                    env, agent, ep,
                    keep_memory_across_subtasks=not args.no_memory,
                )
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
            save_results(all_results, args.output_dir, config=run_config)
            print(f"  [checkpoint] {len(all_results)} subtasks saved to "
                  f"{args.output_dir}")

    # Save results
    if all_results:
        save_results(all_results, args.output_dir, config=run_config)
    else:
        print("\nNo results collected!")

    print("\nDone.")


if __name__ == "__main__":
    main()
