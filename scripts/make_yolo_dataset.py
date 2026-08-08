#!/usr/bin/env python3
"""Render an HM3D -> YOLO detection dataset for the GOAT vocabulary.

Runs on a machine that has habitat-sim installed plus the HM3D **train** scenes
and their HM3D-Semantics annotations (``*.semantic.glb`` + ``*.semantic.txt``).

For each scene it samples navigable camera poses, renders an RGB frame and the
matching semantic instance map, converts every instance whose category is a GOAT
goal category into a tight YOLO bounding box, and writes an Ultralytics-style
dataset (images/labels split by scene) plus a ``data.yaml``.

Usage:
    python scripts/make_yolo_dataset.py \
        --hm3d-root /data/hm3d/train \
        --out data/yolo_goat \
        --frames-per-scene 60

Send only the trained weights back; the dataset itself can stay on this machine.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.goat_classes import GOAT_CATEGORIES, name_to_id


def find_scenes(hm3d_root: str) -> list[tuple[str, str]]:
    """Return (scene_glb, semantic_glb) pairs for scenes that have semantics."""
    scenes = []
    for entry in sorted(os.listdir(hm3d_root)):
        scene_dir = os.path.join(hm3d_root, entry)
        if not os.path.isdir(scene_dir):
            continue
        glb = sem = None
        for f in os.listdir(scene_dir):
            if f.endswith(".semantic.glb"):
                sem = os.path.join(scene_dir, f)
            elif f.endswith(".basis.glb"):
                glb = os.path.join(scene_dir, f)
        if glb and sem:
            scenes.append((glb, sem))
    return scenes


def make_sim(scene_glb: str, resolution: int, seed: int):
    import habitat_sim

    backend = habitat_sim.SimulatorConfiguration()
    backend.scene_id = scene_glb
    backend.scene_dataset_config_file = "default"
    backend.enable_physics = False
    backend.random_seed = seed

    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [resolution, resolution]
    rgb.position = [0.0, 1.41, 0.0]

    sem = habitat_sim.CameraSensorSpec()
    sem.uuid = "semantic"
    sem.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem.resolution = [resolution, resolution]
    sem.position = [0.0, 1.41, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, sem]

    cfg = habitat_sim.Configuration(backend, [agent_cfg])
    return habitat_sim.Simulator(cfg)


def instance_to_class(sim) -> dict[int, int]:
    """Map semantic-sensor instance id -> GOAT class id (only for GOAT categories)."""
    mapping: dict[int, int] = {}
    scene = sim.semantic_scene
    if scene is None:
        return mapping
    for obj in scene.objects:
        if obj is None:
            continue
        try:
            cat = obj.category.name()
        except Exception:
            continue
        cls_id = name_to_id(cat)
        if cls_id is None:
            continue
        raw = getattr(obj, "semantic_id", None)
        if raw is None:
            raw = obj.id.split("_")[-1]
        mapping[int(raw)] = cls_id
    return mapping


def boxes_from_semantic(
    semantic: np.ndarray,
    inst_to_cls: dict[int, int],
    min_box_px: int,
    max_area_frac: float,
) -> list[tuple[int, int, int, int, int]]:
    """Return [(cls_id, x1, y1, x2, y2), ...] for GOAT instances in a frame."""
    h, w = semantic.shape[:2]
    boxes = []
    present = np.unique(semantic)
    for inst_id in present:
        cls_id = inst_to_cls.get(int(inst_id))
        if cls_id is None:
            continue
        ys, xs = np.where(semantic == inst_id)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        if bw < min_box_px or bh < min_box_px:
            continue
        # Reject boxes that cover almost the whole frame (mislabeled walls/floor).
        if (bw * bh) / float(w * h) > max_area_frac:
            continue
        boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def to_yolo_line(cls_id: int, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> str:
    cx = ((x1 + x2 + 1) / 2.0) / w
    cy = ((y1 + y2 + 1) / 2.0) / h
    bw = (x2 - x1 + 1) / w
    bh = (y2 - y1 + 1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def write_data_yaml(out_dir: str) -> None:
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(GOAT_CATEGORIES))
    content = (
        f"path: {os.path.abspath(out_dir)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names}\n"
    )
    with open(os.path.join(out_dir, "data.yaml"), "w") as f:
        f.write(content)


def main() -> None:
    import cv2
    import habitat_sim

    ap = argparse.ArgumentParser(description="Render HM3D -> YOLO GOAT dataset")
    ap.add_argument("--hm3d-root", required=True, help="HM3D train dir (scenes with *.semantic.glb)")
    ap.add_argument("--out", default="data/yolo_goat")
    ap.add_argument("--frames-per-scene", type=int, default=60)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--min-box-px", type=int, default=12)
    ap.add_argument("--max-area-frac", type=float, default=0.6)
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of scenes held out for val")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    scenes = find_scenes(args.hm3d_root)
    if not scenes:
        raise SystemExit(
            f"No scenes with semantics under {args.hm3d_root}. "
            "Need HM3D train GLBs + HM3D-Semantics (*.semantic.glb/.txt)."
        )
    random.shuffle(scenes)
    n_val = max(1, int(len(scenes) * args.val_frac))
    val_scenes = set(range(n_val))
    print(f"Found {len(scenes)} scenes with semantics ({n_val} -> val).")

    for split in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)

    total_imgs = total_boxes = 0
    for si, (scene_glb, _sem_glb) in enumerate(scenes):
        split = "val" if si in val_scenes else "train"
        try:
            sim = make_sim(scene_glb, args.resolution, args.seed + si)
        except Exception as e:
            print(f"  [skip] {scene_glb}: {e}")
            continue

        inst_to_cls = instance_to_class(sim)
        if not inst_to_cls:
            print(f"  [skip] {os.path.basename(scene_glb)}: no GOAT instances")
            sim.close()
            continue

        agent = sim.get_agent(0)
        pf = sim.pathfinder
        kept = 0
        for fi in range(args.frames_per_scene):
            pt = pf.get_random_navigable_point()
            if not np.isfinite(pt).all():
                continue
            state = habitat_sim.AgentState()
            state.position = pt
            yaw = random.uniform(-np.pi, np.pi)
            state.rotation = np.quaternion(np.cos(yaw / 2), 0, np.sin(yaw / 2), 0)
            agent.set_state(state)

            obs = sim.get_sensor_observations()
            semantic = np.asarray(obs["semantic"])
            if semantic.ndim == 3:
                semantic = semantic[..., 0]
            boxes = boxes_from_semantic(
                semantic, inst_to_cls, args.min_box_px, args.max_area_frac
            )
            if not boxes:
                continue

            rgb = np.asarray(obs["rgb"])[:, :, :3]
            h, w = rgb.shape[:2]
            stem = f"{si:04d}_{fi:04d}"
            cv2.imwrite(
                os.path.join(args.out, "images", split, stem + ".jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            with open(os.path.join(args.out, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(to_yolo_line(*b, w, h) for b in boxes) + "\n")
            kept += 1
            total_boxes += len(boxes)

        total_imgs += kept
        print(f"[{si + 1}/{len(scenes)}] {os.path.basename(scene_glb)} ({split}): {kept} frames")
        sim.close()

    write_data_yaml(args.out)
    print(f"\nDone. {total_imgs} images, {total_boxes} boxes -> {args.out}")
    print(f"data.yaml written. Next: python scripts/finetune_yolo.py --data {args.out}/data.yaml")


if __name__ == "__main__":
    main()
