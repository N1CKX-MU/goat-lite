#!/usr/bin/env python3
"""Finetune YOLOv8n on the GOAT vocabulary and export weights.

Run after scripts/make_yolo_dataset.py has produced a dataset + data.yaml.
Trains from the COCO-pretrained yolov8n.pt and copies the best checkpoint to
checkpoints/yolo_goat.pt, which is the single file to send back to the main box.

Usage:
    python scripts/finetune_yolo.py --data data/yolo_goat/data.yaml --epochs 60
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.perception.goat_classes import NUM_CLASSES


def main() -> None:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description="Finetune YOLOv8n for GOAT categories")
    ap.add_argument("--data", required=True, help="path to data.yaml from make_yolo_dataset.py")
    ap.add_argument("--weights", default="yolov8n.pt", help="starting checkpoint")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/goat_yolo")
    ap.add_argument("--name", default="finetune")
    ap.add_argument("--out", default="checkpoints/yolo_goat.pt")
    args = ap.parse_args()

    print(f"Finetuning {args.weights} on {args.data} "
          f"({NUM_CLASSES} GOAT classes, {args.epochs} epochs, imgsz={args.imgsz})")

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=15,
        exist_ok=True,
    )

    best = os.path.join(args.project, args.name, "weights", "best.pt")
    if not os.path.exists(best):
        raise SystemExit(f"Training finished but {best} not found.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    shutil.copyfile(best, args.out)
    print(f"\nDone. Best weights -> {args.out}")
    print("Send this single file back to the main machine (checkpoints/yolo_goat.pt).")


if __name__ == "__main__":
    main()
