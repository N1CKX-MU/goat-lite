"""End-to-end per-frame perception pipeline.

Takes a raw Observation, runs detection + CLIP encoding + back-projection,
and returns a list of PerceivedInstance ready for the instance memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.mapping.transforms import bbox_center_depth_to_world_xyz
from src.perception.detector import YOLODetector
from src.perception.encoder import ClipEncoder
from src.sim.env import Observation


@dataclass
class PerceivedInstance:
    cls_id: int
    cls_name: str
    conf: float
    bbox: tuple[int, int, int, int]
    crop_thumbnail: np.ndarray   # 64x64x3 uint8
    clip_embed: np.ndarray       # [D], L2-normalized
    world_xyz: np.ndarray        # [3], meters, world frame
    seen_step: int


class PerceptionPipeline:
    def __init__(
        self,
        detector: YOLODetector,
        encoder: ClipEncoder,
        intrinsics: np.ndarray,
        min_crop_size: int = 32,
        thumbnail_size: int = 64,
    ):
        self._detector = detector
        self._encoder = encoder
        self._intrinsics = intrinsics
        self._min_crop_size = min_crop_size
        self._thumbnail_size = thumbnail_size

    def process(self, obs: Observation, step: int) -> list[PerceivedInstance]:
        detections = self._detector.detect(obs.rgb)
        if not detections:
            return []

        h, w = obs.rgb.shape[:2]
        crops = []
        valid_dets = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Clip to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            cw = x2 - x1
            ch = y2 - y1
            if cw < self._min_crop_size or ch < self._min_crop_size:
                continue

            # Back-project to world coordinates
            world_xyz = bbox_center_depth_to_world_xyz(
                (x1, y1, x2, y2), obs.depth, self._intrinsics, obs.pose
            )
            if world_xyz is None:
                continue

            crop = obs.rgb[y1:y2, x1:x2]
            crops.append(crop)
            valid_dets.append((det, (x1, y1, x2, y2), world_xyz))

        if not valid_dets:
            return []

        # Batch CLIP encoding
        embeds = self._encoder.encode_image(crops)

        results = []
        for i, (det, bbox, world_xyz) in enumerate(valid_dets):
            thumb = cv2.resize(
                crops[i], (self._thumbnail_size, self._thumbnail_size),
                interpolation=cv2.INTER_AREA,
            )
            results.append(PerceivedInstance(
                cls_id=det.cls_id,
                cls_name=det.cls_name,
                conf=det.conf,
                bbox=bbox,
                crop_thumbnail=thumb,
                clip_embed=embeds[i],
                world_xyz=world_xyz,
                seen_step=step,
            ))

        return results
