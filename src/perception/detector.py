from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: np.ndarray | None = None


class YOLODetector:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        device: str = "cuda",
        fp16: bool = True,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 256,
    ):
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._model.to(device)
        self._model.fuse()
        self._device = device
        self._fp16 = fp16
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            rgb,
            imgsz=self._imgsz,
            half=self._fp16,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
        )
        detections = []
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            cls_id = int(boxes.cls[i].item())
            cls_name = self._model.names[cls_id]
            conf = float(boxes.conf[i].item())
            detections.append(Detection(
                cls_id=cls_id,
                cls_name=cls_name,
                conf=conf,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
            ))
        return detections

    @property
    def class_names(self) -> dict[int, str]:
        return self._model.names
