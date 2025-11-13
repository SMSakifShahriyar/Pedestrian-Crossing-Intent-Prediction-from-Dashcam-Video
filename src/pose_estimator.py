# src/pose_estimator.py
from __future__ import annotations
from typing import List, Dict, Any

import numpy as np
from ultralytics import YOLO


def xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter + 1e-6
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def pose_crossing_likeness(keypoints: np.ndarray) -> float:

    if keypoints is None or keypoints.shape[0] < 7:
       
        return 0.5


    ls = keypoints[5]
    rs = keypoints[6]

    
    shoulder_w = abs(rs[0] - ls[0])


    xs = keypoints[:, 0]
    body_w = float(xs.max() - xs.min())
    if body_w < 1e-3:
        return 0.5

    ratio = float(np.clip(shoulder_w / (body_w + 1e-6), 0.0, 1.0))

    frontal_score = ratio
    profile_score = 1.0 - frontal_score

    
    crossing_score = float(np.clip(profile_score ** 1.5, 0.0, 1.0))
    return crossing_score


class PoseEstimator:


    def __init__(
        self,
        model_path: str = "yolov8x-pose.pt",
        device: str | None = None,
        conf: float = 0.25,
    ):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf

    def infer(self, frame) -> List[Dict[str, Any]]:
        """
        Run pose model on a BGR frame (numpy array).
        """
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.conf,
            verbose=False,
        )
        if not results:
            return []

        r = results[0]
        if r.boxes is None or r.keypoints is None:
            return []

     
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        kpts = r.keypoints.xy.cpu().numpy() 

        dets: List[Dict[str, Any]] = []
        for i in range(boxes.shape[0]):
          
            side = pose_crossing_likeness(kpts[i])
            dets.append(
                {
                    "box_xyxy": boxes[i],
                    "score": float(scores[i]),
                    "side": side,
                }
            )
        return dets

    def match_to_tracks(self, frame, tracks: List[Dict[str, Any]]) -> Dict[int, float]:

        if not tracks:
            return {}

        pose_dets = self.infer(frame)
        if not pose_dets:
            return {}

        pose_boxes = np.stack([d["box_xyxy"] for d in pose_dets], axis=0)

        tid_to_side: Dict[int, float] = {}
        for t in tracks:
            tid = t["id"]
            box_xywh = t["xywh"]
            tb = xywh_to_xyxy(box_xywh)

    
            ious = np.array([iou_xyxy(tb, pb) for pb in pose_boxes], dtype=np.float32)
            j = int(np.argmax(ious))
            if ious[j] < 0.3:
                continue  
            tid_to_side[tid] = float(pose_dets[j]["side"])
        return tid_to_side
