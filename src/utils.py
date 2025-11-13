# src/utils.py
from collections import deque
from typing import Deque, Dict, Tuple, List
import numpy as np

class TrackBuffer:
    """Keeps last N crops and box features per track ID."""
    def __init__(self, maxlen: int = 16):
        self.crops: Deque[np.ndarray] = deque(maxlen=maxlen)
        self.feats: Deque[Tuple[float, float, float, float]] = deque(maxlen=maxlen)

    def push(self, crop: np.ndarray, xywh: Tuple[float, float, float, float]):
        self.crops.append(crop)
        self.feats.append(xywh)

    def ready(self) -> bool:
        return len(self.crops) == self.crops.maxlen

    def get_crops(self) -> List[np.ndarray]:
        return list(self.crops)

    def get_feats(self) -> np.ndarray:
        return np.array(self.feats, dtype=np.float32)


def crop_person(frame, xywh, out_size=224):
    import cv2
    x, y, w, h = xywh
    H, W = frame.shape[:2]
    x1 = max(0, int(x)); y1 = max(0, int(y))
    x2 = min(W, int(x + w)); y2 = min(H, int(y + h))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((out_size, out_size, 3), dtype=frame.dtype)
    patch = frame[y1:y2, x1:x2]
    patch = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return patch


def is_near_road(xywh, frame_h, margin_ratio=0.2):
    """Very simple curb proxy: bottom of box in bottom 20% of frame."""
    x, y, w, h = xywh
    bottom = y + h
    return bottom >= (1.0 - margin_ratio) * frame_h
