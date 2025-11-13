# src/road_seg_deeplab.py
from __future__ import annotations
from typing import Dict, List

import numpy as np
import cv2
import torch
import torchvision


class CityscapesRoadSeg:

    ROAD_CLASS = 0
    SIDEWALK_CLASS = 1

    def __init__(
        self,
        weights_path: str,
        device: str | int | None = None,
        net_height: int = 512,
        net_width: int = 1024,
    ):

        if device is None:
            dev_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, int):
            dev_str = f"cuda:{device}"
        else:
            ds = str(device).lower()
            if ds == "cpu":
                dev_str = "cpu"
            elif ds.isdigit():
                dev_str = f"cuda:{ds}"
            elif ds.startswith("cuda"):
                dev_str = ds
            else:
                dev_str = ds

        if dev_str.startswith("cuda") and not torch.cuda.is_available():
            dev_str = "cpu"

        self.device = torch.device(dev_str)

        self.net_h = int(net_height)
        self.net_w = int(net_width)

        self.model = torchvision.models.segmentation.deeplabv3_resnet50(
            num_classes=19,
            weights=None,
        )

        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict and "state_dict" not in state_dict:
            state_dict = state_dict["model"]
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()


        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        self._last_labels: np.ndarray | None = None  # cache of last label map (H,W)


    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """
        Convert BGR frame to normalized tensor [1,3,H,W] in RGB.
        """
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.net_w, self.net_h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img).unsqueeze(0)  
        return tensor.to(self.device)



    def infer_labels(self, frame_bgr: np.ndarray) -> np.ndarray:

        H, W = frame_bgr.shape[:2]

        with torch.no_grad():
            inp = self._preprocess(frame_bgr)
            out = self.model(inp)["out"]  
            logits = out[0]             
            labels_small = torch.argmax(logits, dim=0).cpu().numpy().astype(np.int32)

   
        labels = cv2.resize(
            labels_small,
            (W, H),
            interpolation=cv2.INTER_NEAREST,
        )
        self._last_labels = labels
        return labels

    def get_last_labels(self) -> np.ndarray | None:
        """
        Return last computed label map or None if not available.
        """
        return self._last_labels

 

    def features_for_tracks(
        self,
        labels: np.ndarray,
        tracks: List[Dict],
    ) -> Dict[int, Dict[str, float]]:

        H, W = labels.shape[:2]
        road_id = self.ROAD_CLASS
        sidewalk_id = self.SIDEWALK_CLASS

        tid_to_feat: Dict[int, Dict[str, float]] = {}

        for t in tracks:
            tid = t["id"]
            x, y, w, h = t["xywh"]

            
            y1 = int(y + 0.6 * h)
            y2 = int(y + h)
            x1 = int(x)
            x2 = int(x + w)

       
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H, y2))

            if x2 <= x1 or y2 <= y1:
                continue

            patch = labels[y1:y2, x1:x2]
            patch_area = float(patch.size)
            if patch_area <= 1.0:
                continue

            road_count = float((patch == road_id).sum())
            sidewalk_count = float((patch == sidewalk_id).sum())

            road_frac = road_count / patch_area
            sidewalk_frac = sidewalk_count / patch_area

            is_on_road = 1.0 if road_frac > 0.45 else 0.0
            is_on_sidewalk = 1.0 if sidewalk_frac > 0.45 else 0.0

            tid_to_feat[tid] = {
                "road_frac": float(np.clip(road_frac, 0.0, 1.0)),
                "sidewalk_frac": float(np.clip(sidewalk_frac, 0.0, 1.0)),
                "is_on_road": is_on_road,
                "is_on_sidewalk": is_on_sidewalk,
            }

        return tid_to_feat


    def debug_overlay(
        self,
        frame_bgr: np.ndarray,
        labels: np.ndarray | None = None,
        alpha: float = 0.35,
    ) -> np.ndarray:

        if labels is None:
            labels = self._last_labels
        if labels is None:
            return frame_bgr

        H, W = frame_bgr.shape[:2]
        if labels.shape[0] != H or labels.shape[1] != W:
           
            labels = cv2.resize(labels, (W, H), interpolation=cv2.INTER_NEAREST)

        overlay = np.zeros_like(frame_bgr, dtype=np.uint8)

        road_mask = labels == self.ROAD_CLASS
        sidewalk_mask = labels == self.SIDEWALK_CLASS

        road_color = (255, 0, 0)      # blue
        sidewalk_color = (0, 255, 0)  # green

        overlay[road_mask] = road_color
        overlay[sidewalk_mask] = sidewalk_color

        blended = cv2.addWeighted(frame_bgr, 1.0, overlay, alpha, 0.0)
        return blended
