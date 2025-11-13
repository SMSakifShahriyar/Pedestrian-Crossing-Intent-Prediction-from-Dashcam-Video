# src/det_track.py
from typing import Generator, List, Dict, Optional
import numpy as np
from ultralytics import YOLO

"""
iter_tracks(video_path, ...) yields (frame_bgr, tracks) for each frame.
- frame_bgr: numpy array (BGR) for cv2.imshow()
- tracks: list of {"id": int, "xywh": (x,y,w,h), "conf": float}
Only class=person is kept.
"""

def iter_tracks(
    video_path: str,
    model_path: str = "yolov8n.pt",
    device: Optional[str | int] = None,  # e.g., 0 for CUDA:0, or "cpu"
    conf: float = 0.25,
    iou: float = 0.5,
) -> Generator[tuple[np.ndarray, List[Dict]], None, None]:
    model = YOLO(model_path)

    stream = model.track(
        source=video_path,
        stream=True,
        tracker="bytetrack.yaml",
        persist=True,
        device=device,
        conf=conf,
        iou=iou,
        classes=[0],   # person
        verbose=False,
    )

    for result in stream:
        frame_bgr = result.orig_img
        tracks: List[Dict] = []

        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.data is None or len(boxes) == 0:
            yield frame_bgr, tracks
            continue

        ids = boxes.id
        confs = boxes.conf
        xyxy = boxes.xyxy

        for i in range(len(xyxy)):
            if ids is None or ids[i] is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
            w, h = x2 - x1, y2 - y1
            track_id = int(ids[i].item() if hasattr(ids[i], "item") else ids[i])
            conf_i   = float(confs[i].item() if hasattr(confs[i], "item") else confs[i])

            tracks.append({"id": track_id, "xywh": (x1, y1, w, h), "conf": conf_i})

        yield frame_bgr, tracks
