# src/run.py
import argparse
import os
import cv2
import time
from src.det_track import iter_tracks

def draw_tracks(frame, tracks):
    for t in tracks:
        x, y, w, h = t["xywh"]
        x2, y2 = x + w, y + h
        tid = t["id"]; conf = t["conf"]

        cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid} {conf:.2f}", (int(x), int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    return frame

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to MP4")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights")
    ap.add_argument("--device", default=None, help="GPU index (e.g., 0) or 'cpu'")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.5)
    ap.add_argument("--noshow", action="store_true", help="Do not open a window")
    ap.add_argument("--out", default="tracked.mp4", help="Output video filename")
    args = ap.parse_args()

    out_path = os.path.join(os.path.dirname(args.video), args.out)
    writer = None
    fps_calc_t0 = time.time()
    frames = 0

    for frame, tracks in iter_tracks(
        args.video, model_path=args.model, device=args.device, conf=args.conf, iou=args.iou
    ):
        frame = draw_tracks(frame, tracks)

   
        if writer is None:
            h, w = frame.shape[:2]

            writer = cv2.VideoWriter(
                out_path, cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h)
            )

        writer.write(frame)

        if not args.noshow:
            cv2.imshow("det+track (persons)", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):  
                break

        frames += 1

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = max(1e-6, time.time() - fps_calc_t0)
    print(f"Processed {frames} frames in {elapsed:.2f}s  (~{frames/elapsed:.1f} FPS)")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
