# src/run_intent.py
import argparse, os, time, ctypes, sys
from collections import defaultdict, deque
from typing import Dict, Deque, Tuple, List

import cv2
import numpy as np

from src.det_track import iter_tracks
from src.utils import TrackBuffer, crop_person, is_near_road
from src.intent_infer import load_pretrained_intention_model
from src.overlay import draw_tracks, draw_intent, draw_hud
from src.pose_estimator import PoseEstimator
from src.road_seg_deeplab import CityscapesRoadSeg


def beep():
    try:
        if sys.platform.startswith("win"):
            ctypes.windll.user32.MessageBeep(0xFFFFFFFF)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def estimate_global_flow(buffers: Dict[int, TrackBuffer]) -> Tuple[float, float]:
    dxs: List[float] = []
    dys: List[float] = []
    for buf in buffers.values():
        xywh_seq = buf.get_feats()
        if xywh_seq.shape[0] < 2:
            continue
        xywh = np.asarray(xywh_seq, dtype=np.float32)
        cx = xywh[:, 0] + 0.5 * xywh[:, 2]
        cy = xywh[:, 1] + 0.5 * xywh[:, 3]
        dx = cx[-1] - cx[-2]
        dy = cy[-1] - cy[-2]
        dxs.append(float(dx))
        dys.append(float(dy))
    if not dxs:
        return 0.0, 0.0
    ux = float(np.median(dxs))
    uy = float(np.median(dys))
    return ux, uy


def heuristic_road_score(
    xywh: Tuple[float, float, float, float],
    frame_h: int,
    frame_w: int,
    lane_center_x: float,
) -> float:
    """
    Simple geometric 'on-road' score in [0,1].

    High when:
      - bbox bottom is lower in the frame (closer to car)
      - bbox center is near the horizontal lane center.
    """
    x, y, w, h = xywh
    cx = x + 0.5 * w
    bottom_y = y + h

    y_norm = bottom_y / max(1.0, frame_h)
    x_dev_norm = abs(cx - lane_center_x) / (0.5 * frame_w + 1e-6)  # 0 center, 1 at edges

    # favor lower positions; y_norm ~0.3->0, 1.0->1
    y_term = np.clip((y_norm - 0.3) / 0.7, 0.0, 1.0)
    # favor center horizontally
    x_term = np.clip(1.0 - x_dev_norm, 0.0, 1.0)

    score = float(np.clip(y_term * x_term, 0.0, 1.0))
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to MP4")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO detector weights")
    ap.add_argument("--device", default=None, help="GPU index (e.g., 0) or 'cpu'")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)

    ap.add_argument("--obs", type=int, default=16, help="Observation frames (~1s)")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Base intent threshold (for CROSSING)",
    )
    ap.add_argument(
        "--approach_threshold",
        type=float,
        default=0.40,
        help="Intent threshold to enter APPROACHING state",
    )
    ap.add_argument(
        "--stable_k",
        type=int,
        default=3,
        help="Frames above threshold before alert/state CROSSING",
    )
    ap.add_argument(
        "--pose_interval",
        type=int,
        default=2,
        help="Run pose every N frames (>=1).",
    )
    ap.add_argument(
        "--seg_interval",
        type=int,
        default=3,
        help="Run DeepLab road segmentation every N frames (>=1).",
    )
    ap.add_argument(
        "--track_max_age",
        type=float,
        default=5.0,
        help="Max track age in seconds before cleanup (approximate).",
    )
    ap.add_argument(
        "--lane_band_width_ratio",
        type=float,
        default=0.5,
        help="Fraction of frame width used as ego-lane band (centered).",
    )
    ap.add_argument(
        "--road_weights",
        type=str,
        default="pytorch_model.bin",
        help="Path to DeepLabV3 Cityscapes pytorch_model.bin",
    )
    ap.add_argument(
        "--show_road_debug",
        action="store_true",
        help="Overlay road/sidewalk segmentation on the frame (debug).",
    )
    ap.add_argument("--noshow", action="store_true", help="Do not open a window")
    ap.add_argument(
        "--out",
        default="alerts_stage8_adaptive.mp4",
        help="Output video filename",
    )
    args = ap.parse_args()

    # Input FPS (for writer + track age in frames)
    cap = cv2.VideoCapture(args.video)
    in_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not in_fps or in_fps <= 1e-3:
        in_fps = 25.0
    track_max_age_frames = int(max(1.0, in_fps) * max(0.0, args.track_max_age))

    model = load_pretrained_intention_model(obs_len=args.obs)

    pose_est = PoseEstimator(
        model_path="yolov8x-pose.pt", device=args.device, conf=0.4
    )

    buffers: Dict[int, TrackBuffer] = defaultdict(lambda: TrackBuffer(maxlen=args.obs))
    score_hist: Dict[int, Deque[float]] = defaultdict(
        lambda: deque(maxlen=args.stable_k)
    )

    last_seen_frame: Dict[int, int] = {}
    last_pose_sideways: Dict[int, float] = {}
    track_state: Dict[int, str] = {}

    road_seg: CityscapesRoadSeg | None = None
    last_labels = None

    out_path = os.path.join(os.path.dirname(args.video), args.out)
    writer = None
    t0 = time.time()
    frames = 0
    fps_show = None

    for frame, tracks in iter_tracks(
        args.video,
        model_path=args.model,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    ):
        H, W = frame.shape[:2]
        frames += 1

        lane_width_ratio = float(np.clip(args.lane_band_width_ratio, 0.1, 1.0))
        lane_center_x = W / 2.0
        lane_band_half_width = 0.5 * lane_width_ratio * W

        if road_seg is None:
            road_seg = CityscapesRoadSeg(
                weights_path=args.road_weights,
                device=args.device,
                net_height=512,
                net_width=1024,
            )

        # track bookkeeping
        for t in tracks:
            tid = t["id"]
            last_seen_frame[tid] = frames

        if frames % 30 == 0:
            stale_ids = [
                tid
                for tid, last_f in last_seen_frame.items()
                if frames - last_f > track_max_age_frames
            ]
            for tid in stale_ids:
                last_seen_frame.pop(tid, None)
                buffers.pop(tid, None)
                score_hist.pop(tid, None)
                last_pose_sideways.pop(tid, None)
                track_state.pop(tid, None)

        # update buffers
        for t in tracks:
            tid = t["id"]
            crop = crop_person(frame, t["xywh"], out_size=224)
            buffers[tid].push(crop, t["xywh"])

        global_flow = estimate_global_flow(buffers)

        if args.pose_interval < 1:
            args.pose_interval = 1
        if args.seg_interval < 1:
            args.seg_interval = 1

        # pose every N frames
        if tracks and (frames % args.pose_interval == 0):
            tid_to_side = pose_est.match_to_tracks(frame, tracks)
            for tid, side in tid_to_side.items():
                last_pose_sideways[tid] = side

        # DeepLab segmentation every seg_interval frames
        if frames % args.seg_interval == 0:
            last_labels = road_seg.infer_labels(frame)
        elif last_labels is None and tracks:
            last_labels = road_seg.infer_labels(frame)

        road_feats = {}
        if last_labels is not None and tracks:
            road_feats = road_seg.features_for_tracks(last_labels, tracks)

        # ---- NEW: overlay road/sidewalk if requested ----
        if args.show_road_debug and last_labels is not None and road_seg is not None:
            frame = road_seg.debug_overlay(frame, last_labels, alpha=0.35)

        # then draw tracks & intent on top
        frame = draw_tracks(frame, tracks)

        for t in tracks:
            tid = t["id"]
            buf = buffers[tid]
            p = None
            alert = False
            prev_state = track_state.get(tid, "SAFE")
            new_state = prev_state

            if buf.ready():
                xywh_seq = buf.get_feats()
                pose_sideways = last_pose_sideways.get(tid, None)

                rf = road_feats.get(
                    tid,
                    {
                        "road_frac": 0.0,
                        "sidewalk_frac": 0.0,
                        "is_on_road": 0.0,
                        "is_on_sidewalk": 0.0,
                    },
                )

                cx = t["xywh"][0] + 0.5 * t["xywh"][2]
                inside_lane = abs(cx - lane_center_x) <= lane_band_half_width

                heur_score = heuristic_road_score(
                    t["xywh"], H, W, lane_center_x
                )

                p = model.predict(
                    buf.get_crops(),
                    xywh_seq,
                    global_flow=global_flow,
                    lane_center_x=lane_center_x,
                    lane_band_half_width=lane_band_half_width,
                    pose_sideways=pose_sideways,
                    is_on_road=bool(rf["is_on_road"]),
                    sidewalk_frac=float(rf["sidewalk_frac"]),
                    road_frac=float(rf["road_frac"]),
                    inside_lane=inside_lane,
                    heur_road_score=heur_score,
                )
                score_hist[tid].append(p)

                # ------------- adaptive CROSSING threshold -------------
                base_th = args.threshold
                th = base_th

                # Case: DeepLab road low, heuristic strong → relax threshold
                if heur_score > 0.7 and float(rf["road_frac"]) < 0.2:
                    th = max(0.5, 0.8 * base_th)  # e.g. 0.56 if base_th=0.70

                stable_high = (
                    len(score_hist[tid]) == args.stable_k
                    and all(s >= th for s in score_hist[tid])
                )

                near = is_near_road(t["xywh"], H, margin_ratio=0.2)
                effective_on_road = bool(rf["is_on_road"] or heur_score > 0.7)

                # CROSSING if:
                #  - intent stable above (possibly relaxed) threshold
                #  - (near frame bottom OR clearly on-road)
                #  - inside ego lane band
                if stable_high and (near or effective_on_road) and inside_lane:
                    new_state = "CROSSING"
                else:
                    if p >= args.approach_threshold:
                        if prev_state != "CROSSING":
                            new_state = "APPROACHING"
                    else:
                        if prev_state == "CROSSING":
                            if p >= 0.7 * args.approach_threshold:
                                new_state = "APPROACHING"
                            else:
                                new_state = "SAFE"
                        elif prev_state == "APPROACHING":
                            if p < 0.5 * args.approach_threshold:
                                new_state = "SAFE"

                track_state[tid] = new_state

                if new_state == "CROSSING" and stable_high and (near or effective_on_road) and inside_lane:
                    alert = True
                    beep()

                frame = draw_intent(frame, t, p, alert=alert, state=new_state)

        # HUD + output
        if frames % 10 == 0:
            elapsed = time.time() - t0
            if elapsed > 0:
                fps_show = frames / elapsed
        frame = draw_hud(frame, fps=fps_show, thresh=args.threshold)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                in_fps,
                (w, h),
            )
        writer.write(frame)

        if not args.noshow:
            cv2.imshow("dashcam-intent-stage8-adaptive", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = max(1e-6, time.time() - t0)
    print(f"Processed {frames} frames in {elapsed:.2f}s (~{frames/elapsed:.1f} FPS)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
