# src/overlay.py
import cv2


def draw_tracks(frame, tracks):
    for t in tracks:
        x, y, w, h = t["xywh"]
        x2, y2 = x + w, y + h
        tid = t["id"]
        conf = t["conf"]
        cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {tid} {conf:.2f}",
            (int(x), int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_intent(frame, track, p, alert=False, state=None):
    """
    Draw intent score and state for a track.

    Colors:
      - CROSSING    -> red
      - APPROACHING -> orange
      - SAFE/other  -> yellow-ish
    """
    x, y, w, h = track["xywh"]
    x2, y2 = x + w, y + h

    # Choose color based on state
    if state == "CROSSING":
        color = (0, 0, 255)        # red
    elif state == "APPROACHING":
        color = (0, 165, 255)      # orange
    else:
        color = (0, 255, 255)      # yellow

    cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), color, 2)

    if p is not None:
        label = f"Intent {p:.2f}"
        if state is not None:
            label += f" [{state}]"
        cv2.putText(
            frame,
            label,
            (int(x), int(y) - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def draw_hud(frame, fps=None, thresh=None):
    y = 28
    if fps is not None:
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    if thresh is not None:
        cv2.putText(
            frame,
            f"THRESH: {thresh:.2f}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame
