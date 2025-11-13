# src/intent_infer.py
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


class IntentModel:
    """
    Training-free, geometry + pose + road-based crossing intent model (Stage 7+8).

    Uses:
      - per-track box motion (xywh_seq)
      - optional global ego-motion flow (ux, uy)
      - lane center position (x-coordinate)
      - lane band half-width (x-range for ego lane)
      - road features: is_on_road, sidewalk_frac, road_frac
      - heuristic road score: heur_road_score in [0,1]
      - inside_lane flag (is box center inside ego lane band)
      - optional pose_sideways in [0,1]
        *pose_sideways ~1.0 means more side-on/profile (crossing-like).*
    """

    def __init__(self, obs_len: int = 16):
        self.obs_len = obs_len

    # ----------------- basic kinematics helpers -----------------

    def _center_traj(self, xywh_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xywh = np.asarray(xywh_seq, dtype=np.float32)
        cx = xywh[:, 0] + 0.5 * xywh[:, 2]
        cy = xywh[:, 1] + 0.5 * xywh[:, 3]
        return cx, cy

    def _raw_velocity(
        self, xywh_seq: np.ndarray, k_tail: int = 5
    ) -> Tuple[float, float]:
        if xywh_seq is None or len(xywh_seq) < 2:
            return 0.0, 0.0
        cx, cy = self._center_traj(xywh_seq)
        dx = np.diff(cx)
        dy = np.diff(cy)
        if dx.size == 0:
            return 0.0, 0.0
        k = min(k_tail, dx.size)
        vx = float(np.mean(dx[-k:]))
        vy = float(np.mean(dy[-k:]))
        return vx, vy

    # ----------------- geometric features -----------------

    def _features(
        self,
        xywh_seq: np.ndarray,
        global_flow: Optional[Tuple[float, float]],
        lane_center_x: Optional[float],
        lane_band_half_width: Optional[float],
    ) -> Tuple[float, float, float, float, float]:
        """
        Returns:
            speed_norm       in [0,1]
            lateral_frac     in [0,1]
            along_frac       in [0,1]
            center_approach  in [-1,1]  (positive = moving toward lane center)
            lane_intersect   in [0,1]   (1 ~ future path hits ego lane band)
        """
        vx, vy = self._raw_velocity(xywh_seq)

        # ego-motion correction
        if global_flow is not None:
            ux, uy = global_flow
            vx_corr = vx - float(ux)
            vy_corr = vy - float(uy)
        else:
            vx_corr, vy_corr = vx, vy

        speed = (vx_corr**2 + vy_corr**2) ** 0.5
        speed_norm = np.clip(speed / 10.0, 0.0, 1.0)

        denom = abs(vx_corr) + abs(vy_corr) + 1e-6
        lateral_frac = abs(vx_corr) / denom   # sideways motion fraction
        along_frac = abs(vy_corr) / denom     # along-camera motion fraction

        center_approach = 0.0
        if lane_center_x is not None and len(xywh_seq) >= 2:
            cx, _ = self._center_traj(xywh_seq)
            d_prev = cx[-2] - lane_center_x
            d_cur = cx[-1] - lane_center_x
            toward = abs(d_prev) - abs(d_cur)     # >0 if moving towards center
            center_approach = float(np.clip(toward / 20.0, -1.0, 1.0))

        # --- future path / lane intersection ---
        lane_intersect = 0.0
        if (
            lane_center_x is not None
            and lane_band_half_width is not None
            and lane_band_half_width > 1.0
            and len(xywh_seq) >= 2
        ):
            cx, _ = self._center_traj(xywh_seq)
            cx_last = float(cx[-1])

            N_future = min(self.obs_len, 12)
            vx_corr2, _ = self._raw_velocity(xywh_seq)
            cx_future = cx_last + vx_corr2 * N_future

            band_left = lane_center_x - lane_band_half_width
            band_right = lane_center_x + lane_band_half_width

            inside_now = band_left <= cx_last <= band_right
            inside_future = band_left <= cx_future <= band_right

            if inside_future:
                lane_intersect = 1.0 if not inside_now else 0.8
            else:
                dist = abs(cx_future - lane_center_x)
                lane_intersect = float(
                    np.clip(
                        1.0 - dist / (2.0 * lane_band_half_width + 1e-6),
                        0.0,
                        1.0,
                    )
                )

        return (
            float(speed_norm),
            float(lateral_frac),
            float(along_frac),
            float(center_approach),
            float(lane_intersect),
        )

    # ----------------- core intent -----------------

    def _sigmoid(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def predict(
        self,
        crops: List[np.ndarray],
        xywh_seq: np.ndarray,
        global_flow: Optional[Tuple[float, float]] = None,
        lane_center_x: Optional[float] = None,
        lane_band_half_width: Optional[float] = None,
        pose_sideways: Optional[float] = None,
        is_on_road: bool = False,
        sidewalk_frac: float = 0.0,
        road_frac: float = 0.0,
        inside_lane: bool = False,
        heur_road_score: float = 0.0,
    ) -> float:
        """
        Args:
            crops: unused for now.
            xywh_seq: [T,4] history.
            global_flow: (ux, uy) camera motion.
            lane_center_x: lane center x in pixels.
            lane_band_half_width: half-width of ego lane band in pixels.
            pose_sideways: in [0,1].
            is_on_road: from DeepLab segmentation (feet on road).
            sidewalk_frac: fraction of lower bbox on sidewalk (DeepLab).
            road_frac: fraction of lower bbox on road (DeepLab).
            inside_lane: True if box center is inside ego lane band.
            heur_road_score: [0,1], geometric “likely on road” score.

        Returns:
            intent in [0,1].
        """
        (
            speed_norm,
            lateral_frac,
            along_frac,
            center_approach,
            lane_intersect,
        ) = self._features(
            xywh_seq,
            global_flow,
            lane_center_x,
            lane_band_half_width,
        )

        # --- 1) Geometry-based intent ---
        geom_raw = (
            3.0 * center_approach
            + 2.0 * lane_intersect
            + 0.5 * lateral_frac * max(center_approach, 0.0)
            - 0.5 * along_frac
            + 0.5 * speed_norm
        )

        # --- 2) Hybrid road prior (DeepLab + heuristic) ---
        heur_road_score = float(np.clip(heur_road_score, 0.0, 1.0))
        road_frac = float(np.clip(road_frac, 0.0, 1.0))
        sidewalk_frac = float(np.clip(sidewalk_frac, 0.0, 1.0))

        # DeepLab says road OR geometric heuristic high → treat as on-road
        effective_on_road = bool(is_on_road or heur_road_score > 0.7)

        # Use the stronger of DeepLab road_frac and heuristic as "combined road"
        combined_road = max(road_frac, heur_road_score)

        road_bonus = 0.0
        if effective_on_road:
            road_bonus += 1.4

        road_bonus += 0.8 * combined_road

        # Sidewalk still suppresses, but less if heuristic strongly disagrees
        sidewalk_supp = sidewalk_frac * (1.0 - 0.7 * heur_road_score)
        road_bonus -= 1.0 * sidewalk_supp

        if inside_lane and effective_on_road:
            road_bonus += 0.6

        geom_intent = self._sigmoid(2.0 * (geom_raw + road_bonus))

        # --- 3) Pose gate ---
        if pose_sideways is None:
            pose_sideways = 0.6

        x = float(np.clip((pose_sideways - 0.2) / 0.6, 0.0, 1.0))
        # Gate in [0.4, 1.0] so pose cannot completely zero clear road risk
        pose_gate = 0.4 + 0.6 * (x**2)

        intent = geom_intent * pose_gate

        # --- 4) Hard overrides for clear on-road crossers ---
        if effective_on_road and combined_road > 0.35:
            intent = max(intent, 0.7)

        if (
            effective_on_road
            and inside_lane
            and center_approach > -0.1
            and lateral_frac > 0.35
        ):
            # Clearly moving across / into lane on road
            intent = max(intent, 0.9)

        # Final clamp
        intent = float(np.clip(intent, 0.0, 1.0))
        return intent


def load_pretrained_intention_model(
    obs_len: int = 16,
    model_dir: str | None = None,
) -> IntentModel:
    return IntentModel(obs_len=obs_len)
