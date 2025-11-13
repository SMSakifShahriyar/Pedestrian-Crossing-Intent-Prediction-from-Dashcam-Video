# src/intent_infer.py
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np


class IntentModel:

    def __init__(self, obs_len: int = 16):
        self.obs_len = obs_len


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


    def _features(
        self,
        xywh_seq: np.ndarray,
        global_flow: Optional[Tuple[float, float]],
        lane_center_x: Optional[float],
        lane_band_half_width: Optional[float],
    ) -> Tuple[float, float, float, float, float]:

        vx, vy = self._raw_velocity(xywh_seq)

        if global_flow is not None:
            ux, uy = global_flow
            vx_corr = vx - float(ux)
            vy_corr = vy - float(uy)
        else:
            vx_corr, vy_corr = vx, vy

        speed = (vx_corr**2 + vy_corr**2) ** 0.5
        speed_norm = np.clip(speed / 10.0, 0.0, 1.0)

        denom = abs(vx_corr) + abs(vy_corr) + 1e-6
        lateral_frac = abs(vx_corr) / denom  
        along_frac = abs(vy_corr) / denom     

        center_approach = 0.0
        if lane_center_x is not None and len(xywh_seq) >= 2:
            cx, _ = self._center_traj(xywh_seq)
            d_prev = cx[-2] - lane_center_x
            d_cur = cx[-1] - lane_center_x
            toward = abs(d_prev) - abs(d_cur)  
            center_approach = float(np.clip(toward / 20.0, -1.0, 1.0))

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


        heur_road_score = float(np.clip(heur_road_score, 0.0, 1.0))
        road_frac = float(np.clip(road_frac, 0.0, 1.0))
        sidewalk_frac = float(np.clip(sidewalk_frac, 0.0, 1.0))


        effective_on_road = bool(is_on_road or heur_road_score > 0.7)

     
        combined_road = max(road_frac, heur_road_score)

        road_bonus = 0.0
        if effective_on_road:
            road_bonus += 1.4

        road_bonus += 0.8 * combined_road

        
        sidewalk_supp = sidewalk_frac * (1.0 - 0.7 * heur_road_score)
        road_bonus -= 1.0 * sidewalk_supp

        if inside_lane and effective_on_road:
            road_bonus += 0.6

        geom_intent = self._sigmoid(2.0 * (geom_raw + road_bonus))


        if pose_sideways is None:
            pose_sideways = 0.6

        x = float(np.clip((pose_sideways - 0.2) / 0.6, 0.0, 1.0))

        pose_gate = 0.4 + 0.6 * (x**2)

        intent = geom_intent * pose_gate

 
        if effective_on_road and combined_road > 0.35:
            intent = max(intent, 0.7)

        if (
            effective_on_road
            and inside_lane
            and center_approach > -0.1
            and lateral_frac > 0.35
        ):

            intent = max(intent, 0.9)


        intent = float(np.clip(intent, 0.0, 1.0))
        return intent


def load_pretrained_intention_model(
    obs_len: int = 16,
    model_dir: str | None = None,
) -> IntentModel:
    return IntentModel(obs_len=obs_len)
