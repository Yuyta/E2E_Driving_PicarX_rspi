"""
カメラ画像と超音波から実際の動きを推定する（ROADMAP の M3「指令と実際の動きの食い違い」）

PiCar-X には車輪の回転数を測るセンサーがないので、画像の動き（オプティカルフロー）と
超音波の距離の変化から、動いたか、どちらへどれだけ曲がったかを推定する。

画像の動きのモデル: 前向きのカメラでは、
- 曲がると景色が横へずれる（右へ曲がると景色は左へ動く）
- 前進すると景色が中心から広がる（拡大）
特徴点の動きに「平行移動 + 拡大 + 回転」のモデルを RANSAC で当てはめ、人などの
一部だけの動きに引きずられないようにする。単眼なので前進の距離（cm）は画像だけでは
分からない。距離は超音波が壁や家具を捉えている時だけ求める。

値は時間あたり（毎秒）に直すので、周期が変わっても同じしきい値を使える。
"""

from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

WORK_WIDTH = 160
WORK_HEIGHT = 120

# カメラの水平画角（度）。PiCar-X の標準カメラ（OV5647）の値で、床に印を付けた試験で校正する
DEFAULT_HFOV_DEG = 53.5


@dataclass
class FlowResult:
    """2枚の画像の間の動き（作業解像度 160x120 のピクセル単位）"""

    valid: bool
    reason: Optional[str]
    points: int
    inlier_ratio: float
    shift_x_px: float  # 画像の中心での景色の横のずれ。右へ曲がると負
    shift_y_px: float
    scale: float  # 1 より大きいと拡大（前進）
    rotation_deg: float  # 画像の面内の回転（傾き）


@dataclass
class MotionSample:
    """ある時点の動きの推定値。時間あたりに直した値を持つ"""

    timestamp: float
    interval_s: float
    image_valid: bool
    image_reason: Optional[str]
    points: int
    inlier_ratio: float
    yaw_rate_dps: float  # 向きの変化（度/秒）。正は右回り
    expansion_rate: float  # 拡大の速さ（1/秒）。正は前進、負は後退
    edge_flow_px_s: float  # 画像の動きの大きさ（ピクセル/秒）。動いたかの判定に使う
    image_moving: bool
    distance_cm: Optional[float]
    approach_cm_s: Optional[float]  # 超音波の距離が縮む速さ（cm/秒）。正は近づいている

    def to_record(self) -> dict:
        record = asdict(self)
        for key, value in record.items():
            if isinstance(value, float):
                record[key] = round(value, 4)
        return record


def to_work_gray(frame: np.ndarray) -> np.ndarray:
    """作業解像度のグレースケールにする（BGR でもグレースケールでもよい）"""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.shape[:2] != (WORK_HEIGHT, WORK_WIDTH):
        gray = cv2.resize(gray, (WORK_WIDTH, WORK_HEIGHT), interpolation=cv2.INTER_AREA)
    return gray


def estimate_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    min_points: int = 12,
    max_corners: int = 150,
) -> FlowResult:
    """2枚の画像の間の景色の動きを、平行移動 + 拡大 + 回転のモデルで推定する"""
    prev_gray = to_work_gray(prev_gray)
    curr_gray = to_work_gray(curr_gray)

    def invalid(reason: str, points: int = 0) -> FlowResult:
        return FlowResult(False, reason, points, 0.0, 0.0, 0.0, 1.0, 0.0)

    corners = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=max_corners, qualityLevel=0.01, minDistance=5, blockSize=5
    )
    if corners is None or len(corners) < min_points:
        return invalid("few_features", 0 if corners is None else len(corners))

    lk = dict(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None, **lk)
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, nxt, None, **lk)
    # 行きと帰りで同じ点に戻るものだけを使う（追跡の誤りを除く）
    fb_error = np.linalg.norm(corners - back, axis=2).reshape(-1)
    good = (status.reshape(-1) == 1) & (status_back.reshape(-1) == 1) & (fb_error < 1.0)
    if good.sum() < min_points:
        return invalid("tracking_lost", int(good.sum()))

    center = np.array([WORK_WIDTH / 2.0, WORK_HEIGHT / 2.0], dtype=np.float32)
    src = corners.reshape(-1, 2)[good] - center
    dst = nxt.reshape(-1, 2)[good] - center
    model, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=500
    )
    if model is None or inliers is None:
        return invalid("no_model", len(src))

    a, b = float(model[0, 0]), float(model[1, 0])
    return FlowResult(
        valid=True,
        reason=None,
        points=len(src),
        inlier_ratio=float(inliers.sum()) / len(src),
        shift_x_px=float(model[0, 2]),
        shift_y_px=float(model[1, 2]),
        scale=float(np.hypot(a, b)),
        rotation_deg=float(np.degrees(np.arctan2(b, a))),
    )


@dataclass
class _History:
    timestamp: float
    gray: np.ndarray
    distance_cm: Optional[float]


class MotionTracker:
    """フレームを順に受け取り、一定時間前のフレームと比べて動きを推定する

    隣り合うフレームだけでは動きが小さく、雑音に埋もれる。`interval_s` 前に最も近い
    フレームと比べる。
    """

    def __init__(
        self,
        interval_s: float = 0.3,
        hfov_deg: float = DEFAULT_HFOV_DEG,
        moving_px_s: float = 4.0,
        max_distance_cm: float = 150.0,
        max_approach_cm_s: float = 100.0,
    ):
        """
        Args:
            interval_s: 比べるフレームの時間差
            hfov_deg: カメラの水平画角。ピクセルのずれを向きの変化（度）に直すのに使う
            moving_px_s: 画像が動いたとみなす動きの大きさ（ピクセル/秒）。実機で校正する
            max_distance_cm: 超音波の距離を信用する上限
            max_approach_cm_s: これより速い距離の変化は、別の物を捉えたとみなして使わない
        """
        self.interval_s = interval_s
        self.deg_per_px = hfov_deg / WORK_WIDTH
        self.moving_px_s = moving_px_s
        self.max_distance_cm = max_distance_cm
        self.max_approach_cm_s = max_approach_cm_s
        self._history: Deque[_History] = deque()

    def reset(self) -> None:
        self._history.clear()

    def _reference(self, now: float) -> Optional[_History]:
        """`interval_s` 前に最も近い過去のフレーム。まだ時間がたっていなければ None"""
        target = now - self.interval_s
        candidates = [h for h in self._history if h.timestamp <= now - self.interval_s * 0.5]
        if not candidates:
            return None
        return min(candidates, key=lambda h: abs(h.timestamp - target))

    def _usable_distance(self, distance_cm: Optional[float]) -> Optional[float]:
        if distance_cm is None or not np.isfinite(distance_cm):
            return None
        if distance_cm <= 0 or distance_cm > self.max_distance_cm:
            return None
        return float(distance_cm)

    def update(
        self, frame: np.ndarray, timestamp: float, distance_cm: Optional[float] = None
    ) -> Optional[MotionSample]:
        """フレームを加え、比べられる過去のフレームがあれば推定値を返す"""
        gray = to_work_gray(frame)
        distance = self._usable_distance(distance_cm)
        reference = self._reference(timestamp)

        self._history.append(_History(timestamp, gray, distance))
        # 比べる相手として必要な範囲だけ残す
        while self._history and self._history[0].timestamp < timestamp - self.interval_s * 3:
            self._history.popleft()

        if reference is None:
            return None
        dt = timestamp - reference.timestamp
        if dt <= 0:
            return None

        flow = estimate_flow(reference.gray, gray)
        yaw_rate, expansion_rate, edge_flow = self._rates(flow, dt)

        approach = None
        if distance is not None and reference.distance_cm is not None:
            rate = (reference.distance_cm - distance) / dt
            if abs(rate) <= self.max_approach_cm_s:
                approach = rate

        return MotionSample(
            timestamp=timestamp,
            interval_s=dt,
            image_valid=flow.valid,
            image_reason=flow.reason,
            points=flow.points,
            inlier_ratio=flow.inlier_ratio,
            yaw_rate_dps=yaw_rate,
            expansion_rate=expansion_rate,
            edge_flow_px_s=edge_flow,
            image_moving=flow.valid and edge_flow >= self.moving_px_s,
            distance_cm=distance,
            approach_cm_s=approach,
        )

    def _rates(self, flow: FlowResult, dt: float) -> Tuple[float, float, float]:
        if not flow.valid:
            return 0.0, 0.0, 0.0
        # 右へ曲がると景色は左（負）へずれるので、符号を反転して右回りを正にする
        yaw_rate = -flow.shift_x_px * self.deg_per_px / dt
        expansion_rate = (flow.scale - 1.0) / dt
        # 画像の端での動きの大きさ: 平行移動と、拡大による端の点のずれの大きい方
        edge = max(
            float(np.hypot(flow.shift_x_px, flow.shift_y_px)),
            abs(flow.scale - 1.0) * WORK_WIDTH / 2.0,
        )
        return yaw_rate, expansion_rate, edge / dt
