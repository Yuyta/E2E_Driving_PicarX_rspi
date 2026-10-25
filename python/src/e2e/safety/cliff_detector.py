"""
PiCar-X Cliff Detector (Floor Drop-off & Stair Safety Layer)
家庭内自律走行において、階段・玄関の段差・床の穴を検知し、
車体の落下・破損を防ぐセーフティレイヤーモジュール。
"""

from typing import List, Optional, Tuple


class CliffDetector:
    """
    底面グレースケール（赤外線反射）センサーを用いた段差・階段落下検知
    """

    def __init__(
        self,
        threshold: int = 150,
        escape_speed: int = 35,
        escape_duration_ms: int = 500,
    ):
        """
        Args:
            threshold: このADC値を下回った場合に「段差・空中（反射光激減）」と判定する閾値
            escape_speed: 段差検知時の後退速度
            escape_duration_ms: 後退実行時間 (ミリ秒)
        """
        self.threshold = threshold
        self.escape_speed = escape_speed
        self.escape_duration_ms = escape_duration_ms
        self.floor_baseline: Optional[List[int]] = None

    def calibrate(self, samples: List[List[int]]) -> int:
        """
        起動時の安全な床面で計測した複数のサンプルから、環境に最適な段差検知閾値を自動算出する。
        （通常床面平均値の約 35% を閾値に設定）

        Args:
            samples: [[left, middle, right], ...] のリスト

        Returns:
            算出された新閾値
        """
        if not samples:
            return self.threshold

        avg_left = sum(s[0] for s in samples) / len(samples)
        avg_mid = sum(s[1] for s in samples) / len(samples)
        avg_right = sum(s[2] for s in samples) / len(samples)

        self.floor_baseline = [int(avg_left), int(avg_mid), int(avg_right)]
        min_floor_val = min(avg_left, avg_mid, avg_right)

        # 床面の最低値の35%を崖判定閾値とする（最低でも50を確保）
        self.threshold = max(50, int(min_floor_val * 0.35))
        return self.threshold

    def is_cliff(self, grayscale: List[int]) -> bool:
        """
        3つのセンサー (左, 中央, 右) のいずれかが段差・空中を検知しているか判定

        Args:
            grayscale: [left, middle, right] のADC値 (0〜4095)

        Returns:
            段差・落下危機があれば True
        """
        if len(grayscale) < 3:
            return False

        # いずれかのセンサーが閾値を下回ったら危険
        return any(val < self.threshold for val in grayscale[:3])

    def get_cliff_status(self, grayscale: List[int]) -> Tuple[bool, bool, bool]:
        """
        各センサー (左, 中央, 右) の崖検知状態を取得

        Returns:
            (left_cliff, mid_cliff, right_cliff)
        """
        if len(grayscale) < 3:
            return (False, False, False)

        return (
            grayscale[0] < self.threshold,
            grayscale[1] < self.threshold,
            grayscale[2] < self.threshold,
        )

    def get_escape_action(self, grayscale: List[int]) -> Tuple[int, int, int]:
        """
        段差から安全なエリアへ退避するためのステアリング角度・速度・時間を算出。

        Returns:
            (escape_steer_angle, escape_speed, duration_ms)
        """
        left_cliff, mid_cliff, right_cliff = self.get_cliff_status(grayscale)

        if left_cliff and not right_cliff:
            # 左側が落ちそう → ハンドルを右に切って後退（左前を崖から引き離す）
            steer = 30
        elif right_cliff and not left_cliff:
            # 右側が落ちそう → ハンドルを左に切って後退（右前を崖から引き離す）
            steer = -30
        else:
            # 正面または両側が落ちそう → まっすぐ後退
            steer = 0

        return steer, -self.escape_speed, self.escape_duration_ms
