"""
Lightweight End-to-End CNN model for PiCar-X Steering Prediction
NVIDIA DAVE-2 / PilotNet に着想を得た、CPUでも高速動作する軽量CNNモデル

行きたい向き（意図: any / left / right）ごとに出力の頭（ヘッド）を分ける（条件付きの模倣学習の
分岐型）。画像の特徴は共通で、意図に合う頭の舵角だけを出す。意図を特徴につなげるだけの形より、
意図に従わせやすい（ROADMAP の M3「行きたい向きの入力」）。

もう1つの頭で、前方が塞がっているか（ロジット）を出す（ROADMAP の M3「前方が塞がっているかの
補助出力」）。ラベルは超音波の距離から自動で付ける。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from ..intent import INTENTS
except ImportError:
    from src.e2e.intent import INTENTS


class SmallDrivingCNN(nn.Module):
    """
    入力: 画像 (B, 3, 120, 160) と意図 (B, 3)。意図は any, left, right の順の 0 / 1
    出力: (B, 1) - ステアリング予測値 (-1.0 ~ 1.0)
    `forward_with_blocked` は、前方が塞がっているかのロジット (B, 1) も返す
    """

    def __init__(self):
        super(SmallDrivingCNN, self).__init__()

        # 特徴抽出器 (軽量畳み込み層)
        self.features = nn.Sequential(
            # 120x160 -> 58x78
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            # 58x78 -> 27x37
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 27x37 -> 12x17
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 12x17 -> 10x15
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # 固定サイズ 64 x 4 x 4 = 1024
        )

        # 全結合層（意図で共通）
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        # 意図ごとの頭
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Tanh(),  # 出力を -1.0 〜 1.0 に正規化
            )
            for _ in INTENTS
        )
        # 前方が塞がっているかの頭。乱数の状態を戻し、この頭を加える前と同じ初期値と
        # ドロップアウトで学習が進むようにする（この頭を学習しない時の舵角の結果を変えない）
        with torch.random.fork_rng(devices=[]):
            self.blocked_head = nn.Sequential(
                nn.Linear(128, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )

    def forward(self, x: torch.Tensor, intent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """意図を省くと おまかせ（any）"""
        return self.forward_with_blocked(x, intent)[0]

    def forward_with_blocked(
        self, x: torch.Tensor, intent: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """舵角と、前方が塞がっているかのロジット（確率にはシグモイドを通す）"""
        if intent is None:
            intent = torch.zeros(x.shape[0], len(INTENTS), dtype=x.dtype, device=x.device)
            intent[:, 0] = 1.0
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # 全部の頭を計算し、意図の 0 / 1 で1つを選ぶ（ONNX でも同じ形で出せる）
        outputs = torch.cat([head(x) for head in self.heads], dim=1)  # (B, 3)
        steering = (outputs * intent).sum(dim=1, keepdim=True)
        return steering, self.blocked_head(x)


class ExportModel(nn.Module):
    """ONNX 用。舵角と、前方が塞がっている確率（0〜1）を出す"""

    def __init__(self, model: SmallDrivingCNN):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, intent: torch.Tensor):
        steering, blocked_logit = self.model.forward_with_blocked(x, intent)
        return steering, torch.sigmoid(blocked_logit)
