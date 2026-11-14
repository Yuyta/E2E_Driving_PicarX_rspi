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
        return self.forward_pooled(self.features(x), intent)

    def forward_pooled(
        self, pooled: torch.Tensor, intent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """プーリングの後（64 x 4 x 4）から、舵角と塞がっているかのロジット"""
        x = torch.flatten(pooled, 1)
        x = self.fc(x)
        # 全部の頭を計算し、意図の 0 / 1 で1つを選ぶ（ONNX でも同じ形で出せる）
        outputs = torch.cat([head(x) for head in self.heads], dim=1)  # (B, 3)
        steering = (outputs * intent).sum(dim=1, keepdim=True)
        return steering, self.blocked_head(x)


def adaptive_pool_matrix(in_size: int, out_size: int) -> torch.Tensor:
    """AdaptiveAvgPool と同じ区間で平均する行列

    区間は start = floor(i*in/out)、end = ceil((i+1)*in/out)
    """
    matrix = torch.zeros(out_size, in_size)
    for i in range(out_size):
        start = (i * in_size) // out_size
        end = -((-(i + 1) * in_size) // out_size)
        matrix[i, start:end] = 1.0 / (end - start)
    return matrix


class MatrixAdaptivePool(nn.Module):
    """入力の大きさが決まっている時の AdaptiveAvgPool2d。左右から定数の行列を掛ける

    10 x 15 から 4 x 4 のように割り切れないと、ONNX へ書き出した AdaptiveAvgPool2d は GatherND を
    使う形になり、Pi で使う tract が読めない（2026-11-14 に確認）。行列の掛け算（MatMul）なら
    読める。
    """

    def __init__(self, in_hw: Tuple[int, int], out_hw: Tuple[int, int]):
        super().__init__()
        self.register_buffer("rows", adaptive_pool_matrix(in_hw[0], out_hw[0]))
        self.register_buffer("cols", adaptive_pool_matrix(in_hw[1], out_hw[1]).t().contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(torch.matmul(self.rows, x), self.cols)


class ExportModel(nn.Module):
    """ONNX 用。舵角と、前方が塞がっている確率（0〜1）を出す

    入力は 120 x 160 に決まっているので、最後のプーリングを同じ計算の行列の掛け算に置き換える。
    重みは学習したモデルのものをそのまま使う。
    """

    def __init__(self, model: SmallDrivingCNN, input_hw: Tuple[int, int] = (120, 160)):
        super().__init__()
        self.model = model
        pool = model.features[-1]
        if not isinstance(pool, nn.AdaptiveAvgPool2d):
            raise TypeError("The last feature layer must be AdaptiveAvgPool2d")
        self.convs = nn.Sequential(*list(model.features)[:-1])
        with torch.no_grad():
            was_training = model.training
            model.eval()
            feature_hw = tuple(self.convs(torch.zeros(1, 3, *input_hw)).shape[-2:])
            model.train(was_training)
        output_size = pool.output_size
        out_hw = (output_size, output_size) if isinstance(output_size, int) else output_size
        self.pool = MatrixAdaptivePool(feature_hw, tuple(out_hw))

    def forward(self, x: torch.Tensor, intent: torch.Tensor):
        steering, blocked_logit = self.model.forward_pooled(self.pool(self.convs(x)), intent)
        return steering, torch.sigmoid(blocked_logit)
