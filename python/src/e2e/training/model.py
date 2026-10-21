"""
Lightweight End-to-End CNN model for PiCar-X Steering Prediction
NVIDIA DAVE-2 / PilotNet に着想を得た、CPUでも高速動作する軽量CNNモデル
"""

import torch
import torch.nn as nn


class SmallDrivingCNN(nn.Module):
    """
    入力: (B, 3, 120, 160)
    出力: (B, 1) - ステアリング予測値 (-1.0 ~ 1.0)
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

        # 全結合層
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Tanh(),  # 出力を -1.0 〜 1.0 に正規化
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
