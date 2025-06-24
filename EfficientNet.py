import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

# MBConv 模块（EfficientNet 核心块）
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1, kernel_size=3, se_ratio=0.25):
        super().__init__()
        expanded_channels = in_channels * expansion
        self.stride = stride
        self.use_residual = (in_channels == out_channels) and (stride == 1)

        # 扩展层（1x1 卷积）
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()  # Swish 激活
        ) if expansion != 1 else nn.Identity()

        # 深度可分离卷积
        self.dw_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                      stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU()
        )

        # Squeeze-and-Excitation 模块
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_channels, 1),
            nn.SiLU(),
            nn.Conv2d(se_channels, expanded_channels, 1),
            nn.Sigmoid()
        )

        # 输出层（1x1 卷积）
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = x
        x = self.expand(x)
        x = self.dw_conv(x)
        x = self.se(x) * x  # SE 模块
        x = self.project(x)
        if self.use_residual:
            x += residual
        return x

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # 初始卷积层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        # MBConv 块配置（B0 版本）
        mb_config = [
            # (expansion, out_channels, num_blocks, stride, kernel_size)
            (1, 16, 1, 1, 3),
            (6, 24, 2, 2, 3),
            (6, 40, 2, 2, 5),
            (6, 80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3)
        ]

        # 构建 MBConv 层
        layers = []
        in_channels = 32
        for expansion, out_channels, num_blocks, stride, kernel_size in mb_config:
            layers.append(MBConvBlock(in_channels, out_channels, expansion, stride, kernel_size))
            for _ in range(1, num_blocks):
                layers.append(MBConvBlock(out_channels, out_channels, expansion, 1, kernel_size))
            in_channels = out_channels
        self.blocks = nn.Sequential(*layers)

        # 分类头
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

