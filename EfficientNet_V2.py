import torch
import torch.nn as nn
import math
from typing import List, Callable


def _make_divisible(v: float, divisor: int, min_value: int = None) -> int:
    """
    Ensure that all layers have a channel number that's divisible by divisor.
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SiLU(nn.Module):
    """Sigmoid-weighted Linear Unit (Swish) activation function."""

    def forward(self, x):
        return x * torch.sigmoid(x)


class SqueezeExcite(nn.Module):
    def __init__(self, in_channels: int, se_ratio: float = 0.25):
        super().__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            SiLU(),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block with squeeze-excitation."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            expand_ratio: float,
            se_ratio: float,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.has_skip = stride == 1 and in_channels == out_channels

        expanded_channels = int(in_channels * expand_ratio)
        self.use_res_connect = self.has_skip

        layers = []

        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                SiLU()
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(
                expanded_channels, expanded_channels, kernel_size, stride,
                padding=kernel_size // 2, groups=expanded_channels, bias=False
            ),
            nn.BatchNorm2d(expanded_channels),
            SiLU()
        ])

        # Squeeze and excitation
        if se_ratio is not None and se_ratio > 0:
            layers.append(SqueezeExcite(expanded_channels, se_ratio))

        # Output phase
        layers.extend([
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        # Add dropout if specified
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class FusedMBConvBlock(nn.Module):
    """Fused MBConv block that combines expansion and depthwise convolution."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            expand_ratio: float,
            se_ratio: float = None,
            dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.has_skip = stride == 1 and in_channels == out_channels

        expanded_channels = int(in_channels * expand_ratio)
        self.use_res_connect = self.has_skip

        layers = []

        # Fused expansion + depthwise convolution
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, kernel_size, stride,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(expanded_channels),
                SiLU()
            ])

        # Squeeze and excitation
        if se_ratio is not None and se_ratio > 0 and expand_ratio != 1:
            layers.append(SqueezeExcite(expanded_channels, se_ratio))

        # Projection if expanded
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels)
            ])
        else:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm2d(out_channels)
            ])

        # Add dropout if specified
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class EfficientNetV2(nn.Module):
    def __init__(
            self,
            width_mult: float = 1.0,
            depth_mult: float = 1.0,
            dropout_rate: float = 0.3,
            num_classes: int = 1000,
    ):
        super().__init__()

        # Configuration for EfficientNetV2-L
        # (type, kernel_size, stride, channels, expansion, layers, se_ratio)
        block_configs = [
            # Stage 0 (fused MBConv)
            ['fused_mbconv', 3, 1, 32, 1, 4, None],
            # Stage 1 (fused MBConv)
            ['fused_mbconv', 3, 2, 64, 4, 7, None],
            # Stage 2 (fused MBConv)
            ['fused_mbconv', 3, 2, 96, 4, 7, None],
            # Stage 3 (MBConv with SE)
            ['mbconv', 3, 2, 192, 4, 10, 0.25],
            # Stage 4 (MBConv with SE)
            ['mbconv', 3, 1, 224, 6, 19, 0.25],
            # Stage 5 (MBConv with SE)
            ['mbconv', 3, 2, 384, 6, 25, 0.25],
            # Stage 6 (MBConv with SE)
            ['mbconv', 3, 1, 640, 6, 7, 0.25],
        ]

        # Stem
        out_channels = _make_divisible(32 * width_mult, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            SiLU()
        )

        # Build blocks
        blocks = []
        in_channels = out_channels

        for config in block_configs:
            block_type, kernel_size, stride, channels, expansion, layers, se_ratio = config
            out_channels = _make_divisible(channels * width_mult, 8)
            num_layers = int(math.ceil(layers * depth_mult))

            for i in range(num_layers):
                # Only first layer in each stage uses specified stride
                current_stride = stride if i == 0 else 1

                if block_type == 'fused_mbconv':
                    block = FusedMBConvBlock(
                        in_channels, out_channels, kernel_size, current_stride,
                        expansion, se_ratio, dropout_rate if i == num_layers - 1 else 0.0
                    )
                elif block_type == 'mbconv':
                    block = MBConvBlock(
                        in_channels, out_channels, kernel_size, current_stride,
                        expansion, se_ratio, dropout_rate if i == num_layers - 1 else 0.0
                    )
                else:
                    raise ValueError(f"Unknown block type: {block_type}")

                blocks.append(block)
                in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

        # Head
        last_channels = _make_divisible(1280 * max(1.0, width_mult), 8)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, last_channels, 1, bias=False),
            nn.BatchNorm2d(last_channels),
            SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.Linear(last_channels, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                init_range = 1.0 / math.sqrt(fan_out)
                nn.init.uniform_(m.weight, -init_range, init_range)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def efficientnet_v2_l(num_classes=1000):
    """EfficientNetV2-L model"""
    return EfficientNetV2(
        width_mult=1.0,
        depth_mult=1.0,
        dropout_rate=0.4,
        num_classes=num_classes
    )


# Example usage:
if __name__ == "__main__":
    model = efficientnet_v2_l(num_classes=38)
    x = torch.randn(1, 3, 224, 224)  # Input size for EfficientNetV2-L is 480x480
    y = model(x)
    print(y.shape)  # Expected output: torch.Size([1, 1000])