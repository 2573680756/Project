import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """双卷积模块：包含两个连续的(卷积 => 批归一化 => ReLU激活)结构

    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        mid_channels: 中间通道数（可选，默认为out_channels）
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # 如果没有指定中间通道数，则使用输出通道数
        if not mid_channels:
            mid_channels = out_channels

        # 定义连续的两个卷积层，每层后接批归一化和ReLU激活
        self.double_conv = nn.Sequential(
            # 第一个卷积层：3x3卷积，padding=1保持特征图尺寸不变
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活，inplace=True节省内存

            # 第二个卷积层
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """前向传播"""
        return self.double_conv(x)


class Down(nn.Module):
    """下采样模块：先进行最大池化，然后执行双卷积

    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 最大池化层（2x2窗口，步长2）加上双卷积
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 下采样，特征图尺寸减半
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """前向传播"""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块：先上采样，然后执行双卷积

    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数
        bilinear: 是否使用双线性插值进行上采样（默认True）
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 选择上采样方式
        if bilinear:
            # 使用双线性插值上采样（计算量小但可能不够精确）
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 上采样后接的双卷积，中间通道数减半
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 使用转置卷积上采样（计算量大但可以学习）
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """前向传播
        参数：
            x1: 来自上一层的特征图（需要上采样）
            x2: 来自编码器的特征图（用于跳跃连接）
        """
        # 上采样
        x1 = self.up(x1)

        # 处理尺寸不匹配问题（由于卷积和池化可能导致的尺寸差异）
        # 计算x2和x1在高度和宽度上的差异
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # 对x1进行填充，使其尺寸与x2匹配
        x1 = F.pad(x1, [
            diffX // 2, diffX - diffX // 2,  # 左右填充
            diffY // 2, diffY - diffY // 2  # 上下填充
        ])

        # 沿通道维度拼接x2和x1（跳跃连接）
        x = torch.cat([x2, x1], dim=1)

        # 执行双卷积
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积层：1x1卷积，将特征图映射到所需的类别数

    参数：
        in_channels: 输入通道数
        out_channels: 输出通道数（等于分类数）
    """

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # 1x1卷积，不改变特征图尺寸，只改变通道数
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """前向传播"""
        return self.conv(x)


class UNet(nn.Module):
    """完整的UNet模型

    参数：
        n_channels: 输入图像的通道数（如RGB图像为3）
        n_classes: 需要分类的类别数（输出通道数）
        bilinear: 是否在上采样时使用双线性插值（默认True）
    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器部分（下采样）
        self.inc = DoubleConv(n_channels, 64)  # 初始双卷积
        self.down1 = Down(64, 128)  # 第一次下采样
        self.down2 = Down(128, 256)  # 第二次下采样
        self.down3 = Down(256, 512)  # 第三次下采样
        self.down4 = Down(512, 1024 // (2 if bilinear else 1))  # 第四次下采样

        # 解码器部分（上采样）
        self.up1 = Up(1024, 512 // (2 if bilinear else 1), bilinear)  # 第一次上采样
        self.up2 = Up(512, 256 // (2 if bilinear else 1), bilinear)  # 第二次上采样
        self.up3 = Up(256, 128 // (2 if bilinear else 1), bilinear)  # 第三次上采样
        self.up4 = Up(128, 64, bilinear)  # 第四次上采样

        # 输出层
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """前向传播过程"""
        # 编码器路径
        x1 = self.inc(x)  # 初始双卷积
        x2 = self.down1(x1)  # 第一次下采样
        x3 = self.down2(x2)  # 第二次下采样
        x4 = self.down3(x3)  # 第三次下采样
        x5 = self.down4(x4)  # 第四次下采样

        # 解码器路径（包含跳跃连接）
        x = self.up1(x5, x4)  # 第一次上采样并与x4拼接
        x = self.up2(x, x3)  # 第二次上采样并与x3拼接
        x = self.up3(x, x2)  # 第三次上采样并与x2拼接
        x = self.up4(x, x1)  # 第四次上采样并与x1拼接

        # 输出层
        logits = self.outc(x)  # 1x1卷积得到最终输出
        return logits

if __name__ == '__main__':
    model = UNet(3,1)
    x=torch.randn((1,3,256,256))
    y=model(x)
    # 使用 matplotlib 的 imshow 函数可视化矩阵
    plt.imshow(y.detach().numpy()[0,0,:,:], cmap='viridis', interpolation='nearest')  # cmap 是颜色映射，可以自定义
    plt.colorbar()  # 添加颜色条以表示数值范围
    plt.title("Matrix Visualization")
    plt.show()