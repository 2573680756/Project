import torch.nn as nn
import torch


class InitialBlock(nn.Module):
    """
    初始块由两个分支组成：
    1. 主分支执行步长为2的普通卷积；
    2. 扩展分支执行最大池化。

    并行执行这两种操作并将它们的结果连接起来，可以实现高效的下采样和扩展。
    主分支输出13个特征图，而扩展分支输出3个，连接后总共有16个特征图。

    关键字参数：
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - kernel_size (int, 可选): 卷积层中使用的滤波器的大小。默认值：3。
    - padding (int, 可选): 在输入的两侧添加的零填充。默认值：0。
    - bias (bool, 可选): 如果为 ``True``，则向输出添加可学习的偏置。默认值：False。
    - relu (bool, 可选): 当 ``True`` 时使用ReLU作为激活函数；否则，使用PReLU。默认值：True。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        """
        初始化InitialBlock类。

        参数：
        - in_channels: 输入特征图的通道数。
        - out_channels: 输出特征图的通道数。
        - bias: 是否在卷积层中添加偏置项，默认为False。
        - relu: 是否使用ReLU激活函数，默认为True，否则使用PReLU。
        """
        super().__init__()

        if relu:
            activation = nn.ReLU  # 如果使用ReLU激活函数
        else:
            activation = nn.PReLU  # 否则使用PReLU激活函数

        # 主分支 - 如上所述，这个分支的输出通道数是总通道数减去3，
        # 因为剩余的通道来自扩展分支
        self.main_branch = nn.Conv2d(
            in_channels,  # 输入通道数
            out_channels - 3,  # 输出通道数（减去扩展分支的3个通道）
            kernel_size=3,  # 卷积核大小
            stride=2,  # 步长为2，用于下采样
            padding=1,  # 填充大小
            bias=bias)  # 是否添加偏置

        # 扩展分支 - 使用最大池化进行下采样
        self.ext_branch = nn.MaxPool2d(
            kernel_size=3,  # 池化核大小
            stride=2,  # 步长为2
            padding=1)  # 填充大小

        # 初始化批量归一化层，用于连接分支后的归一化
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # PReLU层，用于在连接分支后应用激活函数
        self.out_activation = activation()

    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入特征图。

        返回值：
        - out: 经过初始块处理后的输出特征图。
        """
        # 主分支的输出
        main = self.main_branch(x)
        # 扩展分支的输出
        ext = self.ext_branch(x)

        # 将主分支和扩展分支的输出在通道维度上连接起来
        out = torch.cat((main, ext), 1)

        # 应用批量归一化
        out = self.batch_norm(out)

        # 应用激活函数
        return self.out_activation(out)


class RegularBottleneck(nn.Module):
    """
    Regular bottlenecks 是 ENet 的主要构建块。
    主分支：
    1. 残差连接。

    扩展分支：
    1. 1x1 卷积，用于减少通道数，也称为投影；
    2. 普通、扩张或非对称卷积；
    3. 1x1 卷积，用于将通道数增加回原来的数量，也称为扩展；
    4. 作为正则化器的 dropout。

    关键字参数：
    - channels (int): 输入和输出通道数。
    - internal_ratio (int, 可选): 应用于 ``channels`` 的缩放因子，用于计算投影后的通道数。
      例如，给定 ``channels`` 等于 128，internal_ratio 等于 2，则投影后的通道数为 64。默认值：4。
    - kernel_size (int, 可选): 扩展分支中第 2 项描述的卷积层中使用的滤波器大小。默认值：3。
    - padding (int, 可选): 在输入的两侧添加的零填充。默认值：0。
    - dilation (int, 可选): 扩展分支中第 2 项描述的卷积中核元素之间的间距。默认值：1。
    - asymmetric (bool, 可选): 标记扩展分支中第 2 项描述的卷积是否为非对称卷积。默认值：False。
    - dropout_prob (float, 可选): 元素被置为零的概率。默认值：0（不使用 dropout）。
    - bias (bool, 可选): 如果为 ``True``，则向输出添加可学习的偏置。默认值：False。
    - relu (bool, 可选): 当 ``True`` 时使用 ReLU 作为激活函数；否则，使用 PReLU。默认值：True。
    """

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        """
        初始化 RegularBottleneck 类。

        参数：
        - channels: 输入和输出通道数。
        - internal_ratio: 用于计算投影后通道数的缩放因子。
        - kernel_size: 卷积层的核大小。
        - padding: 零填充大小。
        - dilation: 膨胀率。
        - asymmetric: 是否使用非对称卷积。
        - dropout_prob: Dropout 的概率。
        - bias: 是否在卷积层中添加偏置。
        - relu: 是否使用 ReLU 激活函数。
        """
        super().__init__()

        # 检查 internal_ratio 是否在有效范围内 [1, channels]
        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_ratio={1}."
                               .format(channels, internal_ratio))

        internal_channels = channels // internal_ratio  # 计算投影后的通道数

        if relu:
            activation = nn.ReLU  # 如果使用 ReLU 激活函数
        else:
            activation = nn.PReLU  # 否则使用 PReLU 激活函数

        # 主分支 - 残差连接

        # 扩展分支 - 1x1 卷积，后接普通、扩张或非对称卷积，再接另一个 1x1 卷积，最后是正则化器（空间 dropout）。
        # 通道数保持不变。

        # 1x1 投影卷积
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels,  # 输入通道数
                internal_channels,  # 输出通道数
                kernel_size=1,  # 卷积核大小
                stride=1,  # 步长
                bias=bias),  # 是否添加偏置
            nn.BatchNorm2d(internal_channels),  # 批量归一化
            activation())  # 激活函数

        # 如果卷积是非对称的，我们将主卷积分成两个卷积。
        # 例如，对于一个 5x5 的非对称卷积，我们有两个卷积：第一个是 5x1，第二个是 1x5。
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,  # 输入通道数
                    internal_channels,  # 输出通道数
                    kernel_size=(kernel_size, 1),  # 卷积核大小
                    stride=1,  # 步长
                    padding=(padding, 0),  # 填充大小
                    dilation=dilation,  # 膨胀率
                    bias=bias),  # 是否添加偏置
                nn.BatchNorm2d(internal_channels),  # 批量归一化
                activation(),  # 激活函数
                nn.Conv2d(
                    internal_channels,  # 输入通道数
                    internal_channels,  # 输出通道数
                    kernel_size=(1, kernel_size),  # 卷积核大小
                    stride=1,  # 步长
                    padding=(0, padding),  # 填充大小
                    dilation=dilation,  # 膨胀率
                    bias=bias),  # 是否添加偏置
                nn.BatchNorm2d(internal_channels),  # 批量归一化
                activation())  # 激活函数
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,  # 输入通道数
                    internal_channels,  # 输出通道数
                    kernel_size=kernel_size,  # 卷积核大小
                    stride=1,  # 步长
                    padding=padding,  # 填充大小
                    dilation=dilation,  # 膨胀率
                    bias=bias),  # 是否添加偏置
                nn.BatchNorm2d(internal_channels),  # 批量归一化
                activation())  # 激活函数

        # 1x1 扩展卷积
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,  # 输入通道数
                channels,  # 输出通道数
                kernel_size=1,  # 卷积核大小
                stride=1,  # 步长
                bias=bias),  # 是否添加偏置
            nn.BatchNorm2d(channels),  # 批量归一化
            activation())  # 激活函数

        self.ext_regul = nn.Dropout2d(p=dropout_prob)  # 空间 dropout

        # 在添加分支后应用的激活函数
        self.out_activation = activation()

    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入特征图。

        返回值：
        - out: 经过瓶颈模块处理后的输出特征图。
        """
        # 主分支 - 残差连接
        main = x

        # 扩展分支
        ext = self.ext_conv1(x)  # 1x1 投影卷积
        ext = self.ext_conv2(ext)  # 主卷积（普通、扩张或非对称）
        ext = self.ext_conv3(ext)  # 1x1 扩展卷积
        ext = self.ext_regul(ext)  # 应用 dropout

        # 将主分支和扩展分支的输出相加
        out = main + ext

        return self.out_activation(out)  # 应用激活函数


class DownsamplingBottleneck(nn.Module):
    """
    下采样瓶颈模块进一步减小特征图的大小。

    主分支：
    1. 步长为2的最大池化；保存索引以便后续上采样。

    扩展分支：
    1. 步长为2的2x2卷积，减少通道数，也称为投影；
    2. 普通卷积（默认为3x3）；
    3. 1x1卷积，将通道数增加到``out_channels``，也称为扩展；
    4. 作为正则化器的dropout。

    关键字参数：
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - internal_ratio (int, 可选): 应用于``channels``的缩放因子，用于计算投影后的通道数。
      例如，给定``channels``等于128，internal_ratio等于2，则投影后的通道数为64。默认值：4。
    - return_indices (bool, 可选): 如果为``True``，将返回最大值索引以及输出。在后续上采样时很有用。
    - dropout_prob (float, 可选): 元素被置为零的概率。默认值：0（不使用dropout）。
    - bias (bool, 可选): 如果为``True``，则向输出添加可学习的偏置。默认值：False。
    - relu (bool, 可选): 当``True``时使用ReLU作为激活函数；否则，使用PReLU。默认值：True。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        """
        初始化DownsamplingBottleneck类。

        参数：
        - in_channels: 输入通道数。
        - out_channels: 输出通道数。
        - internal_ratio: 用于计算投影后通道数的缩放因子。
        - return_indices: 是否返回最大值索引。
        - dropout_prob: Dropout的概率。
        - bias: 是否在卷积层中添加偏置。
        - relu: 是否使用ReLU激活函数。
        """
        super().__init__()

        # 保存后续需要使用的参数
        self.return_indices = return_indices

        # 检查internal_ratio是否在有效范围内 [1, in_channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_ratio={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio  # 计算投影后的通道数

        if relu:
            activation = nn.ReLU  # 如果使用ReLU激活函数
        else:
            activation = nn.PReLU  # 否则使用PReLU激活函数

        # 主分支 - 最大池化后接特征图（通道）填充
        self.main_max1 = nn.MaxPool2d(
            2,  # 池化核大小
            stride=2,  # 步长
            return_indices=return_indices)  # 是否返回索引

        # 扩展分支 - 2x2卷积，后接普通卷积，再接1x1卷积。通道数加倍。

        # 步长为2的2x2投影卷积
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,  # 输入通道数
                internal_channels,  # 输出通道数
                kernel_size=2,  # 卷积核大小
                stride=2,  # 步长
                bias=bias),  # 是否添加偏置
            nn.BatchNorm2d(internal_channels),  # 批量归一化
            activation())  # 激活函数

        # 普通卷积
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,  # 输入通道数
                internal_channels,  # 输出通道数
                kernel_size=3,  # 卷积核大小
                stride=1,  # 步长
                padding=1,  # 填充大小
                bias=bias),  # 是否添加偏置
            nn.BatchNorm2d(internal_channels),  # 批量归一化
            activation())  # 激活函数

        # 1x1扩展卷积
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,  # 输入通道数
                out_channels,  # 输出通道数
                kernel_size=1,  # 卷积核大小
                stride=1,  # 步长
                bias=bias),  # 是否添加偏置
            nn.BatchNorm2d(out_channels),  # 批量归一化
            activation())  # 激活函数

        self.ext_regul = nn.Dropout2d(p=dropout_prob)  # 空间dropout

        # 在连接分支后应用的激活函数
        self.out_activation = activation()

    def forward(self, x):
        """
        前向传播函数。

        参数：
        - x: 输入特征图。

        返回值：
        - out: 经过下采样瓶颈模块处理后的输出特征图。
        - max_indices: 最大值索引（如果return_indices为True）。
        """
        # 主分支 - 残差连接
        if self.return_indices:
            main, max_indices = self.main_max1(x)  # 最大池化并返回索引
        else:
            main = self.main_max1(x)  # 只进行最大池化

        # 扩展分支
        ext = self.ext_conv1(x)  # 2x2投影卷积
        ext = self.ext_conv2(ext)  # 普通卷积
        ext = self.ext_conv3(ext)  # 1x1扩展卷积
        ext = self.ext_regul(ext)  # 应用dropout

        # 主分支通道填充
        n, ch_ext, h, w = ext.size()  # 获取扩展分支的尺寸
        ch_main = main.size()[1]  # 获取主分支的通道数
        padding = torch.zeros(n, ch_ext - ch_main, h, w)  # 创建填充张量

        # 在连接之前，检查主分支是在CPU还是GPU上，并相应地转换填充张量
        if main.is_cuda:
            padding = padding.cuda()

        # 连接主分支和填充张量
        main = torch.cat((main, padding), 1)

        # 将主分支和扩展分支的输出相加
        out = main + ext

        return self.out_activation(out), max_indices  # 应用激活函数并返回输出和索引（如果需要）


class UpsamplingBottleneck(nn.Module):
    """
    上采样瓶颈模块使用从相应下采样瓶颈模块中存储的最大池化索引来上采样特征图的分辨率。

    主分支：
    1. 步长为1的1x1卷积，减少通道数，也称为投影；
    2. 使用从相应下采样最大池化层中获取的最大池化索引进行最大反池化。

    扩展分支：
    1. 步长为1的1x1卷积，减少通道数，也称为投影；
    2. 转置卷积（默认为3x3）；
    3. 1x1卷积，将通道数增加到``out_channels``，也称为扩展；
    4. 作为正则化器的dropout。

    关键字参数：
    - in_channels (int): 输入通道数。
    - out_channels (int): 输出通道数。
    - internal_ratio (int, 可选): 应用于``in_channels``的缩放因子，用于计算投影后的通道数。
      例如，给定``in_channels``等于128，``internal_ratio``等于2，则投影后的通道数为64。默认值：4。
    - dropout_prob (float, 可选): 元素被置为零的概率。默认值：0（不使用dropout）。
    - bias (bool, 可选): 如果为``True``，则向输出添加可学习的偏置。默认值：False。
    - relu (bool, 可选): 当``True``时使用ReLU作为激活函数；否则，使用PReLU。默认值：True。
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        """
        初始化UpsamplingBottleneck类。

        参数：
        - in_channels: 输入通道数。
        - out_channels: 输出通道数。
        - internal_ratio: 用于计算投影后通道数的缩放因子。
        - dropout_prob: Dropout的概率。
        - bias: 是否在卷积层中添加偏置。
        - relu: 是否使用ReLU激活函数。
        """
        super().__init__()

        # 检查internal_ratio是否在有效范围内 [1, in_channels]
        if internal_ratio <= 1 or internal_ratio > in_channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_ratio={1}. "
                               .format(in_channels, internal_ratio))

        internal_channels = in_channels // internal_ratio  # 计算投影后的通道数

        if relu:
            activation = nn.ReLU  # 如果使用ReLU激活函数
        else:
            activation = nn.PReLU  # 否则使用PReLU激活函数

        # 主分支 - 1x1卷积后接特征图（通道）填充
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),  # 1x1卷积
            nn.BatchNorm2d(out_channels))  # 批量归一化

        # 与最大池化层类似，步长等于卷积核大小
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)  # 最大反池化层

        # 扩展分支 - 1x1卷积，后接普通卷积，再接1x1卷积。通道数加倍。

        # 步长为1的1x1投影卷积
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias),  # 1x1卷积
            nn.BatchNorm2d(internal_channels),  # 批量归一化
            activation())  # 激活函数

        # 转置卷积
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)  # 转置卷积
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_channels)  # 批量归一化
        self.ext_tconv1_activation = activation()  # 激活函数

        # 1x1扩展卷积
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias),  # 1x1卷积
            nn.BatchNorm2d(out_channels))  # 批量归一化

        self.ext_regul = nn.Dropout2d(p=dropout_prob)  # 空间dropout

        # 在连接分支后应用的激活函数
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        """
        前向传播函数。

        参数：
        - x: 输入特征图。
        - max_indices: 从下采样瓶颈模块中获取的最大池化索引。
        - output_size: 上采样后的输出大小。

        返回值：
        - out: 经过上采样瓶颈模块处理后的输出特征图。
        """
        # 主分支 - 残差连接
        main = self.main_conv1(x)  # 1x1卷积
        main = self.main_unpool1(
            main, max_indices, output_size=output_size)  # 最大反池化

        # 扩展分支
        ext = self.ext_conv1(x)  # 1x1投影卷积
        ext = self.ext_tconv1(ext, output_size=output_size)  # 转置卷积
        ext = self.ext_tconv1_bnorm(ext)  # 批量归一化
        ext = self.ext_tconv1_activation(ext)  # 激活函数
        ext = self.ext_conv2(ext)  # 1x1扩展卷积
        ext = self.ext_regul(ext)  # 应用dropout

        # 将主分支和扩展分支的输出相加
        out = main + ext

        return self.out_activation(out)  # 应用激活函数