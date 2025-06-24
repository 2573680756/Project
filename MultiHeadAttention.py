import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        初始化 MultiHeadAttention 模块
        :param embed_dim: 输入嵌入的特征维度
        :param num_heads: 注意力头的数量
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个头的特征维度

        # 定义 Query, Key 和 Value 的线性变换
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)

        # 输出的线性变换
        self.out_linear = nn.Linear(embed_dim, 1)

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (batch_size, seq_len, embed_dim)
        :return: 注意力后的输出，形状为 (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()
        # 生成 Query, Key, Value (batch_size, seq_len, embed_dim)
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # 分成多头 (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 计算注意力分数 (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        # 加权求和 (batch_size, num_heads, seq_len, head_dim)
        attention_output = torch.matmul(attention_weights, V)
        # 拼接多头输出 (batch_size, seq_len, embed_dim
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        # 输出线性变换 (batch_size, seq_len, embed_dim)
        output = self.out_linear(attention_output)
        return output,attention_weights
if __name__=="__main__":
    # 设置随机种子以保证结果可复现
    np.random.seed(42)

    # 定义输入数据的大小
    num_samples = 200
    input_dim = 20

    # 生成输入数据
    X = np.random.rand(num_samples, input_dim)  # 随机生成 [0, 1) 范围内的数据


    # 定义复杂的高维函数
    def complex_high_dim_function(x):
        return 2 * x[:, 0] + 3 * x[:, 1] ** 2 + 4 * x[:, 2] ** 3 + 5 * x[:, 3] ** 4 + 6 * x[:, 4] ** 5 + np.sum(
            x[:, 5:], axis=1)


    # 计算输出数据
    y = complex_high_dim_function(X)

    # 加入噪声
    noise = np.random.normal(0, 0.1, size=y.shape)  # 均值为 0，标准差为 0.1 的高斯噪声
    y += noise

    # 转换为 PyTorch 张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # 转换为 (200, 1) 的形状

    # 定义模型
    embed_dim = 20
    num_heads = 4
    model = MultiHeadAttention(embed_dim, num_heads)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 10000
    for epoch in range(num_epochs):
        # 前向传播
        outputs,at = model(X_tensor[:150,:].unsqueeze(0))
        loss = criterion(outputs, y_tensor[:150,:].unsqueeze(0))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # 评估模型
    model.eval()
    with torch.no_grad():
        predictions, at = model(X_tensor[:150,:].unsqueeze(0))  # 预测值
        predictions = predictions.numpy().flatten()
        true_values = y_tensor[:150,:].numpy().flatten()  # 真实值

    # 绘制预测值与真实值的关系图
    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, predictions, alpha=0.6, label='Predictions')
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red',
                linestyle='--',
                label='y=x')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')
    plt.legend()
    plt.grid(True)
    plt.show()

    for i in range(4):
        plt.imshow(at[0,i,:,:].numpy(), cmap='viridis', interpolation='nearest')  # cmap 是颜色映射，可以自定义
        plt.colorbar()  # 添加颜色条以表示数值范围
        plt.title("Matrix Visualization")
        plt.show()
