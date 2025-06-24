import torch
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt

class LayerNormlization(nn.Module):
    def __init__(self,hidden_size,eps=1e-12):
        super(LayerNormlization,self).__init__()
        self.weight=nn.Parameter(torch.ones(hidden_size))
        self.bias=nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self,x):
        u=x.mean(-1,keepdim=True)#沿最后一维求均值
        s=(x-u).pow(2).mean(-1,keepdim=True)#沿最后一维求方差
        x=(x-u)/torch.sqrt(s+self.eps)
        return self.weight*x+self.bias

class SelfAttention(nn.Module):
    def __init__(self,embed_dim):
        super(SelfAttention,self).__init__()
        self.embed_dim=embed_dim
        self.w_q=nn.Linear(embed_dim,embed_dim)
        self.w_k=nn.Linear(embed_dim,embed_dim)
        self.w_v=nn.Linear(embed_dim,embed_dim)
        self.out_linear=nn.Linear(embed_dim,1)

    def forward(self,x):
        """
        前向传播
        :param x: 输入序列，形状为 (batch_size, seq_len, embed_dim)
        :return: 输出序列，形状为 (batch_size, seq_len, embed_dim)
        """
        q=self.w_q(x)# (batch_size, seq_len, embed_dim)
        k=self.w_k(x)# (batch_size, seq_len, embed_dim)
        v=self.w_v(x)# (batch_size, seq_len, embed_dim)

        attention_score=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.embed_dim) #(batch_size, seq_len, seq_len)

        attention_weight=F.softmax(attention_score,dim=-1)#(batch_size, seq_len, seq_len)每一行求softmax

        attention_output=torch.matmul(attention_weight,v)#点乘values (batch_size, seq_len, embed_dim)

        x=self.out_linear(attention_output)
        return x,attention_weight

# 测试单头注意力模块
if __name__ == "__main__":
    # 设置随机种子以保证结果可复现
    np.random.seed(42)

    # 定义输入数据的大小
    num_samples = 100
    input_dim = 3

    # 生成输入数据
    X = np.random.rand(num_samples, input_dim)  # 随机生成 [0, 1) 范围内的数据


    # 定义高维函数
    def high_dim_function(x1, x2, x3):
        return 2 * x1 + 3 * x2 ** 2 + 4 * x3 ** 3

    # 计算输出数据
    y = high_dim_function(X[:, 0], X[:, 1], X[:, 2])

    # 加入噪声
    noise = np.random.normal(0, 0.1, size=y.shape)  # 均值为 0，标准差为 0.1 的高斯噪声
    y += noise


    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # 转换为 (100, 1) 的形状

    input_dim=3
    epochs = 5000

    attention_net=SelfAttention(embed_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(attention_net.parameters(),lr=1e-3)

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred ,at= attention_net(X_tensor)
        loss = criterion(y_pred, y_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print('Epoch:',epoch,'Loss:',loss.item())

    # 评估模型
    attention_net.eval()
    with torch.no_grad():
        predictions,at = attention_net(X_tensor)  # 预测值
        predictions = predictions.numpy().flatten()
        true_values = y_tensor.numpy().flatten()  # 真实值

    # 绘制预测值与真实值的关系图
    plt.figure(figsize=(8, 6))
    plt.scatter(true_values, predictions, alpha=0.6, label='Predictions')
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linestyle='--',
             label='y=x')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs True Values')
    plt.legend()
    plt.grid(True)
    plt.show()


    # 使用 matplotlib 的 imshow 函数可视化矩阵
    plt.imshow(at.numpy(), cmap='viridis', interpolation='nearest')  # cmap 是颜色映射，可以自定义
    plt.colorbar()  # 添加颜色条以表示数值范围
    plt.title("Matrix Visualization")
    plt.show()
