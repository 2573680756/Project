import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PostionalEncoder(nn.Module):
    """
    位置编码层，为输入序列添加位置信息
    公式: PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """
    def __init__(self,d_model, dropout=0.1, max_len=5000):
        super(PostionalEncoder,self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #计算位置编码PE
        PE=torch.zeros(max_len,d_model)
        #计算词向量位置pos
        postion=torch.arange(0,max_len,dtype=float).unsqueeze(1)
        #分母用换底公式变换后表示
        div_term=torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        PE[:,0::2]=torch.sin(postion/div_term)#偶数位用sin表示
        PE[:,1::2]=torch.cos(postion/div_term)#奇数位用cos表示
        #变换形状为[max_len,1,d_model]
        PE=PE.unsqueeze(0).transpose(0,1)
        self.register_buffer('PE', PE)  # 注册为缓冲区，不参与训练

    def forward(self,x):
        """
        参数:
            x: 输入张量，形状为 [seq_len, batch_size, embedding_dim]
        返回:
            添加了位置编码的张量
        """
        x = x + self.PE[:x.size(0), :]  # 只取前x.size(0)个位置编码
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    初始化 MultiHeadAttention 模块
    :param d_model: 输入嵌入的特征维度
    :param n_head: 注意力头的数量
    """
    def __init__(self,d_model,n_head,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        assert d_model%n_head==0,"d_model必须能被n_head整除"

        self.d_model=d_model# 模型维度
        self.n_head=n_head# 注意力头数量
        self.d_k=d_model//n_head# 每个头的维度

        #线性变换矩阵
        self.Wq=nn.Linear(d_model,d_model)#查询矩阵
        self.Wk=nn.Linear(d_model,d_model)#键矩阵
        self.Wv=nn.Linear(d_model,d_model)#值矩阵

        #输出层
        self.out=nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(p=dropout)

        #缩放因子，即公式中根号dk
        self.scale=torch.sqrt(torch.FloatTensor([d_model])).to(device)

    def forward(self,query,key,value,mask=None):
        """
        参数:
            query: 查询张量 [batch_size, q_len, d_model]
            key: 键张量 [batch_size, k_len, d_model]
            value: 值张量 [batch_size, v_len, d_model]
            mask: 掩码张量 [batch_size, 1, 1, k_len] (decoder用) 或 [batch_size, 1, k_len, k_len] (encoder用)
        返回:
            注意力输出 [batch_size, q_len, d_model]
            注意力权重 [batch_size, n_head, q_len, k_len]
        """
        batch_size = query.size(0)
        #1. 线性投影+分头
        # [batch_size, n_head, q_len, d_k]
        Q=self.Wq(query).view(batch_size,-1,self.n_head,self.d_k).transpose(1,2)
        K=self.Wk(key).view(batch_size,-1,self.n_head,self.d_k).transpose(1,2)
        V=self.Wv(value).view(batch_size,-1,self.n_head,self.d_k).transpose(1,2)

        # 2. 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch_size, n_head, q_len, k_len]

        #应用掩码
        if mask is not None:
            scores=scores.masked_fill(mask==0,-1e9)

        #计算注意力权重
        attn_weights=F.softmax(scores,dim=-1)# [batch_size, n_head, q_len, k_len]
        attn_weights=self.dropout(attn_weights)

        # 3. 应用注意力权重到V上
        output = torch.matmul(attn_weights, V)  # [batch_size, n_head, q_len, d_k]

        # 4. 合并多头
        output = output.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)# [batch_size, q_len, d_model]

        # 5. 最终线性变换
        output=self.out(output)

        return output,attn_weights

class PostionalFeedForward(nn.Module):
    """
    位置前馈网络
    两层全连接层，中间有ReLU激活函数
    """
    def __init__(self,d_model,d_ff,dropout=0.1):
        #d_model为输入层，d_ff为隐藏层
        super(PostionalFeedForward,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)

    def forward(self,x):
        """
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
        返回:
            输出张量 [batch_size, seq_len, d_model]
        """
        x=self.fc1(x)
        x=F.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    """
    Transformer编码器层
    包含: 多头注意力 + 残差连接 & 层归一化 + 前馈网络 + 残差连接 & 层归一化
    """
    def __init__(self,d_model,n_head,d_ff,dropout=0.1):
        super(EncoderLayer,self).__init__()
        self.self_attn=MultiHeadAttention(d_model,n_head,dropout=0.1)
        self.feed_forward=PostionalFeedForward(d_model,d_ff,dropout=0.1)

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)

    def forward(self,x,mask=None):
        """
        参数:
            x: 输入张量 [batch_size, src_len, d_model]
            src_mask: 源序列掩码 [batch_size, 1, 1, src_len]
        返回:
            编码后的张量 [batch_size, src_len, d_model]
        """
        # 1. 自注意力子层
        attn_output,_=self.self_attn(x,x,x,mask=mask)
        x=x+self.dropout1(attn_output)
        x=self.norm1(x)

        # 2. 前馈网络子层
        ff_output=self.feed_forward(x)
        x=x+self.dropout2(ff_output)
        x=self.norm2(x)

        return x

class DecoderLayer(nn.Module):
    """
    Transformer解码器层
    包含: 掩码多头注意力 + 残差连接 & 层归一化 + 编码器-解码器注意力 + 残差连接 & 层归一化 + 前馈网络 + 残差连接 & 层归一化
    """
    def __init__(self,d_model,n_head,d_ff,dropout=0.1):
        super(DecoderLayer,self).__init__()
        self.self_attn=MultiHeadAttention(d_model,n_head,dropout=0.1)
        self.enc_attn=MultiHeadAttention(d_model,n_head,dropout=0.1)
        self.feed_forward=PostionalFeedForward(d_model,d_ff,dropout=0.1)

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
        self.dropout3=nn.Dropout(dropout)

    def forward(self,x,enc_output,src_mask,tgt_mask):
        """
        参数:
            x: 解码器输入 [batch_size, tgt_len, d_model]
            enc_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码 [batch_size, 1, 1, src_len]
            tgt_mask: 目标序列掩码 [batch_size, 1, tgt_len, tgt_len]
        返回:
            解码后的张量 [batch_size, tgt_len, d_model]
        """
        # 1. 掩码自注意力子层
        attn_output,_=self.self_attn(x,x,x,mask=tgt_mask)
        x=x+self.dropout1(attn_output)
        x=self.norm1(x)

        attn_output,_=self.enc_attn(x,enc_output,enc_output,mask=src_mask)
        x=x+self.dropout2(attn_output)
        x=self.norm2(x)

        ff_output=self.feed_forward(x)
        x=x+self.dropout3(ff_output)
        x=self.norm3(x)

        return x

class Encoder(nn.Module):
    """
    Transformer编码器
    包含: 输入嵌入 + 位置编码 + N个编码器层
    """
    def __init__(self,vocab_size, d_model, n_head, d_ff, n_layers, dropout=0.1, max_len=5000):
        super(Encoder,self).__init__()
        self.token_embedding=nn.Embedding(vocab_size,d_model) #vocab_size序列长度
        self.postion_encoder=PostionalEncoder(d_model,dropout,max_len) #位置编码
        #n_layers表示编码器个数
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self,src,src_mask):
        """
        参数:
            src: 源序列 [batch_size, src_len]
            src_mask: 源序列掩码 [batch_size, 1, 1, src_len]
        返回:
            编码后的表示 [batch_size, src_len, d_model]
        """
        # 1. 输入嵌入 + 位置编码
        x=self.token_embedding(src) # [batch_size, src_len, d_model]
        x=self.postion_encoder(x)
        x=self.dropout(x)

        # 2. 通过N个编码器层
        for layer in self.layers:
            x=layer(x,src_mask)

        return x

class Decoder(nn.Module):
    """
    Transformer解码器
    包含: 输入嵌入 + 位置编码 + N个解码器层
    """
    def __init__(self,vocab_size, d_model, n_head, d_ff, n_layers, dropout=0.1, max_len=5000):
        super(Decoder,self).__init__()
        self.token_embedding=nn.Embedding(vocab_size,d_model)
        self.postion_encoder=PostionalEncoder(d_model,dropout,max_len)
        self.layers=nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layers)])
        self.dropout=nn.Dropout(dropout)

    def forward(self,tgt,enc_output,src_mask,tgt_mask):
        """
        参数:
            tgt: 目标序列 [batch_size, tgt_len]
            enc_output: 编码器输出 [batch_size, src_len, d_model]
            src_mask: 源序列掩码 [batch_size, 1, 1, src_len]
            tgt_mask: 目标序列掩码 [batch_size, 1, tgt_len, tgt_len]
        返回:
            解码后的表示 [batch_size, tgt_len, d_model]
        """
        # 1. 输入嵌入 + 位置编码
        x=self.token_embedding(tgt)
        x=self.postion_encoder(x)
        x=self.dropout(x)

        # 2. 通过N个解码器层
        for layer in self.layers:
            x=layer(x,enc_output,src_mask,tgt_mask)

        return x

class transformer(nn.Module):
    """
    完整的Transformer模型
    包含: 编码器 + 解码器 + 输出层
    """
    def __init__(self,src_vocab_size, tgt_vocab_size, d_model=512, n_head=8,
                 d_ff=2048, n_layers=6, dropout=0.1, max_len=5000):
        super(transformer,self).__init__()
        self.encoder=Encoder(src_vocab_size, d_model, n_head, d_ff, n_layers, dropout, max_len)
        self.decoder=Decoder(tgt_vocab_size, d_model, n_head, d_ff, n_layers, dropout, max_len)
        self.output_layer=nn.Linear(d_model,tgt_vocab_size)

        # 参数初始化
        self._reset_parameters()

    def forward(self,src,tgt,src_mask,tgt_mask):
        """
        参数:
            src: 源序列 [batch_size, src_len]
            tgt: 目标序列 [batch_size, tgt_len]
            src_mask: 源序列掩码 [batch_size, 1, 1, src_len]
            tgt_mask: 目标序列掩码 [batch_size, 1, tgt_len, tgt_len]
        返回:
            输出logits [batch_size, tgt_len, tgt_vocab_size]
        """
        # 1. 编码器处理源序列
        enc_output = self.encoder(src, src_mask)

        # 2. 解码器处理目标序列
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # 3. 输出层
        output = self.output_layer(dec_output)

        return output
    def _reset_parameters(self):
        """
        使用Xavier均匀初始化参数
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

def create_padding_mask(seq, pad_idx):
    """
    创建填充掩码
    参数:
        seq: 输入序列 [batch_size, seq_len]
        pad_idx: 填充token的索引
    返回:
        掩码张量 [batch_size, 1, 1, seq_len]
    """
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
    return mask


def create_look_ahead_mask(size):
    """
    创建前瞻掩码（用于解码器自注意力）
    参数:
        size: 目标序列长度
    返回:
        掩码张量 [size, size]
    """
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()  # 上三角矩阵
    return mask


# 示例用法
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 超参数
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    d_model = 512
    n_head = 8
    d_ff = 2048
    n_layers = 6
    dropout = 0.1
    batch_size = 32
    src_len = 50
    tgt_len = 40
    pad_idx = 0

    # 创建模型
    model = transformer(src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, n_layers, dropout).to(device)

    # 创建示例数据
    src = torch.randint(0, src_vocab_size, (batch_size, src_len)).to(device)
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len)).to(device)

    # 创建掩码
    src_mask = create_padding_mask(src, pad_idx).to(device)
    tgt_mask = create_padding_mask(tgt, pad_idx).to(device)
    look_ahead_mask = create_look_ahead_mask(tgt_len).to(device)
    combined_tgt_mask = torch.logical_and(tgt_mask, look_ahead_mask)

    # 前向传播
    output = model(src, tgt, src_mask, combined_tgt_mask)

    print("模型输出形状:", output.shape)  # 应该是 [batch_size, tgt_len, tgt_vocab_size]