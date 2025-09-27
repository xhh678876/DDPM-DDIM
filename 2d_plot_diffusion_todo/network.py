import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        时间步嵌入（timestep embedding）模块。

        作用（典型于扩散/DDPM/UNet 等模型）：
        - 将标量/向量形式的“时间步 t”（可以是离散步，也可以是连续标量，如噪声强度）映射成一个高维向量，
          让后续网络（如 UNet）在每一层都能“感知”当前处于哪个扩散/去噪阶段。
        - 采用常见的正弦/余弦位置编码（sin/cos）作为基底特征，再通过一个小 MLP 做到目标隐藏维度。

        参数：
        - hidden_size: 输出嵌入的维度（通常匹配网络中通道数/特征维度，便于与主干网络融合）。
        - frequency_embedding_size: 先生成的 sin/cos 频率编码的维度（作为 MLP 的输入维度）。
          习惯上该值与 UNet 的宽度相关；默认 256 是常见设置。
        """
        super().__init__() 
        # 一个两层的 MLP（线性 -> SiLU -> 线性），将频率域编码映射到 hidden_size
        # bias=True：线性层带偏置项，通常能提升拟合能力（没有特殊理由不关掉）
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),  # SiLU = x * sigmoid(x)，在扩散网络里很常见，表现稳定
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # 记录频率编码的维度，forward 时会用到
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        构造正弦/余弦形式的时间步（timestep）嵌入。
        参考：OpenAI GLIDE 的实现（见链接）。

        参数：
        - t: 形状 (N,) 的一维 Tensor，表示每个样本的时间步索引/标量（允许小数，支持连续时间）。
             例如 DDPM 里 t ∈ {0, 1, ..., T-1}；连续扩散里也可能是 float。
        - dim: 目标编码维度 D（即输出的最后一维大小）。
        - max_period: 控制最低频率（对应最长周期）；越大表示覆盖到更长的周期范围。
                      常用 10000，与 Transformer 位置编码一致的量级。

        返回：
        - 形状为 (N, D) 的张量，按 [cos部分, sin部分] 拼接（若 D 为奇数会在末尾补零对齐）。
        """
        # 从 glide 的实现借鉴：https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        # 一半的维度分给 cos，一半分给 sin
        half = dim // 2

        # 生成从 1 到 1/max_period 的对数均匀分布的频率（共 'half' 个）
        # 频率序列：freqs[k] = (max_period)^(-k/half), k = 0..half-1
        # k=0 时 freqs=1（最高频），k=half-1 时 ~ 1/max_period（最低频，最长周期）
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)  # 将频率张量放到与 t 相同的设备上（CPU/GPU），避免 device mismatch

        # 将 t (N,) 扩展为 (N,1)，与 (1,half) 的 freqs 做广播相乘 -> 得到 (N, half)
        # 对每个样本、每个频率，都计算 t * freq（即不同频率下的角度参数）
        args = t[:, None].float() * freqs[None]

        # 拼接 cos 与 sin 两部分，得到 (N, 2*half)；若 dim 为偶数，则 2*half == dim
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # 若 dim 为奇数，补一个全 0 列到末尾，保证输出维度严格等于 dim
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )

        return embedding  # (N, dim)

    def forward(self, t: torch.Tensor):
        """
        前向计算：
        1) 将时间步 t 变成频率域的 sin/cos 编码（维度 = self.frequency_embedding_size）
        2) 用一个小 MLP 投影到 hidden_size，作为最终的时间步嵌入
        """
        # 如果传入的是标量 0-dim（比如 t=torch.tensor(10)），扩成 1-dim（形状变成 (1,)）
        if t.ndim == 0:
            t = t.unsqueeze(-1)

        # 生成频率编码（N, frequency_embedding_size）
        # 注意：这里 dim 就是 frequency_embedding_size；如果你改了上面的默认值，这里也会跟着用它
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)

        # 通过 MLP（线性->SiLU->线性）映射到目标隐藏维度（N, hidden_size）
        t_emb = self.mlp(t_freq)

        return t_emb  # 通常用于与主干特征做加法/拼接，注入“当前时间”的条件信息



class TimeLinear(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, num_timesteps: int):
        super().__init__()
        """
        一个带时间条件的线性层 (Time-conditioned Linear layer)。

        思路：
        - 输入特征 x 先通过普通的线性层投影到 dim_out。
        - 再通过时间步嵌入 TimeEmbedding 生成一个和 x 相同维度的向量 alpha。
        - 最终输出是 alpha * x，相当于给每个时间步引入一个可学习的缩放调制。
        
        参数：
        - dim_in: 输入特征维度。
        - dim_out: 输出特征维度。
        - num_timesteps: 时间步的数量（通常是扩散过程的 T），这里传进来但未直接用。
                         它的存在更像是为了接口统一或者后续可能扩展。
        """

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_timesteps = num_timesteps

        # 定义一个时间嵌入器，输出维度设为 dim_out，
        # 这样生成的 alpha 和 x 的投影结果可以逐元素相乘。
        self.time_embedding = TimeEmbedding(dim_out)

        # 普通线性层，把输入特征投影到 dim_out 维度
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        前向传播：
        1. 输入特征 x -> 线性投影到 dim_out。
        2. 根据时间步 t 计算时间嵌入 alpha（形状和 x 相同）。
        3. 输出 alpha * x，相当于“给不同时间步的特征加上调制因子”。

        参数：
        - x: 输入特征张量，形状 (..., dim_in)，例如 (batch, dim_in)。
        - t: 时间步张量，形状 (batch,) 或标量，用来生成时间条件。

        返回：
        - 输出张量，形状 (..., dim_out)。
        """

        # 先做一次线性变换，把 x 转换到 dim_out
        x = self.fc(x)

        # 对时间步做嵌入，得到形状 (batch, dim_out) 的 alpha
        # 注意：这里 view(-1, dim_out) 是为了确保和 x 对齐
        alpha = self.time_embedding(t).view(-1, self.dim_out)

        # 按元素相乘：每个样本的特征都根据对应时间步缩放
        return alpha * x



class SimpleNet(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, dim_hids: List[int], num_timesteps: int
    ):
        super().__init__()
        """
        (TODO) Build a noise estimating network.
         
        Args:
            dim_in: dimension of input
            dim_out: dimension of output
            dim_hids: dimensions of hidden features
            num_timesteps: number of timesteps
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        self.num_timesteps = num_timesteps
        
        # 隐藏层尺寸序列，例如 [128,128,128]
        hids = list(dim_hids)
     
        layers: List[nn.Module] = []
        # 第一层: dim_in -> hids[0]
        in_dim = dim_in
        for h in hids:
            layers.append(TimeLinear(in_dim, h, num_timesteps))  # 用时间调制的线性层
            layers.append(nn.SiLU())
            in_dim = h

        # 最后一层: -> dim_out（不加激活）
        layers.append(nn.Linear(in_dim, dim_out))

        self.net = nn.Sequential(*layers)
        ######################
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        (TODO) Implement the forward pass. This should output
        the noise prediction of the noisy input x at timestep t.

        Args:
            x: the noisy data after t period diffusion
            t: the time that the forward diffusion has been running
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        if t.ndim==2 and t.size(-1)==1:
           t= t.squeeze(-1)
        h=x
        for layer in self.net:
            if isinstance(layer,TimeLinear):
                h=layer(h,t)
            else :
                h=layer(h)
        ######################
        return h
               
