from typing import List, Optional  # 从 typing 导入类型注解（本文件里 Optional 未实际使用）

import numpy as np                 # 数值计算库（本文件未直接使用）
import torch                       # PyTorch 主包
import torch.nn as nn              # 神经网络模块 (layers, losses, etc.)
import torch.nn.functional as F    # 一些函数式 API（本文件未直接使用）
from module import DownSample, ResBlock, Swish, TimeEmbedding, UpSample  # 导入自定义模块：下采样、残差块、激活、时间嵌入、上采样
from torch.nn import init          # 权重初始化工具


class UNet(nn.Module):  # 定义一个继承自 nn.Module 的 U-Net 模型（用于扩散模型的噪声预测）
    def __init__(self, T=1000, image_resolution=64, ch=128, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=4, dropout=0.1, use_cfg=False, cfg_dropout=0.1, num_classes=None):
        super().__init__()  # 初始化父类

        self.image_resolution = image_resolution  # 记录输入图像分辨率（可用于校验或导出）
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'  # 校验 attn 的层级索引不越界（attn 指定在哪些尺度层使用注意力）

        tdim = ch * 4                       # 时间步嵌入向量维度（常用设置：通道的 4 倍）
        # self.time_embedding = TimeEmbedding(T, ch, tdim)  # 另一种构造方式（按你的 TimeEmbedding 定义而定）
        self.time_embedding = TimeEmbedding(tdim)  # 实例化时间嵌入模块：将离散/连续 t 映射到 tdim 维的 embedding

        # classifier-free guidance（分类器自由引导）相关设置
        self.use_cfg = use_cfg              # 是否启用 CFG（训练时随机丢弃条件，推理时混合无条件/有条件）
        self.cfg_dropout = cfg_dropout      # 训练阶段将条件变为 null 的概率（p_null），常见取值 0.1~0.2
        if use_cfg:                         # 启用 CFG 时需要类别信息
            assert num_classes is not None  # 必须提供类别数
            cdim = tdim                     # 类别嵌入维度与时间嵌入保持一致，方便相加/融合
            self.class_embedding = nn.Embedding(num_classes+1, cdim)  # +1 预留 null 类别（索引 0 作为“无条件”）

        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)  # 头部卷积：RGB(3通道) → 基础通道 ch，保持分辨率

        self.downblocks = nn.ModuleList()   # 下采样路径的模块列表（包含多个 ResBlock 以及 DownSample）
        chs = [ch]                          # 记录每个阶段的通道数（含 head 输出），为上采样时做 skip-connection 使用
        now_ch = ch                         # 追踪当前特征图的通道数（初始化为 head 的输出通道）

        for i, mult in enumerate(ch_mult):  # 遍历每个尺度 stage（由通道倍率 ch_mult 决定）
            out_ch = ch * mult              # 当前 stage 的目标通道数（例如 [1,2,2,2] × ch）
            for _ in range(num_res_blocks): # 在该尺度堆叠若干个 ResBlock
                self.downblocks.append(ResBlock(
                    in_ch=now_ch,           # 残差块输入通道
                    out_ch=out_ch,          # 残差块输出通道（通道调整在 ResBlock 内部完成）
                    tdim=tdim,              # 时间嵌入维度（ResBlock 内部会把 temb 融入）
                    dropout=dropout,        # 残差块内部可能包含 dropout
                    attn=(i in attn)))      # 指定该尺度是否启用自注意力（通常在较低分辨率层加注意力）
                now_ch = out_ch             # 更新当前通道为残差块输出通道
                chs.append(now_ch)          # 记录该层输出通道（为对称的 up 阶段 skip-connection 预留）
            if i != len(ch_mult) - 1:       # 若不是最后一个尺度，添加下采样层以降低分辨率
                self.downblocks.append(DownSample(now_ch))  # 下采样：通常 stride=2 的方式将 H,W 减半
                chs.append(now_ch)          # 下采样后也记录当前通道（分辨率变了但通道没变）

        self.middleblocks = nn.ModuleList([             # U-Net 的 bottleneck（中间层）
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),   # 一个带注意力的 ResBlock（捕获全局依赖）
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),  # 一个不带注意力的 ResBlock（增强表达）
        ])

        self.upblocks = nn.ModuleList()                 # 上采样路径的模块列表
        for i, mult in reversed(list(enumerate(ch_mult))):  # 逆序遍历尺度，从最深层回到最高分辨率
            out_ch = ch * mult                          # 对应下采样时的通道数（对称）
            for _ in range(num_res_blocks + 1):         # +1 是因为上采样阶段每个尺度多一个 block 处理 concat 后的通道
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch,           # 输入通道 = 当前特征 + 对称位置的 skip 特征（通道拼接）
                    out_ch=out_ch,                      # 输出通道对齐当前尺度
                    tdim=tdim,                          # 时间嵌入维度
                    dropout=dropout,                    # dropout 比例
                    attn=(i in attn)))                  # 指定该尺度是否启用注意力
                now_ch = out_ch                         # 更新当前通道
            if i != 0:                                  # 只要不是最高分辨率（最外层），就还需要上采样放大分辨率
                self.upblocks.append(UpSample(now_ch))  # 上采样：通常最近邻/反卷积/插值+卷积等实现
        assert len(chs) == 0                            # 校验所有 skip 记录都已在 up 路径中消费掉

        self.tail = nn.Sequential(                      # 尾部输出层：归一化 + 激活 + 卷积
            nn.GroupNorm(32, now_ch),                   # GroupNorm：对通道做分组归一化（比 BN 更稳定，且与 batch size 无关）
            Swish(),                                    # Swish 激活函数（x * sigmoid(x)）
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)  # 映射回 3 通道（通常输出为预测噪声 ε̂ 或预测图像残差）
        )
        self.initialize()                               # 自定义权重初始化

    def initialize(self):                               # 权重初始化函数
        init.xavier_uniform_(self.head.weight)          # 头部卷积权重 Xavier 均匀初始化
        init.zeros_(self.head.bias)                     # 头部卷积偏置置零
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)  # 尾部最后一层卷积权重较小的增益（稳定早期训练）
        init.zeros_(self.tail[-1].bias)                 # 尾部卷积偏置置零

    def forward(self, x, timestep, class_label=None):   # 前向传播：x 为噪声图/特征，timestep 为时间步，class_label 为可选类别条件
        # Timestep embedding
        temb = self.time_embedding(timestep)            # 将 t 编码成 tdim 维的向量（内部通常是 sin/cos + MLP）

        if self.use_cfg and class_label is not None:    # 若启用 CFG 且提供了类别标签
            if self.training:                           # 训练阶段
                assert not torch.any(class_label == 0)  # 约定：输入的真实标签不允许为 0（0 保留给 null 类）
                
                ######## TODO ########
                # DO NOT change the code outside this part.
                # Assignment 2. Implement random null conditioning in CFG training.
                # 这里应实现：以 cfg_dropout 概率把 class_label 替换为 0（null），
                # 然后把 class_embedding(null_or_label) 与 temb 融合（加法或 MLP 融合）
                # raise NotImplementedError("TODO")
                if self.cfg_dropout > 0:
                    mask = (torch.rand(class_label.shape, device=class_label.device) < self.cfg_dropout).long()
                    class_label = class_label * (1 - mask)  # 以 cfg_dropout 概率将标签置为 0（null）
                class_emb = self.class_embedding(class_label.to(self.device).long())  # (B, cdim)
                temb = temb + class_emb  # 融合方式：简单相加
                #######################
            
            ######## TODO ########
            # DO NOT change the code outside this part.
            # Assignment 2. Implement class conditioning
            # 推理阶段（eval）：通常会分别做有条件和无条件两次前向，再按 w 做线性组合：
            # eps_hat = (1+w)*eps_cond - w*eps_uncond
            # 这里则需要：基于 class_label 查 embedding，和 temb 融合后再继续前向
            #raise NotImplementedError("TODO")
            else:
                assert class_label is not None
                class_emb = self.class_embedding(class_label.to(self.device).long())  # (B, cdim)
                temb = temb + class_emb  # 融合方式：简单相加
                    
            #######################

        # Downsampling
        h = self.head(x)                                # 头部卷积：将输入图像映射到 ch 通道
        hs = [h]                                        # 用列表保存每个下采样 block 的输出（供上采样阶段做 skip-connection）
        for layer in self.downblocks:                   # 依次通过所有下采样层
            h = layer(h, temb)                          # ResBlock/DownSample 接收 (feature, temb)，内部会注入时间信息
            hs.append(h)                                # 每层输出都压入栈（包含 DownSample 的输出），顺序与添加一致

        # Middle
        for layer in self.middleblocks:                 # 通过 bottleneck 的两个残差块
            h = layer(h, temb)                          # 继续注入时间信息

        # Upsampling
        for layer in self.upblocks:                     # 依次通过所有上采样层
            if isinstance(layer, ResBlock):             # 如果是 ResBlock（非 UpSample 模块）
                h = torch.cat([h, hs.pop()], dim=1)     # 与下采样对称位置的特征做 skip-connection（通道维拼接）
            h = layer(h, temb)                          # 通过该层（ResBlock 或 UpSample），继续注入时间信息
        h = self.tail(h)                                # 尾部：归一化 + 激活 + 卷积，输出 3 通道图（通常代表噪声预测）

        assert len(hs) == 0                             # 校验：所有下采样保存的特征都应被消耗完
        return h                                        # 返回输出（训练时作为 ε̂；或视任务而定）



    