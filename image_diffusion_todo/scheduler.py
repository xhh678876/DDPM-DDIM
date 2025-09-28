from typing import Optional, Union  # 导入类型注解：Optional 表示可为 None，Union 表示联合类型（本文件中未直接用到 Union）

import numpy as np                 # 数值计算库，这里主要用于构造时间步数组与随机采样
import torch                       # PyTorch 主库
import torch.nn as nn              # 神经网络模块基类与层定义（Module、Parameter、层等）

def extract(input, t: torch.Tensor, x: torch.Tensor):
    # 从一维张量 `input`（通常是长度为 T 的调度表，如 betas/alphas/alphas_cumprod）
    # 中按批次索引 t（形状为 (B,) 或标量）提取对应元素，并 reshape 成 (B,1,1,1) 便于后续广播
    if t.ndim == 0:
        t = t.unsqueeze(0)         # 若 t 是标量，升维成 (1,) 以统一处理
    shape = x.shape                # 记录目标张量 x 的形状，用于确定返回的广播形状
    t = t.long().to(input.device)  # 索引必须为 long 且与 input 在同一设备上
    out = torch.gather(input, 0, t)# 从 `input` 的第 0 维按索引 t 收集元素，返回形状 (B,)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)  # 生成 (B,1,1,1,...) 的目标形状，与 x 的维度数对齐
    return out.reshape(*reshape)   # 返回 (B,1,1,1,...)，便于与 (B,C,H,W) 广播运算


class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear"
    ):
        super().__init__()                         # 初始化 nn.Module
        self.num_train_timesteps = num_train_timesteps       # 训练使用的时间步数（例如 1000）
        self.num_inference_timesteps = num_train_timesteps   # 推理默认与训练相同（可按需修改）
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )                                          # 构造反向时间序列张量：T-1, T-2, ..., 0（用于反向采样循环）

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)  # 线性 beta 调度，从 beta_1 均匀过渡到 beta_T
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )                                      # 二次（平方）调度：对 sqrt(beta) 做线性，再平方回去
        elif mode == "cosine":
            ######## TODO ########
            # Implement the cosine beta schedule (Nichol & Dhariwal, 2021).
            # Hint:
            # 1. Define alphā_t = f(t/T) where f is a cosine schedule:
            #       alphā_t = cos^2( ( (t/T + s) / (1+s) ) * (π/2) )
            #    with s = 0.008 (a small constant for stability).
            # 2. Convert alphā_t into betas using:
            #       beta_t = 1 - alphā_t / alphā_{t-1}
            # 3. Return betas as a tensor of shape [num_train_timesteps].
            s = 0.008                               # 论文中的平移量 s，避免 t=0 时过小
            # 注意：按原论文常见实现，会构造长度 T+1 的 ᾱ 表（含 t=0..T），再由其得到长度 T 的 β
            t = torch.arange(0, num_train_timesteps + 1, dtype=torch.float32)
            f_t = torch.cos(((t / num_train_timesteps + s) / (1 + s)) * (np.pi / 2)) ** 2
            alphas_cumprod_full = f_t / f_t[0]      # 归一化，使 ᾱ_0 = 1
            # β_t = 1 - ᾱ_t / ᾱ_{t-1}  for t=1..T；对齐为长度 T
            betas = 1.0 - (alphas_cumprod_full[1:] / alphas_cumprod_full[:-1])
            # 数值稳定裁剪
            betas = torch.clamp(betas, 1e-8, 0.9999)
            #######################
        else:
            raise NotImplementedError(f"{mode} is not implemented.")  # 不支持的调度模式直接报错

        alphas = 1 - betas                      # α_t = 1 - β_t
        alphas_cumprod = torch.cumprod(alphas, dim=0)
                                                # ᾱ_t = ∏_{s=0}^t α_s（从 0 到 t 的累乘）

        self.register_buffer("betas", betas)    # 以 buffer 形式注册到模块（随模型保存/加载，但非参数）
        self.register_buffer("alphas", alphas)  # 同上，保存 α_t 表
        self.register_buffer("alphas_cumprod", alphas_cumprod)
                                                # 保存 ᾱ_t 表

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
                                                # 从 [0, T) 均匀随机采样 batch_size 个时间步（numpy 实现）
        ts = torch.from_numpy(ts)               # 转为 torch 张量（默认在 CPU）
        if device is not None:
            ts = ts.to(device)                  # 若指定设备则拷贝到对应 device
        return ts                                # 返回形状 (B,) 的整型时间步索引


class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)  # 先构造基础表（betas/alphas/alphas_cumprod）
        
        self.schedule_mode = mode       # 记录所用的调度模式，便于调试或分支控制

        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type    # 噪声标准差的类型选择："small" 或 "large"，对应不同后验方差设定
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            # “小方差”设定：σ_t^2 取真实后验方差 \tilde{β}_t = β_t * (1-ᾱ_{t-1})/(1-ᾱ_t)
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0], device=self.alphas_cumprod.device, dtype=self.alphas_cumprod.dtype),
                 self.alphas_cumprod[:-1]]
            )                               # 形成 ᾱ_{t-1} 序列，且定义 ᾱ_{-1}=1 放在开头对齐索引
            tilde_betas = (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            sigmas = torch.sqrt(torch.clamp(tilde_betas, min=1e-20))  # 数值稳定
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = torch.sqrt(torch.clamp(self.betas, min=1e-20))   # “大方差”设定：σ_t^2 直接等于 β_t
        else:
            raise ValueError(f"Unknown sigma_type: {sigma_type}")

        self.register_buffer("sigmas", sigmas)  # 注册 σ_t 表为 buffer，采样时按 t 取用

    
    def step(self, x_t: torch.Tensor, t: int, net_out: torch.Tensor, predictor: str):
        # 根据 predictor 类型决定如何解释网络输出 net_out，并执行对应的一步反向更新
        if predictor == "noise": #### TODO
            return self.step_predict_noise(x_t, t, net_out)  # 视 net_out 为噪声 ε̂_θ
        elif predictor == "x0": #### TODO
            return self.step_predict_x0(x_t, t, net_out)     # 视 net_out 为 x̂₀
        elif predictor == "mean": #### TODO
            return self.step_predict_mean(x_t, t, net_out)   # 视 net_out 为后验均值 μ̂_θ
        else:
            raise ValueError(f"Unknown predictor: {predictor}")  # 未知 predictor 直接报错

    
    def step_predict_noise(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        Noise prediction version (the standard DDPM formulation).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            eps_theta: predicted noise ε̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # 1. Extract beta_t, alpha_t, and alpha_bar_t from the scheduler.
        # 2. Compute the predicted mean μ_θ(x_t, t) = 1/√α_t * (x_t - (β_t/√(1-ᾱ_t)) * ε̂_θ).
        # 3. Compute the posterior variance \tilde{β}_t = ((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t.
        # 4. Add Gaussian noise scaled by √(\tilde{β}_t) unless t == 0.
        # 5. Return the final sample at t-1.
        beta_t      = self._get_teeth(self.betas,          t, x_like=x_t)
        alpha_t     = self._get_teeth(self.alphas,         t, x_like=x_t)
        alpha_bar_t = self._get_teeth(self.alphas_cumprod, t, x_like=x_t)

        # 处理 t-1 的边界（t 可能是 int 或 Tensor）
        if isinstance(t, int):
            t_prev = max(t - 1, 0)
        else:
            t_prev = (t - 1).clamp(min=0)

        alpha_bar_t_prev = self._get_teeth(self.alphas_cumprod, t_prev, x_like=x_t)

        # μ_θ 公式
        mean_theta = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-20))) * eps_theta
        )

        # 后验方差 \tilde{β}_t
        posterior_variance = torch.clamp(
            ((1.0 - alpha_bar_t_prev) / torch.clamp(1.0 - alpha_bar_t, min=1e-20)) * beta_t,
            min=0.0
        )

        if (isinstance(t, int) and t > 0) or (torch.is_tensor(t) and torch.any(t > 0)):
            noise = torch.randn_like(x_t)
            sample_prev = mean_theta + torch.sqrt(torch.clamp(posterior_variance, min=1e-20)) * noise
        else:
            sample_prev = mean_theta
        #######################
        return sample_prev                 # 返回 x_{t-1}

    
    def step_predict_x0(self, x_t: torch.Tensor, t: int, x0_pred: torch.Tensor):
        """
        x0 prediction version (alternative DDPM objective).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            x0_pred: predicted clean image x̂₀(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # 思路：用 x̂₀ 与 x_t 计算 μ_θ 的等价式（或直接用标准式中的替代项），再与 posterior variance 组合加噪（t>0）
        # 常见公式（DDPM 小方差设定下的均值表达）：
        #   μ_θ = ( √ᾱ_{t-1} * β_t / (1-ᾱ_t) ) * x̂₀ + ( √α_t * (1-ᾱ_{t-1}) / (1-ᾱ_t) ) * x_t
        #   \tilde{β}_t 同上小方差定义
        alpha_t     = self._get_teeth(self.alphas,         t, x_like=x_t)
        alpha_bar_t = self._get_teeth(self.alphas_cumprod, t, x_like=x_t)

        if isinstance(t, int):
            t_prev = max(t - 1, 0)
        else:
            t_prev = (t - 1).clamp(min=0)
        alpha_bar_t_prev = self._get_teeth(self.alphas_cumprod, t_prev, x_like=x_t)
        beta_t           = self._get_teeth(self.betas,           t, x_like=x_t)

        one_minus_alpha_bar_t = torch.clamp(1.0 - alpha_bar_t, min=1e-20)

        mean_theta = (
            (torch.sqrt(alpha_bar_t_prev) * beta_t / one_minus_alpha_bar_t) * x0_pred
            + (torch.sqrt(alpha_t) * (1.0 - alpha_bar_t_prev) / one_minus_alpha_bar_t) * x_t
        )

        # 对应 small 方差：σ_t^2 = \tilde{β}_t；large 方差：σ_t^2 = β_t
        sigma_t = self._get_teeth(self.sigmas, t, x_like=x_t)

        if (isinstance(t, int) and t > 0) or (torch.is_tensor(t) and torch.any(t > 0)):
            sample_prev = mean_theta + sigma_t * torch.randn_like(x_t)
        else:
            sample_prev = mean_theta
        
        #######################
        return sample_prev

    
    def step_predict_mean(self, x_t: torch.Tensor, t: int, mean_theta: torch.Tensor):
        """
        Mean prediction version (directly outputting the posterior mean).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            mean_theta: network-predicted posterior mean μ̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # 1. Extract beta_t, alpha_t, and alpha_bar_t from the scheduler.
        # 2. Use the predicted mean μ̂_θ(x_t, t) as is.
        # 3. Add Gaussian noise scaled by √(\tilde{β}_t) unless t == 0.
        # 4. Return the final sample at t-1.

        # 这里不再直接用标量索引，统一用安全取表，且支持 int / Tensor 的 t
        alpha_bar_t = self._get_teeth(self.alphas_cumprod, t, x_like=x_t)

        if isinstance(t, int):
            t_prev = max(t - 1, 0)
        else:
            t_prev = (t - 1).clamp(min=0)
        alpha_bar_t_prev = self._get_teeth(self.alphas_cumprod, t_prev, x_like=x_t)
        beta_t           = self._get_teeth(self.betas,           t,     x_like=x_t)

        # 使用 small 方差的 \tilde{β}_t 作为后验噪声（与 DDPM 论文一致；large 情况应改为 β_t）
        posterior_variance = torch.clamp(
            ((1.0 - alpha_bar_t_prev) / torch.clamp(1.0 - alpha_bar_t, min=1e-20)) * beta_t,
            min=1e-20
        )

        if (isinstance(t, int) and t > 0) or (torch.is_tensor(t) and torch.any(t > 0)):
            eps = torch.randn_like(x_t)  # 采样标准高斯噪声 z ~ N(0,I)
            sample_prev = mean_theta + torch.sqrt(posterior_variance) * eps
        else:
            sample_prev = mean_theta

        return sample_prev                 # 返回 x_{t-1}
        # ⚠️ 若 t 为批量索引（形如 (B,)），以上取表均已与其他函数一致使用 _get_teeth() 批量提取
       

    # https://nn.labml.ai/diffusion/ddpm/utils.html
    # 记得在文件顶部：import torch

    def _get_teeth(self, consts: torch.Tensor, t, x_like: torch.Tensor = None):
        """
        从 1D 表 consts (shape [T]) 中按索引 t 取值。
        - consts: 1D tensor [T]
        - t: int、list、ndarray 或 Tensor（标量或 shape [B]）
        - x_like: 若提供，将输出 reshape 为 [B, 1, 1, ...] 方便后续广播到 x_like
        返回：
        - 若提供 x_like: shape [B, 1, 1, ...]
        - 否则：与 t 的 shape 对齐（标量或 [B]）
        """
        # 统一 consts 为 1D
        if consts.dim() != 1:
            consts = consts.reshape(-1)

        device = consts.device

        # 把 t 规范成 LongTensor，且在同一 device
        if not torch.is_tensor(t):
            t = torch.as_tensor(t, device=device)
        else:
            if t.device != device:
                t = t.to(device)

        if t.dtype != torch.long:
            t = t.long()

        # 展平成 1D，便于 gather 统一处理
        t_flat = t.view(-1)

        # clamp 到合法范围 [0, T-1]
        T = consts.shape[0]
        t_flat = torch.clamp(t_flat, 0, T - 1)

        # 从 dim=0 索引
        out = consts.gather(0, t_flat)  # [N]，其中 N = t 元素数

        # 只有当 t 不是标量时，按 t 的形状还原
        if t.dim() > 0:
            out = out.view(*t.shape)  # [B] 等

        # 如果需要广播到 x_like，就 reshape 成 [B, 1, 1, ...]
        if x_like is not None:
            B = x_like.shape[0]

            # 关键修复：只要元素个数为 1（标量或 [1]），先扩展到 [B]
            if out.numel() == 1:
                out = out.expand(B)
            elif out.dim() == 1 and out.shape[0] != B:
                # 防御性兜底：极少见的不匹配情况
                out = out.reshape(-1)[:1].expand(B)

            return out.reshape(B, *([1] * (x_like.dim() - 1)))

        return out





    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        if eps is None:
            eps       = torch.randn(x_0.shape, device=x_0.device)
                                           # 若未提供噪声，则在 CUDA 上采样一个与 x_0 同形状的高斯噪声
                                           # ⚠️ 若 x_0 不在 cuda 设备，可能引起 device 不一致；通常建议用 device=x_0.device

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        # 1) Extract ᾱ_t from the scheduler.
        # 2) Compute x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε          
        # 3) Add noise: x_t = x_t + √(1 - ᾱ_t) * ε
        # （说明：第 2）式已经包含噪声项，因此不要再重复“加噪一次”，否则会变为两次高斯注入）
        alpha_bar_t = self._get_teeth(self.alphas_cumprod, t, x_like=x_0)
                                           # 安全取表，支持 int/Tensor，返回 (B,1,1,1) 便于广播
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-20)) * eps
                                           # 标准 q(x_t|x_0) 公式：一次性构造（不再额外加噪）
        #######################

        return x_t, eps                     # 返回前向加噪后的 x_t 与所用噪声 eps
