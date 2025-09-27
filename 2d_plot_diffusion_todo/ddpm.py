import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(input, t: torch.Tensor, x: torch.Tensor):
    """
    从一维查表张量 `input`（形如 (T,)）中，按批次时间步 `t` 取出对应的元素，
    并把结果 reshape 成能与 `x` 做广播的形状 (B, 1, 1, ...)。

    参数
    ----
    input : torch.Tensor
        查表用的一维张量（例如 betas、alphas_cumprod 等），形状为 (T,)。
    t : torch.Tensor
        每个样本对应的时间步索引，形状通常为 (B,) 或 (B, 1) 或标量。
    x : torch.Tensor
        用来决定广播形状的张量（比如当前的 x_t），形状为 (B, C, ...)

    返回
    ----
    out : torch.Tensor
        形状为 (B, 1, 1, ...) 的张量，能与 `x` 在逐元素运算中自动广播。
    """
    if t.ndim == 0:
        t = t.unsqueeze(0)                   # 标量 t -> (1,)
    shape = x.shape                          # 记录 x 的形状（例如 B×C×H×W）
    t = t.long().to(input.device)            # gather 的索引需要 long；并放到与 input 相同的设备

    # 从一维表 `input` 的第 0 维按索引 t 取值；得到 (B,)（或与 t 的 batch 相同的一维）
    out = torch.gather(input, 0, t)

    # 把 (B,) reshape 成 (B, 1, 1, ..., 1)，长度与 x 维度对齐，方便后续广播
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


class BaseScheduler(nn.Module):
    """
    DDPM 的方差调度器（variance scheduler）。

    主要职责：
    1) 依据给定的步数 T 与调度模式（线性 / 二次）生成 beta_t；
    2) 派生 alpha_t = 1 - beta_t 及它的累计乘积 alpha_bar_t；
    3) 提供 `timesteps`（通常是从 T-1 到 0 的倒序，用于采样循环）；
    4) 以 buffer 形式保存上述表格，便于随着模型 .to(device) 一起迁移且不参与梯度。
    """

    def __init__(
        self,
        num_train_timesteps: int,     # 训练/采样总步数 T
        beta_1: float = 1e-4,         # 初始噪声强度（t=0）
        beta_T: float = 0.02,         # 末端噪声强度（t=T-1）
        mode: str = "linear",         # 调度模式：'linear' 或 'quad'
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps

        # 生成倒序的时间步索引 [T-1, T-2, ..., 0]，常用于采样阶段的反向循环
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        # 生成 beta 调度表：控制每一步注入的噪声强度
        if mode == "linear":
            # 线性从 beta_1 均匀过渡到 beta_T，长度为 T
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            # “平方到线性”的二次调度：先在 sqrt 空间线性，再平方回来，得到前期更小后期更大的 beta
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        # alpha_t = 1 - beta_t
        alphas = 1 - betas
        # alpha_bar_t = ∏_{i=0}^t alpha_i （累乘）
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 以 buffer 形式注册：保存到 state_dict，但不参与梯度，且随 .to(device) 一起迁移
        self.register_buffer("betas", betas)                         # (T,)
        self.register_buffer("alphas", alphas)                       # (T,)
        self.register_buffer("alphas_cumprod", alphas_cumprod)       # (T,)
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """
        前向加噪：x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
        x0: (B, ...)  干净样本
        t : (B,)      每个样本的时间步
        noise: (B, ...) 可选，给定噪声；不传则内部采样
        """
        if noise is None:
         noise = torch.randn_like(x0)

        alpha_bar_t = extract(self.alphas_cumprod, t, x0)          # \bar{α}_t
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt((1.0 - alpha_bar_t).clamp(min=0.0)) * noise
        return xt

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor = None):
        """
        与 q_sample 等价的辅助接口，返回 (x_t, eps)
        方便某些代码同时需要 x_t 和 真实噪声
        """
        if eps is None:
            eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=eps)
        return x_t, eps


class DiffusionModule(nn.Module):
    """
    A high-level wrapper of DDPM and DDIM.
    If you want to sample data based on the DDIM's reverse process, use `ddim_p_sample()` and `ddim_p_sample_loop()`.
    """

    def __init__(self, network: nn.Module, var_scheduler: BaseScheduler):
        super().__init__()
        self.network = network ### predict noise
        self.var_scheduler = var_scheduler

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        # For image diffusion model.
        return getattr(self.network, "image_resolution", None)

    def q_sample(self, x0, t, noise=None):
        """
        sample x_t from q(x_t | x_0) of DDPM.

        Input:
            x0 (`torch.Tensor`): clean data to be mapped to timestep t in the forward process of DDPM.
            t (`torch.Tensor`): timestep
            noise (`torch.Tensor`, optional): random Gaussian noise. if None, randomly sample Gaussian noise in the function.
        Output:
            xt (`torch.Tensor`): noisy samples
        """
        if noise is None:
            noise = torch.randn_like(x0)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Compute xt.
        alphas_prod_t = extract(self.var_scheduler.alphas_cumprod, t, x0)
        xt = torch.sqrt(alphas_prod_t) * x0 + torch.sqrt(1.0- alphas_prod_t) * noise
        #xt = x0*np.sqrt(alphas_prod_t) +noise*(1-alphas_prod_t)

        #######################
        return xt

    @torch.no_grad()
    def p_sample(self, xt, t):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.
        Input:
            xt (`torch.Tensor`): samples at arbitrary timestep t.
            t (`torch.Tensor`): current timestep in a reverse process.
        Ouptut:
            x_t_prev (`torch.Tensor`): one step denoised sample. (= x_{t-1})
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute x_t_prev.
        if isinstance(t, int):
            t = torch.tensor([t]).to(self.device)
        eps_factor = (1 - extract(self.var_scheduler.alphas, t, xt)) / (
            1 - extract(self.var_scheduler.alphas_cumprod, t, xt)
        ).sqrt()

        beta_t      = extract(self.var_scheduler.betas,           t, xt)         # β_t
        alpha_t     = extract(self.var_scheduler.alphas,          t, xt)         # α_t = 1 - β_t
        alpha_bar_t = extract(self.var_scheduler.alphas_cumprod,  t, xt)         # \bar{α}_t
        t_prev      = (t - 1).clamp(min=0)
        alpha_bar_t_prev = extract(self.var_scheduler.alphas_cumprod, t_prev, xt) # \bar{α}_{t-1}

        # 1. predict noise
        eps_hat = self.network(xt, t)
        # 2. Posterior mean
        Posterior_mean= (xt - eps_factor * eps_hat) / torch.sqrt(alpha_t)#(1/torch.sqrt(alpha_t))*(xt- (beta_t )/torch.sqrt(1.0-alpha_t )* eps_hat)
        # 3. Posterior variance
        Posterior_variance=beta_t*(1.0-alpha_bar_t_prev)/(1.0-alpha_bar_t)
        # 4. Reverse step
        nonzero = (t > 0).float().view(xt.size(0), *([1] * (xt.dim() - 1)))
        z = torch.randn_like(xt)
        x_t_prev = Posterior_mean + nonzero * torch.sqrt(Posterior_variance) * z
        
        #######################
        return x_t_prev
# # 加减/开方/乘除都直接算
# one_minus_abar_t = 1.0 - abar_t
# sqrt_abar_t      = torch.sqrt(abar_t)
# sqrt_one_minus   = torch.sqrt((1.0 - abar_t).clamp(min=0.0))

# # 例：DDPM 前向加噪
# x_t = sqrt_abar_t * x0 + sqrt_one_minus * noise

# # 例：后验方差、均值系数
# posterior_var = (1.0 - abar_tm1) / (1.0 - abar_t) * beta_t
# coef_x_t = torch.sqrt(alpha_t) * (1.0 - abar_tm1) / (1.0 - abar_t)
# coef_x0  = torch.sqrt(abar_tm1) * beta_t           / (1.0 - abar_t)
# mu = coef_x_t * x_t + coef_x0 * x0
    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        The loop of the reverse process of DDPM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # sample x0 based on Algorithm 2 of DDPM paper.
        xt = torch.randn(shape).to(self.device)
        x0_pred = None
        for t_scalar in self.var_scheduler.timesteps.tolist():
        # 做成 (B,) 的步数张量
            t = torch.full((xt.size(0),), t_scalar, device=xt.device, dtype=torch.long)
            xt=self.p_sample(xt,t)
        ######################
        x0_pred=xt
        return x0_pred

    @torch.no_grad()
    def ddim_p_sample(self, xt, t, t_prev, eta=0.0):
        """
        One step denoising function of DDIM: $x_t{\tau_i}$ -> $x_{\tau{i-1}}$.

        Input:
            xt (`torch.Tensor`): noisy data at timestep $\tau_i$.
            t (`torch.Tensor`): current timestep (=\tau_i)
            t_prev (`torch.Tensor`): next timestep in a reverse process (=\tau_{i-1})
            eta (float): correspond to η in DDIM which controls the stochasticity of a reverse process.
        Output:
           x_t_prev (`torch.Tensor`): one step denoised sample. (= $x_{\tau_{i-1}}$)
        """
        ######## TODO ########
        # NOTE: This code is used for assignment 2. You don't need to implement this part for assignment 1.
        # DO NOT change the code outside this part.
        # compute x_t_prev based on ddim reverse process.
        alpha_prod_t = extract(self.var_scheduler.alphas_cumprod, t, xt)
        if t_prev >= 0:
            alpha_prod_t_prev = extract(self.var_scheduler.alphas_cumprod, t_prev, xt)
        else:
            alpha_prod_t_prev = torch.ones_like(alpha_prod_t)

        x_t_prev = xt

        ######################
        return x_t_prev

    @torch.no_grad()
    def ddim_p_sample_loop(self, shape, num_inference_timesteps=50, eta=0.0):
        """
        The loop of the reverse process of DDIM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
            num_inference_timesteps (`int`): the number of timesteps in the reverse process.
            eta (`float`): correspond to η in DDIM which controls the stochasticity of a reverse process.
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """
        ######## TODO ########
        # NOTE: This code is used for assignment 2. You don't need to implement this part for assignment 1.
        # DO NOT change the code outside this part.
        # sample x0 based on Algorithm 2 of DDPM paper.
        step_ratio = self.var_scheduler.num_train_timesteps // num_inference_timesteps
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        timesteps = torch.from_numpy(timesteps)
        prev_timesteps = timesteps - step_ratio

        xt = torch.zeros(shape).to(self.device)
        for t, t_prev in zip(timesteps, prev_timesteps):
            pass

        x0_pred = xt

        ######################

        return x0_pred

    def compute_loss(self, x0):
        """
        The simplified noise matching loss corresponding Equation 14 in DDPM paper.
        Input:
            x0 (`torch.Tensor`): clean data
        Output:
            loss: the computed loss to be backpropagated.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        batch_size = x0.shape[0]
        
        # 1) random choose timestep
        t = (
            torch.randint(0, self.var_scheduler.num_train_timesteps, size=(batch_size,))
            .to(x0.device)
            .long()
        )
        # 2) get GT noise, and use q_sample to get x_t
        eps = torch.randn_like(x0)                                 # 真噪声 ε ~ N(0, I)
        xt  = self.var_scheduler.q_sample(x0, t, noise=eps)
        # 3) predict noise 
        eps_pred=self.network(xt,t)
        # 4) MSE loss (eps, eps_pred)
        
        loss = F.mse_loss(eps_pred, eps)

        ######################
        return loss

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
