from typing import Optional  # Optional 类型注解（可选参数）

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm                  # 进度条显示
from scheduler import extract          # 从调度器中根据 t 抽取系数的工具函数（通常按 batch 对应索引做 gather）

class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network                     # 噪声预测网络（一般是时间条件化 U-Net）
        self.var_scheduler = var_scheduler         # 扩散/反向去噪的调度器（含 betas、alphas_bar、step() 等）
        self.predictor = kwargs.get("predictor", "noise")  # 训练/采样时网络的预测目标："noise" / "x0" / "mean"

    def get_loss_noise(self, x0, class_label=None, noise=None):
        B = x0.shape[0]                                             # batch size
        t = self.var_scheduler.uniform_sample_t(B, x0.device)       # (B,) 均匀采样时间步 t（训练时随机采样）
        x_t, eps = self.var_scheduler.add_noise(x0, t, eps=noise)   # 前向加噪：得到 x_t 和真实噪声 eps（可传入固定 noise 用于可重复性）
        # 条件或无条件前向：根据是否提供 class_label 决定是否把标签传进网络
        eps_pred = self.network(x_t, t, class_label) if class_label is not None else self.network(x_t, t)
        return F.mse_loss(eps_pred, eps)                            # 经典 DDPM 训练损失：预测噪声的 MSE

    def get_loss_x0(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # "predict x0" 版本的训练：
        B = x0.shape[0]    
        t = self.var_scheduler.uniform_sample_t(B, x0.device)
        x_t, eps = self.var_scheduler.add_noise(x0, t, eps=noise)
        x0_pred = self.network(x_t, t, class_label) if class_label is not None else self.network(x_t, t)
        loss = F.mse_loss(x0_pred, x0)
        
        # 1) 采样时间步 t，并调用 var_scheduler.add_noise(x0, t) 得到 (x_t, eps)。
        # 2) 前向：net_out = self.network(x_t, t, class_label?)，此时网络输出应代表 \hat{x}_0。
        # 3) 用 MSE(\hat{x}_0, x0) 作为损失。
        ######################
        
        return loss

    def get_loss_mean(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # "predict mean" 版本的训练：
        # 1) 同样采样 t，并得到 (x_t, eps)。
        # 2) 前向：预测的是后验均值 \mu_\theta(x_t, t)（网络直接回归 mean）。
        # 3) 用 DDPM 的闭式表达式计算真实后验均值 \mu_{\text{true}}(x_t, t)：
        #    \mu_{\text{true}} = (1/sqrt(alpha_t)) * (x_t - ((1 - alpha_t)/sqrt(1 - \bar{alpha}_t)) * eps)
        #    或者从 scheduler 提供的参数化式中得到（注意你用的是离散索引版）。
        # 4) 损失：MSE(\mu_\theta, \mu_{\text{true}})。
        B = x0.shape[0]    
        t = self.var_scheduler.uniform_sample_t(B, x0.device)
        x_t, eps = self.var_scheduler.add_noise(x0, t, eps=noise)
        mean_pred = self.network(x_t, t, class_label) if class_label is not None else self.network(x_t, t)
        alpha_t = self.var_scheduler.alphas[t].reshape(-1, 1, 1, 1)
        alpha_bar_t = self.var_scheduler.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        mean_true = (1.0 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * eps)
        loss = F.mse_loss(mean_pred, mean_true)
        ######################
        
        return loss

    def get_loss(self, x0, class_label=None, noise=None):
        # 根据 self.predictor 路由到不同训练目标
        if self.predictor == "noise":
            return self.get_loss_noise(x0, class_label, noise)
        elif self.predictor == "x0":
            return self.get_loss_x0(x0, class_label, noise)
        elif self.predictor == "mean":
            return self.get_loss_mean(x0, class_label, noise)
        else:
            raise ValueError(f"Unknown predictor: {self.predictor}")  # 非法 predictor 名称

    @property
    def device(self):
        # 便捷属性：返回当前网络所在的设备（通过任意参数的 device 推断）
        return next(self.network.parameters()).device

    @torch.no_grad()  # 采样阶段不需要梯度
    def sample(
        self,
        batch_size,
        return_traj=False,                            # 是否返回整个轨迹（x_T → x_{T-1} → ... → x_0），用于可视化或调试
        class_label: Optional[torch.Tensor] = None,   # (B,) 条件标签（若模型支持条件生成）
        guidance_scale: Optional[float] = 1.0,        # CFG 的引导强度，>1.0 表示使用 classifier-free guidance
    ):
        # 初始噪声：x_T ~ N(0, I)，形状为 (B, 3, H, W)
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        # 当 guidance_scale > 1.0 时，启用 CFG（需要网络/权重支持无条件分支）
        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:
            ######## TODO ########
            # CFG 的 batch 扩充与 null 条件构造：
            # 假设输入 class_label 形状是 (B,) 且取值 1..C（0 预留给 null）。
            # 目标：构造 (2B,) 的标签向量，其中前 B 为全 0（null），后 B 为原始 class_label。
            # 然后在每个时间步，把 x_t 也复制成 (2B, ...) 堆叠，一次性前向得到 uncond/cond 两个预测，再按 scale 组合。
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            # 不要在这里扩大 x_T；仅准备好两路标签，供后面每个时间步临时拼接使用
            class_label = class_label.to(self.device).long()
            cfg_null_label = torch.zeros_like(class_label, device=self.device)      # (B,) 全 0 作为 null 条件
            cfg_class_label = class_label                                           # (B,) 原始条件
            #######################

        traj = [x_T]  # 记录采样轨迹（把 x_T 放进去，后续每步追加 x_{t-1}）
        for t in tqdm(self.var_scheduler.timesteps):  # 反向时间步循环（例如 t: T-1 → 0）
            x_t = traj[-1]                            # 当前步的样本 x_t
            t_scalar = t.item() if torch.is_tensor(t) else t  # 确保 t 是标量
            
            # Convert scalar t to tensor for network input
            t_tensor = torch.tensor([t_scalar], device=self.device, dtype=torch.long)

            if do_classifier_free_guidance:
                ######## TODO ########
                # 1) 构造 x_t_cat = cat([x_t, x_t], dim=0) → 形状 (2B, 3, H, W)
                # 2) 构造 label_cat：前 B 为 0（null），后 B 为 class_label
                # 3) 前向：net_out_cat = self.network(x_t_cat, t, label_cat)
                #    拆分：eps_uncond, eps_cond = net_out_cat[:B], net_out_cat[B:]
                # 4) 组合（以预测噪声为例）：eps_hat = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                #    如果 predictor 不是 "noise"，需要按对应参数化把 cond/uncond 结果做同构组合。
                x_t_cat = torch.cat([x_t, x_t], dim=0)                                   # (2B,3,H,W)
                label_cat = torch.cat([cfg_null_label, cfg_class_label], dim=0)          # (2B,)
                net_out_cat = self.network(x_t_cat, timestep=t_tensor, class_label=label_cat)
                out_uncond, out_cond = net_out_cat.chunk(2, dim=0)                       # 各 (B,...)
                net_out = out_uncond + guidance_scale * (out_cond - out_uncond)          # 统一组合公式
                #######################
            else:
                # 非 CFG 路径：若提供 class_label 则走条件分支，否则无条件
                if class_label is not None:
                    net_out = self.network(x_t, timestep=t_tensor, class_label=class_label.to(self.device).long())
                else:
                    net_out = self.network(x_t, timestep=t_tensor)

            # 由调度器执行一步反向更新：x_{t-1} = step(x_t, t, net_out, predictor=...)
            # step 内部会根据 predictor 类型解释 net_out（噪声 / x0 / mean）并用相应的闭式公式推进
            x_t_prev = self.var_scheduler.step(x_t, t_scalar, net_out, predictor=self.predictor)

            # 为了节省显存，把上一时刻存入 CPU（如果你要可视化完整轨迹，这样做更省 GPU 显存）
            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())  # 追加新的样本（detach 避免无谓的 autograd 图）

        # 根据需要返回最终样本或整个轨迹
        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        # 保存模型：连同 hparams（网络与调度器对象）以及 state_dict 一起保存
        hparams = {
            "network": self.network,              # 直接保存对象引用（注意：这要求对象可被 torch.save 序列化）
            "var_scheduler": self.var_scheduler,
        }
        state_dict = self.state_dict()            # 模块参数字典

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)                # 序列化到 file_path
    
    @property
    def image_resolution(self):
        # 优先从底层网络取；如果网络还没挂上（例如先构造再load），就回退到 _image_resolution
        if hasattr(self, "network") and self.network is not None and hasattr(self.network, "image_resolution"):
            return self.network.image_resolution
        if hasattr(self, "_image_resolution") and self._image_resolution is not None:
            return self._image_resolution
        raise AttributeError("image_resolution is not set; please pass one via network.image_resolution or _image_resolution")

    @property
    def device(self):
        # 便捷地拿到当前网络所在设备；若网络还没就绪则回退到 CPU
        try:
            return next(self.network.parameters()).device
        except Exception:
            return torch.device("cpu")

    def load(self, file_path):
        # 加载模型：从文件中恢复 hparams 与参数
        dic = torch.load(file_path, map_location="cpu")  # 先加载到 CPU，避免无卡环境报错
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]               # 恢复网络对象（注意：这要求保存时以对象方式保存）
        self.var_scheduler = hparams["var_scheduler"]   # 恢复调度器对象
        self.load_state_dict(state_dict)                # 恢复参数权重到当前模块
