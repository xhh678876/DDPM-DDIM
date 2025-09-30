import argparse                      # 用于命令行参数解析，支持 --arg 的形式从命令行传参数
import numpy as np                   # 数值计算库，用来做数学计算（这里主要用于 np.ceil 向上取整）
import torch                         # PyTorch，用于深度学习模型和张量计算
from pathlib import Path              # 处理文件和路径的跨平台库
from dataset import tensor_to_pil_image  # 工具函数：把 Tensor 转换为 PIL.Image 以便保存
from model import DiffusionModule     # 扩散模型的封装类，里面包含 UNet 和采样逻辑
from scheduler import DDPMScheduler   # 扩散调度器，定义 beta 序列（噪声调度方式）


# def main(args):
#     # ============ 0. 设备/随机数种子 ============
#     device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
#     torch.set_grad_enabled(False)

#     # ============ 1. 保存目录 ============
#     save_dir = Path(args.save_dir)
#     save_dir.mkdir(exist_ok=True, parents=True)

#     # ============ 2. 加载模型（注意：model.py 的 load 已修复 weights_only） ============
#     ddpm = DiffusionModule(None, None)     # 保持你工程的构造方式
#     ddpm.load(args.ckpt_path)              # 建议按我之前的做法：先 safe_globals 再兜底 weights_only=False
#     ddpm.eval().to(device)

#     # ============ 3. predictor/mode 对齐（从 ckpt 路径里“猜提示”） ============
#     # 例如 .../results/predictor_x0/beta_quad/.../last.ckpt
#     ckpt_hint = str(args.ckpt_path).lower()
#     hint_predictor = None
#     if "predictor_noise" in ckpt_hint or "/noise/" in ckpt_hint:
#         hint_predictor = "noise"
#     elif "predictor_x0" in ckpt_hint or "/x0/" in ckpt_hint:
#         hint_predictor = "x0"
#     elif "predictor_mean" in ckpt_hint or "/mean/" in ckpt_hint:
#         hint_predictor = "mean"

#     hint_mode = None
#     if "beta_linear" in ckpt_hint or "mode_linear" in ckpt_hint:
#         hint_mode = "linear"
#     elif "beta_cos" in ckpt_hint or "beta_cosine" in ckpt_hint or "mode_cosine" in ckpt_hint:
#         hint_mode = "cosine"
#     elif "beta_quad" in ckpt_hint or "mode_quad" in ckpt_hint:
#         hint_mode = "quad"

#     # 如果用户显式传了，就以命令行为准；否则用 ckpt 提示（避免错配）
#     predictor = args.predictor if args.predictor else (hint_predictor or "noise")
#     mode = args.mode if args.mode else (hint_mode or "linear")

#     if hint_predictor and predictor != hint_predictor:
#         print(f"[WARN] --predictor={predictor} 与 ckpt 提示 {hint_predictor} 不一致，已使用命令行参数。")
#     if hint_mode and mode != hint_mode:
#         print(f"[WARN] --mode={mode} 与 ckpt 提示 {hint_mode} 不一致，已使用命令行参数。")

#     ddpm.predictor = predictor

#     # ============ 4. 重新构建调度器（与训练 T 一致） ============
#     # 取训练时的 T；若模型里没有就用命令行/默认
#     T = getattr(getattr(ddpm, "var_scheduler", None), "num_train_timesteps", None)
#     if T is None:
#         T = 1000
#         print(f"[INFO] 未从 ckpt 得到训练步数，使用默认 T={T}")

#     var_sched = DDPMScheduler(
#         T,
#         beta_1=args.beta_1,
#         beta_T=args.beta_T,
#         mode=mode,
#     )
#     # 有些 scheduler 不是 nn.Module 没有 .to；这里做个保护
#     if hasattr(var_sched, "to"):
#         var_sched = var_sched.to(device)
#     ddpm.var_scheduler = var_sched

#     # ============ 5. 采样循环 ============
#     total_num_samples = getattr(args, "num_samples", None) or 500
#     num_batches = int(np.ceil(total_num_samples / args.batch_size))

#     print(f"[INFO] Sampling: predictor={predictor}, mode={mode}, "
#           f"T={T}, beta_1={args.beta_1}, beta_T={args.beta_T}, "
#           f"num_samples={total_num_samples}, batch_size={args.batch_size}")

#     for i in range(num_batches):
#         sidx = i * args.batch_size
#         eidx = min(sidx + args.batch_size, total_num_samples)
#         B = eidx - sidx

#         # 可选：启用 autocast（若你没用混精度，注释掉也行）
#         # with torch.autocast(device_type=device.type, dtype=torch.float16 if device.type=="cuda" else torch.bfloat16):
#         if args.use_cfg:
#             assert getattr(ddpm.network, "use_cfg", False), \
#                 "This checkpoint wasn't trained with CFG; 请去掉 --use_cfg 或换支持 CFG 的 ckpt。"
#             # 这里假设类别数为 3（按你现在的写法）。如果不确定，可以把 num_classes 存到 ckpt 里并读取。
#             class_label = torch.randint(0, 3, (B,), device=device)
#             samples = ddpm.sample(
#                 B,
#                 class_label=class_label,
#                 guidance_scale=args.cfg_scale,
#                 sample_method=args.sample_method,
#             )
#         else:
#             samples = ddpm.sample(
#                 B,
#                 sample_method=args.sample_method,
#             )

#         # 强制转 CPU 再喂保存函数，避免某些 PIL/transform 在 GPU 张量上出问题
#         samples = samples.detach().to("cpu")

#         for j, img in zip(range(sidx, eidx), tensor_to_pil_image(samples)):
#             img.save(save_dir / f"{j}.png")
#         print(f"[INFO] Saved images [{sidx}, {eidx})")

#     print(f"[DONE] All images saved to: {save_dir}")
def main(args):
    

    def save_tensor_png(x: torch.Tensor, out_path: Path):
        """
        兜底保存：支持 (C,H,W)/(H,W)/(H,W,C)，值域 [-1,1] 或 [0,1]。
        """
        x = x.detach().to("cpu").float()
        if x.ndim == 3 and x.shape[0] in (1, 3, 4):   # (C,H,W)
            # 统一到 [0,1]
            if x.min() < 0:
                x = (x.clamp(-1, 1) + 1) / 2.0
            else:
                x = x.clamp(0, 1)
            x = x[:3]
            x = x.permute(1, 2, 0)  # (H,W,C)
        elif x.ndim == 2:  # (H,W)
            x = x.unsqueeze(-1)     # (H,W,1)
        elif x.ndim == 3 and x.shape[-1] in (1, 3, 4):  # (H,W,C)
            if x.min() < 0:
                x = (x.clamp(-1, 1) + 1) / 2.0
            else:
                x = x.clamp(0, 1)
            x = x[..., :3]
        else:
            raise RuntimeError(f"Unexpected image tensor shape: {tuple(x.shape)}")

        x = (x.clamp(0, 1) * 255.0).round().to(torch.uint8).numpy()
        if x.ndim == 3 and x.shape[-1] == 3:
            img = Image.fromarray(x, mode="RGB")
        elif x.ndim == 3 and x.shape[-1] == 1:
            img = Image.fromarray(x[..., 0], mode="L")
        elif x.ndim == 2:
            img = Image.fromarray(x, mode="L")
        else:
            raise RuntimeError(f"Unexpected array shape: {x.shape}")
        img.save(out_path)

    # ============ 0. 设备 ============
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    # ============ 1. 保存目录 ============
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"[INFO] save_dir = {save_dir.resolve()}", flush=True)

    # ============ 2. 加载模型 ============
    ddpm = DiffusionModule(None, None)
    print(f"[INFO] loading ckpt: {args.ckpt_path}", flush=True)
    ddpm.load(args.ckpt_path)  # 确保 model.py 的 load 已处理 weights_only/safe_globals
    ddpm.eval().to(device)
    print(f"[INFO] model loaded. device={device}", flush=True)

    # ============ 3. predictor / scheduler ============
    ddpm.predictor = args.predictor
    print(f"[INFO] predictor = {ddpm.predictor}", flush=True)

    T = getattr(getattr(ddpm, "var_scheduler", None), "num_train_timesteps", None)
    if T is None:
        T = 1000
        print(f"[WARN] 未从 ckpt 读取到训练步数，默认 T={T}", flush=True)

    ddpm.var_scheduler = DDPMScheduler(
        T, beta_1=args.beta_1, beta_T=args.beta_T, mode=args.mode
    )
    if hasattr(ddpm.var_scheduler, "to"):
        ddpm.var_scheduler = ddpm.var_scheduler.to(device)
    print(f"[INFO] scheduler: mode={args.mode}, T={T}, beta_1={args.beta_1}, beta_T={args.beta_T}", flush=True)

    # ============ 4. 采样批次数 ============
    total_num_samples = args.num_samples
    num_batches = int(np.ceil(total_num_samples / args.batch_size))
    print(f"[INFO] total_num_samples={total_num_samples}, batch_size={args.batch_size}, num_batches={num_batches}", flush=True)

    # ============ 5. 采样循环 ============
    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx
        print(f"[INFO] batch {i+1}/{num_batches}: sidx={sidx}, eidx={eidx}, B={B}", flush=True)

        try:
            if args.use_cfg:
                assert getattr(ddpm.network, "use_cfg", False), "This checkpoint wasn't trained with CFG."
                class_label = torch.randint(0, 3, (B,), device=device)  # 如有 num_classes，可从 ckpt 读
                samples = ddpm.sample(
                    B,
                    class_label=class_label,
                    guidance_scale=args.cfg_scale,
                )
            else:
                samples = ddpm.sample(B)
        except Exception as e:
            print(f"[ERROR] ddpm.sample() failed: {repr(e)}", flush=True)
            raise

        if not isinstance(samples, torch.Tensor):
            raise TypeError(f"samples is not a Tensor: {type(samples)}")
        if samples.ndim == 3:
            samples = samples.unsqueeze(0)  # 兼容 (C,H,W)
        if samples.ndim != 4:
            raise RuntimeError(f"Expect samples 4D (B,C,H,W), got {samples.ndim}D")
        if samples.shape[0] != B:
            print(f"[WARN] samples.shape[0] != B ({samples.shape[0]} != {B})", flush=True)

        # —— 关键：先搬 CPU&float，再把值域规范到 [0,1] ——
        samples = samples.detach().to("cpu").float()
        if samples.min() < 0:
            samples = (samples.clamp(-1, 1) + 1) / 2.0
        else:
            samples = samples.clamp(0, 1)

        # 先试用你原有的转换工具（有些实现吃 (B,C,H,W)[0..1]）
        try:
            imgs = list(tensor_to_pil_image(samples))
        except Exception as e:
            print(f"[WARN] tensor_to_pil_image failed: {repr(e)}. Fallback to manual saver.", flush=True)
            imgs = []

        # 没产出就走兜底保存
        if len(imgs) == 0:
            print("[INFO] tensor_to_pil_image produced 0 images. Using fallback saver.", flush=True)
            for j in range(B):
                save_tensor_png(samples[j], save_dir / f"{sidx + j}.png")
        else:
            for j, img in zip(range(sidx, eidx), imgs):
                img.save(save_dir / f"{j}.png")

        print(f"[INFO] saved [{sidx}, {eidx}) -> {eidx - sidx} files", flush=True)

    print(f"[DONE] All images saved to: {save_dir.resolve()}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # 基础参数
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500, help="总生成数量")

    # 模型预测相关
    parser.add_argument("--predictor", type=str, default="noise",
                        choices=["noise", "x0", "mean"])

    # 调度器参数
    parser.add_argument("--mode", type=str, default="linear",
                        choices=["linear", "cosine", "quad"])
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)

    # CFG（可选）
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    args = parser.parse_args()

    print("[INFO] args parsed, starting main()", flush=True)
    main(args)
