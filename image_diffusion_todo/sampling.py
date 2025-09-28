import argparse                      # 用于命令行参数解析，支持 --arg 的形式从命令行传参数
import numpy as np                   # 数值计算库，用来做数学计算（这里主要用于 np.ceil 向上取整）
import torch                         # PyTorch，用于深度学习模型和张量计算
from pathlib import Path              # 处理文件和路径的跨平台库
from dataset import tensor_to_pil_image  # 工具函数：把 Tensor 转换为 PIL.Image 以便保存
from model import DiffusionModule     # 扩散模型的封装类，里面包含 UNet 和采样逻辑
from scheduler import DDPMScheduler   # 扩散调度器，定义 beta 序列（噪声调度方式）


def main(args):
    # ============ 1. 设置保存目录 ============
    save_dir = Path(args.save_dir)                   # 将字符串路径转换为 Path 对象
    save_dir.mkdir(exist_ok=True, parents=True)      # 如果目录不存在就创建（支持多级父目录）

    device = f"cuda:{args.gpu}"                      # 指定 GPU 设备，例如 "cuda:0"

    # ============ 2. 加载模型 ============
    ddpm = DiffusionModule(None, None)               # 实例化扩散模型（参数具体在 DiffusionModule 内部处理）
    ddpm.load(args.ckpt_path)                        # 从 checkpoint 文件加载训练好的权重
    ddpm.eval().to(device)                           # 切换到 eval 模式（关闭 dropout 等），并移动到 GPU

    # ============ 3. 配置预测器和调度器 ============
    ddpm.predictor = args.predictor                  # 设置预测器类型：
                                                     # - "noise" → 预测噪声 ε
                                                     # - "x0"   → 预测原始图像 x0
                                                     # - "mean" → 预测均值

    T = ddpm.var_scheduler.num_train_timesteps       # 获取训练时的扩散步数（如 T=1000）
    ddpm.var_scheduler = DDPMScheduler(              # 重新构建一个采样调度器
        T,
        beta_1=args.beta_1,                          # β 序列的起始值
        beta_T=args.beta_T,                          # β 序列的结束值
        mode=args.mode,                              # β 的调度模式（linear / cosine / quad）
    ).to(device)                                     # 将调度器移到 GPU

    # ============ 4. 设置采样批次数 ============
    total_num_samples = 500                          # 总共要采样的图像数量（固定为 500）
    num_batches = int(np.ceil(total_num_samples / args.batch_size))  
    # 用 batch_size 把总数分批，向上取整

    # ============ 5. 开始循环采样 ============
    for i in range(num_batches):
        sidx = i * args.batch_size                   # 当前批次起始索引
        eidx = min(sidx + args.batch_size, total_num_samples)  # 当前批次结束索引
        B = eidx - sidx                              # 当前批次大小

        # ----- 5.1 是否启用 CFG（Classifier-Free Guidance） -----
        if args.use_cfg:
            # 确保模型的网络结构支持 CFG（即训练时是带条件的）
            assert getattr(ddpm.network, "use_cfg", False), "This checkpoint wasn't trained with CFG."

            # 调用采样函数，生成带类别条件的样本
            samples = ddpm.sample(
                B,                                   # batch size
                class_label=torch.randint(0, 3, (B,), device=device),  
                # 随机生成类别标签（这里假设类别数为 3：0,1,2）
                guidance_scale=args.cfg_scale,       # CFG 的引导强度（scale 越大越贴合条件）
            )
        else:
            # 无条件采样
            samples = ddpm.sample(B)

        # ----- 5.2 保存当前批次生成的图片 -----
        for j, img in zip(range(sidx, eidx), tensor_to_pil_image(samples)):
            img.save(save_dir / f"{j}.png")          # 保存图片为 j.png
            print(f"Saved the {j}-th image.")        # 打印日志，确认已保存


# ============ 程序入口 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()               # 新建参数解析器

    # 基础参数
    parser.add_argument("--batch_size", type=int, default=64)    # 每批次生成多少张图
    parser.add_argument("--gpu", type=int, default=0)            # 使用哪块 GPU（如 0 表示 "cuda:0"）
    parser.add_argument("--ckpt_path", type=str, required=True)  # 模型 checkpoint 路径
    parser.add_argument("--save_dir", type=str, required=True)   # 输出图像保存路径

    # 模型预测相关
    parser.add_argument("--predictor", type=str, default="noise",
                        choices=["noise", "x0", "mean"])  
    # predictor：选择反向扩散时预测什么量
    #   - noise：预测噪声
    #   - x0：预测干净图像
    #   - mean：预测均值

    # 调度器参数
    parser.add_argument("--mode", type=str, default="linear",
                        choices=["linear", "cosine", "quad"]) 
    # β 的调度模式（线性 / 余弦 / 二次函数）
    parser.add_argument("--beta_1", type=float, default=1e-4)    # β 起始值
    parser.add_argument("--beta_T", type=float, default=0.02)    # β 结束值

    # CFG 相关
    parser.add_argument("--use_cfg", action="store_true")        # 是否启用 CFG
    parser.add_argument("--sample_method", type=str, default="ddpm")  
    # 采样方法（如 ddpm / ddim，可扩展，这里默认 ddpm）
    parser.add_argument("--cfg_scale", type=float, default=7.5)  # CFG 引导强度，常用 3~8，默认 7.5

    args = parser.parse_args()                                   # 从命令行解析参数
    main(args)                                                   # 执行主逻辑

