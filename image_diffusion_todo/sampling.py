import argparse
import numpy as np
import torch
from pathlib import Path
from dataset import tensor_to_pil_image
from model import DiffusionModule
from scheduler import DDPMScheduler

def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"

    # 1) load model
    ddpm = DiffusionModule(None, None)
    ddpm.load(args.ckpt_path)
    ddpm.eval().to(device)

    # 2)  predictor  scheduler 
    ddpm.predictor = args.predictor  # "noise" / "x0" / "mean"

    T = ddpm.var_scheduler.num_train_timesteps
    ddpm.var_scheduler = DDPMScheduler(
        T,
        beta_1=args.beta_1,
        beta_T=args.beta_T,
        mode=args.mode,
    ).to(device)

    total_num_samples = 500
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        if args.use_cfg:
            assert getattr(ddpm.network, "use_cfg", False), "This checkpoint wasn't trained with CFG."

            samples = ddpm.sample(
                B,
                class_label=torch.randint(0, 3, (B,), device=device),
                guidance_scale=args.cfg_scale,
            )
        else:
            samples = ddpm.sample(B)

        for j, img in zip(range(sidx, eidx), tensor_to_pil_image(samples)):
            img.save(save_dir / f"{j}.png")
            print(f"Saved the {j}-th image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    parser.add_argument("--predictor", type=str, default="noise",
                        choices=["noise", "x0", "mean"])
    parser.add_argument("--mode", type=str, default="linear",
                        choices=["linear", "cosine", "quad"])
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)


    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    args = parser.parse_args()
    main(args)
