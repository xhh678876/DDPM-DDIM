import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dataset import AFHQDataModule, get_data_iterator, tensor_to_pil_image
from dotmap import DotMap
from model import DiffusionModule
from network import UNet
from pytorch_lightning import seed_everything
from scheduler import DDPMScheduler
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from PIL import Image

matplotlib.use("Agg")

# === add near the imports ===
import numpy as np
from torchvision.transforms.functional import to_pil_image as _to_pil

def to_pil_unit_range(x: torch.Tensor):
    # x: [C,H,W], value ~ [-1,1] -> [0,1]
    x = x.clamp(-1, 1)
    x = (x + 1.0) / 2.0
    return _to_pil(x.cpu())

def save_traj_strip(save_path: Path, traj, num_frames: int = 10, pad: int = 4):
    """
    traj: list of T+1 tensors, each [B,C,H,W]; we’ll visualize the first in the batch.
    num_frames: how many frames to pick along the trajectory.
    """
    # pick evenly spaced indices
    idxs = np.linspace(0, len(traj) - 1, num_frames).astype(int)
    pil_frames = []
    for i in idxs:
        x_t = traj[i][0]  # take the first sample in batch
        pil_frames.append(to_pil_unit_range(x_t))

    # stitch horizontally
    H, W = pil_frames[0].height, pil_frames[0].width
    strip = Image.new("RGB", (num_frames * W + pad * (num_frames - 1), H), (0, 0, 0))
    x = 0
    for im in pil_frames:
        strip.paste(im, (x, 0))
        x += W + pad
    strip.save(save_path)


def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now

    
def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"
    
    now = get_current_time()
    if args.use_cfg:
        save_dir = Path(
            f"results/cfg_predictor_{args.predictor}/beta_{config.mode}/{now}"
        )
    else:
        save_dir = Path(
            f"results/predictor_{args.predictor}/beta_{config.mode}/{now}"
        )
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    """######"""

    image_resolution = 64
    ds_module = AFHQDataModule(
        "./data",
        batch_size=config.batch_size,
        num_workers=4,
        max_num_images_per_cat=config.max_num_images_per_cat,
        image_resolution=image_resolution
    )

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    # Set up the scheduler
    var_scheduler = DDPMScheduler(
        config.num_diffusion_train_timesteps,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        mode=config.mode,  ### use args.mode instead of hardcoded,  ### Different scheduling
    )

    network = UNet(
        T=config.num_diffusion_train_timesteps,
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        num_classes=getattr(ds_module, "num_classes", None),
    )

    ddpm = DiffusionModule(network, var_scheduler, predictor=config.predictor)
    ddpm = ddpm.to(config.device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    step = 0
    losses = []
    
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            from PIL import Image
            if step % config.log_interval == 0:
                ddpm.eval()
                
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()
                
                samples = ddpm.sample(4, return_traj=False)
                pil_images = tensor_to_pil_image(samples)
                for i, img in enumerate(pil_images):
                    img.save(save_dir / f"step={step}-{i}.png")

                # traj fig
                traj = ddpm.sample(1, return_traj=True)  # traj
                save_traj_strip(save_dir / f"step={step}-traj.png", traj, num_frames=10, pad=4)                            
                            
                ddpm.save(f"{save_dir}/last.ckpt")
                ddpm.train()

            img, label = next(train_it)
            img, label = img.to(config.device), label.to(config.device)
                        
            if args.use_cfg:  # Conditional, CFG training
                loss = ddpm.get_loss(img, class_label=label) #noise
        
            else:  # Unconditional training
                loss = ddpm.get_loss(img)
             
            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # optimizer.step_predict_x0()
            # optimizer.step_predict_x0()
            losses.append(loss.item())

            step += 1
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000, #50000, #100000
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument(
        "--max_num_images_per_cat",
        type=int,
        default=3000,
        help="max number of images per category for AFHQ dataset",
    )
    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
        help="diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=1e-4)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--image_resolution", type=int, default=64)
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--cfg_dropout", type=float, default=0.1)
    parser.add_argument("--predictor", type=str, default="noise",
                    choices=["noise", "x0", "mean"],
                    help="which parameterization the model uses")
    parser.add_argument("--mode", type=str, default="linear",
                        choices=["linear", "cosine", "quad"],
                        help="beta scheduling mode")
    
    args = parser.parse_args()
    config = DotMap()
    config.update(vars(args))
    main(args)
    