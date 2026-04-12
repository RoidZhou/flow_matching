import math
import os
import random
from dataclasses import dataclass, asdict
from config import Config
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import Config
from dataset import TargetPointDataset, plot_dataset
from model import VelocityMLP
from cfm import CurvedPathCFM

# ============================================================
# Utils
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# Train / Infer
# ============================================================
def train_one_epoch(model, loader, optimizer, path_sampler, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for x1 in loader:
        x1 = x1.to(device)
        _, x1, t, xt, ut = path_sampler.sample_training_tuple(x1)

        pred = model(xt, t, x1)
        loss = F.mse_loss(pred, ut)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x1.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)

@torch.no_grad()
def rollout(model, path_sampler, x0, x1, steps=100):
    """
    从给定 x0 出发，条件于目标点 x1，用欧拉法积分。
    x0, x1: [B, 2]
    返回每一步轨迹 points: [steps+1, B, 2]
    """
    model.eval()
    x = x0.clone()
    traj = [x.clone()]
    dt = 1.0 / steps

    for i in range(steps):
        t_value = torch.full((x.shape[0], 1), i / steps, device=x.device, dtype=x.dtype)
        v = model(x, t_value, x1)
        x = x + dt * v
        traj.append(x.clone())

    return torch.stack(traj, dim=0)

# ============================================================
# Main
# ============================================================

TRAIN = True
def main():
    cfg = Config()
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)

    print("Config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")

    dataset = TargetPointDataset(mode=cfg.dataset_mode, size=cfg.dataset_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    plot_dataset(dataset, os.path.join(cfg.out_dir, "dataset.png"))

    model = VelocityMLP(hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    path_sampler = CurvedPathCFM(alpha=cfg.alpha, eps=cfg.eps)
    ckpt_path = os.path.join(cfg.out_dir, cfg.ckpt_name)
    
    if TRAIN:
        losses = []
        best_loss = float("inf")

        for epoch in range(1, cfg.epochs + 1):
            loss = train_one_epoch(model, loader, optimizer, path_sampler, cfg.device)
            losses.append(loss)

            if loss < best_loss:
                best_loss = loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": asdict(cfg),
                        "best_loss": best_loss,
                    },
                    ckpt_path,
                )

            if epoch % cfg.log_interval == 0 or epoch == 1 or epoch == cfg.epochs:
                print(f"Epoch {epoch:04d} | loss = {loss:.6f}")

        # loss curve
        plt.figure(figsize=(7, 4))
        plt.plot(losses)
        plt.xlabel("epoch")
        plt.ylabel("train loss")
        plt.title("Training curve")
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.out_dir, "loss_curve.png"), dpi=180)
        plt.close()

if __name__ == "__main__":
    main()
