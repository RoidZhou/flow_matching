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
from visualization import plot_teacher_and_pred_paths, plot_single_path_progress, plot_velocity_quiver, plot_speed_curve


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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

    state = torch.load(ckpt_path, map_location=cfg.device)
    model.load_state_dict(state["model"])

    plot_teacher_and_pred_paths(
        model,
        path_sampler,
        dataset,
        cfg,
        os.path.join(cfg.out_dir, "teacher_vs_pred_paths.png"),
    )
    plot_single_path_progress(
        model,
        path_sampler,
        dataset,
        cfg,
        os.path.join(cfg.out_dir, "single_path_progress.png"),
    )
    plot_velocity_quiver(
        model,
        path_sampler,
        dataset,
        cfg,
        os.path.join(cfg.out_dir, "velocity_quiver.png"),
    )
    plot_speed_curve(
        model,
        path_sampler,
        dataset,
        cfg,
        os.path.join(cfg.out_dir, "speed_curve.png"),
    )

    print(f"Done. Outputs saved to: {cfg.out_dir}")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
