import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt

class TargetPointDataset(Dataset):
    """
    只存目标点 x1。
    训练时 x0 从标准高斯在线采样，t 在线采样。
    """

    def __init__(self, mode: str = "8gaussians", size: int = 20000):
        super().__init__()
        self.mode = mode
        self.size = size
        self.points = self._make_points(mode, size).float()

    def _make_points(self, mode: str, size: int) -> torch.Tensor:
        if mode == "8gaussians":
            centers = torch.tensor(
                [
                    [1.5, 0.0],
                    [-1.5, 0.0],
                    [0.0, 1.5],
                    [0.0, -1.5],
                    [1.05, 1.05],
                    [1.05, -1.05],
                    [-1.05, 1.05],
                    [-1.05, -1.05],
                ],
                dtype=torch.float32,
            )
            idx = torch.randint(0, len(centers), (size,))
            noise = 0.08 * torch.randn(size, 2)
            return centers[idx] + noise

        if mode == "circle":
            theta = 2 * math.pi * torch.rand(size)
            radius = 1.5 + 0.05 * torch.randn(size)
            x = radius * torch.cos(theta)
            y = radius * torch.sin(theta)
            return torch.stack([x, y], dim=1)

        raise ValueError(f"Unknown dataset_mode: {mode}")

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.points[idx]

@torch.no_grad()
def plot_dataset(dataset, save_path):
    pts = dataset.points.numpy()
    plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.5)
    plt.title("Target point dataset x1")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
