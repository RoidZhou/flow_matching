
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

# ============================================================
# Model
# 输入: x_t, t, x1
# 输出: 速度 v_theta(x_t, t, x1)
# ============================================================

class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim: int = 64):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B, 1]
        half = self.emb_dim // 2
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0), math.log(1000.0), half, device=t.device, dtype=t.dtype
            )
        )
        angles = t * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return emb


class VelocityMLP(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_layers: int = 4, time_dim: int = 64):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        in_dim = 2 + 2 + time_dim  # x_t(2) + x1(2) + t_emb

        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [2]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, xt: torch.Tensor, t: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        temb = self.time_emb(t)
        h = torch.cat([xt, x1, temb], dim=1)
        return self.net(h)