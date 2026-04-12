import torch
from dataclasses import dataclass, asdict

# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    dataset_mode: str = "8gaussians"  # choices: 8gaussians, circle
    dataset_size: int = 20000
    batch_size: int = 512

    # path
    alpha: float = 0.35
    eps: float = 1e-8

    # model
    hidden_dim: int = 256
    num_layers: int = 4

    # train
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 300
    log_interval: int = 20

    # inference / viz
    rollout_steps: int = 100
    num_demo_paths: int = 8
    out_dir: str = "./fm_target_path_outputs"
    ckpt_name: str = "cfm_target_path.pt"