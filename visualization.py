import matplotlib.pyplot as plt
import torch
import numpy as np
from config import Config
from train import rollout
# ============================================================
# Visualization
# ============================================================

@torch.no_grad()
def rollout_with_velocity(model, path_sampler, x0, x1, steps=100):
    """
    除了轨迹，还返回每一步的速度向量和速度模长。
    x0, x1: [B, 2]
    返回:
      traj: [steps+1, B, 2]
      vel:  [steps,   B, 2]
      speed:[steps,   B]
    """
    model.eval()
    x = x0.clone()
    traj = [x.clone()]
    vel_list = []
    speed_list = []
    dt = 1.0 / steps

    for i in range(steps):
        t_value = torch.full((x.shape[0], 1), i / steps, device=x.device, dtype=x.dtype)
        v = model(x, t_value, x1)
        vel_list.append(v.clone())
        speed_list.append(torch.norm(v, dim=1))
        x = x + dt * v
        traj.append(x.clone())

    traj = torch.stack(traj, dim=0)
    vel = torch.stack(vel_list, dim=0)
    speed = torch.stack(speed_list, dim=0)
    return traj, vel, speed


@torch.no_grad()
def plot_velocity_quiver(model, path_sampler, dataset, cfg: Config, save_path: str):
    """
    可视化单条推理路径上每个点的速度向量。
    背景灰色虚线: teacher path
    蓝色实线: predicted rollout
    红色箭头: 每一步速度方向与大小
    """
    device = cfg.device
    idx = torch.randint(0, len(dataset), (1,)).item()
    x1 = dataset.points[idx : idx + 1].to(device)
    x0 = torch.randn_like(x1)

    traj, vel, speed = rollout_with_velocity(model, path_sampler, x0, x1, steps=cfg.rollout_steps)
    traj_np = traj.squeeze(1).cpu().numpy()
    vel_np = vel.squeeze(1).cpu().numpy()

    ts = torch.linspace(0, 1, cfg.rollout_steps + 1, device=device).unsqueeze(1)
    teacher = torch.cat(
        [path_sampler.teacher_path(x0, x1, t.unsqueeze(0)) for t in ts.squeeze(1)], dim=0
    ).cpu().numpy()

    # 抽稀画箭头，避免太密
    step_stride = max(1, cfg.rollout_steps // 50)
    q_pos = traj_np[:-1:step_stride]
    q_vel = vel_np[::step_stride]

    plt.figure(figsize=(8, 8))
    pts = dataset.points.numpy()
    plt.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.12, label="target dataset")
    plt.plot(teacher[:, 0], teacher[:, 1], linestyle="--", linewidth=2, alpha=0.8, label="teacher path")
    plt.plot(traj_np[:, 0], traj_np[:, 1], linewidth=2.5, alpha=0.95, label="predicted rollout")
    plt.quiver(
        q_pos[:, 0],
        q_pos[:, 1],
        q_vel[:, 0],
        q_vel[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.004,
        alpha=0.9,
    )
    plt.scatter(x0[0, 0].item(), x0[0, 1].item(), marker="x", s=80, label="start x0")
    plt.scatter(x1[0, 0].item(), x1[0, 1].item(), marker="o", s=60, label="target x1")
    plt.title("Predicted path with velocity arrows")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


@torch.no_grad()
def plot_speed_curve(model, path_sampler, dataset, cfg: Config, save_path: str):
    """
    可视化推理过程中每一步速度模长随时间变化。
    """
    device = cfg.device
    idx = torch.randint(0, len(dataset), (1,)).item()
    x1 = dataset.points[idx : idx + 1].to(device)
    x0 = torch.randn_like(x1)

    _, _, speed = rollout_with_velocity(model, path_sampler, x0, x1, steps=cfg.rollout_steps)
    speed_np = speed.squeeze(1).cpu().numpy()
    t_np = np.linspace(0.0, 1.0, cfg.rollout_steps, endpoint=False)

    plt.figure(figsize=(7, 4))
    plt.plot(t_np, speed_np, linewidth=2)
    plt.xlabel("t")
    plt.ylabel("||v_t||")
    plt.title("Speed magnitude along one inference trajectory")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()



@torch.no_grad()
def plot_teacher_and_pred_paths(model, path_sampler, dataset, cfg: Config, save_path: str):
    device = cfg.device
    k = cfg.num_demo_paths

    indices = torch.randperm(len(dataset))[:k]
    x1 = dataset.points[indices].to(device)
    x0 = torch.randn_like(x1)

    pred_traj = rollout(model, path_sampler, x0, x1, steps=cfg.rollout_steps).cpu().numpy()

    ts = torch.linspace(0, 1, cfg.rollout_steps + 1, device=device).unsqueeze(1)
    teacher_frames = []
    for i in range(cfg.rollout_steps + 1):
        t = ts[i].repeat(k, 1)
        teacher_frames.append(path_sampler.teacher_path(x0, x1, t))
    teacher_traj = torch.stack(teacher_frames, dim=0).cpu().numpy()

    x0_np = x0.cpu().numpy()
    x1_np = x1.cpu().numpy()

    plt.figure(figsize=(8, 8))
    pts = dataset.points.numpy()
    plt.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.15, label="target dataset")

    for i in range(k):
        plt.plot(
            teacher_traj[:, i, 0],
            teacher_traj[:, i, 1],
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
        )
        plt.plot(
            pred_traj[:, i, 0],
            pred_traj[:, i, 1],
            linewidth=2.0,
            alpha=0.95,
        )
        plt.scatter(x0_np[i, 0], x0_np[i, 1], marker="x", s=60)
        plt.scatter(x1_np[i, 0], x1_np[i, 1], marker="o", s=40)

    plt.title("Dashed: teacher path | Solid: predicted rollout")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


@torch.no_grad()
def plot_single_path_progress(model, path_sampler, dataset, cfg: Config, save_path: str):
    device = cfg.device
    idx = torch.randint(0, len(dataset), (1,)).item()
    x1 = dataset.points[idx : idx + 1].to(device)
    x0 = torch.randn_like(x1)

    pred_traj = rollout(model, path_sampler, x0, x1, steps=cfg.rollout_steps).squeeze(1).cpu().numpy()
    ts = torch.linspace(0, 1, cfg.rollout_steps + 1, device=device).unsqueeze(1)
    teacher = torch.cat(
        [path_sampler.teacher_path(x0, x1, t.unsqueeze(0)) for t in ts.squeeze(1)], dim=0
    ).cpu().numpy()

    chosen = np.linspace(0, cfg.rollout_steps, 8).astype(int)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    for ax, j in zip(axes, chosen):
        ax.scatter(teacher[:, 0], teacher[:, 1], s=8, alpha=0.2)
        ax.plot(teacher[: j + 1, 0], teacher[: j + 1, 1], linestyle="--", linewidth=2)
        ax.plot(pred_traj[: j + 1, 0], pred_traj[: j + 1, 1], linewidth=2)
        ax.scatter(x0[0, 0].item(), x0[0, 1].item(), marker="x", s=70)
        ax.scatter(x1[0, 0].item(), x1[0, 1].item(), marker="o", s=50)
        ax.set_title(f"t={j / cfg.rollout_steps:.2f}")
        ax.axis("equal")
        ax.grid(alpha=0.2)

    plt.suptitle("One path evolution: dashed teacher, solid predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()

