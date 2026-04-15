"""
train.py — DDPM epsilon-prediction training on Fashion-MNIST  (Weeks 1-3)

Usage:
    # Week 1 / legacy — linear schedule, default folders
    python train.py

    # Week 3 — named experiments with explicit schedule selection
    python train.py --schedule linear  --run_name run_01
    python train.py --schedule cosine  --run_name run_01

Outputs per run:
    experiments/<schedule>/<run_name>/checkpoints/ckpt_epoch<N>.pt
    experiments/<schedule>/<run_name>/logs/loss.csv
    experiments/<schedule>/<run_name>/config.json

The ONLY intended experimental variable between runs is --schedule.
All other defaults are identical so comparisons are fair.
"""

import argparse
import csv
import json
import os

import torch
import torch.nn as nn
from torch.optim import Adam

from data      import get_dataloaders
from diffusion import GaussianDiffusion
from model     import SmallUNet
from schedule  import get_beta_schedule, SUPPORTED_SCHEDULES


# ===========================================================================
# Default configuration — all values are identical across schedules so that
# linear vs cosine comparisons are fair.  Override via CLI only.
# ===========================================================================

DEFAULTS = dict(
    schedule      = "linear",
    run_name      = "run_01",
    batch_size    = 128,
    learning_rate = 2e-4,
    epochs        = 10,
    timesteps     = 200,    # T — increase to 1000 for full DDPM quality
    save_every    = 2,      # checkpoint frequency (epochs)
    seed          = 42,
    num_workers   = 2,      # set 0 on Windows if multiprocessing errors occur
    base_channels = 32,
    time_dim      = 128,
    experiment_root = "experiments",
)


# ===========================================================================
# Helpers
# ===========================================================================

def _move_diffusion_to_device(diffusion: GaussianDiffusion, device: torch.device) -> None:
    """Push all precomputed diffusion tensors to device once."""
    for attr in (
        "betas", "alphas", "alpha_bars",
        "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars",
        "sqrt_recip_alphas", "betas_over_sqrt_one_minus_alpha_bars",
        "sqrt_betas",
    ):
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))


def _run_dir(cfg: dict) -> str:
    """Return the root directory for this experiment run."""
    return os.path.join(cfg["experiment_root"], cfg["schedule"], cfg["run_name"])


# ===========================================================================
# Training loop
# ===========================================================================

def train(cfg: dict) -> None:
    run_dir  = _run_dir(cfg)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    log_dir  = os.path.join(run_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    # ── Reproducibility ─────────────────────────────────────────────────
    torch.manual_seed(cfg["seed"])

    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Schedule : {cfg['schedule']}")
    print(f"Run      : {cfg['run_name']}")
    print(f"Device   : {device}")
    print(f"Run dir  : {run_dir}")

    # ── Save config to disk so sample.py can always reconstruct it ───────
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config   : {config_path}")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader = get_dataloaders(
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )
    print(f"Training batches per epoch: {len(train_loader)}")

    # ── Noise schedule — the ONLY thing that changes between runs ────────
    betas     = get_beta_schedule(cfg["schedule"], cfg["timesteps"])
    diffusion = GaussianDiffusion(betas)
    _move_diffusion_to_device(diffusion, device)

    # ── Model and optimiser ─────────────────────────────────────────────
    # Architecture is identical for all schedules — fair comparison.
    model = SmallUNet(
        in_channels=1,
        base_channels=cfg["base_channels"],
        time_dim=cfg["time_dim"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── Loss log ─────────────────────────────────────────────────────────
    # Written incrementally so a crashed run still has partial data.
    loss_csv = os.path.join(log_dir, "loss.csv")
    csv_file  = open(loss_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "avg_loss"])

    # ── Training loop ────────────────────────────────────────────────────
    EPOCHS    = cfg["epochs"]
    TIMESTEPS = cfg["timesteps"]

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for x0, _ in train_loader:
            x0 = x0.to(device)               # (B, 1, 28, 28) in [-1, 1]

            # Sample random timesteps t ~ Uniform[0, T)
            t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=device)

            # Forward process: corrupt x0 → x_t
            x_t, noise = diffusion.q_sample(x0, t)   # both (B, 1, 28, 28)

            # Predict noise ε̂ with the U-Net
            noise_pred = model(x_t, t)                # (B, 1, 28, 28)

            # DDPM simple objective (Ho et al. 2020, Eq. 14): MSE on noise
            loss = nn.functional.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch:>3}/{EPOCHS}]  loss: {avg_loss:.4f}")
        csv_writer.writerow([epoch, f"{avg_loss:.6f}"])
        csv_file.flush()

        # ── Checkpoint ────────────────────────────────────────────────
        if epoch % cfg["save_every"] == 0 or epoch == EPOCHS:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:04d}.pt")
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss":                 avg_loss,
                    # Full config embedded so sample.py can restore everything
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"           checkpoint saved → {ckpt_path}")

    csv_file.close()
    print(f"Loss log saved → {loss_csv}")
    print("Training complete.")


# ===========================================================================

def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="DDPM training — linear vs cosine schedule comparison (Week 3)"
    )
    parser.add_argument("--schedule",       default=DEFAULTS["schedule"],
                        choices=SUPPORTED_SCHEDULES,
                        help="Noise schedule to use (default: linear)")
    parser.add_argument("--run_name",       default=DEFAULTS["run_name"],
                        help="Name for this run, used in folder path (default: run_01)")
    parser.add_argument("--epochs",         type=int,   default=DEFAULTS["epochs"])
    parser.add_argument("--timesteps",      type=int,   default=DEFAULTS["timesteps"])
    parser.add_argument("--batch_size",     type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--learning_rate",  type=float, default=DEFAULTS["learning_rate"])
    parser.add_argument("--save_every",     type=int,   default=DEFAULTS["save_every"])
    parser.add_argument("--seed",           type=int,   default=DEFAULTS["seed"])
    parser.add_argument("--experiment_root",            default=DEFAULTS["experiment_root"])
    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    cfg.update(vars(args))
    return cfg


if __name__ == "__main__":
    train(parse_args())
