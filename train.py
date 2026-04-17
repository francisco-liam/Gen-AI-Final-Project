"""
train.py — DDPM epsilon-prediction training on Fashion-MNIST

Usage:
    # Linear schedule (default)
    python train.py

    # Named experiments with explicit schedule selection
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

from data      import get_dataloaders, DATASET_INFO, SUPPORTED_DATASETS
from diffusion import GaussianDiffusion
from model     import SmallUNet
from schedule  import get_beta_schedule, SUPPORTED_SCHEDULES


# ===========================================================================
# Default configuration — all values are identical across schedules so that
# linear vs cosine comparisons are fair.  Override via CLI only.
# ===========================================================================

DEFAULTS = dict(
    schedule      = "linear",
    dataset       = "fashionmnist",
    run_name      = "run_01",
    batch_size    = 128,
    learning_rate = 2e-4,
    epochs        = 100,
    timesteps     = 1000,   # T — canonical DDPM setting; cosine vs linear difference is clear here
    save_every    = 10,     # checkpoint frequency (epochs)
    seed          = 42,
    num_workers   = 2,      # set 0 on Windows if multiprocessing errors occur
    base_channels = 32,
    time_dim      = 128,
    use_attention  = False,
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
    return os.path.join(cfg["experiment_root"], cfg["dataset"], cfg["schedule"], cfg["run_name"])


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
        dataset=cfg["dataset"],
    )
    print(f"Training batches per epoch: {len(train_loader)}")

    # ── Noise schedule — the ONLY thing that changes between runs ────────
    betas     = get_beta_schedule(cfg["schedule"], cfg["timesteps"])
    diffusion = GaussianDiffusion(betas)
    _move_diffusion_to_device(diffusion, device)

    # ── Model and optimiser ─────────────────────────────────────────────
    # Architecture is identical for all schedules — fair comparison.
    model = SmallUNet(
        in_channels=DATASET_INFO[cfg["dataset"]]["in_channels"],
        base_channels=cfg["base_channels"],
        time_dim=cfg["time_dim"],
        use_attention=cfg["use_attention"],
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg["learning_rate"])

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── Resume from checkpoint ──────────────────────────────────────────
    start_epoch = 1
    if cfg.get("resume"):
        ckpt_files = sorted([
            f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_epoch") and f.endswith(".pt")
        ])
        if ckpt_files:
            latest = os.path.join(ckpt_dir, ckpt_files[-1])
            print(f"Resuming from checkpoint: {latest}")
            ckpt = torch.load(latest, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print("[WARNING] --resume specified but no checkpoints found; starting from scratch.")

    # ── Loss log ─────────────────────────────────────────────────────────
    # Written incrementally so a crashed run still has partial data.
    log_mode  = "a" if cfg.get("resume") and start_epoch > 1 else "w"
    loss_csv = os.path.join(log_dir, "loss.csv")
    csv_file  = open(loss_csv, log_mode, newline="")
    csv_writer = csv.writer(csv_file)
    if log_mode == "w":
        csv_writer.writerow(["epoch", "avg_loss"])

    # ── Gradient norm log ────────────────────────────────────────────────
    gradnorm_csv  = os.path.join(log_dir, "gradnorm.csv")
    gn_file       = open(gradnorm_csv, log_mode, newline="")
    gn_writer     = csv.writer(gn_file)
    if log_mode == "w":
        gn_writer.writerow(["epoch", "avg_gradnorm"])

    # ── Per-timestep-bucket loss log ─────────────────────────────────────
    # We split T timesteps into N_BUCKETS equal-width buckets and record the
    # mean MSE inside each bucket.  This reveals which timestep range each
    # schedule finds hardest (high loss → schedule assigns hard denoising task).
    N_BUCKETS     = 10
    t_loss_csv    = os.path.join(log_dir, "loss_by_t.csv")
    tl_file       = open(t_loss_csv, log_mode, newline="")
    tl_writer     = csv.writer(tl_file)
    if log_mode == "w":
        tl_writer.writerow(["epoch"] + [f"bucket_{i}" for i in range(N_BUCKETS)])

    # ── Training loop ────────────────────────────────────────────────────
    EPOCHS    = cfg["epochs"]
    TIMESTEPS = cfg["timesteps"]

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running_loss   = 0.0
        running_gnorm  = 0.0
        n_batches      = 0

        # Accumulators for per-timestep-bucket MSE
        bucket_loss_sum   = [0.0] * N_BUCKETS
        bucket_loss_count = [0]   * N_BUCKETS

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

            # Gradient clipping + logging — clip at 1.0 (common DDPM practice)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            running_gnorm += grad_norm.item()

            optimizer.step()

            running_loss += loss.item()
            n_batches    += 1

            # Accumulate per-sample MSE into timestep buckets (no grad needed)
            with torch.no_grad():
                per_sample_mse = (noise_pred - noise).pow(2).mean(dim=(1, 2, 3))  # (B,)
                t_cpu = t.cpu()
                bucket_ids = (t_cpu * N_BUCKETS // TIMESTEPS).clamp(0, N_BUCKETS - 1)
                for b_idx in range(N_BUCKETS):
                    mask = (bucket_ids == b_idx)
                    if mask.any():
                        bucket_loss_sum[b_idx]   += per_sample_mse[mask].sum().item()
                        bucket_loss_count[b_idx] += mask.sum().item()

        avg_loss  = running_loss  / n_batches
        avg_gnorm = running_gnorm / n_batches

        print(f"Epoch [{epoch:>3}/{EPOCHS}]  loss: {avg_loss:.4f}  "
              f"grad_norm: {avg_gnorm:.4f}")

        csv_writer.writerow([epoch, f"{avg_loss:.6f}"])
        csv_file.flush()

        gn_writer.writerow([epoch, f"{avg_gnorm:.6f}"])
        gn_file.flush()

        bucket_avgs = [
            f"{bucket_loss_sum[i] / bucket_loss_count[i]:.6f}"
            if bucket_loss_count[i] > 0 else "nan"
            for i in range(N_BUCKETS)
        ]
        tl_writer.writerow([epoch] + bucket_avgs)
        tl_file.flush()

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
    gn_file.close()
    tl_file.close()
    print(f"Loss log saved      → {loss_csv}")
    print(f"Grad norm log saved → {gradnorm_csv}")
    print(f"Loss-by-t log saved → {t_loss_csv}")
    print("Training complete.")


# ===========================================================================

def parse_args() -> dict:
    parser = argparse.ArgumentParser(
        description="DDPM training — linear vs cosine schedule comparison"
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
    parser.add_argument("--base_channels",  type=int,   default=DEFAULTS["base_channels"])
    parser.add_argument("--use_attention",  action="store_true",
                        help="Add self-attention at the bottleneck")
    parser.add_argument("--dataset",                    default=DEFAULTS["dataset"],
                        choices=SUPPORTED_DATASETS,
                        help="Dataset to train on (default: fashionmnist)")
    parser.add_argument("--experiment_root",            default=DEFAULTS["experiment_root"])
    parser.add_argument("--resume",         action="store_true",
                        help="Resume training from the latest checkpoint in the run directory")
    args = parser.parse_args()

    cfg = DEFAULTS.copy()
    cfg.update(vars(args))
    return cfg


if __name__ == "__main__":
    train(parse_args())
