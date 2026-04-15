"""
sample.py — Week 2 inference script for the trained DDPM

Loads a checkpoint produced by train.py, runs the full DDPM reverse
diffusion loop, and saves a grid of generated Fashion-MNIST images.

Usage:
    python sample.py                          # uses latest checkpoint
    python sample.py --ckpt checkpoints/ckpt_epoch0010.pt

Outputs:
    outputs/samples_epoch0010.png   — grid of generated images
    outputs/trajectory_epoch0010.png — optional denoising timeline
"""

import argparse
import os

import torch
import torchvision.utils as vutils

from diffusion import GaussianDiffusion
from model     import SmallUNet
from schedule  import linear_beta_schedule


# ===========================================================================
# Configuration — edit these or override via CLI arguments
# ===========================================================================

CHECKPOINT_PATH = "checkpoints/ckpt_epoch0010.pt"  # which checkpoint to load
NUM_SAMPLES     = 16        # how many images to generate (keep a perfect square)
TIMESTEPS       = 200       # must match the value used during training
OUTPUT_DIR      = "outputs" # where to write PNG grids
GRID_NROW       = 4         # images per row in the output grid
SAVE_TRAJECTORY = True      # also save a denoising timeline strip
TRAJ_EVERY      = 20        # snapshot every N denoising steps for trajectory


# ===========================================================================
# Helpers
# ===========================================================================

def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load a checkpoint saved by train.py and return the raw dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run train.py first to create a checkpoint."
        )
    ckpt = torch.load(path, map_location=device)
    return ckpt


def build_model_from_checkpoint(ckpt: dict, device: torch.device) -> SmallUNet:
    """
    Reconstruct the U-Net from the config embedded in the checkpoint, then
    load the saved weights.  This means sample.py stays in sync with the
    model architecture used during training automatically.
    """
    cfg = ckpt.get("config", {})

    # Fall back to Week-1 defaults if the checkpoint pre-dates the config key
    base_channels = cfg.get("base_channels", 32)
    time_dim      = cfg.get("time_dim",      128)

    model = SmallUNet(
        in_channels=1,             # Fashion-MNIST is always grayscale
        base_channels=base_channels,
        time_dim=time_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # disable dropout / batchnorm training behaviour
    return model


def build_diffusion(ckpt: dict) -> GaussianDiffusion:
    """
    Rebuild the GaussianDiffusion object using the same timestep count
    that was used during training (stored in the checkpoint config).
    """
    cfg = ckpt.get("config", {})
    timesteps = cfg.get("timesteps", TIMESTEPS)
    betas     = linear_beta_schedule(timesteps)
    return GaussianDiffusion(betas)


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = GRID_NROW,
) -> None:
    """
    Save a (B, 1, 28, 28) tensor as a PNG grid.

    Images are expected to be in [-1, 1] (the training normalisation).
    We clamp first to remove any tiny out-of-range values produced by the
    reverse SDE, then rescale to [0, 1] for saving.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    # Clamp before normalising so extreme values don't skew the colour range
    images = images.clamp(-1.0, 1.0)
    # value_range=(-1, 1) maps −1 → 0 and 1 → 1 (correct for [-1,1] training data)
    vutils.save_image(images, path, nrow=nrow, normalize=True, value_range=(-1.0, 1.0))
    print(f"  Saved: {path}")


def save_trajectory_grid(
    trajectory: list[tuple[int, torch.Tensor]],
    path: str,
) -> None:
    """
    Save a single row of snapshots showing denoising progress over time.

    Each snapshot is the first image from the batch, taken every TRAJ_EVERY
    timesteps.  The leftmost frame is the noisiest (high t), the rightmost
    is the final generated image (t=0).
    """
    if not trajectory:
        return

    # Take only the first sample from each batch snapshot
    frames = [img[0:1] for _, img in trajectory]   # each is (1, 1, 28, 28)
    strip  = torch.cat(frames, dim=0)               # (num_frames, 1, 28, 28)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    strip = strip.clamp(-1.0, 1.0)
    vutils.save_image(
        strip, path,
        nrow=len(frames),          # all frames in a single row
        normalize=True,
        value_range=(-1.0, 1.0),
    )
    print(f"  Saved: {path}")


# ===========================================================================
# Main
# ===========================================================================

def generate(checkpoint_path: str) -> None:
    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating on: {device}")

    # ── Load checkpoint ───────────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = load_checkpoint(checkpoint_path, device)
    epoch = ckpt.get("epoch", "?")
    loss  = ckpt.get("loss",  float("nan"))
    print(f"  Epoch {epoch}  |  training loss: {loss:.4f}")

    # ── Rebuild model ─────────────────────────────────────────────────────
    model = build_model_from_checkpoint(ckpt, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # ── Rebuild diffusion (must use the same schedule as training) ────────
    diffusion = build_diffusion(ckpt)

    # Move all precomputed tensors to device so _gather's .to() is a no-op
    # (avoids repeated small host↔device transfers inside the loop)
    for attr in (
        "betas",
        "alphas",
        "alpha_bars",
        "sqrt_alpha_bars",
        "sqrt_one_minus_alpha_bars",
        "sqrt_recip_alphas",
        "betas_over_sqrt_one_minus_alpha_bars",
        "sqrt_betas",
    ):
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))

    # ── Generate samples ──────────────────────────────────────────────────
    print(f"Generating {NUM_SAMPLES} samples with T={diffusion.T} denoising steps …")

    epoch_tag = f"epoch{epoch:04d}" if isinstance(epoch, int) else str(epoch)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if SAVE_TRAJECTORY:
        # sample_with_trajectory returns both final images and snapshots
        samples, trajectory = diffusion.sample_with_trajectory(
            model,
            num_samples=NUM_SAMPLES,
            device=device,
            save_every=TRAJ_EVERY,
        )
    else:
        samples = diffusion.sample(model, num_samples=NUM_SAMPLES, device=device)
        trajectory = []

    # ── Save outputs ──────────────────────────────────────────────────────
    samples_cpu = samples.cpu()

    grid_path = os.path.join(OUTPUT_DIR, f"samples_{epoch_tag}.png")
    save_image_grid(samples_cpu, grid_path, nrow=GRID_NROW)

    # Always write a "latest" alias so external scripts have a stable path
    latest_path = os.path.join(OUTPUT_DIR, "samples_latest.png")
    save_image_grid(samples_cpu, latest_path, nrow=GRID_NROW)

    if SAVE_TRAJECTORY and trajectory:
        traj_path = os.path.join(OUTPUT_DIR, f"trajectory_{epoch_tag}.png")
        save_trajectory_grid(trajectory, traj_path)

    print("Done.")


# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM sample generator (Week 2)")
    parser.add_argument(
        "--ckpt",
        default=CHECKPOINT_PATH,
        help=f"Path to checkpoint .pt file (default: {CHECKPOINT_PATH})",
    )
    args = parser.parse_args()
    generate(args.ckpt)
