"""
train.py — Week 1 training script for DDPM epsilon prediction on Fashion-MNIST

Usage:
    python train.py

What this script does:
    1. Loads Fashion-MNIST (downloads on first run)
    2. Builds a linear beta schedule
    3. Instantiates a small U-Net that predicts noise
    4. Trains the model with the standard DDPM objective:
           L = E_t [ || ε - ε_θ(x_t, t) ||² ]
    5. Prints average loss per epoch
    6. Saves checkpoints to ./checkpoints/

Week 1 goal: verify the loss decreases steadily over the first ~10 epochs.
A healthy run should see the MSE loss drop from ~1.0 toward ~0.4–0.6.
"""

import os

import torch
import torch.nn as nn
from torch.optim import Adam

from data      import get_dataloaders
from diffusion import GaussianDiffusion
from model     import SmallUNet
from schedule  import linear_beta_schedule


# ===========================================================================
# Configuration — tune these to match your hardware
# ===========================================================================

BATCH_SIZE     = 128      # reduce to 64 if you run out of GPU memory
LEARNING_RATE  = 2e-4
EPOCHS         = 10
TIMESTEPS      = 200      # T: number of diffusion steps
                          # 200 is fast for experimentation; use 1000 for full DDPM

CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY     = 2        # save a checkpoint every N epochs

# Set num_workers=0 if you see multiprocessing errors on Windows
NUM_WORKERS    = 2


# ===========================================================================
# Training loop
# ===========================================================================

def train() -> None:
    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # ── Data ────────────────────────────────────────────────────────────
    train_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    print(f"Training batches per epoch: {len(train_loader)}")

    # ── Noise schedule and diffusion helper ─────────────────────────────
    betas     = linear_beta_schedule(TIMESTEPS)
    diffusion = GaussianDiffusion(betas)
    # Move precomputed tensors to device once (avoids repeated host↔device
    # transfers inside q_sample; _gather's .to() becomes a cheap no-op)
    diffusion.sqrt_alpha_bars           = diffusion.sqrt_alpha_bars.to(device)
    diffusion.sqrt_one_minus_alpha_bars = diffusion.sqrt_one_minus_alpha_bars.to(device)

    # ── Model and optimiser ─────────────────────────────────────────────
    model = SmallUNet(
        in_channels=1,     # Fashion-MNIST is grayscale
        base_channels=32,
        time_dim=128,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # ── Checkpoint directory ─────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for x0, _ in train_loader:
            x0 = x0.to(device)               # (B, 1, 28, 28) in [-1, 1]

            # ── Sample random timesteps t ~ Uniform[0, T) ──────────────
            # Each sample in the batch gets its own independent t.
            t = torch.randint(0, TIMESTEPS, (x0.shape[0],), device=device)

            # ── Forward process: corrupt x0 to x_t ────────────────────
            # noise is the ground-truth target we want the model to predict
            x_t, noise = diffusion.q_sample(x0, t)   # both (B, 1, 28, 28)

            # ── Predict noise with the U-Net ───────────────────────────
            noise_pred = model(x_t, t)                # (B, 1, 28, 28)

            # ── DDPM loss: MSE between true and predicted noise ─────────
            # This is the "simple" objective from Eq. 14 of Ho et al. 2020.
            loss = nn.functional.mse_loss(noise_pred, noise)

            # ── Backprop ───────────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch:>3}/{EPOCHS}]  loss: {avg_loss:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"ckpt_epoch{epoch:04d}.pt")
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss":                 avg_loss,
                    # Save config so checkpoints are self-documenting
                    "config": {
                        "timesteps":     TIMESTEPS,
                        "batch_size":    BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "base_channels": 32,
                        "time_dim":      128,
                    },
                },
                ckpt_path,
            )
            print(f"           checkpoint saved → {ckpt_path}")

    print("Training complete.")


# ===========================================================================

if __name__ == "__main__":
    train()
