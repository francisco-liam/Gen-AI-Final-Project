"""
schedule.py — Noise schedules for diffusion models

A beta schedule controls how much noise is added at each timestep.
Starting small (beta_start) and growing (beta_end) means:
  - early timesteps barely disturb the image
  - later timesteps push the image toward pure Gaussian noise

This file is intentionally kept easy to extend: just add new functions
(e.g. cosine_beta_schedule) following the same signature and return type.
"""

import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    """
    Linear beta schedule from Ho et al. (DDPM, 2020).

    Returns a 1-D tensor of shape (timesteps,) where beta[t] grows
    linearly from beta_start to beta_end.

    Args:
        timesteps:  total number of diffusion steps T
        beta_start: noise level at t=0 (very small)
        beta_end:   noise level at t=T-1 (moderate)

    Returns:
        betas: 1-D float32 tensor of shape (timesteps,)

    Example:
        >>> betas = linear_beta_schedule(200)
        >>> betas.shape
        torch.Size([200])
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# ---------------------------------------------------------------------------
# Placeholder for Week 2 — cosine schedule
# ---------------------------------------------------------------------------
# def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
#     """Cosine schedule from Nichol & Dhariwal (2021)."""
#     ...
