"""
schedule.py — Noise schedules for diffusion models

A beta schedule controls how much noise is added at each timestep.
Starting small (beta_start) and growing (beta_end) means:
  - early timesteps barely disturb the image
  - later timesteps push the image toward pure Gaussian noise

This file is intentionally kept easy to extend: just add new functions
(e.g. cosine_beta_schedule) following the same signature and return type.
"""

import math
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
# Cosine schedule
# ---------------------------------------------------------------------------


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
) -> torch.Tensor:
    """
    Cosine noise schedule from Nichol & Dhariwal ("Improved DDPMs", 2021).

    Key idea: instead of specifying β_t directly, we specify the cumulative
    product ᾱ_t as a cosine curve, then derive betas from consecutive ratios:

        f(t)   = cos²( (t/T + s) / (1 + s) · π/2 )
        ᾱ_t    = f(t) / f(0)
        β_t    = 1 − ᾱ_t / ᾱ_{t−1}        (clipped to [0, 0.999])

    The small offset `s` prevents β_t from being near zero at t=0, which
    caused training instability in the linear schedule at very small t.

    Args:
        timesteps: total number of diffusion steps T
        s:         small offset to prevent singularity near t=0 (default 0.008)

    Returns:
        betas: 1-D float32 tensor of shape (timesteps,)

    Reference:
        Nichol & Dhariwal, 2021 — https://arxiv.org/abs/2102.09672
    """
    # We evaluate f at t = 0, 1, …, T  (T+1 points) so we can take differences.
    steps = timesteps + 1
    # t/T in [0, 1], then shifted and scaled so the argument to cos is in [0, π/2]
    x = torch.linspace(0, timesteps, steps)          # (T+1,)
    alphas_bar = torch.cos(
        ((x / timesteps) + s) / (1.0 + s) * math.pi / 2.0
    ) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]          # normalise so ᾱ_0 = 1

    # β_t = 1 − ᾱ_t / ᾱ_{t-1}  for t = 1 … T
    betas = 1.0 - (alphas_bar[1:] / alphas_bar[:-1])

    # Clip to [0, 0.999] — avoids β_t = 1 which would make ᾱ_t = 0 and
    # cause division-by-zero inside GaussianDiffusion.
    return betas.clamp(0.0, 0.999).float()


# ---------------------------------------------------------------------------
# Schedule dispatcher
# ---------------------------------------------------------------------------

SUPPORTED_SCHEDULES = ("linear", "cosine")


def get_beta_schedule(schedule_name: str, timesteps: int, **kwargs) -> torch.Tensor:
    """
    Return betas for the named schedule.

    Args:
        schedule_name: one of "linear" or "cosine"
        timesteps:     number of diffusion steps T
        **kwargs:      forwarded to the underlying schedule function
                       (e.g. beta_start/beta_end for linear, s for cosine)

    Returns:
        betas: 1-D float32 tensor of shape (timesteps,)

    Raises:
        ValueError: if schedule_name is not recognised

    Example:
        >>> betas = get_beta_schedule("cosine", 200)
        >>> betas = get_beta_schedule("linear", 200, beta_start=1e-4, beta_end=2e-2)
    """
    if schedule_name == "linear":
        return linear_beta_schedule(timesteps, **kwargs)
    elif schedule_name == "cosine":
        return cosine_beta_schedule(timesteps, **kwargs)
    else:
        raise ValueError(
            f"Unknown schedule '{schedule_name}'. "
            f"Supported: {SUPPORTED_SCHEDULES}"
        )
