"""
diffusion.py — Gaussian diffusion utilities

Implements the DDPM forward (noising) process.

Key identity used everywhere:
    x_t = sqrt(ᾱ_t) * x_0  +  sqrt(1 - ᾱ_t) * ε
        where  ε ~ N(0, I)
               ᾱ_t = product of (1 - β_s) for s = 1..t

This "reparameterisation trick" lets us jump directly from a clean image x_0
to a noisy image x_t at any timestep t — no need to simulate each step
sequentially.  Training samples random t, noises x_0 to x_t, then asks the
model to predict ε from (x_t, t).
"""

import torch


class GaussianDiffusion:
    """
    Holds all precomputed diffusion coefficients and implements q_sample.

    Attributes are kept as plain tensors (CPU by default).  The helper
    methods move them to the same device as the input on the fly, so you
    never need to worry about device placement when calling q_sample.

    Args:
        betas: 1-D float tensor of shape (T,) from a beta schedule
    """

    def __init__(self, betas: torch.Tensor) -> None:
        self.T = len(betas)          # total number of timesteps

        # β_t  (noise variance added at each step)
        self.betas = betas.float()

        # α_t = 1 - β_t
        self.alphas = 1.0 - self.betas

        # ᾱ_t = ∏_{s=1}^{t} α_s  (cumulative product of alphas)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        # Precompute the two scalar coefficients used in q_sample:
        #   sqrt(ᾱ_t)          — scales the clean image x_0
        #   sqrt(1 - ᾱ_t)      — scales the noise ε
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _gather(self, coeffs: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
        """
        Look up per-timestep coefficients for a batch of timestep indices t,
        then reshape so they broadcast against (B, C, H, W) tensors.

        Args:
            coeffs: 1-D tensor of shape (T,) — one value per timestep
            t:      1-D integer tensor of shape (B,) — one index per sample
            ndim:   number of dimensions in the data tensor (e.g. 4 for images)

        Returns:
            tensor of shape (B, 1, 1, ..., 1)  [ndim - 1 trailing singletons]
        """
        # Move coefficients to the same device as t  (no-op if already there)
        out = coeffs.to(t.device).gather(0, t)   # (B,)
        # Append singleton dims so the result broadcasts over (B, C, H, W)
        return out.view(-1, *([1] * (ndim - 1)))  # (B, 1, 1, 1) for images

    # ------------------------------------------------------------------
    # Forward noising process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a noisy version of x0 at timestep t.

        Implements:
            x_t = sqrt(ᾱ_t) · x_0  +  sqrt(1 − ᾱ_t) · ε

        Args:
            x0:    clean images, shape (B, C, H, W)
            t:     integer timestep indices, shape (B,)  in [0, T)
            noise: optional pre-sampled Gaussian noise with same shape as x0;
                   sampled fresh if not provided

        Returns:
            x_t:   noisy images at timestep t,  shape (B, C, H, W)
            noise: the Gaussian noise that was applied, shape (B, C, H, W)
                   (useful when the caller needs the ground-truth target)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_abar     = self._gather(self.sqrt_alpha_bars,           t, x0.ndim)
        sqrt_one_mabar = self._gather(self.sqrt_one_minus_alpha_bars, t, x0.ndim)

        x_t = sqrt_abar * x0 + sqrt_one_mabar * noise
        return x_t, noise
