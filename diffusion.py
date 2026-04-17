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

        # ── Extra coefficients needed for the reverse (p_sample) process ──
        #
        # 1/sqrt(α_t)  — reciprocal square-root of per-step alpha (not ᾱ_t)
        #   Used to "undo" the forward step scaling.
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # β_t / sqrt(1 - ᾱ_t)  — the epsilon coefficient in the posterior mean
        #   From the DDPM posterior mean equation:
        #       μ_θ(x_t, t) = (1/sqrt(α_t)) · (x_t − β_t/sqrt(1−ᾱ_t) · ε_θ)
        self.betas_over_sqrt_one_minus_alpha_bars = (
            self.betas / self.sqrt_one_minus_alpha_bars
        )

        # sqrt(β_t)  — the noise standard-deviation for the reverse step
        #   DDPM uses σ_t = sqrt(β_t) (the "upper-bound" fixed variance).
        #   At t=0 this is never applied because no noise is added on the
        #   final step.
        self.sqrt_betas = torch.sqrt(self.betas)

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

    # ------------------------------------------------------------------
    # Reverse denoising process
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t_index: int,
    ) -> torch.Tensor:
        """
        Perform one reverse DDPM denoising step: x_t → x_{t-1}.

        DDPM posterior mean (Ho et al. 2020, Eq. 11):
            μ_θ(x_t, t) = (1/√α_t) · (x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t, t))

        Posterior sample:
            x_{t-1} = μ_θ  +  √β_t · z,   z ~ N(0, I)   if t > 0
            x_{t-1} = μ_θ                                 if t = 0

        Note: σ_t = √β_t is the "upper-bound" fixed variance used in the
        original DDPM paper.  It is simple and works well in practice.

        Args:
            model:   the trained U-Net (must output ε̂ of shape (B,1,28,28))
            x_t:     current noisy images, shape (B, 1, 28, 28), on device
            t_index: scalar int in [0, T), the same for every sample in batch

        Returns:
            x_{t-1}: denoised images, shape (B, 1, 28, 28)
        """
        B, device = x_t.shape[0], x_t.device

        # Build a (B,) batch of identical timestep indices so the model
        # and _gather receive the expected tensor shape.
        t = torch.full((B,), t_index, device=device, dtype=torch.long)

        # ── 1. Predict noise ε̂ with the U-Net ──────────────────────────
        eps_theta = model(x_t, t)                        # (B, 1, 28, 28)

        # ── 2. Gather per-timestep scalar coefficients ─────────────────
        betas_t = self._gather(
            self.betas_over_sqrt_one_minus_alpha_bars, t, x_t.ndim
        )
        sqrt_recip_alpha_t = self._gather(
            self.sqrt_recip_alphas, t, x_t.ndim
        )

        # ── 3. Compute posterior mean μ_θ ───────────────────────────────
        # μ_θ = (1/√α_t) · (x_t − β_t/√(1−ᾱ_t) · ε̂)
        model_mean = sqrt_recip_alpha_t * (x_t - betas_t * eps_theta)

        # ── 4. Add stochastic noise (skip at the very last step t=0) ───
        if t_index == 0:
            # Final step: return the mean directly — no extra noise.
            return model_mean
        else:
            sqrt_betas_t = self._gather(self.sqrt_betas, t, x_t.ndim)
            noise = torch.randn_like(x_t)
            return model_mean + sqrt_betas_t * noise

    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        num_samples: int,
        device: torch.device,
        image_shape: tuple = (1, 28, 28),
    ) -> torch.Tensor:
        """
        Full reverse diffusion: generate images from pure Gaussian noise.

        Algorithm:
            x_T ~ N(0, I)
            for t = T-1 down to 0:
                x_{t-1} = p_sample(model, x_t, t)
            return x_0

        Args:
            model:       trained U-Net in eval mode
            num_samples: number of images to generate (B)
            device:      torch device

        Returns:
            generated images, shape (B, 1, 28, 28), values in roughly [-1, 1]
        """
        model.eval()

        # Start from isotropic Gaussian noise  x_T ~ N(0, I)
        x = torch.randn(num_samples, *image_shape, device=device)

        # Iteratively denoise from t = T-1 down to t = 0
        for t_index in reversed(range(self.T)):
            x = self.p_sample(model, x, t_index)

        return x  # x_0: final generated images

    @torch.no_grad()
    def sample_with_trajectory(
        self,
        model: torch.nn.Module,
        num_samples: int,
        device: torch.device,
        save_every: int = 20,
        image_shape: tuple = (1, 28, 28),
    ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        """
        Like sample(), but also returns intermediate denoising states so you
        can visualise how images emerge from noise over time.

        Args:
            model:       trained U-Net in eval mode
            num_samples: number of images to generate
            device:      torch device
            save_every:  record a snapshot every N timesteps (e.g. 20)

        Returns:
            x_0:         final generated images,  shape (B, 1, 28, 28)
            trajectory:  list of (t_index, images_cpu) tuples, ordered from
                         high t (noisy) to low t (clean).  images_cpu are
                         cloned to CPU so GPU memory is not held.
        """
        model.eval()

        x = torch.randn(num_samples, *image_shape, device=device)
        trajectory: list[tuple[int, torch.Tensor]] = []

        for t_index in reversed(range(self.T)):
            x = self.p_sample(model, x, t_index)
            # Record snapshot at regular intervals and always at t=0
            if t_index % save_every == 0:
                trajectory.append((t_index, x.cpu().clone()))

        return x, trajectory
