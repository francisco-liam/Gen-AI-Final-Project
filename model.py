"""
model.py — Small DDPM-style U-Net for Fashion-MNIST

Architecture overview (input: B × 1 × 28 × 28):

    [Encoder]
    enc1:  1  → 32  channels, 28×28   (ResBlock)
    down1: 32 → 64  channels, 14×14   (strided Conv)
    enc2:  64 → 64  channels, 14×14   (ResBlock)
    down2: 64 → 128 channels,  7×7    (strided Conv)

    [Bottleneck]
    bot:   128 → 128 channels, 7×7    (ResBlock)

    [Decoder — skip connections concatenated before each ResBlock]
    up1:   128 → 64  channels, 14×14  (ConvTranspose2d)
    dec1:  128 → 64  channels, 14×14  (ResBlock, after concat with enc2)
    up2:   64  → 32  channels, 28×28  (ConvTranspose2d)
    dec2:  64  → 32  channels, 28×28  (ResBlock, after concat with enc1)

    [Output]
    out:   32 → 1   channels, 28×28   (1×1 Conv)

Timestep conditioning:
    Each ResBlock receives a per-sample timestep embedding added to its
    feature maps.  The embedding is a sinusoidal positional encoding passed
    through a small MLP (similar to the original DDPM paper).
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Produce sinusoidal positional embeddings for a batch of timestep ints.

    Based on the transformer positional encoding (Vaswani et al., 2017),
    adapted for diffusion timesteps.  Each frequency carries information
    about a different "scale" of time.

    Args:
        timesteps: integer tensor of shape (B,)
        dim:       embedding dimensionality (must be even)

    Returns:
        tensor of shape (B, dim)
    """
    assert dim % 2 == 0, "embedding dim must be even"
    half = dim // 2

    # frequencies: 1 / 10000^(2i/dim)  for i in [0, half)
    freqs = torch.exp(
        -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / (half - 1)
    )                                                 # (half,)

    args = timesteps.float()[:, None] * freqs[None, :]  # (B, half)
    emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    return emb


class TimestepEmbedding(nn.Module):
    """
    Lift a raw sinusoidal embedding through a 2-layer MLP.
    The MLP gives the model a chance to learn a richer time representation.

    Args:
        dim: sinusoidal embedding size (also the output size)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: integer tensor (B,)  →  embedding (B, dim)"""
        emb = sinusoidal_embedding(t, self.dim)  # (B, dim)
        return self.mlp(emb)                     # (B, dim)


# ---------------------------------------------------------------------------
# Core building block
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Residual convolutional block with timestep conditioning.

    Structure:
        h = act(norm(conv1(x)))
        h = h + time_projection(t_emb)[:, :, None, None]   ← inject time
        h = act(norm(conv2(h)))
        return h + residual_conv(x)

    GroupNorm is used instead of BatchNorm so that small batch sizes and
    single-sample inference both work correctly.

    Args:
        in_channels:  input channel count
        out_channels: output channel count
        time_dim:     dimensionality of the timestep embedding
        num_groups:   number of groups for GroupNorm (must divide both channel counts)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_groups: int = 8,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act   = nn.SiLU()

        # Project time embedding to channel dimension for additive conditioning
        self.time_proj = nn.Linear(time_dim, out_channels)

        # 1×1 conv to match channel dims in the residual branch (if needed)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     feature map  (B, in_channels, H, W)
            t_emb: timestep embedding  (B, time_dim)

        Returns:
            (B, out_channels, H, W)
        """
        h = self.act(self.norm1(self.conv1(x)))

        # Inject timestep info: project to (B, out_channels), then broadcast
        t = self.act(self.time_proj(t_emb))         # (B, out_channels)
        h = h + t[:, :, None, None]                 # (B, out_channels, H, W)

        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


# ---------------------------------------------------------------------------
# Self-attention block  (bottleneck only)
# ---------------------------------------------------------------------------

class SelfAttention(nn.Module):
    """
    Multi-head self-attention over spatial positions.

    Applied at the bottleneck (smallest spatial resolution) where the
    receptive field of 3×3 convolutions is most limited relative to the
    feature map size.  Global attention here lets the model reason about
    long-range dependencies when denoising at high noise levels — the key
    advantage that allows the cosine schedule to outperform linear.

    Implementation follows Nichol & Dhariwal (2021) / lucidrains:
      - GroupNorm before Q/K/V projections (pre-norm)
      - Scaled dot-product attention
      - Residual connection

    Args:
        channels:   number of input/output channels
        num_heads:  number of attention heads (must divide channels)
        num_groups: GroupNorm groups (must divide channels)
    """

    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8) -> None:
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads
        self.scale     = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv  = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels,     kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        # Project to Q, K, V  —  each (B, C, H, W)
        qkv = self.qkv(h).chunk(3, dim=1)
        q, k, v = [t.reshape(B, self.num_heads, self.head_dim, H * W) for t in qkv]

        # Scaled dot-product attention over spatial positions
        attn = torch.softmax(torch.einsum("bhdi,bhdj->bhij", q, k) * self.scale, dim=-1)
        out  = torch.einsum("bhij,bhdj->bhdi", attn, v)   # (B, heads, head_dim, HW)
        out  = out.reshape(B, C, H, W)

        return x + self.proj(out)   # residual

class SmallUNet(nn.Module):
    """
    Lightweight U-Net that predicts the noise ε from a noisy image x_t and
    a timestep t.

    Designed for Fashion-MNIST (B, 1, 28, 28) and CIFAR-10 (B, 3, 32, 32).
    The spatial resolution is halved twice and then restored with transposed
    convs.

    Args:
        in_channels:    image channels (1 for grayscale, 3 for RGB)
        base_channels:  channel count in the first encoder stage; subsequent
                        stages multiply this by 2 and 4
        time_dim:       dimensionality of the timestep embedding MLP output
        use_attention:  if True, insert a SelfAttention block at the bottleneck
                        (after the ResBlock).  Enables long-range spatial
                        reasoning needed for the cosine schedule to outperform
                        linear on complex datasets.
    """

    def __init__(
        self,
        in_channels:   int  = 1,
        base_channels: int  = 32,
        time_dim:      int  = 128,
        use_attention: bool = False,
    ) -> None:
        super().__init__()

        c = base_channels  # shorthand

        # ── Timestep embedding ──────────────────────────────────────────
        self.time_embedding = TimestepEmbedding(time_dim)

        # ── Encoder ─────────────────────────────────────────────────────
        self.enc1  = ResBlock(in_channels, c, time_dim)
        self.down1 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.enc2  = ResBlock(c * 2, c * 2, time_dim)
        self.down2 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)

        # ── Bottleneck ───────────────────────────────────────────────────
        self.bottleneck = ResBlock(c * 4, c * 4, time_dim)
        # Optional self-attention: num_groups must divide c*4
        self.attn = SelfAttention(c * 4, num_heads=4, num_groups=min(8, c * 4)) \
                    if use_attention else nn.Identity()

        # ── Decoder ─────────────────────────────────────────────────────
        self.up1  = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(c * 4, c * 2, time_dim)
        self.up2  = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(c * 2, c, time_dim)

        # ── Output projection ────────────────────────────────────────────
        self.out_conv = nn.Conv2d(c, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict the noise ε that was added to x_0 to produce x_t.

        Args:
            x: noisy image tensor  (B, 1, 28, 28)
            t: integer timestep tensor  (B,)

        Returns:
            predicted noise  (B, 1, 28, 28)
        """
        # Compute a single timestep embedding for the whole forward pass
        t_emb = self.time_embedding(t)   # (B, time_dim)

        # ── Encoder ─────────────────────────────────────────────
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)

        # ── Bottleneck (+ optional self-attention) ───────────────
        b  = self.bottleneck(self.down2(e2), t_emb)
        b  = self.attn(b)               # identity when use_attention=False

        # ── Decoder with skip connections ────────────────────────
        d1 = self.dec1(torch.cat([self.up1(b),  e2], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1), t_emb)

        return self.out_conv(d2)
