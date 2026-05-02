"""
model.py — Improved DDPM U-Net for CIFAR-10 (32×32) and noise schedule.

Architecture follows Nichol & Dhariwal "Improved DDPM" (2021) with:
  • GroupNorm instead of BatchNorm  (stable across noise levels / batch sizes)
  • SiLU (Swish) activations        (empirically better for diffusion)
  • 256-dim time embed, 2-layer MLP (much richer timestep conditioning)
  • AdaGN scale-shift conditioning  (time modulates each ResBlock via γ, β)
  • True residual shortcuts         (1×1 conv projection when channels differ)
  • Multi-head self-attention at 8×8 and 16×16 resolutions
  • Bottleneck: attn-ResBlock sandwich at lowest (8×8) resolution

Why this matters for MIA:
  A better-trained DDPM memorises training samples more sharply, widening the
  in/out loss gap that LiRA and Strong LiRA exploit.  Attention + AdaGN are the
  two biggest drivers of that improved per-sample memorisation.

SimpleUNet is kept as an alias of ImprovedUNet for backwards compatibility
with existing train.py / modal_train.py imports.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
#  Noise schedule
# ─────────────────────────────────────────────────────────────────

def make_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02,
                  device: str = "cpu") -> torch.Tensor:
    """Linear beta schedule → returns alpha_bar of shape (T,)."""
    betas     = torch.linspace(beta_start, beta_end, T, device=device)
    alpha_bar = torch.cumprod(1.0 - betas, dim=0)
    return alpha_bar


# ─────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────

def _norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with 32 groups (or fewer if channels < 32)."""
    groups = min(32, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class SinusoidalPE(nn.Module):
    """Sinusoidal positional embedding for timestep t."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ─────────────────────────────────────────────────────────────────
#  Core building blocks
# ─────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Residual block with AdaGN time-conditioning (scale + shift).

    Time embedding is projected to 2 * out_ch and split into
    scale (γ) and shift (β) that modulate the second GroupNorm,
    following Improved DDPM §3.3.
    """
    def __init__(self, in_ch: int, out_ch: int, t_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        # Projects time embed → scale + shift for AdaGN
        self.time_proj = nn.Linear(t_dim, 2 * out_ch)

        self.norm2   = _norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2   = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Residual projection when channel dims differ
        self.residual = (nn.Conv2d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # AdaGN: scale + shift from time embedding
        scale_shift = self.time_proj(F.silu(t_emb))          # (B, 2*C)
        scale, shift = scale_shift.chunk(2, dim=1)            # (B, C) each
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))

        return h + self.residual(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Spatial multi-head self-attention (for 2-D feature maps).
    Flattens spatial dims → sequence, applies MHA, reshapes back.
    """
    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        assert channels % n_heads == 0
        self.norm = _norm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.proj = nn.Conv2d(channels, channels, 1)   # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)          # (B, HW, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + self.proj(h)                            # residual


class DownBlock(nn.Module):
    """Two ResBlocks + optional attention + 2× stride-2 downsample."""
    def __init__(self, in_ch: int, out_ch: int, t_dim: int,
                 has_attn: bool = False, dropout: float = 0.1):
        super().__init__()
        self.res1 = ResBlock(in_ch,  out_ch, t_dim, dropout)
        self.res2 = ResBlock(out_ch, out_ch, t_dim, dropout)
        self.attn = MultiHeadSelfAttention(out_ch) if has_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x) if isinstance(self.attn, MultiHeadSelfAttention) else self.attn(x)
        skip = x                          # skip connection *before* downsample
        x    = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    """2× upsample + two ResBlocks (with skip concat) + optional attention."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, t_dim: int,
                 has_attn: bool = False, dropout: float = 0.1):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res1 = ResBlock(in_ch + skip_ch, out_ch, t_dim, dropout)
        self.res2 = ResBlock(out_ch,          out_ch, t_dim, dropout)
        self.attn = MultiHeadSelfAttention(out_ch) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:               # guard for odd sizes
            x = F.interpolate(x, size=skip.shape[-2:])
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x) if isinstance(self.attn, MultiHeadSelfAttention) else self.attn(x)
        return x


class Bottleneck(nn.Module):
    """Mid-UNet: ResBlock → Attention → ResBlock (Improved DDPM §A.1)."""
    def __init__(self, channels: int, t_dim: int, dropout: float = 0.1):
        super().__init__()
        self.res1 = ResBlock(channels, channels, t_dim, dropout)
        self.attn = MultiHeadSelfAttention(channels)
        self.res2 = ResBlock(channels, channels, t_dim, dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


# ─────────────────────────────────────────────────────────────────
#  Improved U-Net
# ─────────────────────────────────────────────────────────────────

class ImprovedUNet(nn.Module):
    """
    Improved DDPM U-Net for CIFAR-10 (32×32 RGB input).

    Resolution path (CIFAR-10, 32×32 input):
      32×32 → 16×16 → 8×8   [bottleneck]  → 16×16 → 32×32

    Channel widths:  base_ch × (1, 2, 4) = (128, 256, 512) by default.
    Attention at 16×16 and the bottleneck 8×8 level.

    Parameters
    ----------
    image_channels : int   (3 for RGB)
    base_ch        : int   Base channel multiplier. 128 → ~70 M params total.
    t_dim          : int   Timestep embedding dimension (projected to 4×t_dim).
    dropout        : float Applied in each ResBlock.
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_ch:        int = 128,
        t_dim:          int = 128,
        dropout:        float = 0.1,
    ):
        super().__init__()
        ch = base_ch
        inner_t = 4 * t_dim          # projected time dim inside the network

        # ── Time embedding ────────────────────────────────────────
        self.time_embed = nn.Sequential(
            SinusoidalPE(t_dim),
            nn.Linear(t_dim, inner_t),
            nn.SiLU(),
            nn.Linear(inner_t, inner_t),
        )

        # ── Stem ──────────────────────────────────────────────────
        self.stem = nn.Conv2d(image_channels, ch, 3, padding=1)

        # ── Encoder (32→16→8) ─────────────────────────────────────
        # Level 0: 32×32  no attn
        self.down0 = DownBlock(ch,      ch * 2, inner_t, has_attn=False, dropout=dropout)
        # Level 1: 16×16  with attn
        self.down1 = DownBlock(ch * 2,  ch * 4, inner_t, has_attn=True,  dropout=dropout)

        # ── Bottleneck 8×8 ────────────────────────────────────────
        self.mid = Bottleneck(ch * 4, inner_t, dropout=dropout)

        # ── Decoder (8→16→32) ─────────────────────────────────────
        # Level 1 up: 8→16, skip from down1 (ch*4)
        self.up1 = UpBlock(ch * 4, ch * 4, ch * 2, inner_t, has_attn=True,  dropout=dropout)
        # Level 0 up: 16→32, skip from down0 (ch*2)
        self.up0 = UpBlock(ch * 2, ch * 2, ch,     inner_t, has_attn=False, dropout=dropout)

        # ── Output head ───────────────────────────────────────────
        self.out_norm = _norm(ch)
        self.out_conv = nn.Conv2d(ch, image_channels, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        t = self.time_embed(timestep)          # (B, inner_t)

        # Stem
        x = self.stem(x)                       # (B, ch, 32, 32)

        # Encoder
        x, s0 = self.down0(x, t)              # x:(B, 2ch, 16, 16)  s0:(B, 2ch, 32, 32)
        x, s1 = self.down1(x, t)              # x:(B, 4ch,  8,  8)  s1:(B, 4ch, 16, 16)

        # Bottleneck
        x = self.mid(x, t)                    # (B, 4ch, 8, 8)

        # Decoder
        x = self.up1(x, s1, t)               # (B, 2ch, 16, 16)
        x = self.up0(x, s0, t)               # (B, ch,  32, 32)

        # Output
        return self.out_conv(F.silu(self.out_norm(x)))


# ─────────────────────────────────────────────────────────────────
#  Backwards-compatibility alias
# ─────────────────────────────────────────────────────────────────

def SimpleUNet(**kwargs) -> ImprovedUNet:           # type: ignore[return-value]
    """
    Legacy alias.  Calling SimpleUNet() now returns an ImprovedUNet instance
    so existing train.py / modal_train.py imports keep working unchanged.
    Keyword arguments from the old API (down_channels, up_channels, t_dim)
    are silently ignored to avoid breakage.
    """
    # Strip old-style kwargs that don't apply to ImprovedUNet
    for legacy_key in ("down_channels", "up_channels"):
        kwargs.pop(legacy_key, None)
    return ImprovedUNet(**kwargs)
