"""
UNet architecture for diffusion models.

Clean, simple implementation with clear skip connection handling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb_scale)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with float32 computation for stability."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class ResBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.1, num_groups: int = 8):
        super().__init__()
        self.norm1 = GroupNorm32(num_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.norm2 = GroupNorm32(num_groups, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.norm = GroupNorm32(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = F.softmax(torch.einsum("bcn,bcm->bnm", q, k) * (C ** -0.5), dim=-1)
        h = torch.einsum("bnm,bcm->bcn", attn, v).reshape(B, C, H, W)
        return x + self.proj(h)


class UNet(nn.Module):
    """
    Simple UNet for diffusion models.

    Structure:
    - Encoder: [ResBlock, ResBlock, Downsample] × num_levels
    - Middle: ResBlock, Attention, ResBlock
    - Decoder: [ResBlock, ResBlock, ResBlock, Upsample] × num_levels (with skip connections)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.1,
        num_groups: int = 8,
        conditional: bool = False,
        image_size: int = 64,
    ):
        super().__init__()

        self.conditional = conditional
        time_dim = base_channels * 4
        num_levels = len(channel_mult)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Conditioning
        if conditional:
            self.cond_conv = nn.Sequential(
                nn.Conv2d(in_channels, base_channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
            )
            self.combine = nn.Conv2d(base_channels * 2, base_channels, 1)

        # Input
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch_list = [base_channels]  # Track channels for decoder
        ch = base_channels
        res = image_size

        for level in range(num_levels):
            out_ch = base_channels * channel_mult[level]

            # ResBlocks
            for _ in range(num_res_blocks):
                self.encoder.append(ResBlock(ch, out_ch, time_dim, dropout, num_groups))
                ch = out_ch
                if res in attention_resolutions:
                    self.encoder.append(AttentionBlock(ch, num_groups))
                ch_list.append(ch)

            # Downsample (except last)
            if level < num_levels - 1:
                self.downsample.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                ch_list.append(ch)
                res //= 2

        # Middle
        self.mid1 = ResBlock(ch, ch, time_dim, dropout, num_groups)
        self.mid_attn = AttentionBlock(ch, num_groups)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout, num_groups)

        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for level in reversed(range(num_levels)):
            out_ch = base_channels * channel_mult[level]

            # ResBlocks with skip connections
            for i in range(num_res_blocks + 1):
                skip_ch = ch_list.pop()
                self.decoder.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout, num_groups))
                ch = out_ch
                if res in attention_resolutions:
                    self.decoder.append(AttentionBlock(ch, num_groups))

            # Upsample (except last)
            if level > 0:
                self.upsample.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(ch, ch, 3, padding=1),
                ))
                res *= 2

        # Output
        self.out_norm = GroupNorm32(num_groups, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

        # Store for forward pass
        self.num_levels = num_levels
        self.num_res_blocks = num_res_blocks
        self.base_channels = base_channels

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_mlp(get_timestep_embedding(t, self.base_channels))

        # Input
        h = self.input_conv(x)

        # Conditioning
        if self.conditional:
            c = self.cond_conv(cond if cond is not None else torch.zeros_like(x))
            h = self.combine(torch.cat([h, c], dim=1))

        # Encoder
        skips = [h]
        enc_idx = 0
        down_idx = 0

        for level in range(self.num_levels):
            for _ in range(self.num_res_blocks):
                h = self.encoder[enc_idx](h, t_emb)
                enc_idx += 1
                # Check for attention
                if enc_idx < len(self.encoder) and isinstance(self.encoder[enc_idx], AttentionBlock):
                    h = self.encoder[enc_idx](h)
                    enc_idx += 1
                skips.append(h)

            if level < self.num_levels - 1:
                h = self.downsample[down_idx](h)
                down_idx += 1
                skips.append(h)

        # Middle
        h = self.mid1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid2(h, t_emb)

        # Decoder
        dec_idx = 0
        up_idx = 0

        for level in reversed(range(self.num_levels)):
            for _ in range(self.num_res_blocks + 1):
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = self.decoder[dec_idx](h, t_emb)
                dec_idx += 1
                # Check for attention
                if dec_idx < len(self.decoder) and isinstance(self.decoder[dec_idx], AttentionBlock):
                    h = self.decoder[dec_idx](h)
                    dec_idx += 1

            if level > 0:
                h = self.upsample[up_idx](h)
                up_idx += 1

        # Output
        return self.out_conv(F.silu(self.out_norm(h)))


def create_model(config) -> UNet:
    """Create UNet model from config."""
    return UNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_channels=config.model.base_channels,
        channel_mult=config.model.channel_mult,
        num_res_blocks=config.model.num_res_blocks,
        attention_resolutions=config.model.attention_resolutions,
        dropout=config.model.dropout,
        num_groups=config.model.num_groups,
        conditional=(config.model.type == "conditional"),
        image_size=config.data.image_size,
    )
