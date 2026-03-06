"""Single-pass CNN architecture for debris detection with variable track/pol inputs.

Fuses raw SAR set features with static terrain channels via a shared-weight
set encoder with spatial attention, then decodes to a debris probability map.

Input patches are 128x128 pixels. The number of SAR track/pol maps varies
per sample (typically 2-8), handled via set encoding + spatial attention.

Usage:
    model = DebrisDetector()
    sar_maps = [torch.randn(B, 2, 128, 128) for _ in range(N)]
    static = torch.randn(B, 8, 128, 128)
    out = model(sar_maps, static)  # (B, 1, 128, 128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sarvalanche.ml.v2.channels import N_STATIC


class ConvBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN-ReLU block with optional stride for downsampling."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SetEncoder(nn.Module):
    """Shared-weight CNN encoding each track/pol map independently.

    Input: (B, 2, 128, 128) per track/pol (change + ANF)
    Output: (B, feat_dim, 8, 8)
    """

    def __init__(self, in_ch: int = 1, feat_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, 16, stride=2),    # → (16, 64, 64)
            ConvBlock(16, 32, stride=2),       # → (32, 32, 32)
            ConvBlock(32, feat_dim, stride=2), # → (64, 16, 16)
            ConvBlock(feat_dim, feat_dim, stride=2),  # → (64, 8, 8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SpatialSetAttention(nn.Module):
    """Per-position attention pooling over a variable-size set of feature maps.

    Given N encoded feature maps of shape (B, feat_dim, H, W), produces a
    single aggregated feature map (B, feat_dim, H, W) via learned attention.
    """

    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.query = nn.Parameter(torch.randn(feat_dim))
        self.scale = feat_dim ** 0.5

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Parameters
        ----------
        features : list of (B, feat_dim, H, W) tensors, length N

        Returns:
        -------
        (B, feat_dim, H, W) aggregated features
        """
        # Stack: (B, N, feat_dim, H, W)
        stacked = torch.stack(features, dim=1)
        B, N, C, H, W = stacked.shape

        # Reshape for attention: (B, N, C, H*W) → (B, H*W, N, C)
        stacked_flat = stacked.reshape(B, N, C, H * W).permute(0, 3, 1, 2).contiguous()

        # Query: (C,) → (1, 1, 1, C)
        q = self.query.reshape(1, 1, 1, C)

        # Attention scores: dot product of query with each set member per position
        # (B, H*W, N, C) * (1, 1, 1, C) → sum over C → (B, H*W, N)
        scores = (stacked_flat * q).sum(dim=-1) / self.scale
        weights = F.softmax(scores, dim=-1)  # (B, H*W, N)

        # Weighted sum: (B, H*W, N, 1) * (B, H*W, N, C) → sum over N → (B, H*W, C)
        out = (weights.unsqueeze(-1) * stacked_flat).sum(dim=2)

        # Reshape back: (B, H*W, C) → (B, C, H, W)
        return out.permute(0, 2, 1).contiguous().reshape(B, C, H, W)


class _DeconvBlock(nn.Module):
    """ConvTranspose2d 2x upsample + Conv-BN-ReLU refinement."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.refine = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.up(x))


class Decoder(nn.Module):
    """ConvTranspose decoder from 8x8 to 128x128.

    4 stages of 2x upsampling: 8→16→32→64→128.
    Each stage: ConvTranspose2d (learned upsample) + Conv-BN-ReLU (refinement).
    The refinement conv after each transpose helps learn sharp boundaries.
    """

    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.decoder = nn.Sequential(
            _DeconvBlock(in_ch, 32),   # 8→16
            _DeconvBlock(32, 16),      # 16→32
            _DeconvBlock(16, 8),       # 32→64
            _DeconvBlock(8, 4),        # 64→128
            nn.Conv2d(4, out_ch, 1),   # 1x1 projection to output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class StaticEncoder(nn.Module):
    """Small CNN encoder for static terrain channels.

    Input: (B, n_static, 128, 128)
    Output: (B, 32, 8, 8)
    """

    def __init__(self, in_ch: int = N_STATIC):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_ch, 16, stride=4),   # → (16, 32, 32)
            ConvBlock(16, 24, stride=2),      # → (24, 16, 16)
            ConvBlock(24, 32, stride=2),      # → (32, 8, 8)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DebrisDetector(nn.Module):
    """Single-pass debris detection model.

    Fuses SAR set features with static terrain features for debris probability.

    Parameters
    ----------
    feat_dim : int
        Feature dimension for set encoder and attention.
    n_static : int
        Number of static terrain channels.
    """

    def __init__(self, feat_dim: int = 64, n_static: int = N_STATIC):
        super().__init__()
        self.set_encoder = SetEncoder(in_ch=2, feat_dim=feat_dim)
        self.attention = SpatialSetAttention(feat_dim=feat_dim)
        self.static_encoder = StaticEncoder(in_ch=n_static)
        # 64 (SAR) + 32 (static) = 96 → 64
        self.fusion = nn.Conv2d(feat_dim + 32, feat_dim, 1)
        self.decoder = Decoder(in_ch=feat_dim, out_ch=1)

    def forward(
        self,
        sar_maps: list[torch.Tensor],
        static: torch.Tensor,
    ) -> torch.Tensor:
        """Parameters
        ----------
        sar_maps : list of (B, 2, 128, 128) tensors
            Per-track/pol backscatter change + ANF maps.
        static : (B, N_STATIC, 128, 128)
            Static terrain channels.

        Returns:
        -------
        (B, 1, 128, 128) probability logits (apply sigmoid for probabilities)
        """
        encoded = [self.set_encoder(m) for m in sar_maps]
        sar_feat = self.attention(encoded)          # (B, 64, 8, 8)
        static_feat = self.static_encoder(static)   # (B, 32, 8, 8)

        fused = torch.cat([sar_feat, static_feat], dim=1)
        fused = F.relu(self.fusion(fused))
        return self.decoder(fused)
