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

    Input: (B, 4, 128, 128) per track/pol (change + ANF + anomaly + edges)
    Output: (B, feat_dim, 8, 8)
    """

    def __init__(self, in_ch: int = 4, feat_dim: int = 64):
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
    """Single-pass debris detection model (no skip connections).

    Fuses SAR set features with static terrain features for debris probability.

    Parameters
    ----------
    feat_dim : int
        Feature dimension for set encoder and attention.
    n_static : int
        Number of static terrain channels.
    """

    def __init__(self, sar_in_ch: int = 2, feat_dim: int = 64, n_static: int = N_STATIC):
        super().__init__()
        self.set_encoder = SetEncoder(in_ch=sar_in_ch, feat_dim=feat_dim)
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
        sar_maps : list of (B, C, 128, 128) tensors
            Per-pair or per-track/pol SAR maps.
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


# ---------------------------------------------------------------------------
# Skip-connection variants
# ---------------------------------------------------------------------------

class SetEncoderWithSkips(nn.Module):
    """SetEncoder that returns intermediate feature maps for skip connections."""

    def __init__(self, in_ch: int = 4, feat_dim: int = 64):
        super().__init__()
        self.block1 = ConvBlock(in_ch, 16, stride=2)    # → (16, 64, 64)
        self.block2 = ConvBlock(16, 32, stride=2)        # → (32, 32, 32)
        self.block3 = ConvBlock(32, feat_dim, stride=2)  # → (64, 16, 16)
        self.block4 = ConvBlock(feat_dim, feat_dim, stride=2)  # → (64, 8, 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.block1(x)   # (B, 16, 64, 64)
        s2 = self.block2(s1)  # (B, 32, 32, 32)
        s3 = self.block3(s2)  # (B, 64, 16, 16)
        s4 = self.block4(s3)  # (B, 64, 8, 8)
        return s4, [s1, s2, s3]


class StaticEncoderWithSkips(nn.Module):
    """Static encoder returning intermediate features for skip connections."""

    def __init__(self, in_ch: int = N_STATIC):
        super().__init__()
        self.block1 = ConvBlock(in_ch, 16, stride=4)  # → (16, 32, 32)
        self.block2 = ConvBlock(16, 24, stride=2)      # → (24, 16, 16)
        self.block3 = ConvBlock(24, 32, stride=2)      # → (32, 8, 8)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.block1(x)   # (16, 32, 32)
        s2 = self.block2(s1)  # (24, 16, 16)
        s3 = self.block3(s2)  # (32, 8, 8)
        return s3, [s1, s2]


class _SkipDeconvBlock(nn.Module):
    """Upsample + concatenate skip features + refine."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False,
        )
        self.refine = nn.Sequential(
            nn.BatchNorm2d(out_ch + skip_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        cat = torch.cat([up, skip], dim=1)
        return self.refine(cat)


class SkipDecoder(nn.Module):
    """Decoder with skip connections from encoder stages.

    Skip dimensions (SAR + static combined):
      stage 1 (16x16): SAR 64 + static 24 = 88
      stage 2 (32x32): SAR 32 + static 16 = 48
      stage 3 (64x64): SAR 16             = 16  (static has no 64x64 skip)
    """

    def __init__(self, in_ch: int, out_ch: int = 1):
        super().__init__()
        self.up1 = _SkipDeconvBlock(in_ch, 88, 32)   # 8→16
        self.up2 = _SkipDeconvBlock(32, 48, 16)       # 16→32
        self.up3 = _SkipDeconvBlock(16, 16, 8)        # 32→64
        self.up4 = _DeconvBlock(8, 4)                  # 64→128 (no skip)
        self.head = nn.Conv2d(4, out_ch, 1)

    def forward(
        self,
        x: torch.Tensor,
        skips: list[torch.Tensor],
    ) -> torch.Tensor:
        """skips: [skip_16x16, skip_32x32, skip_64x64]"""
        x = self.up1(x, skips[0])   # 8→16
        x = self.up2(x, skips[1])   # 16→32
        x = self.up3(x, skips[2])   # 32→64
        x = self.up4(x)             # 64→128
        return self.head(x)


class DebrisDetectorSkip(nn.Module):
    """Debris detector with skip connections from encoders to decoder.

    Same interface as DebrisDetector but passes encoder intermediate features
    through to the decoder for sharper spatial boundaries.
    """

    def __init__(self, sar_in_ch: int = 2, feat_dim: int = 64, n_static: int = N_STATIC):
        super().__init__()
        self.set_encoder = SetEncoderWithSkips(in_ch=sar_in_ch, feat_dim=feat_dim)
        self.attention = SpatialSetAttention(feat_dim=feat_dim)
        self.static_encoder = StaticEncoderWithSkips(in_ch=n_static)
        self.fusion = nn.Conv2d(feat_dim + 32, feat_dim, 1)
        self.decoder = SkipDecoder(in_ch=feat_dim, out_ch=1)

        # Attention modules for pooling skip connections across variable SAR maps
        self.skip_attentions = nn.ModuleList([
            SpatialSetAttention(feat_dim=16),   # 64x64 skips
            SpatialSetAttention(feat_dim=32),   # 32x32 skips
            SpatialSetAttention(feat_dim=64),   # 16x16 skips
        ])

    def forward(
        self,
        sar_maps: list[torch.Tensor],
        static: torch.Tensor,
    ) -> torch.Tensor:
        # Encode each SAR map, collecting skip features
        all_bottlenecks = []
        all_skips: list[list[torch.Tensor]] = [[], [], []]

        for m in sar_maps:
            bottleneck, skips = self.set_encoder(m)
            all_bottlenecks.append(bottleneck)
            for i, s in enumerate(skips):
                all_skips[i].append(s)

        # Attention-pool bottleneck and each skip level across SAR maps
        sar_feat = self.attention(all_bottlenecks)  # (B, 64, 8, 8)
        sar_skips = [
            self.skip_attentions[i](all_skips[i])
            for i in range(3)
        ]

        # Static encoder with skips
        static_feat, static_skips = self.static_encoder(static)

        # Fuse bottleneck
        fused = torch.cat([sar_feat, static_feat], dim=1)
        fused = F.relu(self.fusion(fused))

        # Build combined skip connections for decoder
        skip_16 = torch.cat([sar_skips[2], static_skips[1]], dim=1)  # 88ch
        skip_32 = torch.cat([sar_skips[1], static_skips[0]], dim=1)  # 48ch
        skip_64 = sar_skips[0]                                       # 16ch

        return self.decoder(fused, [skip_16, skip_32, skip_64])
