"""Experiment: Norwegian FCN-inspired improvements to debris detector.

Tests improvements inspired by Eckerstorfer et al.'s FCN for avalanche detection:
  1. dropout       — Add dropout to encoder/decoder blocks (0.2)
  2. terrain_attn  — Attention mask from terrain applied to SAR before encoding
  3. aug+          — Enhanced augmentation (90° rotations + vertical flips)
  4. combined      — All three improvements together
  5. unet_wide     — Norwegian-style layers: MaxPool down, bilinear up, wider filters
                     (32→64→128→256), double-conv decoder, dropout
  6. unet_wide+attn — unet_wide + terrain attention + enhanced augmentation

Compared against the current best: DebrisDetectorSkip + BCE+Dice.

Usage:
    conda run -n sarvalanche python scripts/debris_pixel_classifier/v2/experiment_norwegian.py \
        --data-dir local/issw/v2_patches --epochs 50 --batch-size 4
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from sarvalanche.ml.v2.channels import N_STATIC
from sarvalanche.ml.v2.dataset import V2PatchDataset, v2_collate_fn
from sarvalanche.ml.v2.model import (
    DebrisDetectorSkip,
    SpatialSetAttention,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks with dropout
# ---------------------------------------------------------------------------

class ConvBlockDrop(nn.Module):
    """Conv-BN-ReLU-Conv-BN-ReLU-Dropout block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Encoder/decoder with dropout
# ---------------------------------------------------------------------------

class SetEncoderDrop(nn.Module):
    """SetEncoder with dropout and skip connections."""

    def __init__(self, in_ch: int = 2, feat_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.block1 = ConvBlockDrop(in_ch, 16, stride=2, dropout=dropout)
        self.block2 = ConvBlockDrop(16, 32, stride=2, dropout=dropout)
        self.block3 = ConvBlockDrop(32, feat_dim, stride=2, dropout=dropout)
        # No dropout in bottleneck block (following Norwegian approach)
        self.block4 = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.block1(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        s4 = self.block4(s3)
        return s4, [s1, s2, s3]


class StaticEncoderDrop(nn.Module):
    """Static encoder with dropout and skip connections."""

    def __init__(self, in_ch: int = N_STATIC, dropout: float = 0.2):
        super().__init__()
        self.block1 = ConvBlockDrop(in_ch, 16, stride=4, dropout=dropout)
        self.block2 = ConvBlockDrop(16, 24, stride=2, dropout=dropout)
        self.block3 = ConvBlockDrop(24, 32, stride=2, dropout=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.block1(x)
        s2 = self.block2(s1)
        s3 = self.block3(s2)
        return s3, [s1, s2]


class _DeconvBlockDrop(nn.Module):
    """Upsample + refine with dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.refine = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.refine(self.up(x))


class _SkipDeconvBlockDrop(nn.Module):
    """Upsample + skip concat + refine with dropout."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        self.refine = nn.Sequential(
            nn.BatchNorm2d(out_ch + skip_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        return self.refine(torch.cat([up, skip], dim=1))


class SkipDecoderDrop(nn.Module):
    """Skip decoder with dropout."""

    def __init__(self, in_ch: int, out_ch: int = 1, dropout: float = 0.2):
        super().__init__()
        self.up1 = _SkipDeconvBlockDrop(in_ch, 88, 32, dropout=dropout)
        self.up2 = _SkipDeconvBlockDrop(32, 48, 16, dropout=dropout)
        self.up3 = _SkipDeconvBlockDrop(16, 16, 8, dropout=dropout)
        self.up4 = _DeconvBlockDrop(8, 4, dropout=dropout)
        # Final block: no dropout (like Norwegian model)
        self.head = nn.Sequential(
            nn.Conv2d(4, out_ch, 1),
        )

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        x = self.up1(x, skips[0])
        x = self.up2(x, skips[1])
        x = self.up3(x, skips[2])
        x = self.up4(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Terrain Attention Net
# ---------------------------------------------------------------------------

class TerrainAttentionNet(nn.Module):
    """Small FCN that generates a spatial attention mask from static terrain features.

    Inspired by the Norwegian PAR attention mechanism. Takes static terrain
    channels and outputs a sigmoid attention mask applied element-wise to each
    SAR channel before encoding.

    The attention net has a small receptive field (7 pixels from 3 stacked 3x3
    convs), suitable since terrain features already encode long-range info.
    """

    def __init__(self, in_ch: int = N_STATIC):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        """Returns attention mask (B, 1, H, W) in [0, 1]."""
        return self.net(static)


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

class DebrisDetectorDropout(DebrisDetectorSkip):
    """DebrisDetectorSkip with dropout in all encoder/decoder blocks."""

    def __init__(self, feat_dim: int = 64, n_static: int = N_STATIC, dropout: float = 0.2):
        # Skip DebrisDetectorSkip.__init__, go straight to nn.Module
        nn.Module.__init__(self)
        self.set_encoder = SetEncoderDrop(in_ch=2, feat_dim=feat_dim, dropout=dropout)
        self.attention = SpatialSetAttention(feat_dim=feat_dim)
        self.static_encoder = StaticEncoderDrop(in_ch=n_static, dropout=dropout)
        self.fusion = nn.Conv2d(feat_dim + 32, feat_dim, 1)
        self.decoder = SkipDecoderDrop(in_ch=feat_dim, out_ch=1, dropout=dropout)
        self.skip_attentions = nn.ModuleList([
            SpatialSetAttention(feat_dim=16),
            SpatialSetAttention(feat_dim=32),
            SpatialSetAttention(feat_dim=64),
        ])


class DebrisDetectorTerrainAttn(DebrisDetectorSkip):
    """DebrisDetectorSkip with terrain attention applied to SAR inputs."""

    def __init__(self, feat_dim: int = 64, n_static: int = N_STATIC):
        super().__init__(feat_dim=feat_dim, n_static=n_static)
        self.terrain_attn = TerrainAttentionNet(in_ch=n_static)

    def forward(
        self,
        sar_maps: list[torch.Tensor],
        static: torch.Tensor,
    ) -> torch.Tensor:
        # Generate attention mask from terrain
        attn_mask = self.terrain_attn(static)  # (B, 1, H, W)

        # Apply attention to SAR change channel (channel 0), not ANF (channel 1)
        masked_sar = []
        for m in sar_maps:
            # m is (B, 2, H, W): [change, anf]
            m_masked = m.clone()
            m_masked[:, 0:1] = m[:, 0:1] * attn_mask
            masked_sar.append(m_masked)

        # Run through parent forward with masked SAR
        return super().forward(masked_sar, static)


class DebrisDetectorCombined(nn.Module):
    """All Norwegian improvements: dropout + terrain attention + skip connections."""

    def __init__(self, feat_dim: int = 64, n_static: int = N_STATIC, dropout: float = 0.2):
        super().__init__()
        self.terrain_attn = TerrainAttentionNet(in_ch=n_static)
        self.set_encoder = SetEncoderDrop(in_ch=2, feat_dim=feat_dim, dropout=dropout)
        self.attention = SpatialSetAttention(feat_dim=feat_dim)
        self.static_encoder = StaticEncoderDrop(in_ch=n_static, dropout=dropout)
        self.fusion = nn.Conv2d(feat_dim + 32, feat_dim, 1)
        self.decoder = SkipDecoderDrop(in_ch=feat_dim, out_ch=1, dropout=dropout)
        self.skip_attentions = nn.ModuleList([
            SpatialSetAttention(feat_dim=16),
            SpatialSetAttention(feat_dim=32),
            SpatialSetAttention(feat_dim=64),
        ])

    def forward(
        self,
        sar_maps: list[torch.Tensor],
        static: torch.Tensor,
    ) -> torch.Tensor:
        # Terrain attention on SAR change channels
        attn_mask = self.terrain_attn(static)
        masked_sar = []
        for m in sar_maps:
            m_masked = m.clone()
            m_masked[:, 0:1] = m[:, 0:1] * attn_mask
            masked_sar.append(m_masked)

        # Encode each SAR map
        all_bottlenecks = []
        all_skips: list[list[torch.Tensor]] = [[], [], []]
        for m in masked_sar:
            bottleneck, skips = self.set_encoder(m)
            all_bottlenecks.append(bottleneck)
            for i, s in enumerate(skips):
                all_skips[i].append(s)

        sar_feat = self.attention(all_bottlenecks)
        sar_skips = [self.skip_attentions[i](all_skips[i]) for i in range(3)]

        static_feat, static_skips = self.static_encoder(static)

        fused = torch.cat([sar_feat, static_feat], dim=1)
        fused = F.relu(self.fusion(fused))

        skip_16 = torch.cat([sar_skips[2], static_skips[1]], dim=1)
        skip_32 = torch.cat([sar_skips[1], static_skips[0]], dim=1)
        skip_64 = sar_skips[0]

        return self.decoder(fused, [skip_16, skip_32, skip_64])


# ---------------------------------------------------------------------------
# Norwegian-style U-Net: MaxPool down, bilinear up, wider filters, double-conv
# ---------------------------------------------------------------------------

class UNetEncBlock(nn.Module):
    """Norwegian-style encoder block: Conv-BN-ReLU-Conv-BN-ReLU + MaxPool + Dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.2, pool: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (pooled_output, pre_pool_features_for_skip)."""
        feat = self.conv(x)
        return self.drop(self.pool(feat)), feat


class UNetDecBlock(nn.Module):
    """Norwegian-style decoder block: bilinear upsample + skip concat + double conv + dropout."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.drop(self.conv(torch.cat([x, skip], dim=1)))


class UNetSAREncoder(nn.Module):
    """Wide U-Net encoder for SAR maps: 32→64→128→256 with MaxPool.

    Input: (B, 2, 128, 128)
    Output: bottleneck (B, 256, 8, 8) + skip features at 4 resolutions
    """

    def __init__(self, in_ch: int = 2, dropout: float = 0.2):
        super().__init__()
        self.enc1 = UNetEncBlock(in_ch, 32, dropout=dropout)     # 128→64
        self.enc2 = UNetEncBlock(32, 64, dropout=dropout)         # 64→32
        self.enc3 = UNetEncBlock(64, 128, dropout=dropout)        # 32→16
        self.enc4 = UNetEncBlock(128, 256, dropout=0, pool=False) # 16 (bottleneck, no pool/drop)
        self.pool4 = nn.MaxPool2d(2)                               # 16→8

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x, s1 = self.enc1(x)   # s1: (B, 32, 128, 128), x: (B, 32, 64, 64)
        x, s2 = self.enc2(x)   # s2: (B, 64, 64, 64),   x: (B, 64, 32, 32)
        x, s3 = self.enc3(x)   # s3: (B, 128, 32, 32),  x: (B, 128, 16, 16)
        _, s4 = self.enc4(x)   # s4: (B, 256, 16, 16)
        bottleneck = self.pool4(s4)  # (B, 256, 8, 8)
        return bottleneck, [s1, s2, s3, s4]


class UNetStaticEncoder(nn.Module):
    """Wide encoder for static terrain: 32→64→128 with MaxPool.

    Input: (B, N_STATIC, 128, 128)
    Output: bottleneck (B, 128, 8, 8) + skip features
    """

    def __init__(self, in_ch: int = N_STATIC, dropout: float = 0.2):
        super().__init__()
        # stride=4 equivalent: pool twice in first block
        self.enc1 = UNetEncBlock(in_ch, 32, dropout=dropout)  # 128→64
        self.pool1b = nn.MaxPool2d(2)                          # 64→32
        self.enc2 = UNetEncBlock(32, 64, dropout=dropout)      # 32→16
        self.enc3 = UNetEncBlock(64, 128, dropout=dropout)     # 16→8

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x, s1 = self.enc1(x)    # s1: (B,32,128,128), x: (B,32,64,64)
        x = self.pool1b(x)      # (B,32,32,32)
        x, s2 = self.enc2(x)    # s2: (B,64,32,32), x: (B,64,16,16)
        x, s3 = self.enc3(x)    # s3: (B,128,16,16), x: (B,128,8,8)
        return x, [s1, s2, s3]


class UNetWideDecoder(nn.Module):
    """Wide U-Net decoder with bilinear upsampling and skip connections.

    Skip dims (SAR + static combined):
      8→16:  SAR 256 + static 128 = 384 skip, from bottleneck 384
      16→32: SAR 128 + static 64  = 192 skip
      32→64: SAR 64  + static 32  = 96 skip  (static s1 at 128 pooled to 64 not available,
                                                use SAR s2 only at 64)
      64→128: SAR 32               = 32 skip
    """

    def __init__(self, in_ch: int = 384, dropout: float = 0.2):
        super().__init__()
        # 8→16:  in=384 (fused bottleneck), skip= SAR_s4(256) + static_s3(128) = 384
        self.up1 = UNetDecBlock(in_ch, 384, 128, dropout=dropout)
        # 16→32: in=128, skip= SAR_s3(128) + static_s2(64) = 192
        self.up2 = UNetDecBlock(128, 192, 64, dropout=dropout)
        # 32→64: in=64, skip= SAR_s2(64) + static pooled_s1(32) = 96
        self.up3 = UNetDecBlock(64, 96, 32, dropout=dropout)
        # 64→128: in=32, skip= SAR_s1(32) = 32
        self.up4 = UNetDecBlock(32, 32, 16, dropout=0)  # last dec block: no dropout
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """skips: [skip_16, skip_32, skip_64, skip_128]"""
        x = self.up1(x, skips[0])   # 8→16
        x = self.up2(x, skips[1])   # 16→32
        x = self.up3(x, skips[2])   # 32→64
        x = self.up4(x, skips[3])   # 64→128
        return self.head(x)


class DebrisDetectorUNetWide(nn.Module):
    """Norwegian-style wide U-Net with MaxPool, bilinear upsample, skip connections.

    Filter progression: 32→64→128→256 (SAR), 32→64→128 (static).
    Uses MaxPool2d downsampling and bilinear upsampling (no transposed convolutions).
    Double Conv-BN-ReLU in both encoder and decoder blocks.
    Dropout in all blocks except bottleneck and final decoder block.
    """

    def __init__(self, n_static: int = N_STATIC, dropout: float = 0.2):
        super().__init__()
        self.sar_encoder = UNetSAREncoder(in_ch=2, dropout=dropout)
        self.attention_bottleneck = SpatialSetAttention(feat_dim=256)
        self.static_encoder = UNetStaticEncoder(in_ch=n_static, dropout=dropout)

        # Fuse SAR(256) + static(128) bottlenecks
        self.fusion = nn.Sequential(
            nn.Conv2d(256 + 128, 384, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        self.decoder = UNetWideDecoder(in_ch=384, dropout=dropout)

        # Attention pooling for SAR skip connections at each level
        self.skip_attentions = nn.ModuleList([
            SpatialSetAttention(feat_dim=32),   # 128x128 skips
            SpatialSetAttention(feat_dim=64),   # 64x64 skips
            SpatialSetAttention(feat_dim=128),  # 32x32 skips
            SpatialSetAttention(feat_dim=256),  # 16x16 skips
        ])

        # Downsample static s1 from 128x128 to 64x64 for skip alignment
        self.static_s1_pool = nn.MaxPool2d(2)

    def forward(
        self,
        sar_maps: list[torch.Tensor],
        static: torch.Tensor,
    ) -> torch.Tensor:
        # Encode each SAR map independently (shared weights)
        all_bottlenecks = []
        all_skips: list[list[torch.Tensor]] = [[], [], [], []]

        for m in sar_maps:
            bottleneck, skips = self.sar_encoder(m)
            all_bottlenecks.append(bottleneck)
            for i, s in enumerate(skips):
                all_skips[i].append(s)

        # Attention-pool across SAR maps at each level
        sar_feat = self.attention_bottleneck(all_bottlenecks)  # (B, 256, 8, 8)
        sar_skips = [self.skip_attentions[i](all_skips[i]) for i in range(4)]
        # sar_skips: [(B,32,128,128), (B,64,64,64), (B,128,32,32), (B,256,16,16)]

        # Static encoder
        static_feat, static_skips = self.static_encoder(static)
        # static_feat: (B,128,8,8)
        # static_skips: [(B,32,128,128), (B,64,32,32), (B,128,16,16)]

        # Fuse bottlenecks
        fused = self.fusion(torch.cat([sar_feat, static_feat], dim=1))  # (B, 384, 8, 8)

        # Build combined skip connections
        # 16x16: SAR_s4(256) + static_s3(128) = 384
        skip_16 = torch.cat([sar_skips[3], static_skips[2]], dim=1)
        # 32x32: SAR_s3(128) + static_s2(64) = 192
        skip_32 = torch.cat([sar_skips[2], static_skips[1]], dim=1)
        # 64x64: SAR_s2(64) + static_s1 pooled(32) = 96
        skip_64 = torch.cat([sar_skips[1], self.static_s1_pool(static_skips[0])], dim=1)
        # 128x128: SAR_s1(32)
        skip_128 = sar_skips[0]

        return self.decoder(fused, [skip_16, skip_32, skip_64, skip_128])


class DebrisDetectorUNetWideAttn(nn.Module):
    """Wide U-Net + terrain attention + dropout — the full Norwegian package."""

    def __init__(self, n_static: int = N_STATIC, dropout: float = 0.2):
        super().__init__()
        self.terrain_attn = TerrainAttentionNet(in_ch=n_static)
        self.sar_encoder = UNetSAREncoder(in_ch=2, dropout=dropout)
        self.attention_bottleneck = SpatialSetAttention(feat_dim=256)
        self.static_encoder = UNetStaticEncoder(in_ch=n_static, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Conv2d(256 + 128, 384, 1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        self.decoder = UNetWideDecoder(in_ch=384, dropout=dropout)
        self.skip_attentions = nn.ModuleList([
            SpatialSetAttention(feat_dim=32),
            SpatialSetAttention(feat_dim=64),
            SpatialSetAttention(feat_dim=128),
            SpatialSetAttention(feat_dim=256),
        ])
        self.static_s1_pool = nn.MaxPool2d(2)

    def forward(
        self,
        sar_maps: list[torch.Tensor],
        static: torch.Tensor,
    ) -> torch.Tensor:
        # Terrain attention on SAR change channels
        attn_mask = self.terrain_attn(static)
        masked_sar = []
        for m in sar_maps:
            m_masked = m.clone()
            m_masked[:, 0:1] = m[:, 0:1] * attn_mask
            masked_sar.append(m_masked)

        all_bottlenecks = []
        all_skips: list[list[torch.Tensor]] = [[], [], [], []]
        for m in masked_sar:
            bottleneck, skips = self.sar_encoder(m)
            all_bottlenecks.append(bottleneck)
            for i, s in enumerate(skips):
                all_skips[i].append(s)

        sar_feat = self.attention_bottleneck(all_bottlenecks)
        sar_skips = [self.skip_attentions[i](all_skips[i]) for i in range(4)]

        static_feat, static_skips = self.static_encoder(static)

        fused = self.fusion(torch.cat([sar_feat, static_feat], dim=1))

        skip_16 = torch.cat([sar_skips[3], static_skips[2]], dim=1)
        skip_32 = torch.cat([sar_skips[2], static_skips[1]], dim=1)
        skip_64 = torch.cat([sar_skips[1], self.static_s1_pool(static_skips[0])], dim=1)
        skip_128 = sar_skips[0]

        return self.decoder(fused, [skip_16, skip_32, skip_64, skip_128])


# ---------------------------------------------------------------------------
# Enhanced augmentation dataset wrapper
# ---------------------------------------------------------------------------

class AugmentedDatasetWrapper(Dataset):
    """Wraps V2PatchDataset with enhanced augmentation: 90° rotations + vertical flips."""

    def __init__(self, base_dataset: Dataset):
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]

        # Random 90° rotation (0, 90, 180, 270)
        k = np.random.randint(0, 4)
        if k > 0:
            item['sar_maps'] = [torch.rot90(m, k, dims=[-2, -1]) for m in item['sar_maps']]
            item['static'] = torch.rot90(item['static'], k, dims=[-2, -1])
            item['label'] = torch.rot90(item['label'], k, dims=[-2, -1])

        # Random vertical flip
        if np.random.random() > 0.5:
            item['sar_maps'] = [torch.flip(m, dims=[-2]) for m in item['sar_maps']]
            item['static'] = torch.flip(item['static'], dims=[-2])
            item['label'] = torch.flip(item['label'], dims=[-2])

        return item


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def weighted_bce(logits, targets, pos_weight):
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)


def bce_dice_loss(logits, targets, pos_weight):
    return weighted_bce(logits, targets, pos_weight) + dice_loss(logits, targets)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        sar_maps = [m.to(device) for m in batch['sar_maps']]
        static = batch['static'].to(device)
        targets = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(sar_maps, static)
        loss = loss_fn(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, threshold=0.5):
    model.eval()
    total_bce = 0.0
    n = 0
    intersection = 0
    union = 0
    tp_probs = []
    all_pos_probs = []

    for batch in loader:
        sar_maps = [m.to(device) for m in batch['sar_maps']]
        static = batch['static'].to(device)
        targets = batch['label'].to(device)

        logits = model(sar_maps, static)
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        total_bce += bce.item()
        n += 1

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        intersection += (preds * targets).sum().item()
        union += ((preds + targets) >= 1).float().sum().item()

        pos_mask = targets > 0.5
        if pos_mask.any():
            all_pos_probs.append(probs[pos_mask].cpu())
        tp_mask = (preds > 0.5) & (targets > 0.5)
        if tp_mask.any():
            tp_probs.append(probs[tp_mask].cpu())

    val_bce = total_bce / max(n, 1)
    iou = intersection / max(union, 1)
    mean_pos_conf = torch.cat(all_pos_probs).mean().item() if all_pos_probs else 0.0
    mean_tp_conf = torch.cat(tp_probs).mean().item() if tp_probs else 0.0
    return val_bce, iou, mean_pos_conf, mean_tp_conf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Norwegian FCN-inspired improvements experiment',
    )
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--pos-weight', type=float, default=0.0, help='0 = auto-compute')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    log.info('Device: %s', device)

    # Dataset — shared split
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = V2PatchDataset(args.data_dir, augment=True)
    n_val = max(1, int(len(dataset) * args.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    val_ds.dataset.augment = False  # type: ignore[attr-defined]

    # Standard loaders (for non-aug+ configs)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=v2_collate_fn, num_workers=0,
    )
    # Enhanced augmentation loader
    aug_train_loader = DataLoader(
        AugmentedDatasetWrapper(train_ds), batch_size=args.batch_size, shuffle=True,
        collate_fn=v2_collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=v2_collate_fn, num_workers=0,
    )

    log.info('Train: %d  Val: %d', n_train, n_val)

    # Positive class weight
    if args.pos_weight > 0:
        pw = args.pos_weight
    else:
        n_pos_px = 0
        n_total_px = 0
        for i in range(len(dataset)):
            data = np.load(dataset.files[i], allow_pickle=True)
            if 'label_mask' in data:
                mask = data['label_mask']
                n_pos_px += mask.sum()
                n_total_px += mask.size
            else:
                label = int(data['label'])
                n_pos_px += label * 128 * 128
                n_total_px += 128 * 128
        n_neg_px = n_total_px - n_pos_px
        pw = float(n_neg_px / max(n_pos_px, 1))
        pw = min(pw, 50.0)
    log.info('pos_weight: %.1f', pw)
    pos_weight = torch.tensor([pw], device=device)

    loss_fn = lambda logits, targets: bce_dice_loss(logits, targets, pos_weight)

    # Experiment configurations
    # Baseline = skip+dice from previous experiment (current best)
    configs = {
        'baseline(skip+dice)': {
            'model_cls': DebrisDetectorSkip,
            'loader': train_loader,
        },
        'dropout': {
            'model_cls': lambda: DebrisDetectorDropout(dropout=args.dropout),
            'loader': train_loader,
        },
        'terrain_attn': {
            'model_cls': DebrisDetectorTerrainAttn,
            'loader': train_loader,
        },
        'aug+': {
            'model_cls': DebrisDetectorSkip,
            'loader': aug_train_loader,
        },
        'combined': {
            'model_cls': lambda: DebrisDetectorCombined(dropout=args.dropout),
            'loader': aug_train_loader,
        },
        'unet_wide': {
            'model_cls': lambda: DebrisDetectorUNetWide(dropout=args.dropout),
            'loader': train_loader,
        },
        'unet_wide+attn+aug': {
            'model_cls': lambda: DebrisDetectorUNetWideAttn(dropout=args.dropout),
            'loader': aug_train_loader,
        },
    }

    results = {}

    for name, cfg in configs.items():
        log.info('=' * 60)
        log.info('Running: %s', name)
        log.info('=' * 60)

        torch.manual_seed(args.seed)

        model_cls = cfg['model_cls']
        model = model_cls() if callable(model_cls) and not isinstance(model_cls, type) else model_cls()
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        log.info('  Parameters: %d', n_params)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_iou = 0.0
        best_epoch = 0
        best_state = None

        for epoch in range(args.epochs):
            train_loss = train_epoch(model, cfg['loader'], optimizer, device, loss_fn)
            val_bce, iou, mean_pos_conf, mean_tp_conf = validate(model, val_loader, device)
            scheduler.step()

            if iou > best_iou:
                best_iou = iou
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == 0:
                log.info(
                    '  [%s] epoch %3d: train=%.4f  val_bce=%.4f  IoU=%.4f  '
                    'pos_conf=%.3f  tp_conf=%.3f',
                    name, epoch + 1, train_loss, val_bce, iou,
                    mean_pos_conf, mean_tp_conf,
                )

        # Final validation with best model
        if best_state is not None:
            model.load_state_dict(best_state)
            model = model.to(device)
        final_bce, final_iou, final_pos_conf, final_tp_conf = validate(model, val_loader, device)

        results[name] = {
            'best_iou': best_iou,
            'best_epoch': best_epoch,
            'final_val_bce': final_bce,
            'final_iou': final_iou,
            'mean_pos_confidence': final_pos_conf,
            'mean_tp_confidence': final_tp_conf,
            'n_params': n_params,
        }

        out_path = args.data_dir / f'v2_norwegian_{name.replace("+", "_").replace("(", "").replace(")", "")}.pt'
        if best_state is not None:
            torch.save(best_state, out_path)
        log.info('  Saved %s → %s', name, out_path)

    # Summary
    log.info('')
    log.info('=' * 80)
    log.info('RESULTS SUMMARY — Norwegian-inspired improvements')
    log.info('=' * 80)
    log.info('%-22s  %6s  %8s  %8s  %10s  %10s  %8s',
             'Config', 'Params', 'Val BCE', 'IoU', 'Pos Conf', 'TP Conf', 'BestEp')
    log.info('-' * 80)
    for name, r in results.items():
        log.info(
            '%-22s  %6d  %8.4f  %8.4f  %10.3f  %10.3f  %8d',
            name, r['n_params'], r['final_val_bce'], r['final_iou'],
            r['mean_pos_confidence'], r['mean_tp_confidence'], r['best_epoch'],
        )


if __name__ == '__main__':
    main()
