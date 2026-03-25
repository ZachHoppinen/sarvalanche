"""v4 multi-scale debris detector with FPN.

Three-branch architecture:
  - Fine: 128×128 at full resolution — debris detail (3.8km)
  - Local context: 512×512 downsampled to 128×128 — neighboring paths (15km)
  - Regional: whole scene downsampled to 128×128 — melt/activity patterns

Shared conv weights across scales with per-scale BatchNorm, so each scale
tracks its own running statistics. The first encoder block uses separate
convs per scale since input distributions differ most at the raw level.

Input:
  fine: (B, C, 128, 128)
  local_ctx: (B, C, 128, 128)
  regional: (B, C, 128, 128)

Output: (B, 1, 128, 128) — debris probability at full res
"""

import torch
import torch.nn as nn

N_SCALES = 3  # fine=0, local=1, regional=2


class ScaleBN(nn.Module):
    """Per-scale BatchNorm: one BN per scale, selected by scale_idx."""

    def __init__(self, num_features, n_scales=N_SCALES):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(n_scales)])

    def forward(self, x, scale_idx=0):
        return self.bns[scale_idx](x)


class SharedConvBlock(nn.Module):
    """Conv block with shared conv weights but per-scale BatchNorm."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = ScaleBN(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = ScaleBN(out_ch)

    def forward(self, x, scale_idx=0):
        x = torch.relu(self.bn1(self.conv1(x), scale_idx))
        x = torch.relu(self.bn2(self.conv2(x), scale_idx))
        return x


class ConvBlock(nn.Module):
    """Standard conv block (single-scale, used in decoder/fusion)."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """Shared encoder with per-scale BatchNorm.

    The first block (enc1) uses separate convs per scale since input
    distributions differ most at the raw input level. Deeper blocks
    share conv weights with per-scale BN.
    """

    def __init__(self, in_ch, base_ch):
        super().__init__()
        # Per-scale input projection — raw features differ significantly
        self.enc1 = nn.ModuleList([
            ConvBlock(in_ch, base_ch) for _ in range(N_SCALES)
        ])
        # Shared convs, per-scale BN for deeper layers
        self.enc2 = SharedConvBlock(base_ch, base_ch * 2)
        self.enc3 = SharedConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = SharedConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, scale_idx=0):
        e1 = self.enc1[scale_idx](x)
        e2 = self.enc2(self.pool(e1), scale_idx)
        e3 = self.enc3(self.pool(e2), scale_idx)
        e4 = self.enc4(self.pool(e3), scale_idx)
        return e1, e2, e3, e4


class MultiScaleDetector(nn.Module):
    """Three-scale FPN debris detector.

    Parameters
    ----------
    in_ch : int
        Input channels (SAR + static)
    base_ch : int
        Base channel width
    """

    def __init__(self, in_ch: int = 11, base_ch: int = 16):
        super().__init__()
        C = base_ch

        # Shared encoder with per-scale BN
        self.encoder = Encoder(in_ch, C)

        # FPN lateral connections (1×1 convs to project each branch)
        self.lat_local4 = nn.Conv2d(C * 8, C * 8, 1)
        self.lat_local3 = nn.Conv2d(C * 4, C * 4, 1)
        self.lat_local2 = nn.Conv2d(C * 2, C * 2, 1)

        self.lat_region4 = nn.Conv2d(C * 8, C * 8, 1)
        self.lat_region3 = nn.Conv2d(C * 4, C * 4, 1)
        self.lat_region2 = nn.Conv2d(C * 2, C * 2, 1)

        # Fusion: concatenate fine + local + regional → reduce
        self.fuse4 = ConvBlock(C * 8 * 3, C * 8)
        self.fuse3 = ConvBlock(C * 4 * 3, C * 4)
        self.fuse2 = ConvBlock(C * 2 * 3, C * 2)

        # Learnable gate: starts at 0 so output = fine features initially.
        # As gate grows, FPN fusion is blended in: out = (1-gate)*fine + gate*fused
        self.fpn_gate = nn.Parameter(torch.tensor([-5.0]))

        # Decoder
        self.bottleneck = ConvBlock(C * 8, C * 8)

        self.up4 = nn.ConvTranspose2d(C * 8, C * 8, 2, stride=2)
        self.dec4 = ConvBlock(C * 16, C * 4)

        self.up3 = nn.ConvTranspose2d(C * 4, C * 4, 2, stride=2)
        self.dec3 = ConvBlock(C * 8, C * 2)

        self.up2 = nn.ConvTranspose2d(C * 2, C * 2, 2, stride=2)
        self.dec2 = ConvBlock(C * 4, C)

        self.up1 = nn.ConvTranspose2d(C, C, 2, stride=2)
        self.dec1 = ConvBlock(C * 2, C)

        self.out_conv = nn.Conv2d(C, 1, 1)

    def forward(self, fine, local_ctx, regional):
        """
        Parameters
        ----------
        fine : (B, C, 128, 128) — full resolution
        local_ctx : (B, C, 128, 128) — 4× downsampled from 512×512
        regional : (B, C, 128, 128) — whole scene downsampled

        Returns
        -------
        logits : (B, 1, 128, 128)
        """
        # Encode each scale (shared deeper convs, per-scale BN)
        f1, f2, f3, f4 = self.encoder(fine, scale_idx=0)
        l1, l2, l3, l4 = self.encoder(local_ctx, scale_idx=1)
        r1, r2, r3, r4 = self.encoder(regional, scale_idx=2)

        # FPN fusion at levels 2-4, gated so model starts as single-branch v3
        g = torch.sigmoid(self.fpn_gate)  # 0 at init → pure fine features
        fused4 = (1 - g) * f4 + g * self.fuse4(torch.cat([f4, self.lat_local4(l4), self.lat_region4(r4)], dim=1))
        fused3 = (1 - g) * f3 + g * self.fuse3(torch.cat([f3, self.lat_local3(l3), self.lat_region3(r3)], dim=1))
        fused2 = (1 - g) * f2 + g * self.fuse2(torch.cat([f2, self.lat_local2(l2), self.lat_region2(r2)], dim=1))

        # Bottleneck
        b = self.bottleneck(self.encoder.pool(fused4))

        # Decoder with fused skip connections
        d4 = self.dec4(torch.cat([self.up4(b), fused4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), fused3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), fused2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), f1], dim=1))  # finest level: only fine features

        return self.out_conv(d1)

    def forward_single(self, x):
        """Single-scale fallback — uses same input for all branches."""
        return self.forward(x, x, x)
