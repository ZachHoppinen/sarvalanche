"""v4 multi-scale debris detector with FPN.

Three-branch architecture:
  - Fine: 128×128 at full resolution — debris detail (3.8km)
  - Local context: 512×512 downsampled to 128×128 — neighboring paths (15km)
  - Regional: whole scene downsampled to 128×128 — melt/activity patterns

All branches share the same encoder. FPN fuses features at each level.

Input:
  fine: (B, C, 128, 128)
  local_ctx: (B, C, 128, 128)
  regional: (B, C, 128, 128)

Output: (B, 1, 128, 128) — debris probability at full res
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
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
    """Shared encoder — used by all three branches."""

    def __init__(self, in_ch, base_ch):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
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

        # Shared encoder for all three branches
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
        # Encode all three scales (shared weights)
        f1, f2, f3, f4 = self.encoder(fine)
        l1, l2, l3, l4 = self.encoder(local_ctx)
        r1, r2, r3, r4 = self.encoder(regional)

        # FPN fusion at levels 2-4
        fused4 = self.fuse4(torch.cat([f4, self.lat_local4(l4), self.lat_region4(r4)], dim=1))
        fused3 = self.fuse3(torch.cat([f3, self.lat_local3(l3), self.lat_region3(r3)], dim=1))
        fused2 = self.fuse2(torch.cat([f2, self.lat_local2(l2), self.lat_region2(r2)], dim=1))

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
