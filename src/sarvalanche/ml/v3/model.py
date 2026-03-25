"""v3 single-pair debris detector: simple U-Net.

Input: (B, N_SAR + N_STATIC, 128, 128) — one SAR pair + static terrain
Output: (B, 1, 128, 128) — debris probability per pixel

No set encoder, no attention. Each crossing pair is evaluated independently.
Temporal aggregation (multi-pass confirmation) happens post-inference.
"""

import torch
import torch.nn as nn

from sarvalanche.ml.v3.channels import N_INPUT, N_SAR, N_STATIC


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU."""

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


class SinglePairDetector(nn.Module):
    """Compact U-Net for single-pair debris detection.

    Encoder: 4 levels (128→64→32→16→8)
    Decoder: 4 levels with skip connections (8→16→32→64→128)

    Parameters
    ----------
    in_ch : int
        Total input channels (N_SAR + N_STATIC). Default from channels.py.
    base_ch : int
        Base channel width. Encoder doubles each level.
    """

    def __init__(self, in_ch: int = N_INPUT, base_ch: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_ch, base_ch)           # 128 → 128
        self.enc2 = ConvBlock(base_ch, base_ch * 2)     # 64 → 64
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4) # 32 → 32
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8) # 16 → 16

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)  # 8 → 8

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 4)  # cat(up, enc4)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2)   # cat(up, enc3)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch)        # cat(up, enc2)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)        # cat(up, enc1)

        # Output: logits (apply sigmoid externally or use BCE with logits)
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, N_INPUT, 128, 128)

        Returns
        -------
        logits : (B, 1, 128, 128)
        """
        # Encoder
        e1 = self.enc1(x)              # (B, 32, 128, 128)
        e2 = self.enc2(self.pool(e1))   # (B, 64, 64, 64)
        e3 = self.enc3(self.pool(e2))   # (B, 128, 32, 32)
        e4 = self.enc4(self.pool(e3))   # (B, 256, 16, 16)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))  # (B, 256, 8, 8)

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))    # (B, 128, 16, 16)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))   # (B, 64, 32, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))   # (B, 32, 64, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))   # (B, 32, 128, 128)

        return self.out_conv(d1)  # (B, 1, 128, 128)
