"""Single-pair debris detector: U-Net.

Input: (B, N_INPUT, H, W) — one SAR pair + static terrain. H and W must
       be divisible by 16 (4 pooling steps). Designed for 128×128 patches.
Output: (B, 1, H, W) — raw logits. Apply torch.sigmoid for probabilities.

Each crossing pair is evaluated independently.
Temporal aggregation happens post-inference via temporal_onset.

Architecture notes:
  - The decoder is deliberately narrower than the encoder (dec4 outputs
    base_ch*4, not base_ch*8). With base_ch=16 and a small geographically
    clustered dataset, a full-width decoder would overfit. The narrow
    decoder constrains capacity in lieu of dropout.
  - BatchNorm is used throughout. Optional Dropout2d between encoder stages.
"""

import torch
import torch.nn as nn

from sarvalanche.ml.pairwise_debris_classifier.channels import N_INPUT


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

    Parameters
    ----------
    in_ch : int
        Total input channels (N_SAR + N_STATIC). Default from channels.py.
    base_ch : int
        Base channel width. Encoder doubles each level.
    dropout : float
        Dropout2d rate applied after each encoder block. 0.0 disables.
    """

    def __init__(self, in_ch: int = N_INPUT, base_ch: int = 16, dropout: float = 0.0):
        super().__init__()
        self.dropout_rate = dropout

        # Encoder: channels double each level
        self.enc1 = ConvBlock(in_ch, base_ch)           # H → H
        self.enc2 = ConvBlock(base_ch, base_ch * 2)     # H/2
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4) # H/4
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8) # H/8

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)  # H/16

        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Decoder: narrower than encoder (see module docstring)
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 4)  # cat(up4, e4)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2)   # cat(up3, e3)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch)        # cat(up2, e2)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)        # cat(up1, e1)

        # Raw logits — apply sigmoid externally for probabilities
        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : (B, in_ch, H, W) — H and W must be divisible by 16.

        Returns
        -------
        logits : (B, 1, H, W) — raw logits, not probabilities.
        """
        _, _, H, W = x.shape
        assert H % 16 == 0 and W % 16 == 0, (
            f"Input spatial dims must be divisible by 16, got ({H}, {W})")

        e1 = self.drop(self.enc1(x))
        e2 = self.drop(self.enc2(self.pool(e1)))
        e3 = self.drop(self.enc3(self.pool(e2)))
        e4 = self.drop(self.enc4(self.pool(e3)))

        b = self.drop(self.bottleneck(self.pool(e4)))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)

    def predict_proba(self, x):
        """Run forward pass and return sigmoid probabilities."""
        return torch.sigmoid(self.forward(x))
