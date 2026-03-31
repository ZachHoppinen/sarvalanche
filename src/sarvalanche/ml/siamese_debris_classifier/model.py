"""Siamese debris detector: weight-sharing U-Net encoders with feature-level fusion.

Instead of pre-computing pixel-level dB diffs, this model receives raw
pre-event and post-event SAR images (VV, VH, ANF per branch) and learns
what change representations to extract.

Architecture:
  - Two weight-sharing encoders process pre/post patches independently
  - Feature subtraction at each encoder level (element-wise diff)
  - Difference features are concatenated with static terrain channels
  - Decoder produces per-pixel debris probability

Input: pre (B, 3, H, W) + post (B, 3, H, W) + static (B, 5, H, W)
Output: (B, 1, H, W) raw logits

H and W must be divisible by 16 (4 pooling steps).
"""

import torch
import torch.nn as nn

from sarvalanche.ml.siamese_debris_classifier.channels import N_BRANCH, N_STATIC


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


class SiameseDebrisDetector(nn.Module):
    """Siamese U-Net for change detection.

    Parameters
    ----------
    branch_ch : int
        Channels per branch (VV, VH, ANF). Default 3.
    static_ch : int
        Static terrain channels. Default 5. Set to 0 for SAR-only.
    base_ch : int
        Base channel width for encoder/decoder.
    """

    def __init__(self, branch_ch: int = N_BRANCH, static_ch: int = N_STATIC,
                 base_ch: int = 16):
        super().__init__()
        self.branch_ch = branch_ch
        self.static_ch = static_ch
        self.base_ch = base_ch

        # Shared encoder (weight-tied for pre and post)
        self.enc1 = ConvBlock(branch_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck operates on the difference features
        # Input: diff features (base_ch*8) from encoder subtraction
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8)

        # Decoder: takes diff features from skip connections
        # dec4 input: bottleneck_up (base_ch*8) + diff_e4 (base_ch*8) = base_ch*16
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 4)

        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 2)

        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch)

        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        # dec1 input: up1 (base_ch) + diff_e1 (base_ch) + static (static_ch)
        self.dec1 = ConvBlock(base_ch * 2 + static_ch, base_ch)

        self.out_conv = nn.Conv2d(base_ch, 1, 1)

    def _encode(self, x):
        """Run shared encoder, return feature maps at each level."""
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def forward(self, pre, post, static=None):
        """
        Parameters
        ----------
        pre : (B, branch_ch, H, W) — pre-event SAR patch
        post : (B, branch_ch, H, W) — post-event SAR patch
        static : (B, static_ch, H, W) or None — terrain channels

        Returns
        -------
        logits : (B, 1, H, W)
        """
        _, _, H, W = pre.shape
        assert H % 16 == 0 and W % 16 == 0, (
            f"Input spatial dims must be divisible by 16, got ({H}, {W})")

        # Shared encoder on both branches
        pre_e1, pre_e2, pre_e3, pre_e4 = self._encode(pre)
        post_e1, post_e2, post_e3, post_e4 = self._encode(post)

        # Feature-level difference at each encoder stage
        diff_e1 = post_e1 - pre_e1
        diff_e2 = post_e2 - pre_e2
        diff_e3 = post_e3 - pre_e3
        diff_e4 = post_e4 - pre_e4

        # Bottleneck on deepest difference features
        b = self.bottleneck(self.pool(diff_e4))

        # Decoder with diff skip connections
        d4 = self.dec4(torch.cat([self.up4(b), diff_e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), diff_e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), diff_e2], dim=1))

        # Inject static channels at full resolution
        up1_out = torch.cat([self.up1(d2), diff_e1], dim=1)
        if static is not None:
            up1_out = torch.cat([up1_out, static], dim=1)
        d1 = self.dec1(up1_out)

        return self.out_conv(d1)

    def predict_proba(self, pre, post, static=None):
        """Run forward pass and return sigmoid probabilities."""
        return torch.sigmoid(self.forward(pre, post, static))
