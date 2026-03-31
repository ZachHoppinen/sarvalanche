"""TinyCD-inspired debris detector.

Siamese encoder with shared weights on pre/post SAR patches.
Key differences from our plain Siamese U-Net:
  - Depthwise separable convolutions for efficiency
  - Squeeze-excite channel attention at each encoder stage
  - Mix module: learned feature fusion (not just subtraction)
  - Lightweight decoder with pixel-wise MLP head

Input: pre (B, 3, H, W) + post (B, 3, H, W) + optional static (B, 5, H, W)
Output: (B, 1, H, W) raw logits

H and W must be divisible by 16.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sarvalanche.ml.tinycd_debris_classifier.channels import N_BRANCH, N_STATIC


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv: depthwise + pointwise. Much fewer params."""

    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=padding,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class SqueezeExcite(nn.Module):
    """Channel attention via squeeze-excite."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class EncoderBlock(nn.Module):
    """Depthwise separable conv block with squeeze-excite attention."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_ch, out_ch)
        self.conv2 = DepthwiseSeparableConv(out_ch, out_ch)
        self.se = SqueezeExcite(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.se(x)


class MixModule(nn.Module):
    """Learned feature mixing between pre and post branches.

    Instead of simple subtraction, uses grouped convolution on
    concatenated features to learn cross-temporal correlations,
    followed by channel attention.
    """

    def __init__(self, channels):
        super().__init__()
        # Grouped conv on concatenated [pre, post] features
        self.mix_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, groups=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.se = SqueezeExcite(channels)

    def forward(self, pre_feat, post_feat):
        concat = torch.cat([pre_feat, post_feat], dim=1)
        mixed = self.mix_conv(concat)
        return self.se(mixed)


class DecoderBlock(nn.Module):
    """Lightweight decoder: upsample + depthwise separable conv."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class TinyCDDebrisDetector(nn.Module):
    """TinyCD-inspired change detection for avalanche debris.

    Parameters
    ----------
    branch_ch : int
        Channels per branch (VV, VH, ANF). Default 3.
    static_ch : int
        Static terrain channels. Default 5. Set to 0 for SAR-only.
    base_ch : int
        Base channel width.
    """

    def __init__(self, branch_ch: int = N_BRANCH, static_ch: int = N_STATIC,
                 base_ch: int = 24):
        super().__init__()
        self.branch_ch = branch_ch
        self.static_ch = static_ch
        self.base_ch = base_ch

        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Shared encoder
        self.enc1 = EncoderBlock(branch_ch, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        self.enc4 = EncoderBlock(c3, c4)

        self.pool = nn.MaxPool2d(2)

        # Mix modules at each level
        self.mix1 = MixModule(c1)
        self.mix2 = MixModule(c2)
        self.mix3 = MixModule(c3)
        self.mix4 = MixModule(c4)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv(c4, c4),
            SqueezeExcite(c4),
        )

        # Decoder
        self.dec4 = DecoderBlock(c4, c4, c3)
        self.dec3 = DecoderBlock(c3, c3, c2)
        self.dec2 = DecoderBlock(c2, c2, c1)
        self.dec1 = DecoderBlock(c1, c1 + static_ch, c1)

        # Pixel-wise MLP head
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, 1, 1),
        )

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def forward(self, pre, post, static=None):
        """
        Parameters
        ----------
        pre : (B, branch_ch, H, W)
        post : (B, branch_ch, H, W)
        static : (B, static_ch, H, W) or None

        Returns
        -------
        logits : (B, 1, H, W)
        """
        _, _, H, W = pre.shape
        assert H % 16 == 0 and W % 16 == 0, (
            f"Input must be divisible by 16, got ({H}, {W})")

        # Shared encoder on both branches
        pre_e1, pre_e2, pre_e3, pre_e4 = self._encode(pre)
        post_e1, post_e2, post_e3, post_e4 = self._encode(post)

        # Learned mixing at each level
        m1 = self.mix1(pre_e1, post_e1)
        m2 = self.mix2(pre_e2, post_e2)
        m3 = self.mix3(pre_e3, post_e3)
        m4 = self.mix4(pre_e4, post_e4)

        # Bottleneck on deepest mixed features
        b = self.bottleneck(self.pool(m4))

        # Decoder with mixed skip connections
        d4 = self.dec4(b, m4)
        d3 = self.dec3(d4, m3)
        d2 = self.dec2(d3, m2)

        # Inject static at full resolution
        if static is not None:
            skip1 = torch.cat([m1, static], dim=1)
        else:
            skip1 = m1
        d1 = self.dec1(d2, skip1)

        return self.head(d1)

    def predict_proba(self, pre, post, static=None):
        return torch.sigmoid(self.forward(pre, post, static))
