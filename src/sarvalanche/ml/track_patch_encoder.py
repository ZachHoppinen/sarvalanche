from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Weights are saved alongside the XGBoost track predictor
CNN_ENCODER_DIR:      Path = Path(__file__).parent / 'weights' / 'cnn_debris_detector'
CNN_ENCODER_PATH:     Path = CNN_ENCODER_DIR / 'track_patch_encoder.pt'
CNN_SEG_ENCODER_PATH: Path = CNN_ENCODER_DIR / 'track_seg_encoder.pt'

# Must match track_features.N_PATCH_CHANNELS
_IN_CHANNELS: int = 8


def _conv_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Upsample 2x then conv to reduce channels."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.GELU(),
    )


class TrackSegEncoder(nn.Module):
    """
    Lightweight U-Net-style CNN for track polygon segmentation.

    Input: ``(B, 8, H, W)`` — eight-channel patch from ``extract_track_patch``.

    Outputs:
    - ``segment(x)`` / ``forward(x)`` → ``(B, 1, H, W)`` raw segmentation logits

    Architecture
    ------------
    Encoder: three conv blocks (8→16→32→64) with stride-2 downsampling.
    Decoder: U-Net skip connections, upsamples back to input resolution.

    Channels in order: combined_distance, d_empirical, fcf, slope,
    cell_counts, northing, easting, track_mask.
    """

    def __init__(self, in_channels: int = _IN_CHANNELS):
        super().__init__()

        # ── Encoder (named blocks for skip connections) ──────────────────
        self.enc1 = _conv_block(in_channels, 16)          # (B, 16, H,   W  )
        self.enc2 = _conv_block(16, 32, stride=2)          # (B, 32, H/2, W/2)
        self.enc3 = _conv_block(32, 64, stride=2)          # (B, 64, H/4, W/4)

        # ── Decoder (U-Net style) ────────────────────────────────────────
        # up3: upsample H/4→H/2, concat with enc2 skip (64+32→32)
        self.up3 = _up_block(64 + 32, 32)                  # (B, 32, H/2, W/2)
        # up2: upsample H/2→H, concat with enc1 skip (32+16→16)
        self.up2 = _up_block(32 + 16, 16)                  # (B, 16, H,   W  )
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)     # (B,  1, H,   W  )

    def segment(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, 1, H, W)`` raw segmentation logits."""
        e1 = self.enc1(x)   # (B, 16, H,   W  )
        e2 = self.enc2(e1)  # (B, 32, H/2, W/2)
        e3 = self.enc3(e2)  # (B, 64, H/4, W/4)

        # up3: cat e3 with e2 skip at H/4 resolution, then upsample 2x → H/2
        e2_down = F.interpolate(e2, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.up3(torch.cat([e3, e2_down], dim=1))  # → (B, 32, H/2, W/2)

        # up2: cat d3 with e1 skip at H/2 resolution, then upsample 2x → H
        e1_down = F.interpolate(e1, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.up2(torch.cat([d3, e1_down], dim=1))  # → (B, 16, H, W)

        return self.seg_head(d2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segment(x)


# Backward compatibility alias
TrackPatchEncoder = TrackSegEncoder
