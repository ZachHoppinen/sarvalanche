"""
UNet++ segmentation model for avalanche debris detection.

Architecture
------------
A lightweight UNet++ with a ResNet18 encoder trained from random weights.
An input projection layer maps the 11 SAR/terrain channels to the 3-channel
input expected by the encoder.

Input  : (B, 11, 64, 64) float32 — context patches from extract_context_patch
Output : (B, 1, 64, 64)  float32 — raw logits (apply sigmoid for probabilities)

Usage
-----
    model = DebrisSegmenter()
    logits = model(patches)            # (B, 1, 64, 64)
    probs  = torch.sigmoid(logits)     # (B, 1, 64, 64)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from sarvalanche.ml.track_patch_extraction import N_PATCH_CHANNELS


class DebrisSegmenter(nn.Module):
    def __init__(
        self,
        in_channels: int = N_PATCH_CHANNELS,
        encoder_name: str = 'resnet18',
        encoder_weights=None,
        patch_size: int = 64,  # stored for reference, not used by forward
    ):
        super().__init__()
        self.patch_size = patch_size
        self.unetpp = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
            activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unetpp(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))