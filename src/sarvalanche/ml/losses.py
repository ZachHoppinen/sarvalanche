import torch
import numpy as np

def nll_loss(mu, sigma, target):
    """Negative log-likelihood - sigma is already clamped in model"""
    return (0.5 * torch.log(2 * np.pi * sigma**2) +
            (target - mu)**2 / (2 * sigma**2)).mean()

"""
Loss functions for debris segmentation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedDebrisLoss(nn.Module):
    """BCE loss with per-pixel weights based on label source.

    Manual polygon pixels get full weight. Weak supervision pixels
    (unmasked_p_target only) get reduced weight. Background gets
    the lowest weight to handle class imbalance.

    Parameters
    ----------
    manual_weight : float
        Weight for pixels covered by manual debris polygons.
    weak_weight : float
        Weight for pixels with unmasked_p_target > threshold but no
        manual polygon.
    bg_weight : float
        Weight for background pixels (target < threshold).
    weak_threshold : float
        p_target threshold above which a pixel is considered weak debris.
    """

    def __init__(
        self,
        manual_weight: float = 1.0,
        weak_weight: float = 0.3,
        bg_weight: float = 0.1,
        weak_threshold: float = 0.5,
    ):
        super().__init__()
        self.manual_weight = manual_weight
        self.weak_weight = weak_weight
        self.bg_weight = bg_weight
        self.weak_threshold = weak_threshold

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        manual_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : (B, 1, H, W) raw logits
        targets : (B, 1, H, W) blended soft targets in [0, 1]
        manual_masks : (B, 1, H, W) binary mask — 1 where manual polygon exists
            If None, all positive pixels are treated as weak supervision.

        Returns
        -------
        scalar loss
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Build per-pixel weight map
        weights = torch.full_like(targets, self.bg_weight)
        weak_pos = targets >= self.weak_threshold
        weights[weak_pos] = self.weak_weight

        if manual_masks is not None:
            weights[manual_masks > 0.5] = self.manual_weight
        else:
            # If no manual mask provided treat all positives as manual
            weights[weak_pos] = self.manual_weight

        return (bce * weights).mean()