import torch
import numpy as np

def nll_loss(mu, sigma, target):
    """Negative log-likelihood - sigma is already clamped in model"""
    return (0.5 * torch.log(2 * np.pi * sigma**2) +
            (target - mu)**2 / (2 * sigma**2)).mean()