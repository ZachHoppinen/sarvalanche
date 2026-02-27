import numpy as np
import torch
import pytest

from sarvalanche.ml.losses import nll_loss


def test_nll_loss_perfect_prediction():
    """When mu == target, loss should depend only on sigma."""
    mu = torch.tensor([1.0, 2.0, 3.0])
    sigma = torch.ones(3)
    target = torch.tensor([1.0, 2.0, 3.0])
    loss = nll_loss(mu, sigma, target)
    # With mu==target and sigma==1: 0.5 * log(2*pi) ≈ 0.9189
    expected = 0.5 * np.log(2 * np.pi)
    assert abs(loss.item() - expected) < 1e-4


def test_nll_loss_increases_with_error():
    """Loss should increase when predictions are further from targets."""
    sigma = torch.ones(3)
    target = torch.zeros(3)
    loss_close = nll_loss(torch.tensor([0.1, 0.1, 0.1]), sigma, target)
    loss_far = nll_loss(torch.tensor([5.0, 5.0, 5.0]), sigma, target)
    assert loss_far > loss_close


def test_nll_loss_decreases_with_larger_sigma():
    """Larger sigma should reduce the residual term but increase the log term."""
    mu = torch.zeros(3)
    target = torch.tensor([2.0, 2.0, 2.0])
    loss_small_sigma = nll_loss(mu, torch.tensor([0.5, 0.5, 0.5]), target)
    loss_large_sigma = nll_loss(mu, torch.tensor([5.0, 5.0, 5.0]), target)
    # With large error and small sigma, loss should be very high
    assert loss_small_sigma > loss_large_sigma


def test_nll_loss_positive():
    """Loss should be positive for reasonable inputs."""
    mu = torch.randn(10)
    sigma = torch.ones(10) * 0.5
    target = torch.randn(10)
    loss = nll_loss(mu, sigma, target)
    assert loss.item() > 0


def test_nll_loss_gradient_flows():
    """Gradients should flow through mu and sigma."""
    mu = torch.randn(5, requires_grad=True)
    sigma = torch.ones(5, requires_grad=True)
    target = torch.randn(5)
    loss = nll_loss(mu, sigma, target)
    loss.backward()
    assert mu.grad is not None
    assert sigma.grad is not None
    assert not torch.all(mu.grad == 0)
