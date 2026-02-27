import torch
import pytest

from sarvalanche.ml.SARTransformer import SARTransformer


@pytest.fixture
def model():
    return SARTransformer(
        img_size=16, patch_size=8, in_chans=2,
        embed_dim=32, depth=1, num_heads=2,
        min_sigma=0.05, max_seq_len=10, dropout=0.0,
    )


def test_forward_output_shapes(model):
    """Forward pass should return mu and sigma with correct shapes."""
    x = torch.randn(2, 5, 2, 16, 16)  # (B, T, C, H, W)
    mu, sigma = model(x)
    assert mu.shape == (2, 2, 16, 16)
    assert sigma.shape == (2, 2, 16, 16)


def test_sigma_clamped(model):
    """Sigma output must be >= min_sigma."""
    x = torch.randn(1, 3, 2, 16, 16)
    _, sigma = model(x)
    assert torch.all(sigma >= model.min_sigma)


def test_patchify_shape(model):
    """Patchify should split image into n_patches of patch_dim."""
    x = torch.randn(2, 2, 16, 16)  # (B, C, H, W)
    patches = model.patchify(x)
    # 16/8 = 2 patches per dim → 4 patches, each 2*8*8 = 128
    assert patches.shape == (2, 4, 128)


def test_patchify_content(model):
    """Patchify should preserve pixel values."""
    x = torch.arange(2 * 16 * 16, dtype=torch.float32).reshape(1, 2, 16, 16)
    patches = model.patchify(x)
    # Top-left patch should contain x[:, :, 0:8, 0:8]
    expected = x[:, :, 0:8, 0:8].reshape(1, -1)
    assert torch.allclose(patches[0, 0], expected[0])


def test_variable_sequence_lengths(model):
    """Model should handle different temporal lengths up to max_seq_len."""
    for t in [2, 5, 10]:
        x = torch.randn(1, t, 2, 16, 16)
        mu, sigma = model(x)
        assert mu.shape == (1, 2, 16, 16)


def test_single_timestep(model):
    """Model should work with a single timestep."""
    x = torch.randn(1, 1, 2, 16, 16)
    mu, sigma = model(x)
    assert mu.shape == (1, 2, 16, 16)


def test_eval_mode_deterministic(model):
    """Eval mode should produce deterministic outputs."""
    model.eval()
    x = torch.randn(1, 3, 2, 16, 16)
    mu1, sigma1 = model(x)
    mu2, sigma2 = model(x)
    assert torch.allclose(mu1, mu2)
    assert torch.allclose(sigma1, sigma2)


def test_custom_config():
    """Model should accept custom configurations."""
    model = SARTransformer(
        img_size=32, patch_size=16, in_chans=1,
        embed_dim=64, depth=2, num_heads=4,
        min_sigma=0.1, max_seq_len=5, dropout=0.1,
    )
    x = torch.randn(1, 3, 1, 32, 32)
    mu, sigma = model(x)
    assert mu.shape == (1, 1, 32, 32)
    assert torch.all(sigma >= 0.1)
