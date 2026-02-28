import torch
import pytest

from sarvalanche.ml.track_patch_encoder import (
    TrackSegEncoder,
    TrackPatchEncoder,
    CNN_ENCODER_DIR,
    CNN_ENCODER_PATH,
    CNN_SEG_ENCODER_PATH,
    _IN_CHANNELS,
)


@pytest.fixture
def encoder():
    return TrackSegEncoder(in_channels=_IN_CHANNELS)


def test_forward_output_shape(encoder):
    """Forward should return (B, 1, H, W) logits."""
    x = torch.randn(2, _IN_CHANNELS, 64, 64)
    out = encoder(x)
    assert out.shape == (2, 1, 64, 64)


def test_segment_same_as_forward(encoder):
    """segment() and forward() should produce identical output."""
    x = torch.randn(1, _IN_CHANNELS, 32, 32)
    encoder.eval()
    with torch.no_grad():
        seg = encoder.segment(x)
        fwd = encoder(x)
    assert torch.allclose(seg, fwd)


def test_backward_compat_alias():
    """TrackPatchEncoder should be an alias for TrackSegEncoder."""
    assert TrackPatchEncoder is TrackSegEncoder


def test_variable_spatial_sizes(encoder):
    """Should handle different spatial sizes."""
    for size in [16, 32, 48, 64, 128]:
        x = torch.randn(1, _IN_CHANNELS, size, size)
        out = encoder(x)
        assert out.shape == (1, 1, size, size)


def test_gradient_flow(encoder):
    """Gradients should flow to all encoder parameters."""
    x = torch.randn(1, _IN_CHANNELS, 32, 32)
    out = encoder(x)
    loss = out.mean()
    loss.backward()
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_weight_paths_under_ml():
    """Weight paths should point to src/sarvalanche/ml/weights/."""
    assert 'sarvalanche/ml/weights/cnn_debris_detector' in str(CNN_ENCODER_DIR)
    assert str(CNN_ENCODER_PATH).endswith('track_patch_encoder.pt')
    assert str(CNN_SEG_ENCODER_PATH).endswith('track_seg_encoder.pt')
