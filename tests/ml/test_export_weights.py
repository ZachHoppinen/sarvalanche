import numpy as np
import pytest
import torch
from pathlib import Path

from sarvalanche.ml.export_weights import (
    build_filename,
    extract_checkpoint_metadata,
    write_sidecar,
    export_weights,
    WEIGHTS_DIR,
)


def test_build_filename_format():
    """Filename should follow {name}_v{version}_{YYYYMMDD} pattern."""
    result = build_filename('test_model', '1.2.3')
    assert result.startswith('test_model_v1.2.3_')
    assert len(result.split('_')[-1]) == 8  # YYYYMMDD


def test_extract_checkpoint_metadata_full():
    """Should extract known fields from checkpoint dict."""
    ckpt = {
        'epoch': 50,
        'train_loss': 0.123,
        'val_loss': 0.456,
        'model_config': {'embed_dim': 256, 'depth': 4},
        'zones': ['Zone_A', 'Zone_B'],
        'seasons': ['2023-2024'],
        'model_state_dict': {},  # should be ignored
    }
    meta = extract_checkpoint_metadata(ckpt)
    assert meta['epoch'] == 50
    assert meta['train_loss'] == 0.123
    assert meta['val_loss'] == 0.456
    assert meta['model_config'] == {'embed_dim': 256, 'depth': 4}
    assert meta['zones'] == ['Zone_A', 'Zone_B']
    assert 'model_state_dict' not in meta


def test_extract_checkpoint_metadata_minimal():
    """Should handle checkpoints missing optional fields."""
    ckpt = {'model_state_dict': {}}
    meta = extract_checkpoint_metadata(ckpt)
    assert meta == {}


def test_write_sidecar(tmp_path):
    """Sidecar should be written with expected content."""
    path = tmp_path / 'test.txt'
    write_sidecar(
        path=path,
        model_name='test_model',
        version='0.1.0',
        train_samples=100,
        test_samples=25,
        checkpoint_meta={'epoch': 10, 'train_loss': 0.5},
        extra_metrics={'f1': 0.88},
        notes='Test run',
    )
    text = path.read_text()
    assert 'test_model' in text
    assert '0.1.0' in text
    assert 'train samples    : 100' in text
    assert 'test samples     : 25' in text
    assert 'total samples    : 125' in text
    assert 'epoch            : 10' in text
    assert 'f1' in text
    assert 'Test run' in text


def test_write_sidecar_no_extras(tmp_path):
    """Sidecar without optional fields should still work."""
    path = tmp_path / 'test.txt'
    write_sidecar(
        path=path,
        model_name='minimal',
        version='0.0.1',
        train_samples=10,
        test_samples=5,
        checkpoint_meta={},
        extra_metrics=None,
        notes=None,
    )
    text = path.read_text()
    assert 'minimal' in text
    assert 'n/a' in text  # epoch should be n/a


def test_export_weights_roundtrip(tmp_path):
    """Export should copy .pth and create .txt sidecar."""
    # Create a fake checkpoint
    ckpt_path = tmp_path / 'model.pth'
    torch.save({
        'epoch': 5,
        'model_state_dict': {'param': torch.randn(10)},
        'model_config': {'embed_dim': 64},
    }, ckpt_path)

    out_dir = tmp_path / 'weights_out'
    dest = export_weights(
        checkpoint_path=ckpt_path,
        model_name='test_export',
        train_samples=50,
        test_samples=10,
        weights_dir=out_dir,
    )
    assert dest.exists()
    assert dest.suffix == '.pth'
    # Sidecar should also exist
    sidecar = dest.with_suffix('.txt')
    assert sidecar.exists()
    assert 'test_export' in sidecar.read_text()


def test_export_weights_missing_checkpoint(tmp_path):
    """Should raise FileNotFoundError for missing checkpoint."""
    with pytest.raises(FileNotFoundError):
        export_weights(
            checkpoint_path=tmp_path / 'nonexistent.pth',
            model_name='test',
            train_samples=1,
            test_samples=1,
        )


def test_weights_dir_points_to_rtc_predictor():
    """WEIGHTS_DIR should point to rtc_predictor subdirectory."""
    assert str(WEIGHTS_DIR).endswith('weights/rtc_predictor')
