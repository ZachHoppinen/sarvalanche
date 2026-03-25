"""Inference utilities for pairwise debris detector.

Provides model loading, per-pair SAR channel construction, sliding window
inference, and the full per-track pair orchestration.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from tqdm import tqdm

from sarvalanche.ml.pairwise_debris_classifier.channels import sign_log1p
from sarvalanche.ml.pairwise_debris_classifier.model import SinglePairDetector
from sarvalanche.ml.pairwise_debris_classifier.pair_extraction import (
    extract_all_pairs,
    get_track_data,
)

log = logging.getLogger(__name__)

PATCH_SIZE = 128


def _resolve_device(device_str=None):
    if device_str:
        return torch.device(device_str)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(
    weights_path: Path,
    device=None,
) -> tuple[nn.Module, torch.device]:
    """Load a SinglePairDetector from a checkpoint.

    Reads in_ch and base_ch from the checkpoint metadata if available,
    falling back to weight tensor shapes.
    """
    device = _resolve_device(device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        model_config = ckpt.get("model_config", {})
        in_ch = model_config.get("in_ch")
        base_ch = model_config.get("base_ch")
        log.info("Checkpoint epoch=%s, val_loss=%.4f",
                 ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan")))
    else:
        state_dict = ckpt
        in_ch = None
        base_ch = None

    if in_ch is None or base_ch is None:
        first_weight_key = "enc1.block.0.weight"
        if first_weight_key not in state_dict:
            raise KeyError(
                f"Cannot infer model architecture: '{first_weight_key}' not in checkpoint. "
                f"Keys found: {list(state_dict.keys())[:5]}...")
        if in_ch is None:
            in_ch = state_dict[first_weight_key].shape[1]
        if base_ch is None:
            base_ch = state_dict[first_weight_key].shape[0]

    model = SinglePairDetector(in_ch=in_ch, base_ch=base_ch).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    log.info("Loaded model: in_ch=%d, base_ch=%d, device=%s", in_ch, base_ch, device)

    return model, device


def build_sar_channels(
    vv_diff: np.ndarray,
    vh_diff: np.ndarray | None,
    anf_norm: np.ndarray,
) -> np.ndarray:
    """Build (4, H, W) SAR channel stack from pre-denoised dB diffs.

    Channels (matching SAR_CHANNELS): change_vv, change_vh, change_cr, anf.

    Parameters
    ----------
    vv_diff : (H, W) float32
        VV dB difference (after - before). NaN should be filled to 0 by caller.
    vh_diff : (H, W) float32 or None
        VH dB difference. NaN should be filled to 0 by caller.
    anf_norm : (H, W) float32
        Normalized ANF for this track. Range [0, 1].
    """
    H, W = vv_diff.shape
    change_vv = sign_log1p(vv_diff)

    if vh_diff is not None:
        change_vh = sign_log1p(vh_diff)
        cr = vh_diff - vv_diff
        change_cr = sign_log1p(cr)
    else:
        change_vh = np.zeros((H, W), dtype=np.float32)
        change_cr = np.zeros((H, W), dtype=np.float32)

    return np.stack([change_vv, change_vh, change_cr, anf_norm], axis=0)


def sliding_window_inference(
    sar_scene: np.ndarray,
    static_scene: np.ndarray,
    model: nn.Module,
    device: torch.device,
    patch_size: int = PATCH_SIZE,
    stride: int = 32,
    batch_size: int = 16,
) -> np.ndarray:
    """Run model on sliding windows, return (H, W) averaged probability.

    Pads the scene so that edge pixels are covered even when
    (H - patch_size) is not divisible by stride.
    """
    H, W = sar_scene.shape[1], sar_scene.shape[2]
    C_sar = sar_scene.shape[0]
    C_static = static_scene.shape[0]

    pad_h = (patch_size - (H % stride)) % stride if H > patch_size else 0
    pad_w = (patch_size - (W % stride)) % stride if W > patch_size else 0
    if pad_h > 0 or pad_w > 0:
        sar_scene = np.pad(sar_scene, ((0, 0), (0, pad_h), (0, pad_w)),
                           mode='constant', constant_values=0)
        static_scene = np.pad(static_scene, ((0, 0), (0, pad_h), (0, pad_w)),
                              mode='constant', constant_values=0)

    Hp, Wp = sar_scene.shape[1], sar_scene.shape[2]
    prob_sum = np.zeros((Hp, Wp), dtype=np.float64)
    count = np.zeros((Hp, Wp), dtype=np.float64)
    coords = [(y0, x0)
              for y0 in range(0, Hp - patch_size + 1, stride)
              for x0 in range(0, Wp - patch_size + 1, stride)]

    model.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(coords), batch_size),
                                desc="  Inference", leave=False,
                                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
            batch_coords = coords[batch_start:batch_start + batch_size]

            n = len(batch_coords)
            batch_np = np.empty((n, C_sar + C_static, patch_size, patch_size),
                                dtype=np.float32)
            for i, (y0, x0) in enumerate(batch_coords):
                batch_np[i, :C_sar] = sar_scene[:, y0:y0 + patch_size, x0:x0 + patch_size]
                batch_np[i, C_sar:] = static_scene[:, y0:y0 + patch_size, x0:x0 + patch_size]

            batch_tensor = torch.from_numpy(batch_np).to(device)
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

            for idx, (y0, x0) in enumerate(batch_coords):
                prob_sum[y0:y0 + patch_size, x0:x0 + patch_size] += probs[idx]
                count[y0:y0 + patch_size, x0:x0 + patch_size] += 1

    result = np.where(count[:H, :W] > 0,
                      prob_sum[:H, :W] / count[:H, :W],
                      np.nan).astype(np.float32)
    return result


def run_all_pairs_inference(
    ds: xr.Dataset,
    static_scene: np.ndarray,
    model: nn.Module,
    device: torch.device,
    max_span_days: int = 60,
    stride: int = 32,
    batch_size: int = 16,
) -> tuple[list[np.ndarray], list[dict]]:
    """Run per-pair inference across all tracks.

    Uses extract_all_pairs for fast pair generation from pre-denoised imagery.

    Returns
    -------
    pair_probs : list of (H, W) probability arrays
    pair_meta : list of dicts with 'track', 't_start', 't_end', 'span_days'
        t_start and t_end are pd.Timestamp objects.
    """
    pair_diffs, pair_metas, anf_per_track, _ = extract_all_pairs(
        ds, max_span_days=max_span_days)

    pair_probs = []
    for pi, (vv_arr, vh_arr, valid_mask) in enumerate(pair_diffs):
        meta = pair_metas[pi]
        anf_norm = anf_per_track[meta['track']]

        sar = build_sar_channels(vv_arr, vh_arr, anf_norm)

        prob_map = sliding_window_inference(
            sar, static_scene, model, device,
            stride=stride, batch_size=batch_size,
        )

        prob_map[~valid_mask] = np.nan
        pair_probs.append(prob_map)

    log.info("Inference complete: %d pairs", len(pair_probs))
    return pair_probs, pair_metas
