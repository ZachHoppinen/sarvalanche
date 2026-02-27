import requests
import json
import shutil
from pathlib import Path
from shapely.geometry import shape, box
from collections import defaultdict
import logging

import torch
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd

from sarvalanche.utils.projections import resolution_to_degrees
from sarvalanche.utils.validation import validate_crs
from sarvalanche.io import assemble_dataset
from sarvalanche.preprocessing.pipelines import preprocess_rtc
from sarvalanche.preprocessing.radiometric import linear_to_dB
from sarvalanche.ml.SARTimeSeriesDataset import SARTimeSeriesDataset
from sarvalanche.ml.SARTransformer import SARTransformer
from sarvalanche.ml.losses import nll_loss
from sarvalanche.io.export import export_netcdf
from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.ml.export_weights import export_weights

import threading, time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(name)s  %(message)s'
)
logging.getLogger('asf_search').setLevel(logging.WARNING)


# ── Augmentation ──────────────────────────────────────────────────────────────

def random_temporal_mask(baseline, p=0.2):
    """
    Randomly zero out entire timesteps during training.
    Forces the model not to over-rely on any single acquisition.
    The most recent timestep is always kept.

    baseline: (B, T, C, H, W)
    returns:  (B, T, C, H, W) with some timesteps zeroed
    """
    B, T, C, H, W = baseline.shape
    # Sample a binary mask: 1 = keep, 0 = drop
    mask = (torch.rand(B, T, device=baseline.device) > p).float()
    mask[:, -1] = 1.0  # always keep most recent timestep
    return baseline * mask.view(B, T, 1, 1, 1)


def random_flip(baseline, target):
    """
    Random horizontal and vertical flips.
    SAR backscatter is statistically invariant to these within a patch.

    baseline: (B, T, C, H, W)
    target:   (B, C, H, W)
    """
    if torch.rand(1).item() > 0.5:
        baseline = torch.flip(baseline, dims=[-1])  # horizontal
        target   = torch.flip(target,   dims=[-1])
    if torch.rand(1).item() > 0.5:
        baseline = torch.flip(baseline, dims=[-2])  # vertical
        target   = torch.flip(target,   dims=[-2])
    return baseline, target


def add_noise(baseline, std=0.1):
    """
    Additive Gaussian noise ~0.1 dB to simulate thermal noise variation.
    baseline: (B, T, C, H, W)
    """
    return baseline + torch.randn_like(baseline) * std


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_variable_length(batch):
    max_T = max([item['baseline'].shape[0] for item in batch])
    baselines, targets = [], []
    for item in batch:
        baseline = item['baseline']   # (T, C, H, W)
        target   = item['target']     # (C, H, W)
        T, C, H, W = baseline.shape
        if T < max_T:
            padding  = torch.zeros(max_T - T, C, H, W)
            baseline = torch.cat([baseline, padding], dim=0)
        baselines.append(baseline)
        targets.append(target)
    return {
        'baseline': torch.stack(baselines),
        'target':   torch.stack(targets),
    }


# ── Zone helpers ──────────────────────────────────────────────────────────────

def fetch_zones(target_centers, geojson_cache, force_refresh=False):
    if geojson_cache.exists() and not force_refresh:
        print(f"Loading zones from cache: {geojson_cache}")
        geojson = json.loads(geojson_cache.read_text())
    else:
        print("Fetching zones from avalanche.org API...")
        url     = "https://api.avalanche.org/v2/public/products/map-layer"
        headers = {'User-Agent': 'sarvalanche-research/1.0 (your@email.com)'}
        geojson = requests.get(url, headers=headers).json()
        geojson_cache.write_text(json.dumps(geojson, indent=2))
        print(f"  Saved to {geojson_cache}")

    zones = {}
    for feature in geojson['features']:
        props     = feature['properties']
        center_id = props.get('center_id', '')
        if center_id not in target_centers:
            continue
        try:
            key = f"{center_id}_{props['name'].replace(' ', '_').replace('/', '-')}"
            zones[key] = {
                'aoi':       box(*shape(feature['geometry']).bounds),
                'center_id': center_id,
                'name':      props['name'],
            }
        except Exception as e:
            print(f"  Skipping {props.get('name', '?')}: {e}")

    print(f"Found {len(zones)} zones across {target_centers}")
    return zones


def ds_to_stacked_array(ds):
    return xr.concat([ds['VV'], ds['VH']], dim='polarization').transpose('time', 'polarization', 'y', 'x')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # --- CONFIG ---
    RESOLUTION_M   = 20
    CRS            = 'EPSG:4326'
    CACHE_DIR      = Path('/Users/zmhoppinen/Documents/sarvalanche/local/data')
    GEOJSON_CACHE  = CACHE_DIR / 'forecast_zones.geojson'
    TV_WEIGHT      = 0.5
    MIN_SEQ_LEN    = 5
    MAX_SEQ_LEN    = 15          # ← increased from 10; model will see up to 14 timesteps as context
    STRIDE         = 48
    WEIGHTS_DIR    = Path('/Users/zmhoppinen/Documents/sarvalanche/src/sarvalanche/ml/weights/rtc_predictor')
    CHECKPOINT_PATH = WEIGHTS_DIR / 'sar_transformer_best.pth'

    # Training hyperparams
    N_EPOCHS       = 75          # more epochs since LR schedule is longer
    BATCH_SIZE     = 256
    BASE_LR        = 1e-4
    SIGMA_REG      = 0.01        # penalty weight on mean log-sigma to curb blowup
    AUG_MASK_P     = 0.10        # probability of dropping any given timestep
    AUG_NOISE_STD  = 0.05         # dB noise std

    TARGET_CENTERS = ['SNFAC', 'GNFAC', 'UAC', 'FAC', 'BTAC', 'CAIC']

    SEASONS = [
        ('2019-12-01', '2020-03-31'),
        ('2020-12-01', '2021-03-31'),
        ('2021-12-01', '2022-03-31'),
    ]

    TEST_ZONES = {
        'GNFAC_Bridger_Range',
        'SNFAC_Soldier_Mountain',
        'UAC_Abajos',
        'BTAC_Snake_River_Range'
    }
    VAL_ZONES = {
        'GNFAC_Southern_Madison_Range',
        'SNFAC_Banner_Summit',
        'CAIC_CAIC_zone'
        'UAC_Uintas',
        'BTAC_Salt_River_and_Wyoming_Ranges',
        'FAC_Flathead_Range_&_Glacier_NP'
    }
    resolution = resolution_to_degrees(RESOLUTION_M, validate_crs(CRS))

    # --- FETCH ZONES ---
    zones = fetch_zones(TARGET_CENTERS, GEOJSON_CACHE)

    # --- LOAD SAR SCENES ---
    train_paths, val_paths, test_paths = [], [], []
    failed = []

    SCENE_CACHE_DIR = CACHE_DIR / 'scene_cache'
    SCENE_CACHE_DIR.mkdir(exist_ok=True)

    for zone_key, zone_info in zones.items():
        bounds = zone_info['aoi'].bounds
        width_deg  = bounds[2] - bounds[0]
        height_deg = bounds[3] - bounds[1]
        if (width_deg * height_deg) > 0.8:
            print(f"{zone_key:50s}  {width_deg:.2f}° x {height_deg:.2f}°  — SKIPPED (too large)")
            continue

        print(f"{zone_key:50s}  {width_deg:.2f}° x {height_deg:.2f}°")

        for start_date, stop_date in SEASONS:
            season_key  = start_date[:7]
            cache_file  = SCENE_CACHE_DIR / f"{zone_key}__{season_key}.nc"

            print(f"Loading {zone_key} | {season_key}...")

            try:
                if cache_file.exists():
                    ds = load_netcdf_to_dataset(cache_file)
                    print(f"  Loaded from cache: {cache_file.name}")
                else:
                    ds = assemble_dataset(
                        aoi=zone_info['aoi'],
                        crs=CRS,
                        resolution=resolution,
                        start_date=start_date,
                        stop_date=stop_date,
                        cache_dir=CACHE_DIR,
                        sar_only=True,
                    )
                    ds = preprocess_rtc(ds, tv_weight=TV_WEIGHT)
                    for pol in ['VV', 'VH']:
                        ds[pol] = linear_to_dB(ds[pol])
                    export_netcdf(ds[['VV', 'VH']], cache_file)
                    print(f"  Saved to {cache_file.name}")

                tracks = np.unique(ds.track.values)
                print(f"  Found {len(tracks)} tracks: {tracks}")

                existing_tracks = list(SCENE_CACHE_DIR.glob(f"{zone_key}__track*__{season_key}.nc"))
                valid_tracks    = [t for t in tracks if len(ds.sel(time=ds.track == t).time) >= 3]

                if len(existing_tracks) == len(valid_tracks):
                    ds.close()
                    del ds
                    print(f"  Found all {len(existing_tracks)} cached tracks, skipping assembly")
                    for f in existing_tracks:
                        if any(zone_key.startswith(z) for z in TEST_ZONES):
                            test_paths.append(f)
                        elif any(zone_key.startswith(z) for z in VAL_ZONES):
                            val_paths.append(f)
                        else:
                            train_paths.append(f)
                    continue

                for track in tracks:
                    ds_track  = ds.sel(time=ds.track == track)
                    track_key = f"{zone_key}__track{track}__{season_key}"
                    cache_file_track = SCENE_CACHE_DIR / f"{track_key}.nc"

                    if len(ds_track.time) < 3:
                        print(f"  Skipping track {track}: only {len(ds_track.time)} timesteps")
                        continue

                    if not cache_file_track.exists():
                        export_netcdf(ds_track[['VV', 'VH']], cache_file_track)
                        print(f"  Saved track {track} → {cache_file_track.name}")

                    del ds_track

                    if any(zone_key.startswith(z) for z in TEST_ZONES):
                        test_paths.append(cache_file_track)
                    elif any(zone_key.startswith(z) for z in VAL_ZONES):
                        val_paths.append(cache_file_track)
                    else:
                        train_paths.append(cache_file_track)

                import gc
                ds.close()
                del ds
                gc.collect()

            except Exception as e:
                print(f"  FAILED: {e}")
                failed.append((zone_key, season_key, str(e)))

    # --- ZARR CONVERSION ---
    ZARR_CACHE_DIR = CACHE_DIR / 'zarr_cache'
    ZARR_CACHE_DIR.mkdir(exist_ok=True)

    def nc_to_zarr(nc_path: Path, zarr_dir: Path) -> Path:
        zarr_path = zarr_dir / (nc_path.stem + '.zarr')
        if zarr_path.exists():
            return zarr_path
        print(f"  Converting {nc_path.name} → zarr...")
        with xr.open_dataset(nc_path) as ds:
            da = xr.concat([ds['VV'], ds['VH']], dim='polarization') \
                .transpose('time', 'polarization', 'y', 'x')
            da = da.rename('backscatter')
            da = da.drop_vars([v for v in da.coords
                               if da.coords[v].dtype.kind in ('U', 'S', 'O')
                               and v != 'time'])
            da.chunk({'time': 1, 'polarization': -1, 'y': 128, 'x': 128}) \
              .to_zarr(zarr_path, consolidated=False)
        return zarr_path

    print("\nConverting track files to zarr...")
    train_paths = [nc_to_zarr(p, ZARR_CACHE_DIR) for p in tqdm(train_paths, desc='train')]
    val_paths   = [nc_to_zarr(p, ZARR_CACHE_DIR) for p in tqdm(val_paths,   desc='val')]
    test_paths  = [nc_to_zarr(p, ZARR_CACHE_DIR) for p in tqdm(test_paths,  desc='test')]

    print(f"\nSplit summary:")
    print(f"  Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    print(f"  Failed: {len(failed)}")
    for f in failed:
        print(f"    {f[0]} | {f[1]}: {f[2]}")

    assert len(train_paths) > 0, "No training scenes loaded"
    assert len(val_paths)   > 0, "No val scenes loaded"

    # --- DATASETS ---
    train_dataset = SARTimeSeriesDataset(
        train_paths, min_seq_len=MIN_SEQ_LEN, max_seq_len=MAX_SEQ_LEN,
        patch_size=16, stride=STRIDE
    )
    val_dataset = SARTimeSeriesDataset(
        val_paths, min_seq_len=MIN_SEQ_LEN, max_seq_len=MAX_SEQ_LEN,
        patch_size=16, stride=STRIDE
    )
    test_dataset = SARTimeSeriesDataset(
        test_paths, min_seq_len=MIN_SEQ_LEN, max_seq_len=MAX_SEQ_LEN,
        patch_size=16, stride=STRIDE
    )

    print(f"\nDataset sizes — Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # --- DATALOADERS ---
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=False,
        collate_fn=collate_variable_length, persistent_workers=True
    )
    print(f"Created train DataLoader with {train_loader.num_workers} workers")

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=False,
        collate_fn=collate_variable_length, persistent_workers=True
    )
    print(f"Created val DataLoader with {val_loader.num_workers} workers")

    # --- MODEL ---
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SARTransformer(
        img_size=16, patch_size=8, in_chans=2,
        embed_dim=384, depth=6, num_heads=6,
        max_seq_len=MAX_SEQ_LEN,   # ← must match dataset
    )
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)

    # Cosine annealing with warm restarts
    # T_0=10: first restart at epoch 10, then 20, then 40 (T_mult=2)
    # eta_min: floor LR so it never fully dies
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # --- CHECKPOINT ---
    start_epoch   = 0
    best_val_loss = float('inf')

    if CHECKPOINT_PATH.exists():
        print(f"Resuming from {CHECKPOINT_PATH}...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch   = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"  Resumed from epoch {checkpoint['epoch']}, best val loss: {best_val_loss:.4f}")
    else:
        print("No checkpoint found, training from scratch.")

    n_batches = len(train_loader)

    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, N_EPOCHS):

        model.train()
        train_loss = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch:02d} [train]', leave=True)
        for batch_idx, batch in enumerate(train_bar):

            baseline_batch = batch['baseline'].to(device)  # (B, T, C, H, W)
            target_batch   = batch['target'].to(device)    # (B, C, H, W)

            # ── Augmentation (train only) ──────────────────────────────────
            baseline_batch = random_temporal_mask(baseline_batch, p=AUG_MASK_P)
            baseline_batch, target_batch = random_flip(baseline_batch, target_batch)
            baseline_batch = add_noise(baseline_batch, std=AUG_NOISE_STD)
            # ──────────────────────────────────────────────────────────────

            mu, sigma = model(baseline_batch)

            # NLL loss + sigma regularization to prevent uncertainty blowup
            loss = nll_loss(mu, sigma, target_batch)
            loss = loss + SIGMA_REG * torch.mean(torch.log(sigma))

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            # Step scheduler every batch for smooth cosine curve
            scheduler.step(epoch + batch_idx / n_batches)

            train_loss += loss.item()
            avg_so_far  = train_loss / (batch_idx + 1)
            train_bar.set_postfix({
                'loss':      f'{loss.item():.4f}',
                'avg':       f'{avg_so_far:.4f}',
                'grad_norm': f'{grad_norm:.3f}',
            })

        avg_train_loss = train_loss / n_batches

        # -- Validate (no augmentation) --
        model.eval()
        val_loss = 0

        val_bar = tqdm(val_loader, desc=f'Epoch {epoch:02d} [val]  ', leave=True)
        with torch.no_grad():
            for batch in val_bar:
                baseline_batch = batch['baseline'].to(device)
                target_batch   = batch['target'].to(device)
                mu, sigma      = model(baseline_batch)
                # Val loss is pure NLL — no augmentation, no reg penalty
                batch_val_loss = nll_loss(mu, sigma, target_batch).item()
                if not np.isfinite(batch_val_loss):
                    print(f'  [val] skipping non-finite batch loss: {batch_val_loss:.4f}')
                    continue
                val_loss      += batch_val_loss
                val_bar.set_postfix({'val_loss': f'{batch_val_loss:.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        current_lr   = optimizer.param_groups[0]['lr']

        print(
            f'\nEpoch {epoch:02d} summary'
            f'  |  train: {avg_train_loss:.4f}'
            f'  |  val: {avg_val_loss:.4f}'
            f'  |  Δ: {avg_val_loss - avg_train_loss:+.4f}'
            f'  |  lr: {current_lr:.2e}'
            f'  |  best: {best_val_loss:.4f}'
        )

        print(f'\nEpoch {epoch} — train: {avg_train_loss:.4f} | val: {avg_val_loss:.4f}')

        # -- Diagnostics --
        with torch.no_grad():
            print(f"  Target range : [{target_batch.min():.2f}, {target_batch.max():.2f}]")
            print(f"  μ range      : [{mu.min():.2f}, {mu.max():.2f}]")
            print(f"  sigma range  : [{sigma.min():.2f}, {sigma.max():.2f}]  mean sigma: {sigma.mean():.4f}")
            print(f"  MAE          : {torch.abs(target_batch - mu).mean():.4f}")
            corr = torch.corrcoef(torch.stack([target_batch.flatten(), mu.flatten()]))[0, 1]
            print(f"  Correlation  : {corr:.4f}")

        # -- Save best --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss':           avg_train_loss,
                'val_loss':             avg_val_loss,
                'model_config': {
                    'img_size': 16, 'patch_size': 8, 'in_chans': 2,
                    'embed_dim': 384, 'depth': 6, 'num_heads': 6,
                    'max_seq_len': MAX_SEQ_LEN,
                },
                'zones':   list(zones.keys()),
                'seasons': SEASONS,
            }, CHECKPOINT_PATH)
            print(f'  ✓ Saved best model (val loss: {best_val_loss:.4f})')

    # -- Save final --
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss':           avg_train_loss,
        'val_loss':             avg_val_loss,
        'model_config': {
            'img_size': 16, 'patch_size': 8, 'in_chans': 2,
            'embed_dim': 384, 'depth': 6, 'num_heads': 6,
            'max_seq_len': MAX_SEQ_LEN,
        },
        'zones':   list(zones.keys()),
        'seasons': SEASONS,
    }, WEIGHTS_DIR / 'sar_transformer_final.pth')

    # --- TEST EVALUATION ---
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=False,
        collate_fn=collate_variable_length, persistent_workers=True
    )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss = 0
    test_bar  = tqdm(test_loader, desc='Test eval', leave=True)
    with torch.no_grad():
        for batch in test_bar:
            baseline_batch = batch['baseline'].to(device)
            target_batch   = batch['target'].to(device)
            mu, sigma      = model(baseline_batch)
            test_loss     += nll_loss(mu, sigma, target_batch).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest loss: {avg_test_loss:.4f}")

    # --- EXPORT WEIGHTS ---
    for ckpt_path, label in [
        (CHECKPOINT_PATH,                      'sar_transformer_best'),
        (WEIGHTS_DIR / 'sar_transformer_final.pth', 'sar_transformer_final'),
    ]:
        if ckpt_path.exists():
            export_weights(
                checkpoint_path=ckpt_path,
                model_name=label,
                train_samples=len(train_dataset),
                test_samples=len(test_dataset),
                extra_metrics={'test_loss': avg_test_loss},
                notes=(
                    f"v3: cosine LR warmup restarts (T0=10, Tmult=2), "
                    f"temporal dropout (p={AUG_MASK_P}), flip aug, noise aug (std={AUG_NOISE_STD}), "
                    f"sigma reg (λ={SIGMA_REG}), max_seq_len={MAX_SEQ_LEN} | "
                    f"Zones: {list(zones.keys())} | Seasons: {SEASONS}"
                ),
                weights_dir=WEIGHTS_DIR
            )

    print('Training complete.')