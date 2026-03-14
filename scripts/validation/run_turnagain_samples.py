"""
Run the empirical detection pipeline (no ML model) for 5 sample dates in
Turnagain Pass using the existing season dataset. tau=6.

5 sample dates chosen so each falls in a distinct full-swath acquisition window
(unique crossing pairs for the right 2/3 of the scene). Note: Jan15→Mar16 has a
60-day gap so dates in that window share the same full-swath pair.
Outputs GeoTIFFs to local/cnfaic/sample_detections/<date>/probabilities/
"""

from pathlib import Path
import sys
import logging
import gc

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sarvalanche.io.dataset import load_netcdf_to_dataset
from sarvalanche.io.export import export_netcdf
from sarvalanche.preprocessing.pipelines import preprocess_rtc
from sarvalanche.weights.pipelines import get_static_weights
from sarvalanche.probabilities.pipelines import get_static_probabilities
from sarvalanche.detection.backscatter_change import calculate_empirical_backscatter_probability
from sarvalanche.probabilities.combine import combine_probabilities

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path("/Users/zmhoppinen/Documents/sarvalanche/local/cnfaic")
SEASON_NC = BASE / "netcdfs" / "Turnagain_Pass_and_Girdwood" / "season_2025-2026_Turnagain_Pass_and_Girdwood.nc"

TAU = 6

DATES = {
    "2025-11-28": "NOV",   # Nov23→Dec05 window, 4 obs, max d=3.0
    "2025-12-13": "DEC",   # Dec05→Dec17 window, 4 obs, max d=2.0
    "2026-01-11": "JAN",   # Jan10→Feb03 window, 13 obs, max d=3.0
    "2026-02-14": "FEB",   # Feb03→Feb15 window, 3 obs, max d=2.5
    "2026-03-10": "MAR",   # Feb27→Mar11 window, 3 obs, max d=2.0
}


def run_one(ds_preproc, avalanche_date_str, label, cache_dir):
    """Run empirical detection on preprocessed dataset for one date."""
    avalanche_date = pd.Timestamp(avalanche_date_str)
    ds = ds_preproc.copy(deep=True)

    log.info(f"Computing weights (tau={TAU})...")
    ds = get_static_weights(ds, avalanche_date, temporal_decay_factor=TAU)

    log.info("Computing static probabilities...")
    ds = get_static_probabilities(ds, avalanche_date)

    log.info("Computing empirical backscatter probability...")
    ds["p_empirical"], ds["d_empirical"] = calculate_empirical_backscatter_probability(
        ds, avalanche_date,
        smooth_method=None,
        use_agreement_boosting=True,
        agreement_strength=0.8,
        min_prob_threshold=0.2,
    )

    # Combine: prior (static) * likelihood (empirical) via Bayesian update
    p_prior = combine_probabilities(
        xr.concat([ds["p_fcf"], ds["p_runout"], ds["p_slope"]], dim="factor"),
        dim="factor", method="weighted_mean",
    )
    p_likelihood = ds["p_empirical"]

    eps = 1e-10
    denominator = p_prior * p_likelihood + (1 - p_prior) * (1 - p_likelihood)
    p_bayesian = p_prior * p_likelihood / denominator.clip(min=eps)
    p_pixelwise = xr.ufuncs.minimum(p_likelihood, p_bayesian)

    # Soft runout constraint
    runout_scale = (ds["p_runout"] / 0.1).clip(min=0, max=1)
    p_pixelwise = p_pixelwise * runout_scale
    p_pixelwise = p_pixelwise.where(~p_pixelwise.isnull(), 0)
    p_pixelwise.attrs = {"source": "sarvalanche", "units": "probability", "product": "pixel_wise_probability"}
    ds["p_pixelwise"] = p_pixelwise

    # Export GeoTIFFs
    prob_dir = cache_dir / "probabilities"
    prob_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{avalanche_date_str}_{label}"

    for var in ["p_pixelwise", "p_empirical", "d_empirical", "p_fcf", "p_runout", "p_slope", "release_zones"]:
        if var in ds:
            tif_path = prob_dir / f"{stem}_{var}.tif"
            log.info(f"  -> {tif_path.name}")
            ds[var].astype(float).rio.to_raster(tif_path)

    log.info(f"Exported {len(list(prob_dir.glob('*.tif')))} GeoTIFFs to {prob_dir}")
    del ds
    gc.collect()


def main():
    log.info(f"Loading season dataset: {SEASON_NC}")
    ds = load_netcdf_to_dataset(SEASON_NC)
    log.info(f"Dataset: {dict(ds.sizes)}, {len(ds.data_vars)} vars")

    # Check if already preprocessed (dB scale)
    from sarvalanche.utils.validation import check_db_linear
    scale = check_db_linear(ds["VV"])
    if scale == "linear":
        log.info("Preprocessing SAR (TV despeckling)...")
        ds = preprocess_rtc(ds, tv_weight=0.5)
        ds.attrs["preprocessed"] = "rtc_tv"
        log.info("Preprocessing complete")
    else:
        log.info(f"SAR data already in {scale} scale, skipping preprocessing")
        ds.attrs["preprocessed"] = "rtc_tv"

    for date_str, label in DATES.items():
        log.info(f"\n{'='*60}")
        log.info(f"{date_str} ({label} danger), tau={TAU}")
        log.info(f"{'='*60}")

        cache_dir = BASE / "sample_detections" / f"{date_str}_{label}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        run_one(ds, date_str, label, cache_dir)
        log.info(f"Done: {date_str}")


if __name__ == "__main__":
    main()
