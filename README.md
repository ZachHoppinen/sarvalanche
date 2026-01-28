## Current detection probability logic

We are using a bayesian model based on our inputs to predic the probability of debris at a pixel (y, x) and time (t)

`P(debris | SAR change, snow change, terrain context, ...)`

```
logP = (
    w0 * log(P0)
  + w_vv * log(L_vv)
  + w_vh * log(L_vh)
  + w_snow * log(L_snow)
)

P = sigmoid(logP)
```

```
P_prior   = debris_zone
P_vv      = vv_likelihood(ΔVV)
P_vh      = vh_likelihood(ΔVH)
P_snow    = snow_likelihood(Δsnow)
```

combined with:

```
P_debris = normalize(
    P_prior**w0 *
    P_vv**w1 *
    P_vh**w2 *
    P_snow**w3
)
```

## Module workflow

Everything flows:

```Raw sensor → preprocessing → features → masks → detection → products```

and all cross-module objects are tested and defined.

## Canonical Data Model

sarvalanche enforces a strict internal data model to ensure sensor-agnostic, reproducible avalanche detection across Sentinel-1 and NISAR.

All internal processing operates on xarray.DataArray objects with consistent dimensions, coordinates, and metadata.

Dimensions

Raster data must use the following dimension order:

```
(time?, y, x)
```

y, x are always the last two dimensions

time is optional, but if present must be the first dimension and sorted

All arrays must be in a projected CRS

Examples:

Single acquisition: `(y, x)`

Time series: `(time, y, x)`

Required Attributes

Every DataArray must define:

`crs` — Projected coordinate reference system (EPSG code or WKT)

`sensor` — "Sentinel-1" or "NISAR"

`product` — e.g. "SLC", "RSLC", "GRD"

`units` — e.g. "dB", "1"

Additional attributes (orbit, polarization, incidence angle) are encouraged.

Semantic Conventions

Backscatter: `name="sigma0"`, `units="dB"`

Coherence: `name="coherence"`, `units="1"`

Masks: boolean arrays with shape `(y, x)`

Detection outputs: boolean debris masks `(y, x)`

Sensor-specific handling is isolated to the `io/` module. All downstream algorithms assume canonical inputs.

## Other packages used ##

flowpy is repackaged to remove GUI and add a non-command line interface, remove file IO, remove pyqt5 dependency.
`Neuhauser, M., D'Amboise, C., Teich, M., Kofler, A., Huber, A., Fromm, R., and Fischer, J. T.: Flow-Py: routing and stopping of gravitational mass flows, Zenodo [code], https://doi.org/10.5281/zenodo.5027274, 2021`