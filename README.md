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