# SARvalanche: SAR-based Avalanche Debris Detection

A probabilistic avalanche debris detection system leveraging synthetic aperture radar (SAR) time series analysis, primarily designed for Sentinel-1 with planned expansion to NISAR.

## Table of Contents

- [Overview](#overview)
- [Scientific Approach](#scientific-approach)
  - [Statistical Change Detection](#statistical-change-detection)
  - [Multi-Modal Evidence Integration](#multi-modal-evidence-integration)
  - [Spatial Context with CRFs](#spatial-context-with-crfs)
- [Workflow](#workflow)
- [Data Model](#data-model)
- [Detection Algorithm Details](#detection-algorithm-details)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)

---

## Overview

SARvalanche detects avalanche debris deposits by analyzing statistical changes in SAR backscatter and coherence measurements across time. The system operates on a fundamental principle: avalanche debris creates detectable disturbances in the radar signal that differ from the expected statistical distribution of the terrain's normal state.

**Key Features:**
- Multi-temporal SAR change detection using statistical hypothesis testing
- Multi-orbit and multi-polarization fusion for robust detection
- Integration of ancillary terrain and snow information
- Spatial regularization via dense conditional random fields (CRFs)
- Sensor-agnostic architecture designed for Sentinel-1 and NISAR

**Output Products:**
- Debris probability rasters
- Vectorized debris polygons (shapefiles)
- Per-pixel confidence metrics

---

## Scientific Approach

### Statistical Change Detection

The core detection mechanism relies on comparing new SAR observations against the historical distribution of radar returns from the same location. This is a hypothesis testing framework:

**Null Hypothesis (H₀):** The observed backscatter/coherence comes from the same distribution as historical measurements (no avalanche debris present).

**Alternative Hypothesis (H₁):** The observed backscatter/coherence represents a significant deviation (debris present).

#### Distribution Modeling

For each pixel location, we model the historical backscatter (σ⁰) and coherence (γ) distributions across multiple prior acquisitions. These distributions capture the natural variability of the terrain under normal conditions, accounting for:
- Seasonal variation in snow cover
- Dielectric property changes
- Temporal decorrelation effects
- Incidence angle dependencies

#### Likelihood Calculation

When a new SAR acquisition arrives, we compute the likelihood (L) that the observed measurement comes from the historical distribution:

```
L = P(observation | historical_distribution)
```

For significantly negative deviations in backscatter or coherence (typical of debris deposits that increase surface roughness and reduce coherence), we obtain low likelihoods, indicating potential debris presence.

These likelihoods are converted to p-values representing the probability of observing such extreme values under the null hypothesis. Small p-values (e.g., p < 0.05) suggest the null hypothesis should be rejected, indicating potential debris.

#### Multi-Polarization Advantage

Different radar polarizations sense different scattering mechanisms:
- **VV polarization:** More sensitive to surface scattering and vertical structures
- **VH polarization:** Responds to volume scattering and depolarization effects

Avalanche debris typically affects both polarizations but with different magnitudes. By analyzing both, we gain complementary information about surface condition changes.

### Multi-Modal Evidence Integration

A robust detection system must integrate multiple sources of evidence beyond SAR alone:

#### 1. SAR Evidence (Primary)

**Backscatter Changes (ΔVV, ΔVH):**
- Debris deposits often decrease backscatter due to increased surface roughness and water content
- Multiple orbit geometries (ascending/descending) provide independent looks at the same location
- Each acquisition contributes a p-value based on its likelihood

**Coherence Changes (Δγ):**
- Loss of coherence indicates decorrelation between acquisition pairs
- Debris deposition disrupts phase relationships, causing coherence loss
- Particularly effective for wet, dense avalanche debris

#### 2. Snow Cover Evidence

Changes in optical snow cover indices (e.g., NDSI from Sentinel-2/Landsat) can corroborate debris detection:
- Debris often exposes darker underlying material
- Fresh avalanche debris may have different reflectance than surrounding snow
- Provides independent validation when cloud-free optical imagery is available

#### 3. Terrain Context

**Topographic Constraints:**
- **Slope angle:** Debris deposits are constrained to feasible slopes (typically 25-55°)
- **Forest cover:** Dense forests prevent large avalanche formation and runout
- Masks these areas to reduce false positives

**Avalanche Flow Modeling (FlowPy):**
- Physics-based avalanche runout modeling identifies plausible debris deposition zones
- Starting from defined release zones, FlowPy simulates gravitational mass flow paths
- Provides a spatial prior (P_prior) indicating where debris is physically possible

#### Bayesian Integration Framework

We combine these evidence sources using a Bayesian approach. The posterior probability of debris presence is proportional to the product of individual likelihoods weighted by confidence factors:

```
P(debris | evidence) ∝ P_prior^w₀ × L_VV^w₁ × L_VH^w₂ × L_snow^w₃ × ...
```

Taking logarithms converts this to a weighted sum (computationally stable):

```
log P(debris | evidence) = w₀·log(P_prior) + w₁·log(L_VV) + w₂·log(L_VH) + w₃·log(L_snow) + ...
```

The weights (w₀, w₁, w₂, w₃) reflect the relative reliability of each evidence source and can be learned from training data or set based on domain knowledge.

#### Signed Z-Score Combination

For SAR observations across multiple orbit geometries and dates, we combine p-values using a meta-analysis approach:

1. Convert each p-value to a z-score (standard normal deviate)
2. Weight by orbit geometry and polarization confidence
3. Combine into a single signed z-score indicating change direction and significance
4. This z-score represents the aggregate statistical evidence for debris

This approach is more robust than simple p-value multiplication because it properly accounts for the expected distribution under the null hypothesis.

### Spatial Context with CRFs

Individual pixel-based detection is prone to noise and false positives. Avalanche debris deposits are spatially coherent features—they form connected regions, not isolated pixels. We incorporate this spatial constraint using dense Conditional Random Fields (CRFs).

#### Why CRFs?

A CRF models the joint probability of all pixel labels simultaneously, encouraging neighboring pixels with similar properties to have similar labels. This serves to:
- **Smooth spurious noise:** Isolated high-probability pixels in unlikely locations are suppressed
- **Enhance contiguous regions:** Connected areas of high probability are reinforced
- **Respect boundaries:** Edge-preserving kernels ensure transitions at real feature boundaries

#### CRF Formulation

Following Krähenbühl & Koltun (NIPS 2011, ICML 2013), we use a fully connected CRF with Gaussian edge potentials. The energy function consists of:

**Unary Potential (ψᵤ):** 
The per-pixel debris probability from the multi-modal integration (previous section). This represents the likelihood of debris at each pixel independently.

**Pairwise Potential (ψₚ):**
Encourages spatial consistency by penalizing label disagreements between pixels with similar appearance and proximity:

```
ψₚ(xᵢ, xⱼ) = μ(xᵢ, xⱼ) · k(fᵢ, fⱼ)
```

Where:
- μ(xᵢ, xⱼ) is a label compatibility function (penalizes different labels)
- k(fᵢ, fⱼ) is a Gaussian kernel over feature vectors (spatial position, intensity, etc.)

**Gaussian Kernels:**

The pairwise potential uses a mixture of Gaussian kernels:

1. **Appearance kernel:** Pixels with similar intensity/color should have similar labels
   ```
   k_appearance = exp(-|pᵢ - pⱼ|²/2σ_spatial² - |Iᵢ - Iⱼ|²/2σ_intensity²)
   ```

2. **Smoothness kernel:** Nearby pixels should have similar labels, regardless of appearance
   ```
   k_smoothness = exp(-|pᵢ - pⱼ|²/2σ_spatial²)
   ```

These kernels enable efficient inference via permutohedral lattice filtering, making fully connected CRFs tractable even for large rasters.

#### Inference

We perform mean-field inference to find the most probable labeling. The algorithm iteratively updates each pixel's label probability based on its unary potential and messages from all other pixels, converging to a solution that balances per-pixel evidence with spatial consistency.

**Result:** A smoothed debris probability map where isolated false positives are suppressed and spatially coherent debris deposits are emphasized.

---

## Workflow

SARvalanche follows a sequential pipeline from raw sensor data to final products:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Raw Sensor Data                                │
│  (Sentinel-1 SLC/GRD, NISAR RSLC, Optical imagery, DEM)                │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Preprocessing                                    │
│  • Coregistration and terrain correction                                │
│  • Radiometric calibration (σ⁰)                                         │
│  • Coherence estimation                                                 │
│  • Projection to common grid                                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Feature Extraction                                  │
│  • Historical distribution modeling (μ, σ)                              │
│  • Change detection (Δσ⁰, Δγ)                                           │
│  • Snow cover indices (NDSI)                                            │
│  • Terrain attributes (slope, aspect, forest)                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Mask Generation                                  │
│  • Forest cover exclusion                                               │
│  • Slope angle feasibility                                              │
│  • FlowPy runout modeling → avalanche zone prior                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Statistical Detection                                 │
│  • Likelihood computation (per orbit, polarization)                     │
│  • Multi-temporal p-value combination → signed z-scores                 │
│  • Bayesian integration of SAR + snow + terrain evidence                │
│  • Unary potential map (per-pixel debris probability)                   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Spatial Regularization (CRF)                           │
│  • Dense CRF with Gaussian edge potentials                              │
│  • Mean-field inference                                                 │
│  • Output: Smoothed debris probability raster                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Product Generation                                │
│  • Thresholding probability map                                         │
│  • Connected component analysis                                         │
│  • Vectorization to polygons (shapefiles)                               │
│  • Metadata assignment (confidence, area, date)                         │
└─────────────────────────────────────────────────────────────────────────┘
```

All inter-module data transfers use standardized xarray objects conforming to the canonical data model (see next section).

---

## Data Model

SARvalanche enforces a strict internal data model to ensure sensor-agnostic, reproducible avalanche detection across Sentinel-1 and NISAR. This consistency enables modular development and reliable algorithm performance regardless of input sensor.

### Core Principle

All internal processing operates on **xarray.DataArray** objects with consistent:
- Dimension ordering
- Coordinate naming
- Metadata attributes
- Semantic conventions

### Dimension Standards

Raster data must use the following dimension order:

```
(time?, y, x)
```

**Rules:**
- `y`, `x` are **always** the last two dimensions (row, column in projected coordinates)
- `time` is **optional**, but if present must be the **first** dimension and **sorted** chronologically
- All arrays must be in a **projected CRS** (no geographic lat/lon unless explicitly reprojected)

**Examples:**

| Data Type | Dimensions | Example |
|-----------|------------|---------|
| Single acquisition | `(y, x)` | One backscatter image |
| Time series | `(time, y, x)` | Stack of coherence maps |
| Multi-band | `(band, y, x)` | RGB composite (if treating bands as time-like) |

### Required Attributes

Every DataArray **must** define these attributes:

| Attribute | Type | Description | Example Values |
|-----------|------|-------------|----------------|
| `crs` | str/int | Projected coordinate reference system | `"EPSG:32610"` (UTM 10N), WKT string |
| `sensor` | str | Sensor identifier | `"Sentinel-1"`, `"NISAR"` |
| `product` | str | Product type | `"SLC"`, `"GRD"`, `"RSLC"` |
| `units` | str | Measurement units | `"dB"`, `"1"` (unitless), `"m"` |

**Additional recommended attributes:**
- `orbit`: Orbit geometry (`"ascending"`, `"descending"`)
- `polarization`: Radar polarization (`"VV"`, `"VH"`, `"HH"`, `"HV"`)
- `incidence_angle`: Local incidence angle (degrees)
- `acquisition_time`: ISO 8601 timestamp
- `processing_level`: Processing stage identifier

### Semantic Conventions

To enable algorithm interoperability, we follow these naming conventions:

| Data Type | Name | Units | Dimensions | Notes |
|-----------|------|-------|------------|-------|
| Backscatter | `"sigma0"` | `"dB"` | `(time?, y, x)` | Calibrated radar cross-section |
| Coherence | `"coherence"` | `"1"` | `(time?, y, x)` | Interferometric coherence, range [0,1] |
| Intensity | `"intensity"` | `"1"` | `(time?, y, x)` | Linear power, not in dB |
| Phase | `"phase"` | `"radians"` | `(time?, y, x)` | Interferometric phase |
| Masks | boolean | — | `(y, x)` | True = valid/included pixel |
| Detection | `"debris_probability"` | `"1"` | `(y, x)` | Range [0,1], post-CRF |

**Naming rules:**
- Use lowercase with underscores (`"sigma0"`, not `"Sigma0"` or `"SIGMA0"`)
- Be explicit: `"sigma0_vv"` vs `"sigma0_vh"` for polarization-specific layers
- Include geometry in name if ambiguous: `"sigma0_vv_asc"` for ascending VV

### Coordinate Standards

**Spatial coordinates:**
- Use `x` and `y` as coordinate names (not `lon`/`lat`, `col`/`row`)
- Units in meters for projected CRS
- Origin and resolution must be consistent across all layers in a processing chain

**Temporal coordinates:**
- Coordinate name: `time`
- Type: `numpy.datetime64` or `pandas.Timestamp`
- Sorted in ascending order
- Include timezone information when available (UTC preferred)

### Sensor-Specific Isolation

Sensor-specific handling (file formats, metadata extraction, calibration quirks) is **isolated to the `io/` module**. All downstream algorithms assume canonical inputs.

**Design principle:**
```python
# ❌ Bad: Algorithm knows about sensor specifics
def detect_debris(sentinel1_file):
    data = read_sentinel1(sentinel1_file)  # Sensor-specific
    ...

# ✅ Good: Algorithm works with canonical data
def detect_debris(sigma0: xr.DataArray):
    assert sigma0.dims == ('time', 'y', 'x')
    assert 'crs' in sigma0.attrs
    ...
```

By the time data reaches processing modules, it has been normalized to this canonical form.

---

## Detection Algorithm Details

### Current Detection Probability Logic

The detection system uses a Bayesian model to predict the probability of debris at each pixel (y, x) at time t:

```
P(debris | SAR change, snow change, terrain context, ...)
```

#### Step 1: Compute Individual Likelihoods

For each evidence source, calculate the likelihood that the observed data comes from a debris-present state:

```python
P_prior   = debris_zone_probability  # From FlowPy runout model
L_vv      = vv_likelihood(ΔVV)       # From VV backscatter change
L_vh      = vh_likelihood(ΔVH)       # From VH backscatter change
L_snow    = snow_likelihood(Δsnow)   # From optical snow cover change
```

Each likelihood function evaluates how consistent the observation is with the presence of debris:
- **Low backscatter change → high likelihood** (debris typically darkens surface)
- **Low coherence → high likelihood** (debris disrupts phase)
- **Snow cover decrease → moderate likelihood** (debris may expose darker material)

#### Step 2: Weighted Log-Likelihood Combination

Combine likelihoods in log space (numerically stable):

```python
log P_debris = (
    w_prior * log(P_prior)     # Weight for terrain/runout prior
  + w_vv * log(L_vv)           # Weight for VV polarization
  + w_vh * log(L_vh)           # Weight for VH polarization
  + w_snow * log(L_snow)       # Weight for snow evidence
)
```

This is equivalent to the multiplicative Bayesian form but avoids numerical underflow.

#### Step 3: Normalization

Convert log-probability to a proper probability in [0, 1]:

```python
P_debris = sigmoid(log P_debris)
```

The sigmoid function ensures the output is bounded and interpretable as a probability.

#### Alternative Formulation (Multiplicative)

Equivalently, in non-log space:

```python
P_debris_unnormalized = (
    P_prior**w_prior *
    L_vv**w_vv *
    L_vh**w_vh *
    L_snow**w_snow
)

P_debris = normalize(P_debris_unnormalized)  # Scale to [0,1]
```

### Step 4: CRF Spatial Regularization

The per-pixel probabilities from Step 3 form the **unary potential** input to the CRF. The CRF then:

1. Constructs pairwise potentials based on spatial proximity and appearance similarity
2. Runs mean-field inference to find a globally consistent labeling
3. Outputs smoothed debris probabilities that respect spatial coherence

**Penalties:**
- **Single isolated pixels**: High-probability isolated pixels are downweighted (likely noise)
- **Contiguous groups**: Spatially coherent regions of high probability are reinforced (likely real debris)

### Weight Selection

The weights (w_prior, w_vv, w_vh, w_snow) control the relative influence of each evidence source. They can be:

- **Manually set** based on domain expertise (e.g., VV typically more reliable than VH for wet snow)
- **Learned from data** via logistic regression or other supervised methods on labeled avalanche debris
- **Adaptively adjusted** based on data quality metrics (e.g., downweight coherence in low-coherence regions)

Typical values might be:
```python
w_prior = 2.0   # Strong terrain prior
w_vv = 1.5      # Primary SAR evidence
w_vh = 1.0      # Secondary SAR evidence
w_snow = 0.5    # Supplementary optical evidence (often cloud-obscured)
```

---

## Dependencies

SARvalanche relies on several external packages and tools:

### Python Packages

- **xarray**: Canonical data model and N-dimensional arrays
- **numpy**: Numerical operations
- **scipy**: Statistical functions, spatial filters
- **rasterio**: Raster I/O and geospatial operations
- **geopandas**: Vector I/O and spatial operations (shapefiles)
- **pydensecrf**: Dense CRF implementation (Krähenbühl & Koltun)
  - *Note: May require manual build; see installation section*
- **scikit-image**: Image processing and connected component analysis

### Repackaged Modules

**FlowPy** (included, modified):
- Original: GUI-based gravitational mass flow model (Neuhauser et al., 2021)
- SARvalanche modifications:
  - Removed GUI and PyQt5 dependency
  - Removed file I/O (replaced with xarray interface)
  - Added programmatic API for embedding in detection pipeline
- Citation: 
  > Neuhauser, M., D'Amboise, C., Teich, M., Kofler, A., Huber, A., Fromm, R., and Fischer, J. T.: Flow-Py: routing and stopping of gravitational mass flows, Zenodo [code], https://doi.org/10.5281/zenodo.5027274, 2021

### External Tools

- **SNAP/GPT** (optional): For advanced Sentinel-1 preprocessing
- **GDAL/OGR**: Coordinate transformations and format conversions (via rasterio/geopandas)

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/sarvalanche.git
cd sarvalanche

# Create conda environment (recommended)
conda env create -f environment.yml
conda activate sarvalanche

# Install package
pip install -e .

# Note: pydensecrf may require compilation
# If installation fails, try:
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

### Verify Installation

```python
import sarvalanche
print(sarvalanche.__version__)
```

---

## Usage

### Basic Detection Workflow

```python
import sarvalanche as sav
import xarray as xr

# 1. Load and preprocess SAR data
# (assuming preprocessed data already in canonical format)
sigma0_vv = xr.open_dataarray("sigma0_vv_timeseries.nc")
sigma0_vh = xr.open_dataarray("sigma0_vh_timeseries.nc")
coherence = xr.open_dataarray("coherence_timeseries.nc")

# 2. Load ancillary data
dem = xr.open_dataarray("dem.nc")
forest_mask = xr.open_dataarray("forest_mask.nc")  # True = forest
slope = xr.open_dataarray("slope.nc")

# 3. Generate avalanche zone prior with FlowPy
from sarvalanche.models import flowpy_wrapper
release_zones = xr.open_dataarray("release_zones.nc")  # Boolean mask
debris_prior = flowpy_wrapper.run_runout_model(
    dem=dem,
    release_zones=release_zones,
    terrain_roughness=0.1
)

# 4. Compute change detection likelihoods
from sarvalanche.detection import change_detection
likelihoods = change_detection.compute_likelihoods(
    sigma0_vv=sigma0_vv,
    sigma0_vh=sigma0_vh,
    coherence=coherence,
    reference_period=slice("2023-01-01", "2023-03-01"),  # Pre-event baseline
    detection_date="2023-03-15"  # Event date
)

# 5. Combine evidence
from sarvalanche.detection import bayesian_fusion
unary_potential = bayesian_fusion.combine_evidence(
    prior=debris_prior,
    likelihood_vv=likelihoods['vv'],
    likelihood_vh=likelihoods['vh'],
    weights={'prior': 2.0, 'vv': 1.5, 'vh': 1.0}
)

# 6. Apply CRF spatial regularization
from sarvalanche.detection import crf
debris_probability = crf.apply_dense_crf(
    unary=unary_potential,
    reference_image=sigma0_vv.isel(time=-1),  # Use latest acquisition for appearance
    spatial_kernel_size=10,
    appearance_kernel_size=5,
    n_iterations=10
)

# 7. Threshold and vectorize
from sarvalanche.products import vectorize
debris_polygons = vectorize.probability_to_polygons(
    debris_probability,
    threshold=0.5,
    min_area_m2=500
)

# 8. Save outputs
debris_probability.to_netcdf("debris_probability.nc")
debris_polygons.to_file("debris_polygons.shp")
```

### Configuration

Advanced users can customize detection parameters via configuration files:

```yaml
# config.yml
detection:
  weights:
    prior: 2.0
    vv: 1.5
    vh: 1.0
    snow: 0.5
  
  crf:
    spatial_stddev: 10  # meters
    appearance_stddev: 0.5  # normalized units
    smoothness_weight: 1.0
    appearance_weight: 3.0
    n_iterations: 10
  
  thresholds:
    probability: 0.5
    min_area_m2: 500
    max_slope_degrees: 60
```

Load with:
```python
from sarvalanche.config import load_config
config = load_config("config.yml")
```

---

## Module Structure

```
sarvalanche/
├── io/                      # Sensor-specific readers, ensure canonical output
│   ├── sentinel1.py         # Sentinel-1 SLC/GRD readers
│   ├── nisar.py             # NISAR RSLC readers (future)
│   └── optical.py           # Sentinel-2/Landsat readers
├── preprocessing/           # Calibration, coregistration, projection
│   ├── calibration.py
│   ├── coregistration.py
│   └── terrain_correction.py
├── features/                # Derived measurements
│   ├── coherence.py
│   ├── change_metrics.py
│   └── terrain.py
├── masks/                   # Spatial constraint generation
│   ├── forest.py
│   ├── slope.py
│   └── flowpy_wrapper.py    # Avalanche runout modeling
├── detection/               # Core detection algorithms
│   ├── change_detection.py  # Statistical hypothesis testing
│   ├── bayesian_fusion.py   # Multi-modal evidence combination
│   └── crf.py               # Dense CRF spatial regularization
├── products/                # Output generation
│   ├── vectorize.py         # Raster to polygon conversion
│   └── metadata.py          # Attach confidence, date, area
├── utils/                   # Shared utilities
│   ├── spatial.py
│   └── validation.py
├── tests/                   # Unit and integration tests
└── config/                  # Configuration schemas
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Data Model Compliance**: Ensure all new functions accept/return xarray objects conforming to the canonical data model
2. **Type Hints**: Use type hints for function signatures
3. **Documentation**: Include docstrings with parameter descriptions and examples
4. **Tests**: Add unit tests for new functionality
5. **Sensor Agnostic**: Isolate sensor-specific code to `io/` module

Submit pull requests with clear descriptions of changes and scientific justification if modifying detection algorithms.

---

## References

### Conditional Random Fields

- Krähenbühl, P., & Koltun, V. (2011). **Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials.** *Advances in Neural Information Processing Systems (NIPS)*, 24.
- Krähenbühl, P., & Koltun, V. (2013). **Parameter Learning and Convergent Inference for Dense Random Fields.** *International Conference on Machine Learning (ICML)*.

### Avalanche Runout Modeling

- Neuhauser, M., D'Amboise, C., Teich, M., Kofler, A., Huber, A., Fromm, R., & Fischer, J. T. (2021). **Flow-Py: routing and stopping of gravitational mass flows.** Zenodo. https://doi.org/10.5281/zenodo.5027274

### SAR Remote Sensing

- (Add relevant citations for SAR avalanche detection methods, Sentinel-1 processing, etc.)

---

## License

[Specify license here, e.g., MIT, Apache 2.0, GPL]

---

## Contact

For questions or collaboration inquiries, please contact:
- **Author:** [Your Name]
- **Email:** [Your Email]
- **Institution:** [Your Institution]

---

## Acknowledgments

This work was supported by [funding sources]. SAR data provided by ESA Copernicus program (Sentinel-1). DEM data from [source]. We thank [contributors/collaborators].