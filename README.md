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
## Physical Basis: SAR Signatures of Avalanche Debris

Understanding how avalanche debris affects SAR measurements is fundamental to the detection approach. The physical changes in surface properties create distinctive signatures in both backscatter intensity and interferometric coherence.

### Backscatter Changes

Avalanche debris typically **increases** radar backscatter compared to undisturbed snow surfaces. This brightening occurs through two primary mechanisms:

#### 1. Surface Roughness Enhancement

Fresh avalanche debris creates a much rougher surface texture than the smooth snow it disturbs. The chaotic deposition of snow blocks, ice fragments, and entrained material (rocks, vegetation, soil) increases surface scattering at C-band wavelengths (5.6 cm for Sentinel-1). This roughness effect has been consistently documented across multiple studies:

- **Schlaffer & Schlogl (2024)** demonstrated backscatter increases in dual-polarimetric time series analysis, showing that the roughness-induced brightening is detectable in both VV and VH polarizations [1].

- **Sartori & Dabiri (2023)** confirmed this signature in the Southern Tyrolean Alps, where debris deposits showed clear backscatter increases relative to pre-event conditions [2].

- **Eckerstorfer et al. (2019)** leveraged this effect in their near-real-time avalanche monitoring system in Norway, using backscatter brightening as the primary detection criterion [3].

#### 2. Volume Scattering (Dense Debris)

For larger, denser avalanche deposits, increased **volume scattering** may contribute additional backscatter enhancement. The heterogeneous internal structure of compacted debris—with varying densities, air pockets, and ice lenses—can create multiple scattering paths within the deposit volume. While less studied than surface roughness effects, this mechanism likely contributes to the particularly strong signatures observed for large, dense avalanches.

At lower wavelengths this will be less significant but at X and higher frequencies this will be a significant cause of incrased backscatter.

### Detection Performance and Size Limitations

The detectability of avalanche debris depends critically on:
- **Temporal revisit frequency**: Number of days between acquisitions
- **Avalanche size**: Spatial extent and debris volume
- **Regional snow climate**: Wet vs. dry snow conditions affect persistence

**Norway (6-day revisit, Sentinel-1)**: Eckerstorfer et al. (2019) achieved near-real-time detection with high success rates, benefiting from frequent revisits and maritime snow conditions with large dense debris signatures [3].

**Western United States (12-day revisit)**: Detection is more challenging with less frequent coverage. Keskinen et al. (2022) found that in transitional snow climates:
- **D2 avalanches** (size 2 on the 5-point scale): ~50% detection rate
- **D3+ avalanches** (size 3 and larger): Majority detected
- **D1 avalanches** (size 1, small sluffs): Generally undetectable

The 12-day repeat cycle in regions with single-satellite coverage means smaller avalanches may be obscured by subsequent snowfall, surface metamorphism, or wind redistribution before the next acquisition [4]. This temporal limitation emphasizes the importance of multi-orbit and multi-sensor integration to improve revisit frequency.

### Coherence Loss

Interferometric coherence measures the similarity of phase between two SAR acquisitions. Avalanche debris causes **coherence loss** (decorrelation) through the introduction of new scatterers and geometric changes between image pairs.

#### Physical Mechanism

Between two SAR acquisitions (e.g., 6 or 12 days apart), avalanche deposition fundamentally alters the scattering geometry:
- New scatterers appear (debris blocks, exposed rocks, vegetation)
- Previous scatterers are buried or displaced
- Surface height changes disrupt phase relationships
- Dielectric properties change (wet debris vs. dry snow)

This disruption causes the phase to become incoherent, reducing correlation between acquisitions from typical snow values (γ ~ 0.4–0.7) to debris values (γ ~ 0.1–0.3).

#### Empirical Evidence

**Yang et al. (2020)** provided detailed quantitative analysis of coherence changes in C-band SAR data [5]:

**VV Polarization:**
- **Pre-event coherence**: γ = 0.41–0.65 (ascending and descending)
- **Post-event coherence**: γ = 0.14–0.32
- **Median drop**: ~0.30–0.40 (absolute decrease)

**VH Polarization:**
- **Pre-event coherence**: γ = 0.40–0.68
- **Post-event coherence**: γ = 0.20–0.49
- **Smaller decrease than VV**, but still substantial

The coherence loss is particularly pronounced in VV polarization, likely because co-polarized returns are more sensitive to geometric phase disruption than cross-polarized volume scattering.

### Polarimetric Decomposition Signatures

Beyond simple backscatter and coherence, polarimetric decomposition parameters reveal additional information about scattering mechanism changes.

#### Entropy (H) and Alpha Angle (α)

**Yang et al. (2020)** analyzed Cloude-Pottier decomposition parameters, showing distinct patterns [5]:

**Entropy (H) - Randomness of Scattering:**
- **Ascending orbit**: H increased from 0.42–0.69 (undisturbed snow) to 0.84–0.91 (debris)
- **Descending orbit**: H increased from 0.25–0.41 to 0.88–0.94 (even more dramatic)
- **Interpretation**: Debris creates highly random, depolarized scattering due to heterogeneous surface structure

**Alpha Angle (α) - Dominant Scattering Type:**
- **Ascending orbit**: α decreased from 70°–74° to 57.5°–63°
- **Descending orbit**: α decreased from 81°–86° to 54°–62° (larger change)
- **Interpretation**: Shift from volume/multiple-bounce scattering (high α) toward surface scattering (lower α), consistent with roughness dominating the debris signature

The particularly strong changes in the descending orbit suggest geometric effects (local incidence angle) modulate the sensitivity to debris detection, reinforcing the value of multi-orbit observations.

### Implications for Detection Algorithm

These physical signatures directly inform the SARvalanche detection strategy:

1. **Backscatter brightening** is the primary detection criterion, validated across multiple studies and regions
2. **Coherence loss** provides complementary evidence, particularly effective for wet, dense debris
3. **Multi-polarization analysis** (VV + VH) captures both surface and volume scattering changes
4. **Multi-orbit fusion** (ascending + descending) compensates for geometric sensitivities
5. **Size-dependent detection** is expected, with focus on D2+ avalanches in 12-day revisit regions

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

For significantly positive deviations in backscatter or drops in coherence (typical of debris deposits that increase surface roughness and reduce coherence), we obtain low likelihoods for the null hypothesis, indicating potential debris presence.

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
- Debris deposits often increase backscatter due to increased surface roughness and water content
- Multiple orbit geometries (ascending/descending) provide independent looks at the same location
- Each acquisition contributes a p-value based on its likelihood
- Different polarizations are weighted seperately

> Note: Coherence change not yet implemented

**Coherence Changes (Δγ):**
- Loss of coherence indicates decorrelation between acquisition pairs
- Debris deposition adds new scatterers between scenes, causing coherence loss
- Particularly effective for wet, dense avalanche debris

#### 2. SWE Reanalysis Context
 - The UCLA SWE reanalysis is used in historical avalanche detection runs
 - It provides two features: (1) the SWE change over the last 7 days
 - and (2) the presence of absence of snow in start zones.

#### 3. Snow Cover Evidence

> Note: Since we are testing historical datasets currently this is implemented using the UCLA reanalysis product
> but will be implemented for near-real time runs.

Changes in optical snow cover indices (e.g., NDSI from Sentinel-2/Landsat) can corroborate debris detection:
- Debris often exposes darker underlying material
- Fresh avalanche debris may have different reflectance than surrounding snow
- Provides independent validation when cloud-free optical imagery is available

#### 4. Terrain Context

**Topographic Constraints:**
- **Slope angle:** Debris deposits are constrained to feasible slopes (typically <25°)
- **Forest cover:** Dense forests prevent large avalanche formation and runout
- While these areas are not "masked" a lower prior probability is used which
- means larger increases in backscatter are neccessary to cause a "detection"

**Avalanche Flow Modeling (FlowPy):**
- Physics-based avalanche runout modeling identifies plausible debris deposition zones
- Starting from defined release zones, FlowPy simulates gravitational mass flow paths
- Provides a spatial prior (P_prior) indicating where debris is physically possible
- The primary parameter used is "cell_count" incidicating the number of other pixels that flowed to that pixel.
- Again this gives a probability layer that makes detections less likely but not impossible in lower flow pixels.

#### Bayesian Integration Framework

We combine multiple evidence sources using a **weighted geometric mean** approach. Rather than a traditional Bayesian posterior, we compute an aggregate probability from independent probabilistic assessments:

**Evidence Sources:**

1. **p_empirical**: Probability based on weighted backscatter changes across observations
2. **p_ecdf**: Probability derived from empirical cumulative distribution function (ECDF) of historical backscatter
   - Measures how extreme the current observation is relative to the pre-event distribution
3. **p_fcf**: Forest cover probability (forests inhibit avalanche formation/runout)
   - Sigmoid function: high forest cover → low debris probability
   - Parameters: midpoint = 50% cover, slope = 0.1
4. **p_runout**: Avalanche runout model probability (from FlowPy cell counts)
   - Higher cell counts (more simulated flow paths) → higher debris probability
5. **p_slope**: Slope angle probability
   - Debris deposits are constrained to physically feasible slope ranges
   - Too flat (no runout) or too steep (no deposition) both reduce probability

**Combination Formula:**

```
P_debris = (p_empirical^w₁ × p_ecdf^w₂ × p_fcf^w₃ × p_runout^w₄ × p_slope^w₅)^(1/Σwᵢ)
```

**Current weights:**
- w₁ = 1.0 (empirical backscatter changes)
- w₂ = 1.0 (ECDF-based backscatter probability)
- w₃ = 1.0 (forest cover constraint)
- w₄ = 1.0 (runout model prior)
- w₅ = 1.0 (slope angle constraint)

The weighted geometric mean has several advantages:
- **Scale invariance**: Works consistently across probability ranges [0,1]
- **Multiplicative influence**: Any very low probability (e.g., forest = 0.01) strongly suppresses the result
- **Balanced contribution**: Weights control relative importance without dimensional issues
- **Interpretability**: Output remains a valid probability in [0,1]

Null values are replaced with 0 (zero probability) in the final pixel-wise probability map (`p_pixelwise`).

**Key Distinction from Traditional Bayes:**

This is not computing `P(debris | observations)` via Bayes' theorem, but rather combining multiple probabilistic assessments where each evidence source independently estimates debris likelihood. The geometric mean provides a principled way to aggregate these assessments while respecting their scale and allowing differential weighting.

#### Signed Z-Score Combination

For SAR observations across multiple orbit geometries and dates, we combine p-values using a weighted meta-analysis approach that preserves directional information:

**Step 1: Convert p-values to signed z-scores**

For each observation channel (orbit geometry + polarization), we:

1. Clip p-values to avoid numerical instability: `p ∈ [ε, 1-ε]` where ε is a small value (e.g., 10⁻¹⁰)
2. Convert to z-scores using the inverse normal CDF: `z = Φ⁻¹(1 - p)`
3. Apply sign from the observed change direction: `z_signed = z × sign(Δσ⁰)`

The sign indicates whether backscatter increased (+) or decreased (-), with positive values typically indicating potential debris.

**Step 2: Calculate observation weights**

Each observation is weighted based on three quality factors:
```
w = (1/σ) × f(θ) × q_pol
```

Where:
- **1/σ**: Inverse of historical backscatter standard deviation (stable areas with low natural variability get higher weight)
- **f(θ)**: Geometric weighting based on local incidence angle θ
  - Avalanches appear brightest at local incidence angles of 55° ± 20° (Bühler et al., 2021)
  - Implemented as Gaussian: `f(θ) = exp(-(θ - 55°)² / (2 × 25²))`
  - Observations outside the optimal 35-75° range are strongly downweighted
  - This accounts for varying backscatter sensitivity across the swath
- **q_pol**: Polarization quality factor
  - VV polarization: q = 1.0 (primary, more sensitive to surface roughness changes)
  - VH polarization: q = 0.8 (secondary, more sensitive to volume scattering, closer to noise floor)

This weighting scheme emphasizes observations with:
1. Low temporal noise (stable baseline)
2. Optimal viewing geometry (35-75° local incidence angles)
3. Reliable polarization characteristics

**Reference:**
Bühler, Y., Hafner, E. D., Zweifel, B., Zesiger, M., & Heisig, H. (2021). Where are the avalanches? Rapid SPOT6 satellite data acquisition to map an extreme avalanche period over the Swiss Alps. *The Cryosphere*, 15(1), 83-98. https://doi.org/10.5194/tc-15-83-2021

**Step 3: Weighted combination**

The signed z-scores are combined across all observation channels using their weights, producing an aggregate statistical measure of change significance that accounts for both observation quality and geometric viewing conditions.

This approach is more robust than simple p-value multiplication because it:
- Properly accounts for the expected distribution under the null hypothesis
- Preserves directional information (increase vs. decrease in backscatter)
- Down-weights unreliable observations (high variance, poor geometry, less sensitive polarization)

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
│                           Raw Sensor Data                               │
│  (Sentinel-1 SLC/GRD, NISAR RSLC, Optical imagery, DEM)                 │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Preprocessing                                   │
│  • Coregistration and terrain correction                                │
│  • Coherence estimation                                                 │
│  • Projection to common grid                                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Feature Extraction                                 │
│  • Historical distribution modeling (μ, σ)                              │
│  • Change detection (Δσ⁰, Δγ)                                           │
│  • Snow cover indices (NDSI)                                            │
│  • Snow water equiavelent change                                        │
│  • Terrain attributes (slope, aspect, forest)                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Mask Generation                                 │
│  • Forest cover probability                                             │
│  • Slope angle feasibility                                              │
│  • FlowPy runout modeling → avalanche zone prior                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Statistical Detection                                │
│  • Likelihood computation (per orbit, polarization)                     │
│  • Multi-temporal p-value combination → signed z-scores                 │
│  • Bayesian integration of SAR + snow + terrain evidence                │
│  • Unary potential map (per-pixel debris probability)                   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Spatial Regularization (CRF)                          │
│  • Dense CRF with Gaussian edge potentials                              │
│  • Mean-field inference                                                 │
│  • Output: Smoothed debris probability raster                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Product Generation                               │
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
  - updated release zone split for multiprocessing to better balance process loads
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

# Note: pydensecrf used in the dense CRF
# uses python 3.8 at most.
# You will need to run
conda env create -f py38_environment.yml
# this will create the conda environment used internally
# to run the dense crf section.
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
from pathlib import Path
from shapely.geometry import box
import xarray as xr
# aoi = box(-115.0, 43.58, -114.5, 44.13)
# aoi = box(-115.02068191820455, 44.14529196659243,  -114.99119907695828, 44.16379687146455)
# aoi = box(*total_bounds)
aoi = box(-110.772, 43.734, -110.745, 43.756)
crs= 'EPSG:4326'

# convert 30 meters to degrees
from sarvalanche.utils.projections import resolution_to_degrees
from sarvalanche.utils.validation import validate_crs
resolution = resolution_to_degrees(20, validate_crs(crs))

start_date = "2019-10-01"
stop_date = "2020-03-01"
avalanche_date = '2020-01-11'
from sarvalanche.detection.debris import detect_avalanche_debris
cache_dir = Path('/Users/zmhoppinen/Documents/sarvalanche/local/data')
ds = detect_avalanche_debris(aoi, crs, 20, start_date, stop_date, avalanche_date, cache_dir=cache_dir)
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

### SAR-Based Avalanche Detection

1. Schlaffer, S. & Schlogl, M. (2024). **Snow Avalanche Debris Analysis Using Time Series of Dual-Polarimetric Synthetic Aperture Radar Data.** *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, PP, 1–13. https://doi.org/10.1109/JSTARS.2024.3423403

2. Sartori, M. & Dabiri, Z. (2023). **Assessing the Applicability of Sentinel-1 SAR Data for Semi-automatic Detection of Snow-avalanche Debris in the Southern Tyrolean Alps.** *GI_Forum*, 1, 59–68. https://doi.org/10.1553/giscience2023_01_s59

3. Eckerstorfer, M., Vickers, H., Malnes, E. & Grahn, J. (2019). **Near-Real Time Automatic Snow Avalanche Activity Monitoring System Using Sentinel-1 SAR Data in Norway.** *Remote Sensing*, 11, 2863. https://doi.org/10.3390/rs11232863

4. Keskinen, Z., Hendrikx, J., Eckerstorfer, M. & Birkeland, K. (2022). **Satellite detection of snow avalanches using Sentinel-1 in a transitional snow climate.** *Cold Regions Science and Technology*, 199, 103558. https://doi.org/10.1016/j.coldregions.2022.103558

5. Yang, J. et al. (2020). **Automatic Detection of Regional Snow Avalanches with Scattering and Interference of C-band SAR Data.** *Remote Sensing*, 12, 2781. https://doi.org/10.3390/rs12172781

6. Bühler, Y., Hafner, E. D., Zweifel, B., Zesiger, M., & Heisig, H. (2021). Where are the avalanches? Rapid SPOT6 satellite data acquisition to map an extreme avalanche period over the Swiss Alps. *The Cryosphere*, 15(1), 83-98. https://doi.org/10.5194/tc-15-83-2021

---

## License

> Todo: add license

---

## Contact

For questions or collaboration inquiries, please contact:
- **Author:** Zachary Hoppinen
- **Email:** zmhoppinen@alaska.edu
- **Institution:** University of Alaska Fairbanks

---

## Acknowledgments

<!-- This work was supported by [funding sources]. SAR data provided by ESA Copernicus program (Sentinel-1). DEM data from [source]. We thank [contributors/collaborators]. -->
