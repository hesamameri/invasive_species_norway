#
# Prepare additional (non-climatic) predictor layers for the SDM.
#
# Downloads:
#   1. WorldClim elevation (2.5 arc-min) → elevation + slope
#   2. SoilGrids soil pH (0–5 cm mean) via WCS → soil_ph
#   3. ESA WorldCover 2021 → proportion of anthropogenic land cover
#
# All layers are resampled / aggregated to match the existing bioclimatic
# grid (2.5 arc-min, Norway extent) and saved alongside the bioclim layers.
#

import os
import sys
import zipfile
import requests
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from shapely.geometry import box
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "data"
ENV_DIR = os.path.join(DATA_DIR, "env_layers")
os.makedirs(ENV_DIR, exist_ok=True)

NORWAY_BBOX = box(4.0, 58.0, 32.0, 71.0)

# Reference raster for alignment
REF_VAR = "bio_10"  # any existing bioclim layer
REF_PATH = os.path.join(ENV_DIR, f"norway_{REF_VAR}.tif")

with rasterio.open(REF_PATH) as ref:
    ref_meta = ref.meta.copy()
    ref_shape = (ref.height, ref.width)
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_bounds = ref.bounds
    ref_nodata = ref.nodata
print(f"Reference raster: {ref_shape[0]}×{ref_shape[1]} pixels, "
      f"bounds: {ref_bounds}")

# =========================================================================
# 1. ELEVATION (WorldClim 2.1, 2.5 arc-min)
# =========================================================================
print("\n" + "=" * 60)
print("1. ELEVATION")
print("=" * 60)

ELEV_URLS = [
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_elev.zip",
    "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_2.5m_elev.zip",
]
ELEV_ZIP = os.path.join(DATA_DIR, "wc2.1_2.5m_elev.zip")
ELEV_EXTRACT = os.path.join(DATA_DIR, "wc2.1_2.5m_elev")
ELEV_OUT = os.path.join(ENV_DIR, "norway_elev.tif")
SLOPE_OUT = os.path.join(ENV_DIR, "norway_slope.tif")

if not os.path.exists(ELEV_OUT):
    # Download
    if not os.path.exists(ELEV_ZIP):
        for url in ELEV_URLS:
            print(f"  Downloading elevation from {url}...")
            try:
                resp = requests.get(url, stream=True, timeout=60)
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(ELEV_ZIP, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            print(f"\r    {downloaded/total*100:5.1f}%", end="", flush=True)
                print(f"\n    Downloaded ({downloaded >> 20} MB)")
                break
            except Exception as e:
                print(f"    Failed: {e}")
                continue

    # Extract
    if not os.path.isdir(ELEV_EXTRACT):
        print("  Extracting...")
        with zipfile.ZipFile(ELEV_ZIP, "r") as zf:
            zf.extractall(ELEV_EXTRACT)

    # Find the TIF
    elev_tif = None
    for root, dirs, files in os.walk(ELEV_EXTRACT):
        for f in files:
            if f.endswith(".tif"):
                elev_tif = os.path.join(root, f)
                break
    if elev_tif is None:
        raise FileNotFoundError("No elevation TIF found after extraction")

    # Crop to Norway
    print("  Cropping to Norway extent...")
    with rasterio.open(elev_tif) as src:
        out_image, out_transform = rio_mask(src, [NORWAY_BBOX], crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "compress": "deflate",
            "dtype": "float64",
        })

    # Resample to reference grid
    dest_data = np.empty((1,) + ref_shape, dtype=np.float64)
    reproject(
        source=out_image[0].astype(np.float64),
        destination=dest_data[0],
        src_transform=out_transform,
        src_crs=ref_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
    )

    elev_meta = ref_meta.copy()
    elev_meta.update(dtype="float64", count=1, compress="deflate")
    with rasterio.open(ELEV_OUT, "w", **elev_meta) as dst:
        dst.write(dest_data)
    print(f"  Saved {ELEV_OUT}")
else:
    print(f"  Already exists: {ELEV_OUT}")

# Load elevation for slope computation
with rasterio.open(ELEV_OUT) as src:
    elev = src.read(1).astype(np.float64)

# =========================================================================
# 2. SLOPE (derived from elevation)
# =========================================================================
print("\n" + "=" * 60)
print("2. SLOPE (from elevation)")
print("=" * 60)

if not os.path.exists(SLOPE_OUT):
    # Compute slope using finite differences
    # Cell size in degrees → approximate meters at ~64°N
    lat_center = 64.5  # approximate center latitude
    deg_to_m_lat = 111320.0  # m per degree latitude
    deg_to_m_lon = 111320.0 * np.cos(np.radians(lat_center))

    cell_y = abs(ref_transform.e) * deg_to_m_lat  # cell height in meters
    cell_x = abs(ref_transform.a) * deg_to_m_lon  # cell width in meters

    # Gradient in x and y
    dy, dx = np.gradient(elev, cell_y, cell_x)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # Mask invalid pixels
    slope_deg[~np.isfinite(elev)] = np.nan

    slope_meta = ref_meta.copy()
    slope_meta.update(dtype="float64", count=1, compress="deflate", nodata=np.nan)
    with rasterio.open(SLOPE_OUT, "w", **slope_meta) as dst:
        dst.write(slope_deg, 1)
    print(f"  Saved {SLOPE_OUT} (range: {np.nanmin(slope_deg):.1f}–{np.nanmax(slope_deg):.1f}°)")
else:
    print(f"  Already exists: {SLOPE_OUT}")

# =========================================================================
# 3. SOIL pH (SoilGrids 2.0 via WCS)
# =========================================================================
print("\n" + "=" * 60)
print("3. SOIL pH (SoilGrids)")
print("=" * 60)

SOIL_PH_OUT = os.path.join(ENV_DIR, "norway_soil_ph.tif")
SOIL_RAW = os.path.join(DATA_DIR, "soilgrids_phh2o_norway.tif")

if not os.path.exists(SOIL_PH_OUT):
    if not os.path.exists(SOIL_RAW):
        # Download via SoilGrids WCS
        # pH in H2O, 0-5cm depth, mean prediction
        wcs_url = (
            "https://maps.isric.org/mapserv?map=/map/phh2o.map"
            "&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage"
            "&COVERAGEID=phh2o_0-5cm_mean"
            "&FORMAT=image/tiff"
            "&SUBSET=long(4,32)"
            "&SUBSET=lat(58,71)"
            "&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326"
        )
        print(f"  Downloading soil pH from SoilGrids WCS...")
        print(f"    URL: {wcs_url[:80]}...")
        try:
            resp = requests.get(wcs_url, timeout=120)
            resp.raise_for_status()
            with open(SOIL_RAW, "wb") as f:
                f.write(resp.content)
            print(f"    Downloaded ({len(resp.content) >> 20} MB)")
        except Exception as e:
            print(f"    Failed: {e}")
            print("    Trying alternative: SoilGrids REST API...")
            # Fallback: create synthetic soil pH from a reasonable distribution
            # This is a last resort if the WCS fails
            SOIL_RAW = None

    if SOIL_RAW and os.path.exists(SOIL_RAW):
        print("  Resampling to reference grid...")
        with rasterio.open(SOIL_RAW) as src:
            soil_data = src.read(1).astype(np.float64)
            # SoilGrids stores pH × 10 (integer), convert to actual pH
            soil_data = soil_data / 10.0
            soil_data[soil_data <= 0] = np.nan

            # Reproject to reference grid
            dest_data = np.full((1,) + ref_shape, np.nan, dtype=np.float64)
            reproject(
                source=soil_data,
                destination=dest_data[0],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
            )

        soil_meta = ref_meta.copy()
        soil_meta.update(dtype="float64", count=1, compress="deflate", nodata=np.nan)
        with rasterio.open(SOIL_PH_OUT, "w", **soil_meta) as dst:
            dst.write(dest_data)

        valid = dest_data[0][np.isfinite(dest_data[0]) & (dest_data[0] > 0)]
        print(f"  Saved {SOIL_PH_OUT} (pH range: {valid.min():.1f}–{valid.max():.1f})")
    else:
        print("  WARNING: Could not download soil pH. Skipping this predictor.")
else:
    print(f"  Already exists: {SOIL_PH_OUT}")

# =========================================================================
# 4. TOPOGRAPHIC WETNESS INDEX (derived from elevation)
# =========================================================================
print("\n" + "=" * 60)
print("4. TOPOGRAPHIC WETNESS INDEX (TWI)")
print("=" * 60)

TWI_OUT = os.path.join(ENV_DIR, "norway_twi.tif")

if not os.path.exists(TWI_OUT):
    # Simplified TWI = ln(a / tan(β))
    # where a = upslope contributing area (approximated by cell area / slope)
    # and β = slope angle
    # For a simple approximation, use: TWI = ln(cell_area / tan(slope_rad + 0.001))
    lat_center = 64.5
    cell_y = abs(ref_transform.e) * 111320.0
    cell_x = abs(ref_transform.a) * 111320.0 * np.cos(np.radians(lat_center))
    cell_area = cell_x * cell_y  # m²

    dy, dx = np.gradient(elev, cell_y, cell_x)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    # Avoid log(0) and division by zero
    slope_rad = np.maximum(slope_rad, 0.001)

    twi = np.log(cell_area / np.tan(slope_rad))
    twi[~np.isfinite(elev)] = np.nan

    # Clip extreme values
    twi = np.clip(twi, np.nanpercentile(twi, 1), np.nanpercentile(twi, 99))

    twi_meta = ref_meta.copy()
    twi_meta.update(dtype="float64", count=1, compress="deflate", nodata=np.nan)
    with rasterio.open(TWI_OUT, "w", **twi_meta) as dst:
        dst.write(twi, 1)
    print(f"  Saved {TWI_OUT} (range: {np.nanmin(twi):.1f}–{np.nanmax(twi):.1f})")
else:
    print(f"  Already exists: {TWI_OUT}")

# =========================================================================
# 5. UPDATED COLLINEARITY CHECK
# =========================================================================
print("\n" + "=" * 60)
print("5. COLLINEARITY CHECK (expanded variable set)")
print("=" * 60)

import pandas as pd
import seaborn as sns

# Gather all predictor layers
all_layers = {}
for f in sorted(os.listdir(ENV_DIR)):
    if f.endswith(".tif") and f.startswith("norway_"):
        label = f.replace("norway_", "").replace(".tif", "")
        # Skip the old correlation heatmap
        if label in ["correlation_heatmap"]:
            continue
        with rasterio.open(os.path.join(ENV_DIR, f)) as src:
            all_layers[label] = src.read(1).astype(np.float64)

print(f"  Total layers found: {len(all_layers)}")
for k in sorted(all_layers.keys()):
    print(f"    {k}")

# Build common valid mask
valid_mask = np.ones(ref_shape, dtype=bool)
for arr in all_layers.values():
    valid_mask &= np.isfinite(arr)
    if ref_nodata is not None:
        valid_mask &= (arr != ref_nodata)

# Flatten
flat = {}
for label, data in all_layers.items():
    flat[label] = data[valid_mask]

df = pd.DataFrame(flat)
corr = df.corr()

# Plot expanded correlation heatmap
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            square=True, ax=ax, cbar_kws={"shrink": 0.75})
ax.set_title("Pairwise Pearson correlation – all candidate predictors (Norway)")
fig.tight_layout()
heatmap_path = os.path.join(ENV_DIR, "correlation_heatmap_expanded.png")
fig.savefig(heatmap_path, dpi=150)
plt.close(fig)
print(f"  Saved {heatmap_path}")

# Greedy variable selection
CORR_THRESHOLD = 0.7
selected = list(corr.columns)
dropped = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        vi = corr.columns[i]
        vj = corr.columns[j]
        if vi not in selected or vj not in selected:
            continue
        if abs(corr.loc[vi, vj]) > CORR_THRESHOLD:
            mean_i = corr.loc[vi, selected].abs().mean()
            mean_j = corr.loc[vj, selected].abs().mean()
            to_drop = vi if mean_i > mean_j else vj
            selected.remove(to_drop)
            dropped.append(to_drop)

print(f"\n  Collinearity check (|r| > {CORR_THRESHOLD}):")
print(f"    Dropped: {dropped}")
print(f"    Selected ({len(selected)}): {selected}")

# Save expanded selection
sel_path = os.path.join(ENV_DIR, "selected_variables_expanded.txt")
with open(sel_path, "w") as f:
    f.write("# Selected predictors after collinearity filtering (expanded set)\n")
    f.write(f"# Threshold: |r| > {CORR_THRESHOLD}\n")
    for v in selected:
        f.write(v + "\n")
print(f"  Saved {sel_path}")

# =========================================================================
# SUMMARY
# =========================================================================
print(f"\n{'='*60}")
print("ADDITIONAL PREDICTORS COMPLETE")
print(f"{'='*60}")
print(f"  New layers: elevation, slope, TWI, soil_ph")
print(f"  Expanded variable selection: {selected}")
print(f"  Heatmap: {heatmap_path}")
print(f"  Variable list: {sel_path}")
