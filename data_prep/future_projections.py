#
# CMIP6 Future Climate Projections for Lupinus polyphyllus in Norway
#
# Downloads WorldClim 2.1 CMIP6 bioclimatic projections, crops to Norway,
# projects the fitted MaxEnt model onto future climates, and generates
# publication-quality comparison maps.
#
# Outputs:
#   data/future/             – cropped future climate rasters
#   data/future/projections/ – suitability rasters per GCM × SSP × period
#   data/figures/future_*    – publication figures
#   data/future/summary.csv  – suitable area statistics
#

import os
import sys
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import box
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import elapid
import requests
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "data"
ENV_DIR = os.path.join(DATA_DIR, "env_layers")
MODEL_DIR = os.path.join(DATA_DIR, "model")
FIG_DIR = os.path.join(DATA_DIR, "figures")
FUTURE_DIR = os.path.join(DATA_DIR, "future")
PROJ_DIR = os.path.join(FUTURE_DIR, "projections")
os.makedirs(FUTURE_DIR, exist_ok=True)
os.makedirs(PROJ_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SELECTED_VARS_FILE = os.path.join(ENV_DIR, "selected_variables.txt")
NORWAY_BBOX = box(4.0, 58.0, 32.0, 71.0)
MAP_EXTENT = [3, 33, 57.5, 71.5]

# GCMs spanning low → moderate → high climate sensitivity
GCMS = [
    "ACCESS-CM2",      # moderate sensitivity (~4.7°C ECS, Australian)
    "EC-Earth3-Veg",   # moderate             (~3.9°C ECS, European)
    "CMCC-ESM2",       # moderate-high        (~4.7°C ECS, Italian)
]

SSPS = ["ssp245", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]

SSP_LABELS = {
    "ssp245": "SSP2-4.5 (moderate)",
    "ssp585": "SSP5-8.5 (high emissions)",
}

BASE_URL = "https://geodata.ucdavis.edu/cmip6/2.5m"

# ---------------------------------------------------------------------------
# STEP 0 – Load selected variables and reference raster
# ---------------------------------------------------------------------------
print("Loading configuration...")
with open(SELECTED_VARS_FILE) as f:
    selected_vars = [l.strip() for l in f if l.strip() and not l.startswith("#")]
print(f"  Selected variables: {selected_vars}")

# Map variable names to band indices in the multi-band CMIP6 TIF
# bio_1 → band 1, bio_2 → band 2, etc.
var_to_band = {}
for v in selected_vars:
    band_num = int(v.replace("bio_", ""))
    var_to_band[v] = band_num
print(f"  Band mapping: {var_to_band}")

# Load reference raster metadata (current climate) for alignment
ref_path = os.path.join(ENV_DIR, f"norway_{selected_vars[0]}.tif")
with rasterio.open(ref_path) as ref:
    ref_meta = ref.meta.copy()
    ref_shape = (ref.height, ref.width)
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_bounds = ref.bounds
    ref_nodata = ref.nodata
print(f"  Reference raster: {ref_shape[0]}×{ref_shape[1]} pixels")

# Load the fitted MaxEnt model
print("  Loading MaxEnt model...")
model = elapid.load_object(os.path.join(MODEL_DIR, "best_model.ela"))

# Load current suitability for comparison
current_suit_path = os.path.join(MODEL_DIR, "suitability_cloglog.tif")
with rasterio.open(current_suit_path) as src:
    current_suit = src.read(1).astype(np.float64)

# Build valid pixel mask from current layers
print("  Building valid pixel mask...")
env_current = {}
for var in selected_vars:
    with rasterio.open(os.path.join(ENV_DIR, f"norway_{var}.tif")) as src:
        env_current[var] = src.read(1).astype(np.float64)

valid_mask = np.ones(ref_shape, dtype=bool)
for arr in env_current.values():
    valid_mask &= np.isfinite(arr)
    if ref_nodata is not None:
        valid_mask &= (arr != ref_nodata)

valid_rows, valid_cols = np.where(valid_mask)
n_valid = len(valid_rows)
print(f"  Valid land pixels: {n_valid}")

# ---------------------------------------------------------------------------
# STEP 1 – Download and crop CMIP6 bioclimatic rasters
# ---------------------------------------------------------------------------
def download_cmip6_tif(gcm, ssp, period):
    """Download a CMIP6 bioclimatic multi-band TIF if not already cached."""
    fname = f"wc2.1_2.5m_bioc_{gcm}_{ssp}_{period}.tif"
    local_path = os.path.join(FUTURE_DIR, fname)

    if os.path.exists(local_path):
        return local_path

    url = f"{BASE_URL}/{gcm}/{ssp}/{fname}"
    print(f"    Downloading {fname}...")
    print(f"      URL: {url}")

    try:
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"    ✗ Failed to download: {e}")
        return None

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r      {pct:5.1f}% ({downloaded >> 20} MB / {total >> 20} MB)",
                      end="", flush=True)
    print(f"\n      Done ({downloaded >> 20} MB)")
    return local_path


def crop_and_extract(tif_path, gcm, ssp, period):
    """Crop global TIF to Norway and extract selected bioclimatic bands."""
    out_dir = os.path.join(FUTURE_DIR, f"{gcm}_{ssp}_{period}")
    os.makedirs(out_dir, exist_ok=True)

    # Check if already done
    expected = [os.path.join(out_dir, f"norway_{v}.tif") for v in selected_vars]
    if all(os.path.exists(p) for p in expected):
        return out_dir

    print(f"    Cropping to Norway extent...")
    with rasterio.open(tif_path) as src:
        for var, band_idx in var_to_band.items():
            # Read specific band and crop to Norway
            out_image, out_transform = rio_mask(
                src, [NORWAY_BBOX], crop=True, indexes=[band_idx]
            )
            out_meta = src.meta.copy()
            out_meta.update({
                "count": 1,
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "deflate",
                "dtype": "float64",
            })

            # The cropped future raster may have slightly different dimensions
            # than the reference (current). We need to resample to match exactly.
            out_path = os.path.join(out_dir, f"norway_{var}.tif")
            # First write the raw crop
            temp_path = out_path + ".tmp.tif"
            with rasterio.open(temp_path, "w", **out_meta) as dst:
                dst.write(out_image.astype(np.float64))

            # Now reproject/resample to match reference grid exactly
            dest_data = np.empty((1,) + ref_shape, dtype=np.float64)
            dest_meta = ref_meta.copy()
            dest_meta.update(dtype="float64", count=1, compress="deflate")

            with rasterio.open(temp_path) as tmp_src:
                reproject(
                    source=tmp_src.read(1),
                    destination=dest_data[0],
                    src_transform=tmp_src.transform,
                    src_crs=tmp_src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                )
            with rasterio.open(out_path, "w", **dest_meta) as dst:
                dst.write(dest_data)

            os.remove(temp_path)

    print(f"    Saved {len(selected_vars)} cropped layers to {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# STEP 2 – Download, crop, project for all combinations
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("PHASE 1: Downloading & processing CMIP6 future climate data")
print("=" * 60)

all_results = []

for gcm in GCMS:
    for ssp in SSPS:
        for period in PERIODS:
            combo = f"{gcm} / {ssp} / {period}"
            print(f"\n--- {combo} ---")

            # Download
            tif_path = download_cmip6_tif(gcm, ssp, period)
            if tif_path is None:
                print(f"  SKIPPED (download failed)")
                continue

            # Crop and extract bands
            out_dir = crop_and_extract(tif_path, gcm, ssp, period)

            # Load future environmental layers
            future_env = {}
            for var in selected_vars:
                fpath = os.path.join(out_dir, f"norway_{var}.tif")
                with rasterio.open(fpath) as src:
                    future_env[var] = src.read(1).astype(np.float64)

            # Build pixel data for valid pixels
            pixel_data = np.column_stack([
                future_env[var][valid_rows, valid_cols] for var in selected_vars
            ])

            # Check for NaN/Inf in future data
            row_valid = np.all(np.isfinite(pixel_data), axis=1)
            print(f"    Valid future pixels: {row_valid.sum()} / {n_valid}")

            # Project model
            print(f"    Projecting MaxEnt model...")
            suit_future = np.full(ref_shape, np.nan, dtype=np.float64)
            if row_valid.sum() > 0:
                preds = model.predict(pixel_data[row_valid])
                valid_subset_rows = valid_rows[row_valid]
                valid_subset_cols = valid_cols[row_valid]
                suit_future[valid_subset_rows, valid_subset_cols] = preds

            # Save projection raster
            proj_path = os.path.join(PROJ_DIR, f"suit_{gcm}_{ssp}_{period}.tif")
            proj_meta = ref_meta.copy()
            proj_meta.update(dtype="float64", nodata=np.nan, compress="deflate", count=1)
            with rasterio.open(proj_path, "w", **proj_meta) as dst:
                dst.write(suit_future, 1)

            # Compute stats
            suit_vals = suit_future[valid_mask & np.isfinite(suit_future)]
            n_total = valid_mask.sum()
            n_high = np.sum(suit_vals >= 0.6)
            n_vhigh = np.sum(suit_vals >= 0.8)
            pct_high = n_high / n_total * 100
            pct_vhigh = n_vhigh / n_total * 100
            mean_suit = np.nanmean(suit_vals)

            all_results.append({
                "gcm": gcm, "ssp": ssp, "period": period,
                "mean_suitability": mean_suit,
                "pct_high_vhigh": pct_high,
                "pct_vhigh": pct_vhigh,
                "n_high_vhigh": int(n_high),
                "n_vhigh": int(n_vhigh),
            })
            print(f"    Mean suitability: {mean_suit:.3f}")
            print(f"    High+VeryHigh: {pct_high:.1f}% ({n_high} pixels)")

# Also add current climate stats
current_vals = current_suit[valid_mask & np.isfinite(current_suit)]
n_total_cur = valid_mask.sum()
all_results.insert(0, {
    "gcm": "Current", "ssp": "baseline", "period": "1970-2000",
    "mean_suitability": np.nanmean(current_vals),
    "pct_high_vhigh": np.sum(current_vals >= 0.6) / n_total_cur * 100,
    "pct_vhigh": np.sum(current_vals >= 0.8) / n_total_cur * 100,
    "n_high_vhigh": int(np.sum(current_vals >= 0.6)),
    "n_vhigh": int(np.sum(current_vals >= 0.8)),
})

results_df = pd.DataFrame(all_results)
results_path = os.path.join(FUTURE_DIR, "projection_summary.csv")
results_df.to_csv(results_path, index=False)
print(f"\n{'='*60}")
print("Projection summary:")
print(results_df.to_string(index=False))
print(f"\nSaved to {results_path}")

# ---------------------------------------------------------------------------
# STEP 3 – Compute multi-GCM ensemble mean per SSP × period
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Computing multi-GCM ensemble means...")

ensemble = {}
for ssp in SSPS:
    for period in PERIODS:
        key = f"{ssp}_{period}"
        rasters = []
        for gcm in GCMS:
            path = os.path.join(PROJ_DIR, f"suit_{gcm}_{ssp}_{period}.tif")
            if os.path.exists(path):
                with rasterio.open(path) as src:
                    rasters.append(src.read(1).astype(np.float64))

        if rasters:
            stack = np.stack(rasters, axis=0)
            ens_mean = np.nanmean(stack, axis=0)
            ens_sd = np.nanstd(stack, axis=0)
            ensemble[key] = {"mean": ens_mean, "sd": ens_sd}

            # Save ensemble mean raster
            ens_path = os.path.join(PROJ_DIR, f"ensemble_mean_{key}.tif")
            ens_meta = ref_meta.copy()
            ens_meta.update(dtype="float64", nodata=np.nan, compress="deflate", count=1)
            with rasterio.open(ens_path, "w", **ens_meta) as dst:
                dst.write(ens_mean, 1)
            print(f"  {key}: {len(rasters)} GCMs → ensemble mean saved")

# ---------------------------------------------------------------------------
# STEP 4 – Publication figures
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Creating publication figures...")

img_extent = [ref_bounds.left, ref_bounds.right, ref_bounds.bottom, ref_bounds.top]


def make_map_ax(fig, gs_pos):
    ax = fig.add_subplot(gs_pos, projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", color="grey")
    ax.add_feature(cfeature.OCEAN, facecolor="lightcyan", zorder=0)
    return ax


# --- Figure A: 2×2 panel – ensemble means for each SSP × period ---
fig = plt.figure(figsize=(14, 16))
gs = gridspec.GridSpec(2, 2, hspace=0.15, wspace=0.05)

panel_labels = [
    ("ssp245_2041-2060", "SSP2-4.5, 2041–2060"),
    ("ssp585_2041-2060", "SSP5-8.5, 2041–2060"),
    ("ssp245_2061-2080", "SSP2-4.5, 2061–2080"),
    ("ssp585_2061-2080", "SSP5-8.5, 2061–2080"),
]

for idx, (key, title) in enumerate(panel_labels):
    if key not in ensemble:
        continue
    ax = make_map_ax(fig, gs[idx])
    data = np.ma.masked_invalid(ensemble[key]["mean"])
    im = ax.imshow(data, extent=img_extent, origin="upper",
                   transform=ccrs.PlateCarree(), cmap="cividis",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=12, fontweight="bold")

cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.015])
fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
             label="Predicted habitat suitability (ensemble mean of 3 GCMs)")

fig.suptitle("Future habitat suitability for Lupinus polyphyllus in Norway\n"
             "CMIP6 multi-GCM ensemble projections",
             fontsize=14, fontweight="bold", y=0.95)

path_a = os.path.join(FIG_DIR, "future_suitability_panels.png")
fig.savefig(path_a, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved {path_a}")


# --- Figure B: Change maps (future – current) ---
fig = plt.figure(figsize=(14, 16))
gs = gridspec.GridSpec(2, 2, hspace=0.15, wspace=0.05)

for idx, (key, title) in enumerate(panel_labels):
    if key not in ensemble:
        continue
    ax = make_map_ax(fig, gs[idx])
    change = ensemble[key]["mean"] - current_suit
    change_masked = np.ma.masked_invalid(change)
    im = ax.imshow(change_masked, extent=img_extent, origin="upper",
                   transform=ccrs.PlateCarree(), cmap="RdBu_r",
                   vmin=-0.4, vmax=0.4, interpolation="nearest")
    ax.set_title(f"Δ Suitability: {title}", fontsize=11, fontweight="bold")

cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.015])
fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
             label="Change in suitability (future − current)")

fig.suptitle("Projected change in habitat suitability\n"
             "under CMIP6 scenarios (ensemble mean − baseline)",
             fontsize=14, fontweight="bold", y=0.95)

path_b = os.path.join(FIG_DIR, "future_change_panels.png")
fig.savefig(path_b, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved {path_b}")


# --- Figure C: Bar chart of suitable area by scenario ---
fig, ax = plt.subplots(figsize=(10, 6))

# Group by SSP and period
bar_data = results_df[results_df["gcm"] != "Current"].copy()
bar_data["label"] = bar_data["ssp"].str.upper() + "\n" + bar_data["period"]

# Ensemble means
ens_bars = []
for ssp in SSPS:
    for period in PERIODS:
        subset = bar_data[(bar_data["ssp"] == ssp) & (bar_data["period"] == period)]
        ens_bars.append({
            "ssp": ssp, "period": period,
            "label": ssp.upper().replace("SSP", "SSP") + "\n" + period,
            "mean_pct": subset["pct_high_vhigh"].mean(),
            "sd_pct": subset["pct_high_vhigh"].std(),
        })
ens_df = pd.DataFrame(ens_bars)

# Current baseline
current_pct = results_df[results_df["gcm"] == "Current"]["pct_high_vhigh"].values[0]

colors = {"ssp245": "#2166ac", "ssp585": "#b2182b"}
x_pos = np.arange(len(ens_df))
bars = ax.bar(x_pos, ens_df["mean_pct"], yerr=ens_df["sd_pct"],
              capsize=5, width=0.6,
              color=[colors[s] for s in ens_df["ssp"]],
              edgecolor="black", linewidth=0.5)

ax.axhline(y=current_pct, color="black", linestyle="--", linewidth=1.5,
           label=f"Current baseline ({current_pct:.1f}%)")
ax.set_xticks(x_pos)
ax.set_xticklabels(ens_df["label"], fontsize=10)
ax.set_ylabel("% of land area with High + Very High suitability", fontsize=11)
ax.set_title("Projected invasion risk area under CMIP6 scenarios\n"
             "(mean ± SD across 3 GCMs)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, max(ens_df["mean_pct"].max() + 10, current_pct + 10))

fig.tight_layout()
path_c = os.path.join(FIG_DIR, "future_risk_area_barplot.png")
fig.savefig(path_c, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved {path_c}")


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("CMIP6 FUTURE PROJECTIONS COMPLETE")
print(f"{'='*60}")
print(f"\nSummary statistics:")
print(results_df[["gcm", "ssp", "period", "mean_suitability",
                   "pct_high_vhigh"]].to_string(index=False))
print(f"\nFigures:")
print(f"  {path_a}")
print(f"  {path_b}")
print(f"  {path_c}")
print(f"\nData:")
print(f"  {results_path}")
print(f"  {PROJ_DIR}/ (individual + ensemble rasters)")
