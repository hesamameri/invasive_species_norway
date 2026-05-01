#
# Create bias surface and sample background points for the MaxEnt SDM.
#
# Invasive-species occurrence records are spatially biased toward roads,
# urban areas, and other accessible places.  If we use uniformly random
# background points, MaxEnt will confuse "accessible" with "suitable."
#
# This script builds a 2-D Gaussian kernel density estimate (KDE) from
# the cleaned Lupinus polyphyllus occurrences, converts it to a bias
# raster aligned with the environmental layers, and then samples 10 000
# background points weighted by that surface.
#
# Outputs:
#   data/bias_surface.tif          – KDE-based bias raster (same grid as env)
#   data/background_points.csv     – 10 000 background lon/lat + env values
#   data/occurrence_env.csv        – occurrence lon/lat + env values
#   data/bias_surface_plot.png     – visual check of the bias raster

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
ENV_DIR = os.path.join("data", "env_layers")
DATA_DIR = "data"
SELECTED_VARS_FILE = os.path.join(ENV_DIR, "selected_variables.txt")
OCCURRENCE_CSV = os.path.join(DATA_DIR, "lupinus_polyphyllus_no_clean.csv")
N_BACKGROUND = 10_000
SEED = 42

# ---------------------------------------------------------------------------
# STEP 1 – Load selected variable names
# ---------------------------------------------------------------------------
with open(SELECTED_VARS_FILE) as f:
    selected_vars = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]
print(f"Selected variables: {selected_vars}")

# Build a mapping: variable name → file path
var_files = {}
for var in selected_vars:
    fname = f"norway_{var}.tif"
    path = os.path.join(ENV_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing raster: {path}")
    var_files[var] = path
print(f"Found {len(var_files)} raster files.")

# ---------------------------------------------------------------------------
# STEP 2 – Read reference raster (for grid info) and all env layers
# ---------------------------------------------------------------------------
ref_path = list(var_files.values())[0]
with rasterio.open(ref_path) as ref:
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_shape = (ref.height, ref.width)
    ref_bounds = ref.bounds
    ref_nodata = ref.nodata

# read all selected layers into a dict
env_arrays = {}
for var, path in var_files.items():
    with rasterio.open(path) as src:
        env_arrays[var] = src.read(1).astype(np.float64)

# build a common valid-pixel mask (True where ALL layers have data)
valid_mask = np.ones(ref_shape, dtype=bool)
for var, arr in env_arrays.items():
    valid_mask &= np.isfinite(arr)
    if ref_nodata is not None:
        valid_mask &= (arr != ref_nodata)

print(f"Reference raster shape: {ref_shape}, valid pixels: {valid_mask.sum()}")

# ---------------------------------------------------------------------------
# STEP 3 – Load occurrence points and extract env values
# ---------------------------------------------------------------------------
occ = pd.read_csv(OCCURRENCE_CSV)
# the cleaned CSV from step 1 has decimalLongitude / decimalLatitude
occ = occ[["decimalLongitude", "decimalLatitude"]].dropna()
occ = occ.rename(columns={"decimalLongitude": "lon", "decimalLatitude": "lat"})
print(f"Loaded {len(occ)} occurrence points.")

def extract_values(lons, lats, env_arrays, transform, valid_mask):
    """Extract env values at lon/lat locations; return DataFrame + keep mask."""
    rows, cols = rowcol(transform, lons, lats)
    rows = np.array(rows)
    cols = np.array(cols)

    h, w = valid_mask.shape
    in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    # also check the pixel is valid
    valid = in_bounds.copy()
    valid[in_bounds] &= valid_mask[rows[in_bounds], cols[in_bounds]]

    records = []
    for i in range(len(lons)):
        if not valid[i]:
            continue
        r, c = rows[i], cols[i]
        rec = {"lon": lons[i], "lat": lats[i]}
        for var, arr in env_arrays.items():
            rec[var] = arr[r, c]
        records.append(rec)
    return pd.DataFrame(records), valid

occ_env, occ_valid = extract_values(
    occ["lon"].values, occ["lat"].values, env_arrays, ref_transform, valid_mask
)
print(f"Occurrences with valid env data: {len(occ_env)}")

# ---------------------------------------------------------------------------
# STEP 4 – Build KDE bias surface
# ---------------------------------------------------------------------------
print("Building Gaussian KDE bias surface...")
coords = np.vstack([occ_env["lon"].values, occ_env["lat"].values])
kde = gaussian_kde(coords, bw_method="scott")

# evaluate KDE on the raster grid (only valid pixels for speed)
valid_rows, valid_cols = np.where(valid_mask)
# convert pixel centres to lon/lat
xs = ref_transform.c + (valid_cols + 0.5) * ref_transform.a
ys = ref_transform.f + (valid_rows + 0.5) * ref_transform.e
grid_coords = np.vstack([xs, ys])

kde_vals = kde(grid_coords)

# create full bias raster
bias_raster = np.full(ref_shape, np.nan, dtype=np.float64)
bias_raster[valid_rows, valid_cols] = kde_vals

# normalise so that valid pixels sum to 1 (probability surface)
total = np.nansum(bias_raster)
bias_raster[valid_rows, valid_cols] /= total

print("KDE bias surface built.")

# save bias raster
bias_path = os.path.join(DATA_DIR, "bias_surface.tif")
with rasterio.open(ref_path) as ref:
    meta = ref.meta.copy()
meta.update(dtype="float64", nodata=np.nan, compress="deflate")
with rasterio.open(bias_path, "w", **meta) as dst:
    dst.write(bias_raster, 1)
print(f"Saved bias surface to {bias_path}")

# quick plot
fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(
    bias_raster,
    extent=[ref_bounds.left, ref_bounds.right, ref_bounds.bottom, ref_bounds.top],
    origin="upper",
    cmap="YlOrRd",
)
occ_env.plot.scatter(x="lon", y="lat", ax=ax, s=2, c="blue", alpha=0.3, label="occurrences")
ax.set_title("KDE bias surface + occurrence points")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
fig.colorbar(im, ax=ax, shrink=0.6, label="sampling probability")
fig.tight_layout()
plot_path = os.path.join(DATA_DIR, "bias_surface_plot.png")
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved bias plot to {plot_path}")

# ---------------------------------------------------------------------------
# STEP 5 – Sample background points weighted by bias surface
# ---------------------------------------------------------------------------
print(f"Sampling {N_BACKGROUND} background points weighted by bias surface...")
rng = np.random.default_rng(SEED)

# probability weights for each valid pixel
probs = bias_raster[valid_rows, valid_cols]
probs = probs / probs.sum()  # ensure exact normalisation

chosen_idx = rng.choice(len(valid_rows), size=N_BACKGROUND, replace=True, p=probs)
bg_rows = valid_rows[chosen_idx]
bg_cols = valid_cols[chosen_idx]

# convert to lon/lat (pixel centres)
bg_lons = ref_transform.c + (bg_cols + 0.5) * ref_transform.a
bg_lats = ref_transform.f + (bg_rows + 0.5) * ref_transform.e

# extract env values at background locations
bg_records = []
for i in range(N_BACKGROUND):
    r, c = bg_rows[i], bg_cols[i]
    rec = {"lon": bg_lons[i], "lat": bg_lats[i]}
    for var, arr in env_arrays.items():
        rec[var] = arr[r, c]
    bg_records.append(rec)

bg_env = pd.DataFrame(bg_records)
print(f"Background points sampled: {len(bg_env)}")

# ---------------------------------------------------------------------------
# STEP 6 – Save outputs
# ---------------------------------------------------------------------------
occ_csv = os.path.join(DATA_DIR, "occurrence_env.csv")
bg_csv = os.path.join(DATA_DIR, "background_points.csv")
occ_env.to_csv(occ_csv, index=False)
bg_env.to_csv(bg_csv, index=False)
print(f"Saved occurrence + env values to {occ_csv}")
print(f"Saved background points to {bg_csv}")

print(f"\n✓ Bias correction complete.")
print(f"  Occurrence points with env data: {len(occ_env)}")
print(f"  Background points:               {len(bg_env)}")
print(f"  Selected predictors:             {selected_vars}")
print(f"\nNext step: fit the MaxEnt model.")
