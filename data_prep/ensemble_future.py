#
# Ensemble Future Projections
#
# Projects all 4 algorithms (MaxEnt, RF, GBM, GLM) to CMIP6 future climates,
# computes AUC-weighted ensemble mean, and generates publication figures.
#
# Static predictors (slope, soil_ph) are held constant; only bioclimatic
# variables change with CMIP6 scenarios.
#

import os
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import elapid
import joblib
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
ENS_PROJ_DIR = os.path.join(FUTURE_DIR, "ensemble_projections")
os.makedirs(ENS_PROJ_DIR, exist_ok=True)

EXPANDED_VARS_FILE = os.path.join(ENV_DIR, "selected_variables_expanded.txt")
NORWAY_BBOX = box(4.0, 58.0, 32.0, 71.0)
MAP_EXTENT = [3, 33, 57.5, 71.5]
SEED = 42

GCMS = ["ACCESS-CM2", "EC-Earth3-Veg", "CMCC-ESM2"]
SSPS = ["ssp245", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]

# ---------------------------------------------------------------------------
# STEP 0 – Load configuration
# ---------------------------------------------------------------------------
print("Loading configuration...")
with open(EXPANDED_VARS_FILE) as f:
    expanded_vars = [l.strip() for l in f if l.strip() and not l.startswith("#")]
print(f"  Expanded predictors: {expanded_vars}")

# Identify bioclimatic vs static predictors
BIOCLIM_VARS = [v for v in expanded_vars if v.startswith("bio_")]
STATIC_VARS = [v for v in expanded_vars if not v.startswith("bio_")]
print(f"  Bioclimatic (change with climate): {BIOCLIM_VARS}")
print(f"  Static (held constant):            {STATIC_VARS}")

# Reference raster
ref_path = os.path.join(ENV_DIR, f"norway_{expanded_vars[0]}.tif")
with rasterio.open(ref_path) as ref:
    ref_meta = ref.meta.copy()
    ref_shape = (ref.height, ref.width)
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_bounds = ref.bounds

# Valid mask from current layers
env_current = {}
for var in expanded_vars:
    with rasterio.open(os.path.join(ENV_DIR, f"norway_{var}.tif")) as src:
        env_current[var] = src.read(1).astype(np.float64)

valid_mask = np.ones(ref_shape, dtype=bool)
for arr in env_current.values():
    valid_mask &= np.isfinite(arr)
valid_rows, valid_cols = np.where(valid_mask)
n_valid = len(valid_rows)
print(f"  Valid pixels: {n_valid}")

# ---------------------------------------------------------------------------
# STEP 1 – Re-fit all models (same as ensemble_models.py)
# ---------------------------------------------------------------------------
print("\nRe-fitting models on full training data...")

occ = pd.read_csv(os.path.join(DATA_DIR, "occurrence_env_expanded.csv"))
bg = pd.read_csv(os.path.join(DATA_DIR, "background_points_expanded.csv"))
X_occ = occ[expanded_vars].values.astype(np.float64)
X_bg = bg[expanded_vars].values.astype(np.float64)
X_all = np.vstack([X_occ, X_bg])
y_all = np.concatenate([np.ones(len(X_occ)), np.zeros(len(X_bg))])

# AUC weights from CV
summary = pd.read_csv(os.path.join(MODEL_DIR, "ensemble_summary.csv"))
auc_weights = dict(zip(summary["algorithm"], summary["mean_auc"]))
print(f"  AUC weights: {auc_weights}")

models = {}
scaler = StandardScaler()

# MaxEnt
m = elapid.MaxentModel(feature_types=["linear"], beta_multiplier=3.0,
                        n_hinge_features=10, transform="cloglog")
m.fit(X_all, y_all)
models["MaxEnt"] = m

# RF
m = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=5,
                           class_weight="balanced", random_state=SEED, n_jobs=-1)
m.fit(X_all, y_all)
models["RF"] = m

# GBM
m = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                min_samples_leaf=10, subsample=0.8, random_state=SEED)
m.fit(X_all, y_all)
models["GBM"] = m

# GLM
X_scaled = scaler.fit_transform(X_all)
m = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="lbfgs",
                       class_weight="balanced", random_state=SEED)
m.fit(X_scaled, y_all)
models["GLM"] = m

print("  All 4 models fitted.")

# Current suitability (ensemble)
def predict_all(pixel_data):
    """Get predictions from all 4 models and weighted ensemble."""
    preds = {}
    for name, model in models.items():
        if name == "GLM":
            p = model.predict_proba(scaler.transform(pixel_data))[:, 1]
        elif name == "MaxEnt":
            p = model.predict(pixel_data)
        else:
            p = model.predict_proba(pixel_data)[:, 1]
        preds[name] = p

    # Weighted ensemble
    total_w = sum(auc_weights.values())
    ens = np.zeros(len(pixel_data))
    for name, p in preds.items():
        ens += (auc_weights[name] / total_w) * p
    preds["Ensemble"] = ens
    return preds


# ---------------------------------------------------------------------------
# STEP 2 – Extract bio_19 from CMIP6 multi-band TIFs (if not done)
# ---------------------------------------------------------------------------
print("\nExtracting bio_19 from CMIP6 TIFs...")

# bio_19 = band 19 in the multi-band TIF
NEW_BIOCLIM = [v for v in BIOCLIM_VARS if v not in
               ["bio_10", "bio_11", "bio_15", "bio_18", "bio_2", "bio_3", "bio_8"]]
print(f"  New bioclim vars to extract: {NEW_BIOCLIM}")

for gcm in GCMS:
    for ssp in SSPS:
        for period in PERIODS:
            out_dir = os.path.join(FUTURE_DIR, f"{gcm}_{ssp}_{period}")
            if not os.path.isdir(out_dir):
                continue

            for var in NEW_BIOCLIM:
                out_path = os.path.join(out_dir, f"norway_{var}.tif")
                if os.path.exists(out_path):
                    continue

                band_num = int(var.replace("bio_", ""))
                tif_name = f"wc2.1_2.5m_bioc_{gcm}_{ssp}_{period}.tif"
                tif_path = os.path.join(FUTURE_DIR, tif_name)
                if not os.path.exists(tif_path):
                    print(f"    SKIP {gcm}/{ssp}/{period} — TIF not found")
                    continue

                with rasterio.open(tif_path) as src:
                    out_image, out_transform = rio_mask(
                        src, [NORWAY_BBOX], crop=True, indexes=[band_num]
                    )

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
                dest_meta = ref_meta.copy()
                dest_meta.update(dtype="float64", count=1, compress="deflate")
                with rasterio.open(out_path, "w", **dest_meta) as dst:
                    dst.write(dest_data)

            print(f"  {gcm}/{ssp}/{period}: bio_19 extracted")


# ---------------------------------------------------------------------------
# STEP 3 – Project ensemble to all CMIP6 scenarios
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("PROJECTING ENSEMBLE TO CMIP6 FUTURES")
print(f"{'='*60}")

all_results = []

# Current baseline
current_pixels = np.column_stack([
    env_current[var][valid_rows, valid_cols] for var in expanded_vars
])
current_preds = predict_all(current_pixels)
current_ens = current_preds["Ensemble"]
current_ens_raster = np.full(ref_shape, np.nan)
current_ens_raster[valid_rows, valid_cols] = current_ens

current_vals = current_ens[np.isfinite(current_ens)]
all_results.append({
    "gcm": "Current", "ssp": "baseline", "period": "1970-2000",
    "mean_suit": np.mean(current_vals),
    "pct_high": np.sum(current_vals >= 0.6) / len(current_vals) * 100,
})

for gcm in GCMS:
    for ssp in SSPS:
        for period in PERIODS:
            combo = f"{gcm}/{ssp}/{period}"
            out_dir = os.path.join(FUTURE_DIR, f"{gcm}_{ssp}_{period}")

            # Load future bioclim + static current predictors
            future_env = {}
            skip = False
            for var in expanded_vars:
                if var in STATIC_VARS:
                    future_env[var] = env_current[var]
                else:
                    fpath = os.path.join(out_dir, f"norway_{var}.tif")
                    if not os.path.exists(fpath):
                        print(f"  SKIP {combo} — {var} missing")
                        skip = True
                        break
                    with rasterio.open(fpath) as src:
                        future_env[var] = src.read(1).astype(np.float64)
            if skip:
                continue

            # Build pixel array
            pixel_data = np.column_stack([
                future_env[var][valid_rows, valid_cols] for var in expanded_vars
            ])
            row_valid = np.all(np.isfinite(pixel_data), axis=1)

            # Predict
            preds = predict_all(pixel_data[row_valid])

            # Save ensemble raster
            ens_raster = np.full(ref_shape, np.nan)
            ens_raster[valid_rows[row_valid], valid_cols[row_valid]] = preds["Ensemble"]

            out_path = os.path.join(ENS_PROJ_DIR, f"ens_{gcm}_{ssp}_{period}.tif")
            out_meta = ref_meta.copy()
            out_meta.update(dtype="float64", nodata=np.nan, compress="deflate", count=1)
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(ens_raster, 1)

            vals = preds["Ensemble"]
            mean_s = np.mean(vals)
            pct_h = np.sum(vals >= 0.6) / len(vals) * 100
            all_results.append({
                "gcm": gcm, "ssp": ssp, "period": period,
                "mean_suit": mean_s, "pct_high": pct_h,
            })
            print(f"  {combo}: mean={mean_s:.3f}, High+VHigh={pct_h:.1f}%")

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(ENS_PROJ_DIR, "ensemble_future_summary.csv"), index=False)
print(f"\n{results_df.to_string(index=False)}")

# ---------------------------------------------------------------------------
# STEP 4 – Compute ensemble means per SSP × period
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Computing multi-GCM ensemble means...")

ensemble_futures = {}
for ssp in SSPS:
    for period in PERIODS:
        key = f"{ssp}_{period}"
        rasters = []
        for gcm in GCMS:
            path = os.path.join(ENS_PROJ_DIR, f"ens_{gcm}_{ssp}_{period}.tif")
            if os.path.exists(path):
                with rasterio.open(path) as src:
                    rasters.append(src.read(1).astype(np.float64))
        if rasters:
            stack = np.stack(rasters)
            ensemble_futures[key] = np.nanmean(stack, axis=0)
            print(f"  {key}: {len(rasters)} GCMs averaged")

# ---------------------------------------------------------------------------
# STEP 5 – Publication figures
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

# --- Fig: 2×2 ensemble future suitability ---
fig = plt.figure(figsize=(14, 16))
gs = gridspec.GridSpec(2, 2, hspace=0.15, wspace=0.05)

panel_labels = [
    ("ssp245_2041-2060", "SSP2-4.5, 2041\u20132060"),
    ("ssp585_2041-2060", "SSP5-8.5, 2041\u20132060"),
    ("ssp245_2061-2080", "SSP2-4.5, 2061\u20132080"),
    ("ssp585_2061-2080", "SSP5-8.5, 2061\u20132080"),
]

for idx, (key, title) in enumerate(panel_labels):
    if key not in ensemble_futures:
        continue
    ax = make_map_ax(fig, gs[idx])
    data = np.ma.masked_invalid(ensemble_futures[key])
    im = ax.imshow(data, extent=img_extent, origin="upper",
                   transform=ccrs.PlateCarree(), cmap="cividis",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=12, fontweight="bold")

cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.015])
fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
             label="Ensemble habitat suitability (4 algorithms × 3 GCMs)")
fig.suptitle("Future habitat suitability — multi-algorithm, multi-GCM ensemble\n"
             "Lupinus polyphyllus in Norway",
             fontsize=14, fontweight="bold", y=0.95)

path_a = os.path.join(FIG_DIR, "ensemble_future_suitability.png")
fig.savefig(path_a, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved {path_a}")

# --- Fig: Change maps (ensemble future – ensemble current) ---
fig = plt.figure(figsize=(14, 16))
gs = gridspec.GridSpec(2, 2, hspace=0.15, wspace=0.05)

for idx, (key, title) in enumerate(panel_labels):
    if key not in ensemble_futures:
        continue
    ax = make_map_ax(fig, gs[idx])
    change = ensemble_futures[key] - current_ens_raster
    change_masked = np.ma.masked_invalid(change)
    im = ax.imshow(change_masked, extent=img_extent, origin="upper",
                   transform=ccrs.PlateCarree(), cmap="RdBu_r",
                   vmin=-0.4, vmax=0.4, interpolation="nearest")
    ax.set_title(f"\u0394 Suitability: {title}", fontsize=11, fontweight="bold")

cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.015])
fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
             label="Change in ensemble suitability (future \u2212 current)")
fig.suptitle("Projected change in ensemble habitat suitability\n"
             "under CMIP6 scenarios",
             fontsize=14, fontweight="bold", y=0.95)

path_b = os.path.join(FIG_DIR, "ensemble_future_change.png")
fig.savefig(path_b, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved {path_b}")

# --- Fig: Bar plot comparing current vs future suitable area ---
fig, ax = plt.subplots(figsize=(10, 6))

ens_bars = []
for ssp in SSPS:
    for period in PERIODS:
        subset = results_df[(results_df["ssp"] == ssp) & (results_df["period"] == period)]
        if len(subset) == 0:
            continue
        ens_bars.append({
            "label": ssp.upper() + "\n" + period,
            "ssp": ssp,
            "mean_pct": subset["pct_high"].mean(),
            "sd_pct": subset["pct_high"].std(),
        })
ens_df = pd.DataFrame(ens_bars)

current_pct = results_df[results_df["gcm"] == "Current"]["pct_high"].values[0]
colors = {"ssp245": "#2166ac", "ssp585": "#b2182b"}
x_pos = np.arange(len(ens_df))
ax.bar(x_pos, ens_df["mean_pct"], yerr=ens_df["sd_pct"], capsize=5, width=0.6,
       color=[colors[s] for s in ens_df["ssp"]], edgecolor="black", linewidth=0.5)
ax.axhline(y=current_pct, color="black", linestyle="--", linewidth=1.5,
           label=f"Current ensemble baseline ({current_pct:.1f}%)")
ax.set_xticks(x_pos)
ax.set_xticklabels(ens_df["label"], fontsize=10)
ax.set_ylabel("% of land area with High + Very High suitability", fontsize=11)
ax.set_title("Projected ensemble invasion risk under CMIP6\n"
             "(mean ± SD across 3 GCMs × 4 algorithms)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(0, max(ens_df["mean_pct"].max() + 15, current_pct + 15))
fig.tight_layout()

path_c = os.path.join(FIG_DIR, "ensemble_future_barplot.png")
fig.savefig(path_c, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved {path_c}")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("ENSEMBLE FUTURE PROJECTIONS COMPLETE")
print(f"{'='*60}")
print(f"\nResults:")
print(results_df.to_string(index=False))
print(f"\nFigures: {path_a}, {path_b}, {path_c}")
