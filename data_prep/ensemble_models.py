#
# Ensemble Species Distribution Modeling
#
# Compares four algorithms using the expanded predictor set and spatial CV:
#   1. MaxEnt  (elapid)
#   2. Random Forest  (sklearn)
#   3. Gradient Boosted Trees  (sklearn)
#   4. Logistic Regression / GLM  (sklearn)
#
# Outputs:
#   data/model/ensemble_cv_results.csv     – per-fold AUC/Boyce per algorithm
#   data/model/ensemble_summary.csv        – mean metrics per algorithm
#   data/figures/ensemble_comparison.png    – bar chart comparing algorithms
#   data/figures/ensemble_suitability.png   – 4-panel suitability maps
#   data/model/ensemble_mean_suitability.tif – weighted ensemble mean
#   data/figures/ensemble_change_maps.png   – future projection comparison
#

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import elapid
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = "data"
ENV_DIR = os.path.join(DATA_DIR, "env_layers")
MODEL_DIR = os.path.join(DATA_DIR, "model")
FIG_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

OCC_CSV = os.path.join(DATA_DIR, "occurrence_env.csv")
BG_CSV = os.path.join(DATA_DIR, "background_points.csv")
EXPANDED_VARS_FILE = os.path.join(ENV_DIR, "selected_variables_expanded.txt")

SEED = 42
N_SPATIAL_FOLDS = 4
np.random.seed(SEED)

# MaxEnt best params from tuning
BEST_BETA = 3.0
BEST_FEATURES = ["linear"]

MAP_EXTENT = [3, 33, 57.5, 71.5]

# ---------------------------------------------------------------------------
# STEP 0 – Load expanded variable list
# ---------------------------------------------------------------------------
print("Loading configuration...")
with open(EXPANDED_VARS_FILE) as f:
    expanded_vars = [l.strip() for l in f if l.strip() and not l.startswith("#")]
print(f"  Expanded predictors ({len(expanded_vars)}): {expanded_vars}")

# Identify which variables are new (not in original CSVs)
ORIGINAL_VARS = ["bio_10", "bio_11", "bio_15", "bio_18", "bio_2", "bio_3", "bio_8"]
new_vars = [v for v in expanded_vars if v not in ORIGINAL_VARS]
print(f"  New variables to extract: {new_vars}")

# ---------------------------------------------------------------------------
# STEP 1 – Extract new predictor values at point locations
# ---------------------------------------------------------------------------
print("\nExtracting new predictor values at point locations...")
occ = pd.read_csv(OCC_CSV)
bg = pd.read_csv(BG_CSV)

for var in new_vars:
    raster_path = os.path.join(ENV_DIR, f"norway_{var}.tif")
    if not os.path.exists(raster_path):
        print(f"  WARNING: {raster_path} not found, skipping {var}")
        continue

    with rasterio.open(raster_path) as src:
        # Extract values at occurrence points
        occ_vals = []
        for _, row in occ.iterrows():
            try:
                r, c = src.index(row["lon"], row["lat"])
                if 0 <= r < src.height and 0 <= c < src.width:
                    val = src.read(1)[r, c]
                    occ_vals.append(float(val) if np.isfinite(val) else np.nan)
                else:
                    occ_vals.append(np.nan)
            except Exception:
                occ_vals.append(np.nan)
        occ[var] = occ_vals

        # Extract values at background points
        bg_vals = []
        for _, row in bg.iterrows():
            try:
                r, c = src.index(row["lon"], row["lat"])
                if 0 <= r < src.height and 0 <= c < src.width:
                    val = src.read(1)[r, c]
                    bg_vals.append(float(val) if np.isfinite(val) else np.nan)
                else:
                    bg_vals.append(np.nan)
            except Exception:
                bg_vals.append(np.nan)
        bg[var] = bg_vals

    print(f"  {var}: occ NaN={occ[var].isna().sum()}, bg NaN={bg[var].isna().sum()}")

# Drop rows with NaN in any expanded predictor
occ_clean = occ.dropna(subset=expanded_vars)
bg_clean = bg.dropna(subset=expanded_vars)
print(f"\n  After NaN removal: occ={len(occ_clean)} (was {len(occ)}), "
      f"bg={len(bg_clean)} (was {len(bg)})")

# Save expanded CSVs
occ_clean.to_csv(os.path.join(DATA_DIR, "occurrence_env_expanded.csv"), index=False)
bg_clean.to_csv(os.path.join(DATA_DIR, "background_points_expanded.csv"), index=False)

# ---------------------------------------------------------------------------
# STEP 2 – Prepare data arrays
# ---------------------------------------------------------------------------
print("\nPreparing data arrays...")
X_occ = occ_clean[expanded_vars].values.astype(np.float64)
X_bg = bg_clean[expanded_vars].values.astype(np.float64)
X_all = np.vstack([X_occ, X_bg])
y_all = np.concatenate([np.ones(len(X_occ)), np.zeros(len(X_bg))])

print(f"  X shape: {X_all.shape}, y shape: {y_all.shape}")
print(f"  Prevalence: {y_all.mean():.3f}")

# ---------------------------------------------------------------------------
# STEP 3 – Spatial folds
# ---------------------------------------------------------------------------
print("\nBuilding spatial folds...")
all_lons = np.concatenate([occ_clean["lon"].values, bg_clean["lon"].values])
all_lats = np.concatenate([occ_clean["lat"].values, bg_clean["lat"].values])
points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(all_lons, all_lats)],
                        crs="EPSG:4326")

gkf = elapid.GeographicKFold(n_splits=N_SPATIAL_FOLDS, random_state=SEED)
folds = list(gkf.split(points))

for i, (tr, te) in enumerate(folds):
    n_occ_test = (y_all[te] == 1).sum()
    print(f"  Fold {i+1}: train={len(tr)}, test={len(te)} (occ_test={n_occ_test})")


# ---------------------------------------------------------------------------
# Boyce index
# ---------------------------------------------------------------------------
def boyce_index(preds_all, preds_presence, n_bins=10):
    """Continuous Boyce index (Spearman correlation of P/E ratio)."""
    bin_edges = np.linspace(preds_all.min(), preds_all.max(), n_bins + 1)
    pe_ratios = []
    bin_centers = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        prop_predicted = np.mean((preds_all >= lo) & (preds_all < hi))
        prop_presence = np.mean((preds_presence >= lo) & (preds_presence < hi))
        if prop_predicted > 0:
            pe_ratios.append(prop_presence / prop_predicted)
            bin_centers.append((lo + hi) / 2)
    if len(pe_ratios) < 3:
        return np.nan
    rho, _ = spearmanr(bin_centers, pe_ratios)
    return rho


# ---------------------------------------------------------------------------
# STEP 4 – Define algorithms
# ---------------------------------------------------------------------------
def make_maxent():
    return elapid.MaxentModel(
        feature_types=BEST_FEATURES,
        beta_multiplier=BEST_BETA,
        n_hinge_features=10,
        transform="cloglog",
    )

def make_rf():
    return RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )

def make_gbm():
    return GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_samples_leaf=10, subsample=0.8, random_state=SEED,
    )

def make_glm():
    return LogisticRegression(
        penalty="l2", C=1.0, max_iter=2000, solver="lbfgs",
        class_weight="balanced", random_state=SEED,
    )


ALGORITHMS = {
    "MaxEnt": make_maxent,
    "RF": make_rf,
    "GBM": make_gbm,
    "GLM": make_glm,
}

# ---------------------------------------------------------------------------
# STEP 5 – Spatial cross-validation for all algorithms
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("SPATIAL CROSS-VALIDATION – 4 algorithms × 4 folds")
print(f"{'='*60}")

cv_results = []

# StandardScaler for GLM (fitted on training data per fold)
for alg_name, make_model in ALGORITHMS.items():
    print(f"\n--- {alg_name} ---")

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        # Scale data for GLM (other algorithms don't need it but it doesn't hurt)
        scaler = StandardScaler()
        if alg_name == "GLM":
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
        else:
            X_train_s = X_train
            X_test_s = X_test

        model = make_model()

        try:
            if alg_name == "MaxEnt":
                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)
            else:
                model.fit(X_train_s, y_train)
                preds = model.predict_proba(X_test_s)[:, 1]

            auc = roc_auc_score(y_test, preds)

            # Boyce on test presence vs all test predictions
            test_pres = preds[y_test == 1]
            boyce = boyce_index(preds, test_pres)

            print(f"  Fold {fold_i+1}: AUC={auc:.3f}, Boyce={boyce:.3f}")
        except Exception as e:
            auc = np.nan
            boyce = np.nan
            print(f"  Fold {fold_i+1}: FAILED – {e}")

        cv_results.append({
            "algorithm": alg_name, "fold": fold_i + 1,
            "auc": auc, "boyce": boyce,
        })

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv(os.path.join(MODEL_DIR, "ensemble_cv_results.csv"), index=False)

# Summary
summary = cv_df.groupby("algorithm").agg(
    mean_auc=("auc", "mean"), sd_auc=("auc", "std"),
    mean_boyce=("boyce", "mean"), sd_boyce=("boyce", "std"),
).reset_index()
summary = summary.sort_values("mean_auc", ascending=False)
summary.to_csv(os.path.join(MODEL_DIR, "ensemble_summary.csv"), index=False)

print(f"\n{'='*60}")
print("CV SUMMARY:")
print(summary.to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 6 – Fit final models on ALL data
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Fitting final models on all data...")

final_models = {}
scaler_all = StandardScaler()

for alg_name, make_model in ALGORITHMS.items():
    model = make_model()
    if alg_name == "GLM":
        X_scaled = scaler_all.fit_transform(X_all)
        model.fit(X_scaled, y_all)
    elif alg_name == "MaxEnt":
        model.fit(X_all, y_all)
    else:
        model.fit(X_all, y_all)
    final_models[alg_name] = model
    print(f"  {alg_name}: fitted")

# Save MaxEnt model
elapid.save_object(final_models["MaxEnt"],
                   os.path.join(MODEL_DIR, "best_model_expanded.ela"))

# ---------------------------------------------------------------------------
# STEP 7 – Project all models to current climate raster
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Projecting all models to current climate raster...")

# Load raster data
ref_path = os.path.join(ENV_DIR, f"norway_{expanded_vars[0]}.tif")
with rasterio.open(ref_path) as ref:
    ref_meta = ref.meta.copy()
    ref_shape = (ref.height, ref.width)
    ref_transform = ref.transform
    ref_bounds = ref.bounds

env_rasters = {}
for var in expanded_vars:
    with rasterio.open(os.path.join(ENV_DIR, f"norway_{var}.tif")) as src:
        env_rasters[var] = src.read(1).astype(np.float64)

valid_mask = np.ones(ref_shape, dtype=bool)
for arr in env_rasters.values():
    valid_mask &= np.isfinite(arr)
valid_rows, valid_cols = np.where(valid_mask)
n_valid = len(valid_rows)

pixel_data = np.column_stack([
    env_rasters[var][valid_rows, valid_cols] for var in expanded_vars
])
print(f"  Valid pixels: {n_valid}")

# Predict with each algorithm
suitability_maps = {}
for alg_name, model in final_models.items():
    print(f"  Projecting {alg_name}...")
    suit = np.full(ref_shape, np.nan, dtype=np.float64)

    if alg_name == "GLM":
        preds = model.predict_proba(scaler_all.transform(pixel_data))[:, 1]
    elif alg_name == "MaxEnt":
        preds = model.predict(pixel_data)
    else:
        preds = model.predict_proba(pixel_data)[:, 1]

    suit[valid_rows, valid_cols] = preds
    suitability_maps[alg_name] = suit

    # Save raster
    out_meta = ref_meta.copy()
    out_meta.update(dtype="float64", nodata=np.nan, compress="deflate", count=1)
    with rasterio.open(os.path.join(MODEL_DIR, f"suit_{alg_name.lower()}.tif"),
                       "w", **out_meta) as dst:
        dst.write(suit, 1)

# Compute weighted ensemble mean (weights = AUC)
auc_weights = {}
for _, row in summary.iterrows():
    auc_weights[row["algorithm"]] = row["mean_auc"]

total_weight = sum(auc_weights.values())
ensemble_mean = np.zeros(ref_shape, dtype=np.float64)
for alg_name, suit in suitability_maps.items():
    w = auc_weights[alg_name] / total_weight
    ensemble_mean += w * np.nan_to_num(suit, nan=0.0)
ensemble_mean[~valid_mask] = np.nan

suitability_maps["Ensemble"] = ensemble_mean

# Save ensemble raster
out_meta = ref_meta.copy()
out_meta.update(dtype="float64", nodata=np.nan, compress="deflate", count=1)
with rasterio.open(os.path.join(MODEL_DIR, "ensemble_mean_suitability.tif"),
                   "w", **out_meta) as dst:
    dst.write(ensemble_mean, 1)
print("  Ensemble mean saved")

# ---------------------------------------------------------------------------
# STEP 8 – Publication figures
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


# --- Fig A: Algorithm comparison bar chart ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# AUC
alg_order = summary["algorithm"].tolist()
colors = {"MaxEnt": "#2166ac", "RF": "#4daf4a", "GBM": "#ff7f00", "GLM": "#984ea3"}
x = np.arange(len(alg_order))

axes[0].bar(x, summary["mean_auc"], yerr=summary["sd_auc"], capsize=5, width=0.6,
            color=[colors.get(a, "grey") for a in alg_order],
            edgecolor="black", linewidth=0.5)
axes[0].set_xticks(x)
axes[0].set_xticklabels(alg_order, fontsize=11)
axes[0].set_ylabel("Mean AUC (spatial CV)", fontsize=11)
axes[0].set_title("AUC comparison", fontsize=12, fontweight="bold")
axes[0].set_ylim(0.5, 1.0)

# Boyce
axes[1].bar(x, summary["mean_boyce"], yerr=summary["sd_boyce"], capsize=5, width=0.6,
            color=[colors.get(a, "grey") for a in alg_order],
            edgecolor="black", linewidth=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(alg_order, fontsize=11)
axes[1].set_ylabel("Mean Boyce index (spatial CV)", fontsize=11)
axes[1].set_title("Boyce index comparison", fontsize=12, fontweight="bold")
axes[1].set_ylim(-0.5, 1.0)

fig.suptitle("Algorithm comparison – spatial cross-validation",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
path_comp = os.path.join(FIG_DIR, "ensemble_comparison.png")
fig.savefig(path_comp, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved {path_comp}")


# --- Fig B: 5-panel suitability maps (4 algorithms + ensemble) ---
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.1, wspace=0.05)

panel_order = ["MaxEnt", "RF", "GBM", "GLM", "Ensemble"]
for idx, alg_name in enumerate(panel_order):
    if idx < 3:
        ax = make_map_ax(fig, gs[0, idx])
    elif idx == 3:
        ax = make_map_ax(fig, gs[1, 0])
    else:
        ax = make_map_ax(fig, gs[1, 1])

    data = np.ma.masked_invalid(suitability_maps[alg_name])
    im = ax.imshow(data, extent=img_extent, origin="upper",
                   transform=ccrs.PlateCarree(), cmap="cividis",
                   vmin=0, vmax=1, interpolation="nearest")
    auc_str = f" (AUC={auc_weights.get(alg_name, 0):.3f})" if alg_name != "Ensemble" else " (weighted mean)"
    ax.set_title(f"{alg_name}{auc_str}", fontsize=11, fontweight="bold")

cbar_ax = fig.add_axes([0.15, 0.03, 0.7, 0.015])
fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
             label="Predicted habitat suitability")

fig.suptitle("Habitat suitability for Lupinus polyphyllus in Norway\n"
             "Individual algorithms and AUC-weighted ensemble",
             fontsize=14, fontweight="bold", y=0.97)

path_maps = os.path.join(FIG_DIR, "ensemble_suitability.png")
fig.savefig(path_maps, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"  Saved {path_maps}")


# --- Fig C: Variable importance (RF permutation) ---
if "RF" in final_models:
    rf = final_models["RF"]
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(expanded_vars)),
            importances[sorted_idx],
            color="#4daf4a", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(expanded_vars)))
    ax.set_yticklabels([expanded_vars[i] for i in sorted_idx], fontsize=10)
    ax.set_xlabel("Feature importance (Gini)", fontsize=11)
    ax.set_title("Random Forest variable importance (expanded predictor set)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path_imp = os.path.join(FIG_DIR, "rf_variable_importance.png")
    fig.savefig(path_imp, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path_imp}")


# ---------------------------------------------------------------------------
# STEP 9 – Ensemble statistics
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("ENSEMBLE MODELING COMPLETE")
print(f"{'='*60}")
print(f"\nAlgorithm performance (spatial CV):")
print(summary.to_string(index=False))

# Suitable area comparison
print(f"\nHigh+VeryHigh suitability area (≥0.6) by algorithm:")
for alg_name, suit in suitability_maps.items():
    vals = suit[valid_mask & np.isfinite(suit)]
    pct = np.sum(vals >= 0.6) / len(vals) * 100
    print(f"  {alg_name}: {pct:.1f}%")

print(f"\nOutputs:")
print(f"  {os.path.join(MODEL_DIR, 'ensemble_cv_results.csv')}")
print(f"  {os.path.join(MODEL_DIR, 'ensemble_summary.csv')}")
print(f"  {os.path.join(MODEL_DIR, 'ensemble_mean_suitability.tif')}")
print(f"  {path_comp}")
print(f"  {path_maps}")
