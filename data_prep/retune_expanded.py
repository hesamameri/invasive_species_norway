#
# Re-tune MaxEnt on the expanded 10-predictor set.
#
# The original tuning was on 7 bioclimatic variables only. This script
# repeats the grid search on the full 10-predictor set (8 bioclim + slope
# + soil_ph) to verify/update optimal hyperparameters.
#
# Also computes the data-driven max(sensitivity + specificity) threshold
# for risk classification, addressing reviewer critique of arbitrary
# equal-interval thresholds.
#

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from sklearn.metrics import roc_auc_score, roc_curve
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

EXPANDED_VARS_FILE = os.path.join(ENV_DIR, "selected_variables_expanded.txt")
SEED = 42
N_SPATIAL_FOLDS = 4

BETA_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
FEATURE_COMBOS = {
    "L":    ["linear"],
    "LQ":   ["linear", "quadratic"],
    "LQH":  ["linear", "quadratic", "hinge"],
    "LQHP": ["linear", "quadratic", "hinge", "product"],
}

# ---------------------------------------------------------------------------
# STEP 1 – Load data
# ---------------------------------------------------------------------------
print("Loading data...")

with open(EXPANDED_VARS_FILE) as f:
    expanded_vars = [l.strip() for l in f if l.strip() and not l.startswith("#")]
print(f"  Expanded predictors ({len(expanded_vars)}): {expanded_vars}")

occ = pd.read_csv(os.path.join(DATA_DIR, "occurrence_env_expanded.csv"))
bg = pd.read_csv(os.path.join(DATA_DIR, "background_points_expanded.csv"))

X_occ = occ[expanded_vars].values.astype(np.float64)
X_bg = bg[expanded_vars].values.astype(np.float64)
X_all = np.vstack([X_occ, X_bg])
y_all = np.concatenate([np.ones(len(X_occ)), np.zeros(len(X_bg))])

print(f"  Occurrences: {len(X_occ)}, Background: {len(X_bg)}")

# ---------------------------------------------------------------------------
# STEP 2 – Spatial folds
# ---------------------------------------------------------------------------
print("Building spatial folds...")
all_lons = np.concatenate([occ["lon"].values, bg["lon"].values])
all_lats = np.concatenate([occ["lat"].values, bg["lat"].values])
points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(all_lons, all_lats)],
                        crs="EPSG:4326")
gkf = elapid.GeographicKFold(n_splits=N_SPATIAL_FOLDS, random_state=SEED)
folds = list(gkf.split(points))
print(f"  {N_SPATIAL_FOLDS} spatial folds created")


# ---------------------------------------------------------------------------
# Boyce index helper
# ---------------------------------------------------------------------------
def boyce_index(preds_all, preds_presence, n_bins=10):
    bin_edges = np.linspace(preds_all.min(), preds_all.max(), n_bins + 1)
    pe_ratios, bin_centers = [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        prop_pred = np.mean((preds_all >= lo) & (preds_all < hi))
        prop_pres = np.mean((preds_presence >= lo) & (preds_presence < hi))
        if prop_pred > 0:
            pe_ratios.append(prop_pres / prop_pred)
            bin_centers.append((lo + hi) / 2)
    if len(pe_ratios) < 3:
        return np.nan
    rho, _ = spearmanr(bin_centers, pe_ratios)
    return rho


# ---------------------------------------------------------------------------
# STEP 3 – Grid search on EXPANDED predictors
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"GRID SEARCH: {len(BETA_MULTIPLIERS)} betas × {len(FEATURE_COMBOS)} features")
print(f"{'='*60}")

results = []
for feat_name, feat_types in FEATURE_COMBOS.items():
    for beta in BETA_MULTIPLIERS:
        fold_aucs = []
        fold_boyces = []
        for fold_i, (train_idx, test_idx) in enumerate(folds):
            X_tr, X_te = X_all[train_idx], X_all[test_idx]
            y_tr, y_te = y_all[train_idx], y_all[test_idx]

            try:
                m = elapid.MaxentModel(
                    feature_types=feat_types,
                    beta_multiplier=beta,
                    n_hinge_features=10,
                    transform="cloglog",
                )
                m.fit(X_tr, y_tr)
                preds = m.predict(X_te)
                auc = roc_auc_score(y_te, preds)
                boyce = boyce_index(preds, preds[y_te == 1])
            except Exception as e:
                auc = np.nan
                boyce = np.nan

            fold_aucs.append(auc)
            fold_boyces.append(boyce)

        mean_auc = np.nanmean(fold_aucs)
        mean_boyce = np.nanmean(fold_boyces)
        results.append({
            "features": feat_name, "beta": beta,
            "mean_auc": mean_auc, "sd_auc": np.nanstd(fold_aucs),
            "mean_boyce": mean_boyce,
        })
        print(f"  {feat_name:5s} β={beta:.1f}: AUC={mean_auc:.4f} ± {np.nanstd(fold_aucs):.4f}, "
              f"Boyce={mean_boyce:.3f}")

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(MODEL_DIR, "tuning_expanded_results.csv"), index=False)

# Best combo
best = res_df.loc[res_df["mean_auc"].idxmax()]
print(f"\n  BEST: {best['features']} β={best['beta']:.1f} → "
      f"AUC={best['mean_auc']:.4f}, Boyce={best['mean_boyce']:.3f}")

# Tuning heatmap
pivot = res_df.pivot(index="features", columns="beta", values="mean_auc")
pivot = pivot.reindex(["L", "LQ", "LQH", "LQHP"])
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
            cbar_kws={"label": "Mean spatial CV AUC"})
ax.set_title("MaxEnt tuning on expanded 10-predictor set", fontweight="bold")
ax.set_ylabel("Feature classes")
ax.set_xlabel("Regularization multiplier (β)")
fig.tight_layout()
heatmap_path = os.path.join(MODEL_DIR, "tuning_expanded_heatmap.png")
fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {heatmap_path}")

# ---------------------------------------------------------------------------
# STEP 4 – Refit best model, compute optimal threshold
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Refitting best model + computing data-driven threshold...")

best_feat = FEATURE_COMBOS[best["features"]]
best_beta = best["beta"]

model = elapid.MaxentModel(
    feature_types=best_feat,
    beta_multiplier=best_beta,
    n_hinge_features=10,
    transform="cloglog",
)
model.fit(X_all, y_all)
elapid.save_object(model, os.path.join(MODEL_DIR, "best_model_expanded.ela"))

# Compute max(sensitivity + specificity) threshold from CV predictions
all_preds = np.zeros(len(y_all))
for fold_i, (train_idx, test_idx) in enumerate(folds):
    m = elapid.MaxentModel(
        feature_types=best_feat,
        beta_multiplier=best_beta,
        n_hinge_features=10,
        transform="cloglog",
    )
    m.fit(X_all[train_idx], y_all[train_idx])
    all_preds[test_idx] = m.predict(X_all[test_idx])

# ROC-based optimal threshold
fpr, tpr, thresholds = roc_curve(y_all, all_preds)
# max(sensitivity + specificity) = max(tpr + (1-fpr)) = max(tpr - fpr)
j_index = tpr - fpr
optimal_idx = np.argmax(j_index)
optimal_threshold = thresholds[optimal_idx]
optimal_sens = tpr[optimal_idx]
optimal_spec = 1 - fpr[optimal_idx]

print(f"  Optimal threshold (max sens+spec): {optimal_threshold:.3f}")
print(f"    Sensitivity: {optimal_sens:.3f}")
print(f"    Specificity: {optimal_spec:.3f}")

# Also compute 10-percentile training presence threshold
pres_preds = all_preds[y_all == 1]
p10_threshold = np.percentile(pres_preds, 10)
print(f"  10th percentile training presence threshold: {p10_threshold:.3f}")

# Save thresholds
thresholds_df = pd.DataFrame([
    {"method": "max_sens_spec", "threshold": optimal_threshold,
     "sensitivity": optimal_sens, "specificity": optimal_spec},
    {"method": "p10_training_presence", "threshold": p10_threshold,
     "sensitivity": np.nan, "specificity": np.nan},
])
thresholds_df.to_csv(os.path.join(MODEL_DIR, "optimal_thresholds.csv"), index=False)
print(f"  Saved {os.path.join(MODEL_DIR, 'optimal_thresholds.csv')}")

# ---------------------------------------------------------------------------
# STEP 5 – Project to raster with both threshold schemes
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Projecting best model to raster...")

ref_path = os.path.join(ENV_DIR, f"norway_{expanded_vars[0]}.tif")
with rasterio.open(ref_path) as ref:
    ref_meta = ref.meta.copy()
    ref_shape = (ref.height, ref.width)

env_rasters = {}
for var in expanded_vars:
    with rasterio.open(os.path.join(ENV_DIR, f"norway_{var}.tif")) as src:
        env_rasters[var] = src.read(1).astype(np.float64)

valid_mask = np.ones(ref_shape, dtype=bool)
for arr in env_rasters.values():
    valid_mask &= np.isfinite(arr)
valid_rows, valid_cols = np.where(valid_mask)

pixel_data = np.column_stack([
    env_rasters[var][valid_rows, valid_cols] for var in expanded_vars
])

preds = model.predict(pixel_data)
suit = np.full(ref_shape, np.nan)
suit[valid_rows, valid_cols] = preds

# Binary suitable/unsuitable using optimal threshold
binary = np.full(ref_shape, np.nan)
binary[valid_rows, valid_cols] = (preds >= optimal_threshold).astype(float)

pct_suitable = np.nansum(binary) / np.sum(valid_mask) * 100
print(f"  Suitable area (max sens+spec threshold {optimal_threshold:.3f}): {pct_suitable:.1f}%")

# Save
out_meta = ref_meta.copy()
out_meta.update(dtype="float64", nodata=np.nan, compress="deflate", count=1)
with rasterio.open(os.path.join(MODEL_DIR, "suitability_expanded_best.tif"),
                   "w", **out_meta) as dst:
    dst.write(suit, 1)

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("RE-TUNING COMPLETE")
print(f"{'='*60}")
print(f"  Best config: {best['features']} β={best_beta}")
print(f"  Mean spatial CV AUC: {best['mean_auc']:.4f}")
print(f"  Optimal threshold (max sens+spec): {optimal_threshold:.3f}")
print(f"  10th-percentile presence threshold: {p10_threshold:.3f}")
print(f"  Suitable area (max sens+spec): {pct_suitable:.1f}%")
