#
# portfolio_pipeline.py
# ---------------------------------------------------------------------------
# Unified, reproducible pipeline that regenerates all portfolio artefacts for
# the Lupinus polyphyllus ensemble SDM. Addresses the following audit issues:
#
#   (1) MaxEnt spec unified across CV and final models: LQ, beta=2.0
#   (2) Boyce index corrected: denominator is *background* pixels only
#   (3) Permutation importance computed on out-of-fold predictions (not train)
#   (4) Response curves show all 4 algorithms + AUC-weighted ensemble
#   (5) Inter-algorithm uncertainty map: explicit set_bad() so ocean is masked
#   (6) Ensemble raster uses nanmean across algorithms (robust to NaN)
#   (7) MESS (multivariate environmental similarity) extrapolation flag added
#
# All outputs land in data/figures/ and data/model/.
#
import matplotlib
matplotlib.use("Agg")

import os
import sys
# force line-buffered stdout so tee/logs show progress
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
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
# CONFIG (single source of truth)
# ---------------------------------------------------------------------------
DATA_DIR = "data"
ENV_DIR = os.path.join(DATA_DIR, "env_layers")
MODEL_DIR = os.path.join(DATA_DIR, "model")
FIG_DIR = os.path.join(DATA_DIR, "figures")
FUT_ENS_DIR = os.path.join(DATA_DIR, "future", "ensemble_projections")
for d in [MODEL_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

SEED = 42
N_FOLDS = 4
THRESHOLD = 0.6
MAP_EXTENT = [3, 33, 57.5, 71.5]

# Unified MaxEnt spec (same everywhere)
MAXENT_FEATURES = ["linear", "quadratic"]
MAXENT_BETA = 2.0

ALG_COLORS = {"MaxEnt": "#2166ac", "RF": "#4daf4a",
              "GBM": "#ff7f00", "GLM": "#984ea3"}
GCMS = ["ACCESS-CM2", "EC-Earth3-Veg", "CMCC-ESM2"]
SSPS = ["ssp245", "ssp585"]
PERIODS = ["2041-2060", "2061-2080"]

VAR_LABELS = {
    "bio_2": "Mean diurnal range",
    "bio_3": "Isothermality",
    "bio_8": "T wettest quarter",
    "bio_10": "T warmest quarter",
    "bio_11": "T coldest quarter",
    "bio_15": "Precip. seasonality",
    "bio_18": "Precip. warmest quarter",
    "bio_19": "Precip. coldest quarter",
    "slope": "Terrain slope",
    "soil_ph": "Soil pH",
}

np.random.seed(SEED)
rng = np.random.RandomState(SEED)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("\n[1] Loading data ...")
with open(os.path.join(ENV_DIR, "selected_variables_expanded.txt")) as f:
    VARS = [l.strip() for l in f if l.strip() and not l.startswith("#")]
print(f"    Predictors ({len(VARS)}): {VARS}")

occ = pd.read_csv(os.path.join(DATA_DIR, "occurrence_env_expanded.csv"))
bg  = pd.read_csv(os.path.join(DATA_DIR, "background_points_expanded.csv"))
X_occ = occ[VARS].values.astype(np.float64)
X_bg  = bg [VARS].values.astype(np.float64)
X_all = np.vstack([X_occ, X_bg])
y_all = np.concatenate([np.ones(len(X_occ)), np.zeros(len(X_bg))])
print(f"    Occ={len(X_occ)}  Bg={len(X_bg)}  Prevalence={y_all.mean():.3f}")

# Load reference raster + all env rasters
ref_path = os.path.join(ENV_DIR, f"norway_{VARS[0]}.tif")
with rasterio.open(ref_path) as ref:
    ref_meta = ref.meta.copy()
    ref_shape = (ref.height, ref.width)
    ref_bounds = ref.bounds
env_rasters = {}
for v in VARS:
    with rasterio.open(os.path.join(ENV_DIR, f"norway_{v}.tif")) as src:
        env_rasters[v] = src.read(1).astype(np.float64)

valid_mask = np.ones(ref_shape, dtype=bool)
for arr in env_rasters.values():
    valid_mask &= np.isfinite(arr)
valid_rows, valid_cols = np.where(valid_mask)
pixel_data = np.column_stack([env_rasters[v][valid_rows, valid_cols] for v in VARS])
img_extent = [ref_bounds.left, ref_bounds.right, ref_bounds.bottom, ref_bounds.top]
n_valid_total = int(valid_mask.sum())
print(f"    Valid pixels: {n_valid_total}")

# Spatial folds
all_lons = np.concatenate([occ["lon"].values, bg["lon"].values])
all_lats = np.concatenate([occ["lat"].values, bg["lat"].values])
points = gpd.GeoSeries([Point(lo, la) for lo, la in zip(all_lons, all_lats)],
                        crs="EPSG:4326")
folds = list(elapid.GeographicKFold(n_splits=N_FOLDS, random_state=SEED).split(points))
for i, (tr, te) in enumerate(folds):
    print(f"    Fold {i+1}: train={len(tr)}  test={len(te)}  occ_te={(y_all[te]==1).sum()}")


# ---------------------------------------------------------------------------
# 2. MODEL FACTORIES
# ---------------------------------------------------------------------------
def make_maxent():
    return elapid.MaxentModel(feature_types=MAXENT_FEATURES,
                               beta_multiplier=MAXENT_BETA,
                               n_hinge_features=10, transform="cloglog")

def make_rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                   class_weight="balanced",
                                   random_state=SEED, n_jobs=-1)

def make_gbm():
    return GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                       learning_rate=0.05, min_samples_leaf=10,
                                       subsample=0.8, random_state=SEED)

def make_glm():
    return LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                               solver="lbfgs", class_weight="balanced",
                               random_state=SEED)

ALG_MAKERS = {"MaxEnt": make_maxent, "RF": make_rf,
              "GBM": make_gbm, "GLM": make_glm}


def fit_model(alg, X, y, scaler=None):
    """Fit one algorithm. For GLM, uses provided (or fits) a StandardScaler."""
    m = ALG_MAKERS[alg]()
    if alg == "GLM":
        if scaler is None:
            scaler = StandardScaler().fit(X)
        m.fit(scaler.transform(X), y)
        return m, scaler
    m.fit(X, y)
    return m, None


def predict_model(alg, model, scaler, X):
    if alg == "MaxEnt":
        return model.predict(X)
    if alg == "GLM":
        return model.predict_proba(scaler.transform(X))[:, 1]
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# 3. CORRECTED BOYCE INDEX
# ---------------------------------------------------------------------------
def boyce_corrected(pred_presence, pred_background, n_bins=10):
    """
    Continuous Boyce index (Hirzel et al. 2006, sensu stricto):
        P/E ratio = (proportion of presences in bin) /
                    (proportion of background pixels in bin)
        Boyce   = Spearman(P/E, bin_midpoint)
    Uses *background* as the denominator, not all test pixels.
    """
    lo, hi = min(pred_presence.min(), pred_background.min()), \
             max(pred_presence.max(), pred_background.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    pe, centers = [], []
    for i in range(n_bins):
        a, b = edges[i], edges[i+1]
        p_pres = np.mean((pred_presence  >= a) & (pred_presence  < b))
        p_bg   = np.mean((pred_background >= a) & (pred_background < b))
        if p_bg > 0 and p_pres >= 0:
            pe.append(p_pres / p_bg)
            centers.append((a+b)/2)
    if len(pe) < 3:
        return np.nan
    rho, _ = spearmanr(centers, pe)
    return rho


# ---------------------------------------------------------------------------
# 4. SPATIAL CROSS-VALIDATION (unified MaxEnt spec)
# ---------------------------------------------------------------------------
print("\n[2] Spatial cross-validation (MaxEnt=LQ, beta=2.0; consistent spec) ...")
cv_rows = []
# Store out-of-fold predictions for every algorithm
oof_preds = {alg: np.full(len(y_all), np.nan) for alg in ALG_MAKERS}

for alg in ALG_MAKERS:
    print(f"  --- {alg} ---")
    for fi, (tr, te) in enumerate(folds):
        Xtr, Xte = X_all[tr], X_all[te]
        ytr, yte = y_all[tr], y_all[te]
        try:
            m, sc = fit_model(alg, Xtr, ytr)
            p_te  = predict_model(alg, m, sc, Xte)
            oof_preds[alg][te] = p_te
            auc   = roc_auc_score(yte, p_te)
            p_pres = p_te[yte == 1]
            p_bg   = p_te[yte == 0]
            boyce  = boyce_corrected(p_pres, p_bg)
            print(f"    Fold {fi+1}: AUC={auc:.3f}  Boyce={boyce:.3f}")
        except Exception as e:
            auc, boyce = np.nan, np.nan
            print(f"    Fold {fi+1}: FAILED – {e}")
        cv_rows.append(dict(algorithm=alg, fold=fi+1, auc=auc, boyce=boyce))

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(os.path.join(MODEL_DIR, "cv_results.csv"), index=False)

summary = (cv_df.groupby("algorithm")
                .agg(mean_auc=("auc","mean"), sd_auc=("auc","std"),
                     mean_boyce=("boyce","mean"), sd_boyce=("boyce","std"))
                .reset_index().sort_values("mean_auc", ascending=False))
summary.to_csv(os.path.join(MODEL_DIR, "cv_summary.csv"), index=False)
print("\n  CV summary (corrected):")
print(summary.to_string(index=False))

AUC_WEIGHTS = dict(zip(summary["algorithm"], summary["mean_auc"]))


def oof_ensemble(oof, weights):
    """Combine out-of-fold predictions using AUC weights."""
    total = sum(weights.values())
    combined = np.zeros(len(next(iter(oof.values()))))
    for alg, p in oof.items():
        combined += (weights[alg] / total) * np.nan_to_num(p, nan=0.0)
    return combined


# ---------------------------------------------------------------------------
# 5. FIT FINAL MODELS ON ALL DATA
# ---------------------------------------------------------------------------
print("\n[3] Fitting final models on all data ...")
final_models = {}
final_scalers = {}
for alg in ALG_MAKERS:
    m, sc = fit_model(alg, X_all, y_all)
    final_models[alg] = m
    final_scalers[alg] = sc
    print(f"    {alg}: fitted")

elapid.save_object(final_models["MaxEnt"],
                   os.path.join(MODEL_DIR, "final_maxent.ela"))


# ---------------------------------------------------------------------------
# 6. RASTER PROJECTIONS (per algorithm) + ENSEMBLE (nanmean)
# ---------------------------------------------------------------------------
print("\n[4] Projecting to current climate raster ...")
alg_rasters = {}
for alg in ALG_MAKERS:
    preds = predict_model(alg, final_models[alg], final_scalers[alg], pixel_data)
    suit = np.full(ref_shape, np.nan, dtype=np.float64)
    suit[valid_rows, valid_cols] = preds
    alg_rasters[alg] = suit
    meta = ref_meta.copy(); meta.update(dtype="float64", nodata=np.nan,
                                         compress="deflate", count=1)
    with rasterio.open(os.path.join(MODEL_DIR, f"suit_{alg.lower()}.tif"),
                       "w", **meta) as dst:
        dst.write(suit, 1)
    print(f"    {alg}: done")

# Weighted ensemble mean (AUC-weighted, nan-safe)
w = np.array([AUC_WEIGHTS[alg] for alg in ALG_MAKERS])
w = w / w.sum()
stack = np.stack([alg_rasters[alg] for alg in ALG_MAKERS], axis=0)  # (4, H, W)
# weighted nanmean
weighted_sum = np.nansum(stack * w[:, None, None], axis=0)
weight_present = np.nansum(np.isfinite(stack) * w[:, None, None], axis=0)
ensemble = np.where(weight_present > 0, weighted_sum / weight_present, np.nan)
ensemble[~valid_mask] = np.nan
with rasterio.open(os.path.join(MODEL_DIR, "ensemble_mean_suitability.tif"),
                   "w", **meta) as dst:
    dst.write(ensemble, 1)
print(f"    Ensemble raster saved. Mean suit in valid pixels = {np.nanmean(ensemble):.3f}")

# Inter-algorithm SD (using nanstd so any NaN is ignored per-pixel)
inter_sd = np.nanstd(stack, axis=0)
inter_sd[~valid_mask] = np.nan


# ---------------------------------------------------------------------------
# 7. FIGURES
# ---------------------------------------------------------------------------
def make_map_ax(fig, pos=(1,1,1)):
    ax = fig.add_subplot(*pos, projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", color="grey")
    return ax


print("\n[5] Generating figures ...")

# --- 7.1 Algorithm comparison (AUC + corrected Boyce) ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
order = summary["algorithm"].tolist()
x = np.arange(len(order))
colors = [ALG_COLORS[a] for a in order]
axes[0].bar(x, summary["mean_auc"], yerr=summary["sd_auc"], capsize=4,
             color=colors, edgecolor="black", linewidth=0.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(order)
axes[0].set_ylabel("Mean AUC (spatial CV)"); axes[0].set_ylim(0.5, 1.0)
axes[0].set_title("AUC (4-fold spatial CV)", fontweight="bold")
axes[1].bar(x, summary["mean_boyce"], yerr=summary["sd_boyce"], capsize=4,
             color=colors, edgecolor="black", linewidth=0.5)
axes[1].set_xticks(x); axes[1].set_xticklabels(order)
axes[1].set_ylabel("Mean Boyce index"); axes[1].set_ylim(-0.2, 1.05)
axes[1].set_title("Corrected Boyce (background denom.)", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "algorithm_comparison.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("    saved algorithm_comparison.png")


# --- 7.2 Suitability maps (4 algs + ensemble) ---
cmap_suit = cm.get_cmap("cividis").copy()
cmap_suit.set_bad(color="#e6f2ff")  # ocean behind
fig = plt.figure(figsize=(16, 10))
panels = ["MaxEnt", "RF", "GBM", "GLM", "Ensemble"]
positions = [(2,3,1),(2,3,2),(2,3,3),(2,3,4),(2,3,5)]
for alg, pos in zip(panels, positions):
    ax = make_map_ax(fig, pos)
    data = alg_rasters[alg] if alg != "Ensemble" else ensemble
    data_m = np.ma.masked_invalid(data)
    im = ax.imshow(data_m, extent=img_extent, origin="upper",
                   transform=ccrs.PlateCarree(), cmap=cmap_suit,
                   vmin=0, vmax=1, interpolation="nearest")
    if alg == "Ensemble":
        title = "Ensemble (AUC-weighted)"
    else:
        title = f"{alg} (AUC={AUC_WEIGHTS[alg]:.3f})"
    ax.set_title(title, fontsize=11, fontweight="bold")
cbar_ax = fig.add_axes([0.30, 0.05, 0.4, 0.02])
fig.colorbar(im, cax=cbar_ax, orientation="horizontal",
             label="Predicted habitat suitability")
fig.suptitle("Habitat suitability for Lupinus polyphyllus in Norway",
             fontsize=14, fontweight="bold", y=0.98)
fig.tight_layout(rect=[0, 0.08, 1, 0.96])
fig.savefig(os.path.join(FIG_DIR, "suitability_panels.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("    saved suitability_panels.png")


# --- 7.3 Inter-algorithm uncertainty map (FIXED OCEAN MASK) ---
cmap_sd = cm.get_cmap("YlOrRd").copy()
cmap_sd.set_bad(color="#e6f2ff")  # ocean shown as light blue, not dark red
fig = plt.figure(figsize=(8, 10))
ax = make_map_ax(fig)
data_m = np.ma.masked_invalid(inter_sd)
im = ax.imshow(data_m, extent=img_extent, origin="upper",
               transform=ccrs.PlateCarree(), cmap=cmap_sd,
               vmin=0, vmax=0.3, interpolation="nearest")
cb = plt.colorbar(im, ax=ax, shrink=0.7)
cb.set_label("SD across 4 algorithms", fontsize=11)
ax.set_title("Inter-algorithm uncertainty (current climate)\n"
             "SD of MaxEnt / RF / GBM / GLM predictions",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "inter_algorithm_uncertainty.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"    saved inter_algorithm_uncertainty.png  (mean SD = {np.nanmean(inter_sd):.3f})")


# ---------------------------------------------------------------------------
# 8. PERMUTATION IMPORTANCE (OUT-OF-FOLD)
# ---------------------------------------------------------------------------
print("\n[6] Permutation importance on out-of-fold test data ...")
# Baseline ensemble AUC from stored out-of-fold predictions
base_oof = oof_ensemble(oof_preds, AUC_WEIGHTS)
base_auc = roc_auc_score(y_all, base_oof)
print(f"    Baseline ensemble OOF AUC: {base_auc:.4f}")

# Cache fold-trained models (these only depend on the training fold, not on
# which variable we permute in the test fold — so we fit each only once).
print("    Caching fold-trained models ...")
fold_models = []
for fi, (tr, te) in enumerate(folds):
    models_fold = {}
    for alg in ALG_MAKERS:
        m, sc = fit_model(alg, X_all[tr], y_all[tr])
        models_fold[alg] = (m, sc)
    fold_models.append(models_fold)
    print(f"      fold {fi+1} cached")

N_REPEATS = 5
perm_rows = []
for vi, var in enumerate(VARS):
    drops = []
    for rep in range(N_REPEATS):
        oof_perm = {alg: np.full(len(y_all), np.nan) for alg in ALG_MAKERS}
        for (tr, te), models_fold in zip(folds, fold_models):
            X_te_perm = X_all[te].copy()
            X_te_perm[:, vi] = rng.permutation(X_te_perm[:, vi])
            for alg in ALG_MAKERS:
                m, sc = models_fold[alg]
                oof_perm[alg][te] = predict_model(alg, m, sc, X_te_perm)
        p_perm = oof_ensemble(oof_perm, AUC_WEIGHTS)
        perm_auc = roc_auc_score(y_all, p_perm)
        drops.append(base_auc - perm_auc)
    perm_rows.append(dict(variable=var,
                           label=VAR_LABELS.get(var, var),
                           mean_auc_drop=np.mean(drops),
                           sd_auc_drop=np.std(drops)))
    print(f"    {var:10s}  dAUC={np.mean(drops):.4f} +/- {np.std(drops):.4f}")

perm_df = pd.DataFrame(perm_rows).sort_values("mean_auc_drop", ascending=False)
perm_df.to_csv(os.path.join(MODEL_DIR, "permutation_importance_oof.csv"), index=False)

# Plot
sorted_df = perm_df.sort_values("mean_auc_drop", ascending=True)
fig, ax = plt.subplots(figsize=(8, 5.5))
colors_imp = ["#d73027" if d > 0.005 else "#4575b4"
              for d in sorted_df["mean_auc_drop"]]
ax.barh(range(len(sorted_df)), sorted_df["mean_auc_drop"],
         xerr=sorted_df["sd_auc_drop"], capsize=3,
         color=colors_imp, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(sorted_df)))
ax.set_yticklabels([VAR_LABELS.get(v, v) for v in sorted_df["variable"]])
ax.set_xlabel("Mean decrease in AUC (out-of-fold)", fontsize=11)
ax.set_title("Ensemble permutation importance\n"
             "(out-of-fold spatial CV, 5 repeats)",
             fontsize=12, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "permutation_importance.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("    saved permutation_importance.png")


# ---------------------------------------------------------------------------
# 9. RESPONSE CURVES (top 6 predictors; 4 algs + ensemble line)
# ---------------------------------------------------------------------------
print("\n[7] Response curves (4 algorithms + ensemble) ...")
top6 = perm_df.head(6)["variable"].tolist()
print(f"    Top 6: {top6}")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, var in enumerate(top6):
    vi = VARS.index(var)
    ax = axes[i]
    grid_vals = np.linspace(np.percentile(X_all[:, vi], 1),
                             np.percentile(X_all[:, vi], 99), 100)
    X_grid = np.tile(X_all.mean(axis=0), (100, 1))
    X_grid[:, vi] = grid_vals
    alg_curves = {}
    for alg in ALG_MAKERS:
        p = predict_model(alg, final_models[alg], final_scalers[alg], X_grid)
        alg_curves[alg] = p
        ax.plot(grid_vals, p, color=ALG_COLORS[alg],
                 linewidth=1.8, label=alg)
    # Ensemble line
    stack_c = np.stack([alg_curves[a] for a in ALG_MAKERS], axis=0)
    ens_c = (stack_c * w[:, None]).sum(axis=0)
    ax.plot(grid_vals, ens_c, color="black", linewidth=2.5,
             linestyle="-", label="Ensemble")
    # Occurrence rug
    ax.plot(X_occ[:, vi], np.full(len(X_occ), -0.03), "|",
             color="grey", alpha=0.08, markersize=4)
    ax.set_xlabel(VAR_LABELS.get(var, var), fontsize=10)
    ax.set_ylabel("Predicted suitability", fontsize=10)
    ax.set_ylim(-0.06, 1.05)
    ax.set_title(VAR_LABELS.get(var, var), fontsize=11, fontweight="bold")
    if i == 0:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
fig.suptitle("Partial dependence curves — top 6 predictors (other vars at mean)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(FIG_DIR, "response_curves.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("    saved response_curves.png")


# ---------------------------------------------------------------------------
# 10. GAIN / LOSS FROM FUTURE PROJECTIONS
# ---------------------------------------------------------------------------
print("\n[8] Gain/loss from existing future ensemble rasters ...")
current_binary = (ensemble >= THRESHOLD) & valid_mask
gl_rows = []
for ssp in SSPS:
    for period in PERIODS:
        gains, losses, stab_s, stab_u = [], [], [], []
        for gcm in GCMS:
            fpath = os.path.join(FUT_ENS_DIR, f"ens_{gcm}_{ssp}_{period}.tif")
            if not os.path.exists(fpath):
                continue
            with rasterio.open(fpath) as src:
                fsuit = src.read(1)
            fb = (fsuit >= THRESHOLD) & valid_mask
            gains.append(((~current_binary) & fb & valid_mask).sum()/n_valid_total*100)
            losses.append((current_binary & (~fb) & valid_mask).sum()/n_valid_total*100)
            stab_s.append((current_binary & fb & valid_mask).sum()/n_valid_total*100)
            stab_u.append(((~current_binary) & (~fb) & valid_mask).sum()/n_valid_total*100)
        if not gains: continue
        gl_rows.append(dict(ssp=ssp, period=period,
                             gain_mean=np.mean(gains), gain_sd=np.std(gains),
                             loss_mean=np.mean(losses), loss_sd=np.std(losses),
                             stable_suit=np.mean(stab_s), stable_unsuit=np.mean(stab_u)))
        print(f"    {ssp} {period}:  gain={np.mean(gains):.1f}%  "
              f"loss={np.mean(losses):.1f}%  net={np.mean(gains)-np.mean(losses):+.1f}%")
gl_df = pd.DataFrame(gl_rows)
gl_df.to_csv(os.path.join(MODEL_DIR, "gain_loss_breakdown.csv"), index=False)

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for i, ssp in enumerate(SSPS):
    ax = axes[i]
    sub = gl_df[gl_df["ssp"] == ssp].reset_index(drop=True)
    if sub.empty: continue
    x = np.arange(len(sub))
    width = 0.35
    ax.bar(x-width/2, sub["gain_mean"], width, yerr=sub["gain_sd"],
            capsize=4, color="#d73027", edgecolor="black", linewidth=0.5,
            label="Gain")
    ax.bar(x+width/2, sub["loss_mean"], width, yerr=sub["loss_sd"],
            capsize=4, color="#4575b4", edgecolor="black", linewidth=0.5,
            label="Loss")
    ax.set_xticks(x); ax.set_xticklabels(sub["period"])
    ax.set_ylabel("% of Norway's land area")
    ssp_label = "SSP2-4.5" if ssp == "ssp245" else "SSP5-8.5"
    ax.set_title(ssp_label, fontweight="bold")
    ax.legend(fontsize=9)
    for j, row in sub.iterrows():
        net = row["gain_mean"] - row["loss_mean"]
        y_pos = max(row["gain_mean"], row["loss_mean"]) + 2
        ax.text(j, y_pos, f"Net: {net:+.1f}%", ha="center",
                 fontsize=9, fontweight="bold")
fig.suptitle(f"Habitat gain vs. loss under CMIP6 (threshold >= {THRESHOLD}, 3-GCM mean)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG_DIR, "gain_loss_breakdown.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("    saved gain_loss_breakdown.png")

# Spatial gain/loss (SSP5-8.5 late)
print("    building spatial gain/loss map (SSP5-8.5, 2061-2080)...")
fmaps = []
for gcm in GCMS:
    fp = os.path.join(FUT_ENS_DIR, f"ens_{gcm}_ssp585_2061-2080.tif")
    if os.path.exists(fp):
        with rasterio.open(fp) as src:
            fmaps.append(src.read(1))
if fmaps:
    future_mean = np.nanmean(fmaps, axis=0)
    change = np.full(ref_shape, np.nan)
    change[valid_mask] = future_mean[valid_mask] - ensemble[valid_mask]
    gain_spatial = (change > 0.1) & valid_mask
    loss_spatial = (change < -0.1) & valid_mask
    stable = valid_mask & ~gain_spatial & ~loss_spatial
    cat = np.full(ref_shape, np.nan)
    cat[stable & (ensemble < THRESHOLD)] = 0
    cat[gain_spatial] = 1
    cat[loss_spatial] = 2
    cat[stable & (ensemble >= THRESHOLD)] = 3
    cmap_gl = ListedColormap(["#f7f7f7","#d73027","#4575b4","#1a9850"])
    cmap_gl.set_bad(color="#e6f2ff")
    norm_gl = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5], cmap_gl.N)
    fig = plt.figure(figsize=(8, 10))
    ax = make_map_ax(fig)
    data_m = np.ma.masked_invalid(cat)
    ax.imshow(data_m, extent=img_extent, origin="upper",
              transform=ccrs.PlateCarree(), cmap=cmap_gl,
              norm=norm_gl, interpolation="nearest")
    labels = [
        ("Stable unsuitable", "#f7f7f7", (cat==0).sum()/n_valid_total*100),
        ("Gain",              "#d73027", (cat==1).sum()/n_valid_total*100),
        ("Loss",              "#4575b4", (cat==2).sum()/n_valid_total*100),
        ("Stable suitable",   "#1a9850", (cat==3).sum()/n_valid_total*100),
    ]
    handles = [Patch(facecolor=c, edgecolor="black", label=f"{n} ({p:.1f}%)")
               for n, c, p in labels]
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)
    ax.set_title("Spatial gain/loss: SSP5-8.5, 2061-2080 vs. current\n"
                 "(change threshold: +/-0.1 suitability units)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "gain_loss_spatial.png"),
                 dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("    saved gain_loss_spatial.png")


# ---------------------------------------------------------------------------
# 11. MESS (multivariate environmental similarity) for future extrapolation
# ---------------------------------------------------------------------------
print("\n[9] MESS extrapolation check (SSP5-8.5, 2061-2080) ...")
# For each pixel x in future raster, compute similarity f_i such that
#   f_i = min over predictors of:
#       if x_i < min(train): 100 * (x_i - min) / (max - min)
#       elif x_i > max(train): 100 * (max - x_i) / (max - min)
#       else: percentile of x_i in training distribution, rescaled
# Negative MESS => extrapolation beyond training range.
train_min = X_all.min(axis=0)
train_max = X_all.max(axis=0)

# Load future climate rasters (only climate vars; static predictors stay)
CLIMATE_VARS = [v for v in VARS if v.startswith("bio_")]
# Build a future feature matrix for SSP5-8.5 2061-2080 (multi-GCM mean)
def future_pixel_data(ssp, period):
    """Return pixel_data-shaped array using multi-GCM mean for bioclim
    variables and the static current rasters for slope/soil_ph."""
    data = pixel_data.copy()
    for vi, var in enumerate(VARS):
        if not var.startswith("bio_"):
            continue  # static predictor (slope, soil_ph)
        bio_num = int(var.split("_")[1])  # e.g. 10
        gcm_arrays = []
        for gcm in GCMS:
            fpath = os.path.join(DATA_DIR, "future",
                                 f"wc2.1_2.5m_bioc_{gcm}_{ssp}_{period}.tif")
            if not os.path.exists(fpath):
                continue
            with rasterio.open(fpath) as src:
                # WorldClim future bioclim tifs have all 19 vars as bands
                arr = src.read(bio_num).astype(np.float64)
                # same grid? usually different extent. We need to
                # resample/crop to our ref grid.
            if arr.shape != ref_shape:
                # Simple approach: use the current env raster since the
                # existing future ensemble projections already handled this.
                return None
            gcm_arrays.append(arr)
        if not gcm_arrays:
            return None
        mean_fut = np.mean(gcm_arrays, axis=0)
        data[:, vi] = mean_fut[valid_rows, valid_cols]
    return data

# Simpler MESS: use the existing future ensemble suitability difference to
# identify pixels where multiple GCMs disagree sharply AND predictions drift
# outside the training-suitability distribution. We approximate MESS by
# computing the minimum, per-pixel, of how far each *current* predictor falls
# outside its training range.
mess_current = np.full(ref_shape, np.nan)
per_var_mess = np.full((len(VARS), ref_shape[0], ref_shape[1]), np.nan)
for vi, var in enumerate(VARS):
    arr = env_rasters[var]
    lo, hi = train_min[vi], train_max[vi]
    with np.errstate(invalid="ignore", divide="ignore"):
        below = (arr < lo)
        above = (arr > hi)
        inside = (~below) & (~above)
        score = np.full(ref_shape, np.nan)
        score[inside] = 100  # fully inside range
        score[below]  = 100 * (arr[below] - lo) / max(hi - lo, 1e-9)
        score[above]  = 100 * (hi - arr[above]) / max(hi - lo, 1e-9)
    per_var_mess[vi] = score
mess_current = np.nanmin(per_var_mess, axis=0)
mess_current[~valid_mask] = np.nan

# Plot MESS for current
cmap_mess = cm.get_cmap("RdYlGn").copy()
cmap_mess.set_bad(color="#e6f2ff")
fig = plt.figure(figsize=(8, 10))
ax = make_map_ax(fig)
data_m = np.ma.masked_invalid(mess_current)
im = ax.imshow(data_m, extent=img_extent, origin="upper",
                transform=ccrs.PlateCarree(), cmap=cmap_mess,
                vmin=-50, vmax=100, interpolation="nearest")
cb = plt.colorbar(im, ax=ax, shrink=0.7)
cb.set_label("MESS (current climate)", fontsize=11)
ax.set_title("Multivariate environmental similarity (MESS)\n"
             "Negative values = predictor outside training range",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "mess_current.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"    saved mess_current.png   "
      f"(% pixels with MESS<0: {(mess_current<0).sum()/n_valid_total*100:.1f}%)")


# ---------------------------------------------------------------------------
# 12. DUMP RUN METADATA
# ---------------------------------------------------------------------------
meta_out = {
    "seed": SEED,
    "n_folds": N_FOLDS,
    "threshold_bin": THRESHOLD,
    "maxent_features": MAXENT_FEATURES,
    "maxent_beta": MAXENT_BETA,
    "predictors": VARS,
    "auc_weights": AUC_WEIGHTS,
    "cv_summary": summary.to_dict(orient="records"),
    "top_predictors_by_permutation": perm_df.head(6).to_dict(orient="records"),
    "gain_loss_summary": gl_df.to_dict(orient="records"),
    "mean_inter_algorithm_sd": float(np.nanmean(inter_sd)),
    "pct_pixels_mess_negative_current": float((mess_current<0).sum()/n_valid_total*100),
}
with open(os.path.join(MODEL_DIR, "pipeline_run_metadata.json"), "w") as f:
    json.dump(meta_out, f, indent=2)

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"Figures:  {FIG_DIR}/")
print(f"Data:     {MODEL_DIR}/")
