#
# quicklook.py
# ---------------------------------------------------------------------------
# Lightweight "try your own invasive species" module.
#
# Reuses the Norway environmental raster stack already prepared for the main
# pipeline but runs a fast (~3-5 min) subset of the modelling:
#
#    GBIF download -> 1-km spatial thinning -> env extraction ->
#    KDE bias-corrected background -> MaxEnt + GLM with spatial CV ->
#    current-climate ensemble suitability map.
#
# Skips: RF, GBM, CMIP6 future projections, MESS. Those require longer runtime
# and/or per-species CMIP6 raster extraction. Use the full pipeline if you
# want them.
#
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from scipy.stats import gaussian_kde, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import elapid
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "data"
ENV_DIR = os.path.join(DATA_DIR, "env_layers")
SEED = 42
N_FOLDS = 4
MAP_EXTENT = [3, 33, 57.5, 71.5]

# Guardrails
MIN_OCC = 200    # below: spatial CV unreliable
MAX_OCC = 50000  # above: download is slow

# Suggested example species for demo
EXAMPLE_SPECIES = [
    "Lupinus polyphyllus",        # garden lupin (the main case study)
    "Heracleum mantegazzianum",   # giant hogweed
    "Rosa rugosa",                # Japanese rose
    "Impatiens glandulifera",     # Himalayan balsam
    "Solidago canadensis",        # Canadian goldenrod
]


def download_gbif(species, country="NO", max_records=MAX_OCC, verbose=True):
    """Query GBIF for a species+country, apply basic filters, return a DataFrame."""
    from pygbif import occurrences
    if verbose:
        print(f"[GBIF] Querying '{species}' in {country} ...")
    records, offset, limit = [], 0, 300
    while True:
        res = occurrences.search(
            scientificName=species, country=country, hasCoordinate=True,
            limit=limit, offset=offset,
        )
        batch = res.get("results", [])
        if not batch: break
        records.extend(batch)
        offset += limit
        if verbose and offset % 3000 == 0:
            print(f"  downloaded {len(records)}...")
        if len(records) >= max_records: break
        if offset >= res.get("count", 0): break
    if verbose:
        print(f"  total: {len(records)}")
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records)
    # Keep only useful columns if they exist
    keep = [c for c in ["decimalLatitude", "decimalLongitude",
                         "basisOfRecord", "occurrenceStatus",
                         "coordinateUncertaintyInMeters",
                         "hasGeospatialIssues", "year"]
            if c in df.columns]
    return df[keep].copy()


def clean_and_thin(df, thin_deg=0.01, verbose=True):
    """Apply quality filters and ~1 km grid thinning."""
    before = len(df)
    if "hasGeospatialIssues" in df.columns:
        df = df[~df["hasGeospatialIssues"]]
    df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])
    df = df[~((df.decimalLatitude == 0) & (df.decimalLongitude == 0))]
    if "coordinateUncertaintyInMeters" in df.columns:
        df = df[df.coordinateUncertaintyInMeters.fillna(0) <= 5000]
    if "basisOfRecord" in df.columns:
        df = df[df["basisOfRecord"].isin({"HUMAN_OBSERVATION", "PRESERVED_SPECIMEN"})]
    if "occurrenceStatus" in df.columns:
        df = df[df["occurrenceStatus"] == "PRESENT"]
    df = df.drop_duplicates(subset=["decimalLatitude", "decimalLongitude"])

    # Simple 1-km grid thinning: snap to grid, keep one per cell
    df = df.assign(
        _gx=(df.decimalLongitude / thin_deg).round().astype(int),
        _gy=(df.decimalLatitude  / thin_deg).round().astype(int),
    ).drop_duplicates(subset=["_gx", "_gy"]).drop(columns=["_gx", "_gy"])
    if verbose:
        print(f"[clean] kept {len(df)} / {before} after filtering + thinning")
    return df.rename(columns={"decimalLongitude": "lon",
                               "decimalLatitude": "lat"}).reset_index(drop=True)


def load_env_stack():
    """Load all env rasters + valid mask. Returns a dict of arrays + meta."""
    with open(os.path.join(ENV_DIR, "selected_variables_expanded.txt")) as f:
        variables = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    ref_path = os.path.join(ENV_DIR, f"norway_{variables[0]}.tif")
    with rasterio.open(ref_path) as ref:
        meta = ref.meta.copy()
        shape = (ref.height, ref.width)
        bounds = ref.bounds
        transform = ref.transform
    rasters = {}
    for v in variables:
        with rasterio.open(os.path.join(ENV_DIR, f"norway_{v}.tif")) as src:
            rasters[v] = src.read(1).astype(np.float64)
    valid_mask = np.ones(shape, dtype=bool)
    for arr in rasters.values():
        valid_mask &= np.isfinite(arr)
    return dict(variables=variables, rasters=rasters, valid_mask=valid_mask,
                shape=shape, bounds=bounds, transform=transform, meta=meta)


def extract_env_at_points(lons, lats, env):
    """Extract predictor values at point locations. Returns array (N, D) with NaN for invalid."""
    from rasterio.transform import rowcol
    rows, cols = rowcol(env["transform"], lons, lats)
    rows = np.asarray(rows); cols = np.asarray(cols)
    H, W = env["shape"]
    out = np.full((len(lons), len(env["variables"])), np.nan)
    inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    for i, v in enumerate(env["variables"]):
        arr = env["rasters"][v]
        vals = np.full(len(lons), np.nan)
        good = inside.copy()
        if good.any():
            vals[good] = arr[rows[good], cols[good]]
        out[:, i] = vals
    return out


def kde_background(occ_df, env, n_bg=5000, seed=SEED):
    """Sample n_bg background points weighted by a 2D KDE of occurrences.
    Returns (bg_df, X_bg)."""
    mask = env["valid_mask"]
    vrows, vcols = np.where(mask)
    transform = env["transform"]
    xs = transform.c + (vcols + 0.5) * transform.a
    ys = transform.f + (vrows + 0.5) * transform.e

    coords = np.vstack([occ_df["lon"].values, occ_df["lat"].values])
    kde = gaussian_kde(coords, bw_method="scott")
    probs = kde(np.vstack([xs, ys]))
    probs = probs / probs.sum()

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(vrows), size=n_bg, replace=True, p=probs)
    bg_lons = xs[idx]; bg_lats = ys[idx]
    bg_df = pd.DataFrame({"lon": bg_lons, "lat": bg_lats})
    X_bg = np.column_stack([env["rasters"][v][vrows[idx], vcols[idx]]
                             for v in env["variables"]])
    return bg_df, X_bg


def boyce_corrected(p_pres, p_bg, n_bins=10):
    lo = min(p_pres.min(), p_bg.min()); hi = max(p_pres.max(), p_bg.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    pe, c = [], []
    for i in range(n_bins):
        a, b = edges[i], edges[i+1]
        pp = np.mean((p_pres >= a) & (p_pres < b))
        pb = np.mean((p_bg   >= a) & (p_bg   < b))
        if pb > 0:
            pe.append(pp / pb); c.append((a+b)/2)
    if len(pe) < 3: return np.nan
    return spearmanr(c, pe)[0]


def run_quicklook(species, country="NO", n_bg=5000, verbose=True, show=True):
    """
    End-to-end quick-look modelling for an invasive species in Norway.
    Returns a dict with cv results, ensemble map, and key metrics.
    """
    # 1. Download + clean
    raw = download_gbif(species, country=country, verbose=verbose)
    if raw.empty:
        raise RuntimeError(f"GBIF returned zero records for '{species}' in {country}.")
    occ = clean_and_thin(raw, verbose=verbose)
    if len(occ) < MIN_OCC:
        raise RuntimeError(
            f"Only {len(occ)} occurrences after cleaning — need >= {MIN_OCC} "
            f"for reliable spatial cross-validation. Try a more common species."
        )
    if len(occ) > MAX_OCC:
        occ = occ.sample(MAX_OCC, random_state=SEED)

    # 2. Env extraction + filter invalid
    env = load_env_stack()
    X_occ = extract_env_at_points(occ["lon"].values, occ["lat"].values, env)
    keep = np.isfinite(X_occ).all(axis=1)
    occ = occ.loc[keep].reset_index(drop=True)
    X_occ = X_occ[keep]
    if len(occ) < MIN_OCC:
        raise RuntimeError(f"Only {len(occ)} occurrences inside env grid. Aborting.")
    if verbose:
        print(f"[env] {len(occ)} occurrences with valid predictors")

    # 3. Bias-corrected background
    bg, X_bg = kde_background(occ, env, n_bg=n_bg)
    X_all = np.vstack([X_occ, X_bg])
    y_all = np.concatenate([np.ones(len(X_occ)), np.zeros(len(X_bg))])

    # 4. Spatial folds
    pts = gpd.GeoSeries(
        [Point(lo, la) for lo, la in
         zip(np.concatenate([occ["lon"].values, bg["lon"].values]),
             np.concatenate([occ["lat"].values, bg["lat"].values]))],
        crs="EPSG:4326",
    )
    folds = list(elapid.GeographicKFold(n_splits=N_FOLDS, random_state=SEED).split(pts))

    # 5. Spatial CV (MaxEnt + GLM only)
    cv_rows = []
    oof = {"MaxEnt": np.full(len(y_all), np.nan),
           "GLM":    np.full(len(y_all), np.nan)}
    for fi, (tr, te) in enumerate(folds):
        # MaxEnt
        m = elapid.MaxentModel(feature_types=["linear", "quadratic"],
                                beta_multiplier=2.0, n_hinge_features=10,
                                transform="cloglog")
        m.fit(X_all[tr], y_all[tr])
        p_mx = m.predict(X_all[te])
        oof["MaxEnt"][te] = p_mx
        # GLM
        sc = StandardScaler().fit(X_all[tr])
        g = LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                                solver="lbfgs", class_weight="balanced",
                                random_state=SEED)
        g.fit(sc.transform(X_all[tr]), y_all[tr])
        p_gl = g.predict_proba(sc.transform(X_all[te]))[:, 1]
        oof["GLM"][te] = p_gl
        for name, p in (("MaxEnt", p_mx), ("GLM", p_gl)):
            a = roc_auc_score(y_all[te], p)
            b = boyce_corrected(p[y_all[te] == 1], p[y_all[te] == 0])
            cv_rows.append(dict(algorithm=name, fold=fi+1, auc=a, boyce=b))
            if verbose:
                print(f"[cv] {name} fold {fi+1}: AUC={a:.3f}  Boyce={b:.3f}")
    cv = pd.DataFrame(cv_rows)
    summary = (cv.groupby("algorithm")
                 .agg(mean_auc=("auc","mean"), sd_auc=("auc","std"),
                      mean_boyce=("boyce","mean"), sd_boyce=("boyce","std"))
                 .reset_index())

    # 6. Fit final models & project
    mx = elapid.MaxentModel(feature_types=["linear", "quadratic"],
                             beta_multiplier=2.0, n_hinge_features=10,
                             transform="cloglog")
    mx.fit(X_all, y_all)
    sc_f = StandardScaler().fit(X_all)
    gl = LogisticRegression(penalty="l2", C=1.0, max_iter=2000,
                             solver="lbfgs", class_weight="balanced",
                             random_state=SEED)
    gl.fit(sc_f.transform(X_all), y_all)

    vrows, vcols = np.where(env["valid_mask"])
    pixel_data = np.column_stack([env["rasters"][v][vrows, vcols]
                                   for v in env["variables"]])
    p_mx = mx.predict(pixel_data)
    p_gl = gl.predict_proba(sc_f.transform(pixel_data))[:, 1]

    w_mx = summary.set_index("algorithm").loc["MaxEnt", "mean_auc"]
    w_gl = summary.set_index("algorithm").loc["GLM", "mean_auc"]
    w_sum = w_mx + w_gl
    ens = (w_mx * p_mx + w_gl * p_gl) / w_sum

    ensemble = np.full(env["shape"], np.nan)
    ensemble[vrows, vcols] = ens

    # 7. Map
    if show:
        _plot_quicklook_map(ensemble, env, species, summary)

    return dict(species=species, n_occurrences=len(occ),
                cv_results=cv, cv_summary=summary,
                ensemble_map=ensemble, env=env,
                occ=occ, bg=bg)


def _plot_quicklook_map(ensemble, env, species, summary):
    img_extent = [env["bounds"].left, env["bounds"].right,
                  env["bounds"].bottom, env["bounds"].top]
    cmap = cm.get_cmap("cividis").copy(); cmap.set_bad("#e6f2ff")
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="#e6f2ff", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle="--", color="grey")
    im = ax.imshow(np.ma.masked_invalid(ensemble), extent=img_extent,
                    origin="upper", transform=ccrs.PlateCarree(),
                    cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Predicted habitat suitability")
    sub = summary.set_index("algorithm")
    subtitle = (f"MaxEnt AUC={sub.loc['MaxEnt','mean_auc']:.3f}  "
                f"GLM AUC={sub.loc['GLM','mean_auc']:.3f}  "
                f"(4-fold spatial CV)")
    ax.set_title(f"Habitat suitability — {species}\n{subtitle}",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()
