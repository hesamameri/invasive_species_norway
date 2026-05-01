#
# Prepare environmental predictor layers for the Lupinus polyphyllus SDM.
#
# This script:
#   1. Downloads WorldClim 2.1 bioclimatic variables (bio1–bio19) at 2.5 arc-min
#   2. Crops each layer to the Norway extent (lon 4–32°E, lat 58–71°N)
#   3. Saves cropped layers as individual GeoTIFFs in data/env_layers/
#   4. Checks pairwise correlation and drops highly collinear variables (|r| > 0.7)
#   5. Saves a correlation heatmap and a final "selected variables" list
#
# Resolution note: 2.5 arc-min ≈ 4.5 km at 60°N — a good balance for a
# national-scale invasive-species model.

import os
import zipfile
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join("data", "env_layers")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# WorldClim 2.1 bioclimatic variables – 2.5 arc-min resolution
# Primary URL (current domain) + fallback (old domain)
WORLDCLIM_URLS = [
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_2.5m_bio.zip",
    "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_2.5m_bio.zip",
]
ZIP_PATH = os.path.join("data", "wc2.1_2.5m_bio.zip")

# Norway bounding box (slightly buffered)
NORWAY_BBOX = box(4.0, 58.0, 32.0, 71.0)

# Correlation threshold for dropping collinear variables
CORR_THRESHOLD = 0.7

# ---------------------------------------------------------------------------
# STEP 1  –  Download the WorldClim archive if it doesn't already exist
# ---------------------------------------------------------------------------
def download_worldclim():
    if os.path.exists(ZIP_PATH):
        print(f"ZIP already exists at {ZIP_PATH}, skipping download.")
        return

    last_error = None
    for url in WORLDCLIM_URLS:
        print(f"Trying to download from:\n  {url}")
        print("This file is ~300 MB — it may take a few minutes...")
        try:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
        except (requests.ConnectionError, requests.Timeout) as e:
            print(f"  ✗ Failed: {e}\n  Trying next mirror...")
            last_error = e
            continue

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(ZIP_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {pct:5.1f}% ({downloaded >> 20} MB / {total >> 20} MB)", end="")
        print("\nDownload complete.")
        return  # success

    # if we get here, all URLs failed
    print("\n" + "=" * 60)
    print("ERROR: Could not download from any mirror.")
    print("Please download the file manually:")
    for url in WORLDCLIM_URLS:
        print(f"  {url}")
    print(f"\nSave it as: {os.path.abspath(ZIP_PATH)}")
    print("Then re-run this script.")
    print("=" * 60)
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# STEP 2  –  Extract the .tif files from the zip
# ---------------------------------------------------------------------------
EXTRACT_DIR = os.path.join("data", "wc2.1_2.5m_bio")

def extract_worldclim():
    if os.path.isdir(EXTRACT_DIR) and any(f.endswith(".tif") for f in os.listdir(EXTRACT_DIR)):
        print(f"TIF files already extracted in {EXTRACT_DIR}, skipping.")
        return
    print("Extracting ZIP archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(EXTRACT_DIR)
    print(f"Extracted to {EXTRACT_DIR}")


# ---------------------------------------------------------------------------
# STEP 3  –  Crop each bioclim layer to the Norway extent and save
# ---------------------------------------------------------------------------
def crop_to_norway():
    tif_files = sorted(
        f for f in os.listdir(EXTRACT_DIR)
        if f.endswith(".tif") and "bio" in f.lower()
    )
    if not tif_files:
        raise FileNotFoundError(f"No bioclim TIFs found in {EXTRACT_DIR}")

    print(f"Found {len(tif_files)} bioclim layers. Cropping to Norway extent...")

    for fname in tif_files:
        src_path = os.path.join(EXTRACT_DIR, fname)
        dst_path = os.path.join(OUTPUT_DIR, fname.replace("wc2.1_2.5m_", "norway_"))

        with rasterio.open(src_path) as src:
            # clip to Norway bbox
            out_image, out_transform = rio_mask(
                src, [NORWAY_BBOX], crop=True, nodata=src.nodata
            )
            out_meta = src.meta.copy()
            out_meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "compress": "deflate",
            })
            with rasterio.open(dst_path, "w", **out_meta) as dst:
                dst.write(out_image)

        print(f"  Saved {dst_path}")

    print("All layers cropped and saved.")


# ---------------------------------------------------------------------------
# STEP 4  –  Check collinearity and select variables
# ---------------------------------------------------------------------------
def check_collinearity():
    cropped_files = sorted(
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".tif") and f.startswith("norway_bio")
    )
    if not cropped_files:
        raise FileNotFoundError("No cropped bioclim TIFs found in " + OUTPUT_DIR)

    # read all layers into a dict of 1-D arrays (valid pixels only)
    arrays = {}
    valid_mask = None
    for fname in cropped_files:
        with rasterio.open(os.path.join(OUTPUT_DIR, fname)) as src:
            data = src.read(1).astype(np.float64)
            nodata = src.nodata
        # build a common valid-pixel mask across all layers
        layer_valid = np.isfinite(data)
        if nodata is not None:
            layer_valid &= (data != nodata)
        if valid_mask is None:
            valid_mask = layer_valid.copy()
        else:
            valid_mask &= layer_valid
        arrays[fname] = data

    # flatten to 1-D using the common mask
    flat = {}
    for fname, data in arrays.items():
        label = fname.replace("norway_", "").replace(".tif", "")
        flat[label] = data[valid_mask]

    df = pd.DataFrame(flat)
    corr = df.corr()

    # ---- plot correlation heatmap ----
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".1f", cmap="RdBu_r", vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={"shrink": 0.75})
    ax.set_title("Pairwise Pearson correlation – bioclimatic variables (Norway)")
    fig.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"Saved correlation heatmap to {heatmap_path}")

    # ---- greedy variable selection (drop highly correlated) ----
    # keep variable with lowest mean absolute correlation when a pair exceeds threshold
    selected = list(corr.columns)
    dropped = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            vi = corr.columns[i]
            vj = corr.columns[j]
            if vi not in selected or vj not in selected:
                continue
            if abs(corr.loc[vi, vj]) > CORR_THRESHOLD:
                # drop the one with higher mean |r| to the rest
                mean_i = corr.loc[vi, selected].abs().mean()
                mean_j = corr.loc[vj, selected].abs().mean()
                to_drop = vi if mean_i > mean_j else vj
                selected.remove(to_drop)
                dropped.append(to_drop)

    print(f"\nCollinearity check (|r| > {CORR_THRESHOLD}):")
    print(f"  Dropped variables: {dropped}")
    print(f"  Selected variables ({len(selected)}): {selected}")

    # save the selection list
    sel_path = os.path.join(OUTPUT_DIR, "selected_variables.txt")
    with open(sel_path, "w") as f:
        f.write("# Selected bioclim variables after collinearity filtering\n")
        f.write(f"# Threshold: |r| > {CORR_THRESHOLD}\n")
        for v in selected:
            f.write(v + "\n")
    print(f"  Saved variable list to {sel_path}")

    return selected


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    download_worldclim()
    extract_worldclim()
    crop_to_norway()
    selected = check_collinearity()
    print("\n✓ Environmental layers are ready for modeling.")
    print(f"  Cropped GeoTIFFs:  {OUTPUT_DIR}/")
    print(f"  Selected predictors: {selected}")
    print("\nNext step: create bias / background sampling layer.")
