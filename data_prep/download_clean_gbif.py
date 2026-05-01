# 
# Download and clean GBIF occurrence data for Lupinus polyphyllus in Norway.

# This script uses pygbif to fetch records, applies basic quality filters,
# removes duplicates, and spatially thins points to reduce sampling bias.
# The output is saved as a CSV and GeoPackage for further modeling steps.

# Adapt it step-by-step in a notebook or script; print intermediate
# results to inspect the cleaning progress.


import os
import pandas as pd
import geopandas as gpd
from pygbif import occurrences
from shapely.geometry import Point

# 1. define query parameters ------------------------------------------------
SPECIES = "Lupinus polyphyllus"
COUNTRY = "NO"  # ISO2 code for Norway
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. download occurrences from GBIF -----------------------------------------
# we'll request a reasonably large page size and loop until all records
records = []
offset = 0
limit = 300  # gbif max is 300

print("Querying GBIF for occurrences...")
while True:
    res = occurrences.search(
        scientificName=SPECIES,
        country=COUNTRY,
        hasCoordinate=True,
        limit=limit,
        offset=offset,
    )
    batch = res.get("results", [])
    if not batch:
        break
    records.extend(batch)
    offset += limit
    print(f"  downloaded {len(records)} records...")
    if offset >= res.get("count", 0):
        break

print(f"Finished download: {len(records)} total records")

# convert to DataFrame ------------------------------------------------------
df = pd.json_normalize(records)

# 3. basic quality filtering -------------------------------------------------
# remove any with flagged issues, zero coordinates, or huge uncertainty
if "hasGeospatialIssues" in df.columns:
    df = df[~df["hasGeospatialIssues"]]

# drop records without lat/long
for col in ["decimalLatitude", "decimalLongitude"]:
    df = df.dropna(subset=[col])

# remove obviously wrong coords (0,0)
df = df[~((df.decimalLatitude == 0) & (df.decimalLongitude == 0))]

# optionally filter by coordinateUncertaintyInMeters
if "coordinateUncertaintyInMeters" in df.columns:
    df = df[df.coordinateUncertaintyInMeters <= 5000]
    # choose a threshold that makes sense for your data

# restrict to field observations and museum specimens (exclude cultivated)
if "basisOfRecord" in df.columns:
    allowed_basis = {"HUMAN_OBSERVATION", "PRESERVED_SPECIMEN"}
    before = len(df)
    df = df[df["basisOfRecord"].isin(allowed_basis)]
    print(f"  Removed {before - len(df)} records with excluded basisOfRecord")

# remove absence records
if "occurrenceStatus" in df.columns:
    before = len(df)
    df = df[df["occurrenceStatus"] == "PRESENT"]
    print(f"  Removed {before - len(df)} absence records")

print(f"Records after quality filtering: {len(df)}")

# 4. drop duplicates (exact lat/long repetition) ----------------------------
df = df.drop_duplicates(subset=["decimalLatitude", "decimalLongitude"])
print(f"Records after de-duplication: {len(df)}")

# 5. convert to GeoDataFrame -------------------------------------------------
geometry = [Point(xy) for xy in zip(df.decimalLongitude, df.decimalLatitude)]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# 6. optional spatial thinning
# simple approach: use scikit-learn's BallTree or geopandas sjoin
# here we demonstrate a naive 1-km grid-based thinning

# create a buffer grid and sample one point per cell
cell_size = 0.01  # ≈1 km at mid-latitudes
xmin, ymin, xmax, ymax = gdf.total_bounds
grid_cells = []
x = xmin
while x < xmax:
    y = ymin
    while y < ymax:
        grid_cells.append(Point(x, y).buffer(cell_size / 2))
        y += cell_size
    x += cell_size

grid = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")
# use the modern `predicate` parameter instead of deprecated `op`
# https://geopandas.org/en/stable/docs/reference/api/geopandas.sjoin.html
joined = gpd.sjoin(gdf, grid, how="left", predicate="within")
thinned = joined.drop_duplicates(subset=["index_right"])
thinned = thinned.reset_index(drop=True)
print(f"Records after simple grid thinning: {len(thinned)}")

# use `thinned` as the cleaned set to save
final_gdf = thinned.copy()

# 7. save cleaned points -----------------------------------------------------
clean_csv = os.path.join(OUTPUT_DIR, "lupinus_polyphyllus_no_clean.csv")
clean_gpkg = os.path.join(OUTPUT_DIR, "lupinus_polyphyllus_no_clean.gpkg")
final_gdf.to_csv(clean_csv, index=False)
final_gdf.to_file(clean_gpkg, layer="occurrences", driver="GPKG")

print(f"Saved cleaned data to {clean_csv} and {clean_gpkg}")

# 8. quick sanity check (plot with geopandas) --------------------------------
# wrap plotting in try/except so script still runs on minimal installs
# and guard against geopandas versions without `datasets`.

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # (optional) if you have a country basemap available, plot it here;
    # removing geopandas.datasets avoids Pylance warnings.  We'll just draw points.
    # e.g. world = gpd.read_file("path/to/norway.shp")
    # world.plot(ax=ax, color="lightgrey")
    final_gdf.plot(ax=ax, markersize=5, color="red")
    ax.set_title("Cleaned Lupinus polyphyllus occurrences in Norway")
    plt.show()
except ImportError:
    # matplotlib missing; skip plotting silently
    pass

# next steps:
# - inspect the output CSV/GeoPackage
# - consider using specialized packages like `spatial_thinning` or
#   `blockCV` equivalence for more rigorous bias handling
# - prepare environmental layers for modeling

