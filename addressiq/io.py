# addressiq/io.py
from __future__ import annotations
import pyarrow as pa, pyarrow.dataset as ds, pyarrow.compute as pc, pyarrow.fs as fs
import polars as pl
#import geopandas as gpd         # only for one CRS call
import geopolars as gpl         # alpha: pip install --pre geopolars

# ---------- internal helpers ---------- #
def _s3_dataset(path: str) -> ds.Dataset:
    """Return anon S3 dataset (Parquet)."""
    if not path.startswith("s3://"):
        raise ValueError("Path must start with s3://")
    s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
    return ds.dataset(path, filesystem=s3, format="parquet")

def _bbox_filter(bbox: tuple[float, float, float, float]):
    xmin, ymin, xmax, ymax = bbox
    return (
        (pc.field("bbox", "xmin") < xmax) &
        (pc.field("bbox", "xmax") > xmin) &
        (pc.field("bbox", "ymin") < ymax) &
        (pc.field("bbox", "ymax") > ymin)
    )

# ---------- public API ---------- #
def fetch_addresses(
    bbox: tuple[float, float, float, float],
    release: str = "2025-04-23.0",
    columns: list[str] | None = None,
) -> gpl.GeoDataFrame:
    """
    Return Overture *address* rows inside bbox as a **GeoPolars GeoDataFrame**.
    """
    path = (
        f"s3://overturemaps-us-west-2/release/{release}"
        "/theme=place/type=address/"
    )

    dataset = _s3_dataset(path)
    filt    = _bbox_filter(bbox)

    # Arrow ➜ Polars
    table   = dataset.to_table(filter=filt, columns=columns)
    df      = pl.from_arrow(table)

    # AddressIQ convention: geometry column stays WKB bytes called 'geometry'
    gdf = gpl.from_polars(df, geometry_column="geometry", crs="EPSG:4326")
    return gdf
