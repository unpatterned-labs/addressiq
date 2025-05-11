import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.fs as fs
import shapely.wkb
from typing import Tuple

def convert_map_to_list_of_structs(table: pa.Table) -> pa.Table:
    """Convert Map types to List[Struct] for Polars compatibility."""
    def _convert_type(data_type):
        if isinstance(data_type, pa.MapType):
            return pa.list_(pa.struct([
                pa.field("key", data_type.key_type),
                pa.field("value", data_type.item_type)
            ]))
        elif isinstance(data_type, pa.StructType):
            return pa.struct([pa.field(f.name, _convert_type(f.type)) for f in data_type])
        elif isinstance(data_type, pa.ListType):
            return pa.list_(_convert_type(data_type.value_type))
        return data_type

    new_schema = pa.schema([
        pa.field(field.name, _convert_type(field.type))
        for field in table.schema
    ])
    return table.cast(new_schema)

def parse_geometry(wkb_bytes: bytes) -> Tuple[float, float]:
    """Convert WKB geometry to (lon, lat) tuple"""
    geom = shapely.wkb.loads(wkb_bytes)
    return (geom.x, geom.y) if geom else (None, None)

def process_address_levels(address_levels: list) -> dict:
    """Unnest address levels into named columns"""
    return {f"admin_level_{i+1}": level["value"]
            for i, level in enumerate(address_levels[:5])}

def fetch_overture_data(s3_path: str, bbox: Tuple[float]) -> pl.DataFrame:
    """Fetch and process Overture Maps address data"""
    # Load data with bounding box filter
    dataset = ds.dataset(s3_path, filesystem=fs.S3FileSystem(anonymous=True))
    bbox_filter = (
        (pc.field("bbox", "xmin") < bbox[2]) &
        (pc.field("bbox", "xmax") > bbox[0]) &
        (pc.field("bbox", "ymin") < bbox[3]) &
        (pc.field("bbox", "ymax") > bbox[1])
    )

    table = dataset.to_table(filter=bbox_filter)
    if table.num_rows == 0:
        raise ValueError("No data found in bounding box")

    # Convert complex types and process
    converted_table = convert_map_to_list_of_structs(table)
    df = pl.from_arrow(converted_table)

    # Process geometry and address levels
    return df.with_columns(
        pl.col("geometry").map_elements(
            lambda x: parse_geometry(x)[0], return_dtype=pl.Float64
        ).alias("lon"),
        pl.col("geometry").map_elements(
            lambda x: parse_geometry(x)[1], return_dtype=pl.Float64
        ).alias("lat"),
        pl.col("address_levels").map_elements(
            lambda x: process_address_levels(x) if x else None
        ).struct.rename_fields([f"admin_level_{i+1}" for i in range(5)])
        .alias("address_levels"),
    ).unnest("address_levels")

# Example usage
if __name__ == "__main__":
    s3_path = "overturemaps-us-west-2/release/2023-04-23-beta.0/theme=addresses/type=address/"
    bbox = (-71.068, 42.353, -71.058, 42.363)  # Boston area

    df = fetch_overture_data(s3_path, bbox)

    # Select and rename columns to match target schema
    processed_df = df.select([
        "id", "lon", "lat",
        pl.col("country").alias("country"),
        pl.col("postcode").alias("postcode"),
        pl.col("street").alias("street"),
        pl.col("number").alias("number"),
        pl.col("unit").alias("unit"),
        *[pl.col(f"admin_level_{i+1}").alias(f"admin_{i+1}") for i in range(5)],
        pl.col("postal_city").alias("postal_city")
    ])

    print("Processed DataFrame Schema:")
    print(processed_df.schema)
    print("\nSample data:")
    print(processed_df.head(3))




def generate_us_address(df: pl.DataFrame) -> pl.DataFrame:
    """Generate full U.S. mailing addresses."""


    return (
        df
        # Clean text fields
        # Build address components
        .with_columns(
            street_part=pl.concat_str(
                pl.col("number"),
                pl.col("street"),
                separator=" "
            ).str.replace_all(r"\s+", " "),
            postal_part=pl.concat_str(
                pl.col("state"),
                pl.col("postcode"),
                separator=" "
            ).str.replace_all(r"\s+", " ")
        )
        # Final formatting
        .with_columns(
            full_address=pl.concat_str(
                pl.col("street_part"),
                pl.col("city"),
                pl.col("postal_part"),
                pl.col("country"),
                separator=", "
            )
            .str.replace(r",\s+,", ", ", literal=False)
            .str.strip_chars(", ")
        )
        # Select output columns
        .select(
            "full_address",
            "country",
            "lat",
            "lon"
        )
    )
