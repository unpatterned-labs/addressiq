import polars as pl
import pyarrow as pa
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.fs as fs
import pyarrow.parquet as pq
import shapely.wkb
import time


def convert_map_to_list_of_structs(table: pa.Table) -> pa.Table:
    """Convert all Map types in the table to List[Struct] for compatibility with Polars."""

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

    new_schema = [pa.field(field.name, _convert_type(field.type)) for field in table.schema]
    return table.cast(pa.schema(new_schema))


def fetch_overture_data(s3_path: str, bbox: tuple) -> pl.DataFrame:
    """Fetch data from Overture Maps within a bounding box and return as Polars DataFrame."""
    dataset = ds.dataset(s3_path, filesystem=fs.S3FileSystem(anonymous=True, region="us-west-2"))

    # Apply bounding box filter
    bbox_filter = (
            (pc.field("bbox", "xmin") < bbox[2]) &
            (pc.field("bbox", "xmax") > bbox[0]) &
            (pc.field("bbox", "ymin") < bbox[3]) &
            (pc.field("bbox", "ymax") > bbox[1])
    )

    batches = dataset.to_batches(filter=bbox_filter)
    non_empty_batches = [b for b in batches if b.num_rows > 0]

    if not non_empty_batches:
        raise ValueError("No data found within the specified bounding box.")

    # Convert to Polars DataFrame
    table = pa.Table.from_batches(non_empty_batches)
    converted_table = convert_map_to_list_of_structs(table)
    return pl.from_arrow(converted_table)


s3_path = "overturemaps-us-west-2/release/2025-04-23.0/theme=addresses/type=address/"
start_time = time.time()

## setting bounding box
bbox = (-1.921470205,52.4741678281,-1.9128475077,52.4784319842)

bbox = (-71.068,42.353,-71.058,42.363)
    ##(-2.1099092231, 52.3556570611, -1.6529557739, 52.5474833593)
print(f"📡 Fetching data for bbox: {bbox} ...")
df = fetch_overture_data(s3_path, bbox)

#(-1.921470205,52.4741678281,-1.9128475077,52.4784319842)
