"""Convert the validated Singapore rainfall CSV files to partitioned Parquet.

Run from the project root after activating the virtual environment:
python code/prepare_base_data.py \
    --data-dir data \
    --output-dir data/processed/rainfall_base

This script preserves all observations and original data fields. It adds source
provenance, writes Snappy-compressed Parquet partitioned by source year, and
verifies the output row counts. Existing output is never overwritten.
"""

import argparse
import csv
from pathlib import Path

from pyspark.sql import SparkSession, functions as F


SCHEMA = """date DATE, timestamp TIMESTAMP, update_timestamp TIMESTAMP,
station_id STRING, station_name STRING, station_device_id STRING,
location_longitude DOUBLE, location_latitude DOUBLE,
reading_update_timestamp TIMESTAMP, reading_value DOUBLE,
reading_type STRING, reading_unit STRING"""

EXPECTED_COLUMNS = [field.strip().split()[0] for field in SCHEMA.split(",")]

# These counts come from the completed data-quality audit for this assignment.
EXPECTED_YEAR_COUNTS = {
    2017: 5_256_106,
    2018: 4_594_246,
    2019: 5_026_862,
    2020: 6_334_473,
    2021: 7_044_646,
    2022: 7_133_230,
    2023: 6_742_890,
    2024: 6_406_260,
}

BASE_COLUMNS = [
    "source_year",
    "source_file",
    "date",
    "timestamp",
    "update_timestamp",
    "station_id",
    "station_name",
    "station_device_id",
    "location_longitude",
    "location_latitude",
    "reading_update_timestamp",
    "reading_value",
    "reading_type",
    "reading_unit",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/rainfall_base"),
    )
    parser.add_argument("--master", default="local[4]")
    return parser.parse_args()


def validate_input_files(data_dir):
    """Return the eight expected files after validating their CSV headers."""
    files = [
        data_dir.resolve() / f"HistoricalRainfallacrossSingapore{year}.csv"
        for year in range(2017, 2025)
    ]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing annual file: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            if next(csv.reader(handle), None) != EXPECTED_COLUMNS:
                raise ValueError(f"Unexpected column names or order: {path}")
    return files


def validate_output(parquet_data):
    """Verify the output schema and audited row counts without collecting rows."""
    expected_types = {
        "source_year": "int",
        "source_file": "string",
        "date": "date",
        "timestamp": "timestamp",
        "update_timestamp": "timestamp",
        "station_id": "string",
        "station_name": "string",
        "station_device_id": "string",
        "location_longitude": "double",
        "location_latitude": "double",
        "reading_update_timestamp": "timestamp",
        "reading_value": "double",
        "reading_type": "string",
        "reading_unit": "string",
    }
    actual_types = {field.name: field.dataType.simpleString()
                    for field in parquet_data.schema.fields}
    if actual_types != expected_types:
        raise RuntimeError(
            f"Parquet schema mismatch. Expected {expected_types}, got {actual_types}"
        )

    count_rows = parquet_data.groupBy("source_year").count().orderBy("source_year").collect()
    actual_counts = {row["source_year"]: row["count"] for row in count_rows}
    if actual_counts != EXPECTED_YEAR_COUNTS:
        raise RuntimeError(
            f"Parquet row-count mismatch. Expected {EXPECTED_YEAR_COUNTS}, "
            f"got {actual_counts}"
        )

    print("\nValidated Parquet row counts:", flush=True)
    for year, count in actual_counts.items():
        print(f"  {year}: {count:,}", flush=True)
    print(f"  Total: {sum(actual_counts.values()):,}", flush=True)


def main():
    args = parse_args()
    files = validate_input_files(args.data_dir)
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("Prepare Singapore rainfall base data")
        .master(args.master)
        .config("spark.sql.session.timeZone", "Asia/Singapore")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        raw = (
            spark.read
            .option("header", True)
            .option("mode", "FAILFAST")
            .schema(SCHEMA)
            .csv([str(path) for path in files])
        )

        base_data = (
            raw
            .withColumn(
                "source_file",
                F.regexp_extract(F.input_file_name(), r"([^/]+)$", 1),
            )
            .withColumn(
                "source_year",
                F.regexp_extract("source_file", r"(\d{4})", 1).cast("int"),
            )
            .select(*BASE_COLUMNS)
        )

        print(f"Writing partitioned Parquet to: {output_dir}", flush=True)
        (
            base_data.write
            .mode("errorifexists")
            .option("compression", "snappy")
            .partitionBy("source_year")
            .parquet(str(output_dir))
        )

        parquet_data = spark.read.parquet(str(output_dir))
        validate_output(parquet_data)
        print("\nBase-data preparation completed successfully.", flush=True)
        print(f"Output: {output_dir}", flush=True)
        print("No observations were removed or imputed.", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
