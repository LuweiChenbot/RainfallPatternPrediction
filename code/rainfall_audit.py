"""Audit Singapore rainfall data without modifying or removing observations.

Usage (activate your virtual environment with PySpark installed first):
python code/rainfall_audit.py --data-dir data --output-dir audit_results

Only summary tables are exported. Each run creates a separate output directory.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession, functions as F


SCHEMA = """date DATE, timestamp TIMESTAMP, update_timestamp TIMESTAMP,
station_id STRING, station_name STRING, station_device_id STRING,
location_longitude DOUBLE, location_latitude DOUBLE,
reading_update_timestamp TIMESTAMP, reading_value DOUBLE,
reading_type STRING, reading_unit STRING"""
EXPECTED_COLUMNS = [field.strip().split()[0] for field in SCHEMA.split(",")]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("audit_results"))
    parser.add_argument("--master", default="local[4]")
    return parser.parse_args()


def count_if(condition, name):
    """Count matching rows by assigning 1 to a match and 0 otherwise."""
    return F.sum(F.when(condition, 1).otherwise(0)).alias(name)


def save_summary(frame, directory, name):
    """Collect only bounded summary tables, never the full observation dataset."""
    rows = frame.collect()
    with (directory / (name + ".csv")).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(frame.columns)
        writer.writerows(tuple(row) for row in rows)
    print(f"\n{name}: saved {len(rows)} summary rows", flush=True)
    for row in rows[:10]:
        print(row.asDict(), flush=True)


def main():
    args = parse_args()
    # Read exactly eight annual files to avoid including unrelated CSV files.
    files = [args.data_dir.resolve() / f"HistoricalRainfallacrossSingapore{year}.csv"
             for year in range(2017, 2025)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing annual file: {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            if next(csv.reader(handle), None) != EXPECTED_COLUMNS:
                raise ValueError(f"Unexpected column names or order: {path}")

    result_dir = args.output_dir.resolve() / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    result_dir.mkdir(parents=True, exist_ok=False)
    spark = (SparkSession.builder.appName("Singapore rainfall data audit")
             .master(args.master)
             .config("spark.sql.session.timeZone", "Asia/Singapore")
             .config("spark.ui.showConsoleProgress", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        # FAILFAST raises an error on malformed input instead of silently accepting it.
        data = (spark.read.option("header", True).option("mode", "FAILFAST")
                .schema(SCHEMA).csv([str(path) for path in files]))
        data = (data.withColumn("source_file", F.regexp_extract(F.input_file_name(), r"([^/]+)$", 1))
                .withColumn("source_year", F.regexp_extract("source_file", r"(\d{4})", 1).cast("int")))
        data = (data
                .withColumn(
                    "update_delay_seconds",
                    F.col("update_timestamp").cast("long") - F.col("timestamp").cast("long"))
                .withColumn(
                    "reading_update_delay_seconds",
                    F.col("reading_update_timestamp").cast("long") - F.col("timestamp").cast("long")))
        print("Auditing 2017–2024. Full scans and duplicate checks may take several minutes.", flush=True)
        data.printSchema()

        # Combine annual checks in one aggregation to reduce repeated file scans.
        missing_fields = {
            "timestamp": "missing_timestamp", "station_id": "missing_station",
            "reading_value": "missing_rainfall", "location_longitude": "missing_longitude",
            "location_latitude": "missing_latitude", "date": "missing_date",
            "update_timestamp": "missing_update_time",
            "reading_update_timestamp": "missing_reading_update_time",
            "station_name": "missing_station_name",
            "station_device_id": "missing_device_id",
        }
        annual = data.groupBy("source_year").agg(
            F.count("*").alias("rows"),
            F.countDistinct("station_id").alias("stations"),
            F.min("timestamp").alias("first_time"), F.max("timestamp").alias("last_time"),
            *[count_if(F.col(column).isNull(), label) for column, label in missing_fields.items()],
            F.min("reading_value").alias("minimum_rainfall"),
            F.max("reading_value").alias("maximum_rainfall"),
            count_if(F.col("reading_value") < 0, "negative_values"),
            count_if(F.isnan("reading_value") | F.col("reading_value").isin(float("inf"), float("-inf")), "nonfinite_rainfall"),
            count_if(F.trim(F.col("station_id")) == "", "blank_station"),
            count_if(F.trim(F.col("station_name")) == "", "blank_station_name"),
            count_if(F.trim(F.col("station_device_id")) == "", "blank_device_id"),
            F.countDistinct("reading_unit").alias("unit_types"),
            F.countDistinct("reading_type").alias("reading_types"),
            count_if(F.col("reading_unit").isNull(), "missing_unit"),
            count_if(F.col("reading_type").isNull(), "missing_reading_type"),
            count_if(F.col("date") != F.to_date("timestamp"), "date_mismatch"),
            count_if(F.year("timestamp") != F.col("source_year"), "year_mismatch"),
            count_if(F.col("update_delay_seconds") < 0, "update_before_observation"),
            count_if(F.col("reading_update_delay_seconds") < 0,
                     "reading_update_before_observation"),
            count_if(F.col("update_timestamp") != F.col("reading_update_timestamp"),
                     "different_update_times"),
            F.min("update_delay_seconds").alias("minimum_update_delay_seconds"),
            F.percentile_approx("update_delay_seconds", 0.5).alias("median_update_delay_seconds"),
            F.max("update_delay_seconds").alias("maximum_update_delay_seconds"),
        ).orderBy("source_year")
        save_summary(annual, result_dir, "yearly_quality")

        # Exclude source_year from the key to detect duplicates across files too.
        duplicates = (data.groupBy("station_id", "timestamp").count()
                      .filter(F.col("count") > 1))
        duplicate_summary = duplicates.agg(
            F.count("*").alias("duplicate_keys"),
            F.coalesce(F.sum(F.col("count") - 1), F.lit(0)).alias("extra_rows"))
        save_summary(duplicate_summary, result_dir, "duplicate_summary")
        save_summary(duplicates.orderBy("station_id", "timestamp").limit(20),
                     result_dir, "duplicate_examples")

        # One unit per year does not imply the same unit across all eight years.
        categories = data.groupBy("source_year").agg(
            F.sort_array(F.collect_set("reading_unit")).alias("units"),
            F.sort_array(F.collect_set("reading_type")).alias("types"),
        ).orderBy("source_year")
        save_summary(categories, result_dir, "units_and_reading_types")

        # Station ID is the stable identity. Names and coordinates can be revised.
        # The latest name is for reporting only and must not be used as a model feature.
        station_metadata = data.groupBy("station_id").agg(
            F.countDistinct("station_name").alias("name_versions"),
            F.countDistinct("station_device_id").alias("device_versions"),
            F.countDistinct(F.struct("location_longitude", "location_latitude"))
            .alias("location_versions"),
            F.max_by("station_name", "timestamp").alias("canonical_station_name"),
            F.max("timestamp").alias("latest_observation"),
        ).orderBy("station_id")
        save_summary(station_metadata, result_dir, "station_metadata")

        # Preserve original timestamps; inspect the early 2017 five-minute rhythm.
        early = (data.filter(F.col("source_year") == 2017).select("timestamp")
                 .distinct().orderBy("timestamp").limit(15))
        save_summary(early, result_dir, "early_2017_timestamps")

        # Coverage is diagnostic: missing observations must not be filled with zero.
        # This assumes a regular five-minute grid; irregular times make it approximate.
        stations = data.groupBy("source_year", "station_id").agg(
            F.count("*").alias("records"), F.min("timestamp").alias("first_time"),
            F.max("timestamp").alias("last_time"))
        stations = (stations.withColumn("expected_records_in_span",
                    F.floor((F.col("last_time").cast("long") - F.col("first_time").cast("long")) / 300) + 1)
                    .withColumn("coverage_ratio_in_span", F.round(F.col("records") / F.col("expected_records_in_span"), 4))
                    .orderBy("source_year", "station_id")
                    .cache())
        save_summary(stations, result_dir, "station_coverage")

        coverage_by_year = stations.groupBy("source_year").agg(
            F.count("*").alias("stations"),
            F.round(F.avg("coverage_ratio_in_span"), 4).alias("average_coverage"),
            F.round(F.percentile_approx("coverage_ratio_in_span", 0.5), 4)
            .alias("median_coverage"),
            count_if(F.col("coverage_ratio_in_span") < 0.8, "stations_below_80pct"),
            count_if(F.col("coverage_ratio_in_span") < 0.5, "stations_below_50pct"),
        ).orderBy("source_year")
        save_summary(coverage_by_year, result_dir, "coverage_by_year")
        stations.unpersist()
        print(f"\nAudit complete. Results: {result_dir}\nNo observations removed; original CSV files unchanged.", flush=True)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
