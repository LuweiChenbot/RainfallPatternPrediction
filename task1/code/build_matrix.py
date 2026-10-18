"""Build a dense five-minute station matrix from the aligned rainfall Parquet.

Spark reads, validates and pivots the 48.5 million aligned readings into one
row per five-minute slot (Asia/Singapore, 2017-01-01 00:00 to 2024-12-31 23:55)
and one column per station. A missing reading stays null; it is never zero.

The Task 1 analyses (lagged cross-correlation, rain-event segmentation and
multi-resolution aggregation) are matrix operations, which numpy performs on
this array in seconds, whereas pairwise self-joins in Spark SQL would not be
practical on a laptop. Spark still performs every row-level step.

Run from the repository root:
python task1/code/build_matrix.py
"""

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

SGT = timezone(timedelta(hours=8))
ORIGIN = datetime(2017, 1, 1, tzinfo=SGT)
ORIGIN_EPOCH = int(ORIGIN.timestamp())
SLOT_SECONDS = 300
YEARS = range(2017, 2025)
EXPECTED_ROWS = 48_538_627


def slot_of(moment):
    return int((moment.timestamp() - ORIGIN_EPOCH) // SLOT_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/rainfall_aligned"))
    parser.add_argument("--output-dir", type=Path, default=Path("task1/data/matrix_5min"))
    parser.add_argument("--master", default="local[4]")
    parser.add_argument("--driver-memory", default="4g")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output}")

    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", f"--driver-memory {args.driver_memory} pyspark-shell")
    spark = (SparkSession.builder.appName("Task 1 revision: five-minute station matrix")
             .master(args.master)
             .config("spark.sql.session.timeZone", "Asia/Singapore")
             .config("spark.sql.shuffle.partitions", "32")
             .config("spark.ui.showConsoleProgress", "true").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    try:
        base = spark.read.parquet(str(args.input_dir.resolve()))
        slot = ((F.col("time_5min").cast("long") - F.lit(ORIGIN_EPOCH)) / SLOT_SECONDS)
        if base.filter(F.pmod(F.col("time_5min").cast("long") - F.lit(ORIGIN_EPOCH), F.lit(SLOT_SECONDS)) != 0).limit(1).count():
            raise ValueError("Off-grid time_5min found; run the temporal preparation first.")

        # Latest coordinates and name per station, as in Task 1's spatial table.
        stations = (base.groupBy("station_id").agg(
            F.max(F.struct("time_5min", "location_longitude", "location_latitude", "station_name")).alias("latest"),
            F.min("time_5min").alias("first_time"), F.max("time_5min").alias("last_time"),
            F.count("*").alias("observations"))
            .select("station_id", "latest.location_longitude", "latest.location_latitude",
                    "latest.station_name", "first_time", "last_time", "observations")
            .orderBy("station_id").toPandas())
        station_ids = stations.station_id.tolist()
        output.mkdir(parents=True)
        stations.to_csv(output / "stations.csv", index=False)

        counts = {}
        for year in YEARS:
            start = slot_of(datetime(year, 1, 1, tzinfo=SGT))
            end = slot_of(datetime(year + 1, 1, 1, tzinfo=SGT))
            readings = (base.filter(F.col("source_year") == year)
                        .select(slot.cast("int").alias("slot"), "station_id",
                                F.col("reading_value").cast("float").alias("value")))
            observed = readings.count()
            # Keys are unique after alignment, so first() returns the single reading.
            wide = readings.groupBy("slot").pivot("station_id", station_ids).agg(F.first("value"))
            grid = spark.range(start, end).select(F.col("id").cast("int").alias("slot"))
            full = grid.join(wide, "slot", "left").orderBy("slot")
            target = output / f"source_year={year}"
            full.coalesce(1).write.parquet(str(target))
            written = spark.read.parquet(str(target))
            cells = written.select([F.count(F.col(f"`{s}`")).alias(s) for s in station_ids]).first()
            rows = written.count()
            if rows != end - start or sum(cells) != observed:
                raise RuntimeError(f"{year}: expected {end - start} slots and {observed} readings, "
                                   f"got {rows} slots and {sum(cells)} readings")
            counts[year] = {"slots": rows, "readings": observed}
            print(f"  {year}: {rows:,} slots, {observed:,} readings", flush=True)

        total = sum(c["readings"] for c in counts.values())
        if total != EXPECTED_ROWS:
            raise RuntimeError(f"Expected {EXPECTED_ROWS:,} readings in total, got {total:,}")
        (output / "manifest.json").write_text(json.dumps({
            "origin_local": ORIGIN.isoformat(), "slot_seconds": SLOT_SECONDS,
            "stations": len(station_ids), "year_counts": counts, "total_readings": total,
            "note": "Rows are complete five-minute slots; null means no reading, not zero rain.",
        }, indent=2) + "\n")
        print(f"Matrix saved: {output} ({len(station_ids)} stations, {total:,} readings)", flush=True)
    finally:
        try:
            spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
