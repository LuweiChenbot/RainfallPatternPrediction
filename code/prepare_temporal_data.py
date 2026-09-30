"""Create a unique five-minute rainfall time grid from the base Parquet data.

Run from the project root after activating the virtual environment:
python code/prepare_temporal_data.py \
    --input-dir data/processed/rainfall_base \
    --output-dir data/processed/rainfall_aligned

The original timestamp is preserved. A new time_5min column shifts the known
2017 :59 timestamps forward by one second. At the two documented transition
times, an exact :00 record and a shifted :59 record represent the same reading;
the exact :00 record is retained deterministically. Existing output is never
overwritten.
"""

import argparse
from pathlib import Path

from pyspark.sql import SparkSession, functions as F


EXPECTED_INPUT_COUNTS = {
    2017: 5_256_106,
    2018: 4_594_246,
    2019: 5_026_862,
    2020: 6_334_473,
    2021: 7_044_646,
    2022: 7_133_230,
    2023: 6_742_890,
    2024: 6_406_260,
}

# The alignment creates 86 duplicate station-time keys in 2017. Their rainfall
# values were verified to be identical, so one row per key is retained.
EXPECTED_OUTPUT_COUNTS = {
    **EXPECTED_INPUT_COUNTS,
    2017: EXPECTED_INPUT_COUNTS[2017] - 86,
}

TRANSITION_EXACT_TIMES = [
    "2017-04-25 11:20:00",
    "2017-04-25 11:25:00",
]
EXPECTED_TRANSITION_PAIRS = 86


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/rainfall_base"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/rainfall_aligned"),
    )
    parser.add_argument("--master", default="local[4]")
    return parser.parse_args()


def collect_year_counts(frame):
    rows = frame.groupBy("source_year").count().orderBy("source_year").collect()
    return {row["source_year"]: row["count"] for row in rows}


def print_year_counts(label, counts):
    print(f"\n{label}:", flush=True)
    for year, count in counts.items():
        print(f"  {year}: {count:,}", flush=True)
    print(f"  Total: {sum(counts.values()):,}", flush=True)


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input Parquet directory not found: {input_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("Prepare aligned Singapore rainfall data")
        .master(args.master)
        .config("spark.sql.session.timeZone", "Asia/Singapore")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        base = spark.read.parquet(str(input_dir))

        input_counts = collect_year_counts(base)
        if input_counts != EXPECTED_INPUT_COUNTS:
            raise RuntimeError(
                f"Unexpected input row counts. Expected {EXPECTED_INPUT_COUNTS}, "
                f"got {input_counts}"
            )
        print_year_counts("Validated input row counts", input_counts)

        valid_exact_grid = (
            (F.second("timestamp") == 0)
            & (F.pmod(F.minute("timestamp"), F.lit(5)) == 0)
        )
        valid_2017_legacy_grid = (
            (F.col("source_year") == 2017)
            & (F.second("timestamp") == 59)
            & (F.pmod(F.minute("timestamp"), F.lit(5)) == 4)
        )
        unexpected_timestamp_count = base.filter(
            ~(valid_exact_grid | valid_2017_legacy_grid)
        ).count()
        if unexpected_timestamp_count:
            raise RuntimeError(
                f"Found {unexpected_timestamp_count} timestamps outside the "
                "validated five-minute patterns"
            )

        print(f"\nWriting aligned Parquet to: {output_dir}", flush=True)
        # Write one explicit year partition at a time. This preserves partition
        # pruning and avoids Spark's memory-heavy dynamic-partition sort across
        # all 48 million rows on a local machine.
        for year in sorted(EXPECTED_INPUT_COUNTS):
            year_data = base.filter(F.col("source_year") == year)
            aligned_year = year_data.withColumn(
                "time_5min",
                F.when(
                    F.second("timestamp") == 59,
                    F.expr("timestamp + INTERVAL 1 SECOND"),
                ).otherwise(F.col("timestamp")),
            )

            if year == 2017:
                # During the documented transition, 86 exact :00 rows have
                # matching legacy :59 rows with identical rainfall values.
                # Four other legacy rows have no exact counterpart and remain.
                exact_transition_keys = (
                    year_data
                    .filter(F.col("timestamp").isin(*TRANSITION_EXACT_TIMES))
                    .select(
                        "station_id",
                        F.col("timestamp").alias("time_5min"),
                    )
                    .distinct()
                    .withColumn("_has_exact_transition_record", F.lit(True))
                    .cache()
                )
                exact_transition_count = exact_transition_keys.count()
                if exact_transition_count != EXPECTED_TRANSITION_PAIRS:
                    raise RuntimeError(
                        f"Expected {EXPECTED_TRANSITION_PAIRS} exact transition "
                        f"keys, found {exact_transition_count}"
                    )

                aligned_year = (
                    aligned_year
                    .join(
                        F.broadcast(exact_transition_keys),
                        ["station_id", "time_5min"],
                        "left",
                    )
                    .filter(
                        F.col("_has_exact_transition_record").isNull()
                        | (F.col("timestamp") == F.col("time_5min"))
                    )
                    .select(*base.columns, "time_5min")
                )

            year_output = output_dir / f"source_year={year}"
            print(f"  Writing {year}...", flush=True)
            (
                aligned_year
                .drop("source_year")
                .write
                .mode("errorifexists")
                .option("compression", "snappy")
                .parquet(str(year_output))
            )

            if year == 2017:
                exact_transition_keys.unpersist()

        verified = spark.read.parquet(str(output_dir))
        output_counts = collect_year_counts(verified)
        if output_counts != EXPECTED_OUTPUT_COUNTS:
            raise RuntimeError(
                f"Unexpected output row counts. Expected {EXPECTED_OUTPUT_COUNTS}, "
                f"got {output_counts}"
            )

        off_grid_count = verified.filter(
            (F.second("time_5min") != 0)
            | (F.pmod(F.minute("time_5min"), F.lit(5)) != 0)
        ).count()
        if off_grid_count:
            raise RuntimeError(f"Found {off_grid_count} output rows off the five-minute grid")

        # Original station-timestamp keys were already unique. Alignment can
        # only create collisions at the two transition times validated above,
        # so restrict this check to that small documented boundary.
        duplicate_key_count = (
            verified
            .filter(
                (F.col("source_year") == 2017)
                & F.col("time_5min").isin(*TRANSITION_EXACT_TIMES)
            )
            .groupBy("station_id", "time_5min")
            .count()
            .filter(F.col("count") > 1)
            .count()
        )
        if duplicate_key_count:
            raise RuntimeError(
                f"Found {duplicate_key_count} duplicate station-time keys after alignment"
            )

        print_year_counts("Validated aligned row counts", output_counts)
        print("\nTemporal preparation completed successfully.", flush=True)
        print(f"Output: {output_dir}", flush=True)
        print("Original timestamps preserved; 86 duplicate aligned rows removed.", flush=True)
    finally:
        try:
            spark.stop()
        except Exception:
            # If the JVM has already terminated, preserve the original error.
            pass


if __name__ == "__main__":
    main()
