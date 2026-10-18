"""Shared configuration, validation, and small-output helpers."""

import csv
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession, functions as F

TIME_ZONE = "Asia/Singapore"


def spark_options(parser):
    parser.add_argument("--master", default="local[4]")
    parser.add_argument("--driver-memory", default="4g")
    parser.add_argument("--shuffle-partitions", type=int, default=32)


def start_spark(name, args):
    # Set driver memory before the Java gateway starts when using python.
    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", f"--driver-memory {args.driver_memory} pyspark-shell")
    spark = (SparkSession.builder.appName(name).master(args.master)
             .config("spark.sql.session.timeZone", TIME_ZONE)
             .config("spark.sql.shuffle.partitions", str(args.shuffle_partitions))
             .config("spark.ui.showConsoleProgress", "true")
             .config("spark.sql.adaptive.enabled", "true").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def fresh_directory(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(frame, path):
    """Stream only aggregated tables to a single human-readable CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(frame.columns)
        for row in frame.toLocalIterator():
            writer.writerow(list(row))


def environment(spark, args):
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "spark": spark.version,
        "java": spark.sparkContext._jvm.java.lang.System.getProperty("java.version"),
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "master": spark.sparkContext.master,
        "driver_memory": spark.sparkContext.getConf().get("spark.driver.memory", "unspecified"),
        "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
        "time_zone": spark.conf.get("spark.sql.session.timeZone"),
        "arguments": vars(args),
    }


def read_aligned(spark, path):
    frame = spark.read.parquet(str(Path(path).resolve()))
    required = {"source_year", "station_id", "time_5min", "timestamp", "reading_value",
                "update_timestamp", "reading_update_timestamp"}
    if required - set(frame.columns):
        raise ValueError(f"Missing aligned columns: {sorted(required - set(frame.columns))}")
    return frame


def validate_observations(frame):
    bad = (F.col("station_id").isNull() | (F.trim("station_id") == "")
           | F.col("time_5min").isNull() | F.col("reading_value").isNull()
           | F.isnan("reading_value") | (F.col("reading_value") < 0)
           | (F.abs("reading_value") == float("inf"))
           | (F.pmod(F.col("time_5min").cast("long"), F.lit(300)) != 0))
    if frame.filter(bad).limit(1).count():
        raise ValueError("Invalid observation or off-grid timestamp. Run the audit first.")
    if frame.groupBy("station_id", "time_5min").count().filter("count > 1").limit(1).count():
        raise ValueError("Duplicate aligned station-time keys. Run temporal preparation first.")


def stop_spark(spark):
    try:
        spark.stop()
    except Exception:
        pass
