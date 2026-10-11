"""Here we setup configurations for task2, including the path fixing, bin labeling, evaluation thresholds, ..."""
from pathlib import Path

# find the paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "task2" / "data"
LABELS = DATA / "labels"
ASSEMBLED = DATA / "assembled"
STATION_STATIC = DATA / "station_static.csv"
MODELS = ROOT / "task2" / "models"
RESULTS = ROOT / "task2" / "results"

# fix the time grid
ORIGIN_EPOCH = 1483200000 # slot 0 = 2017-01-01 00:00 Asia/Singapore
SLOT_SECONDS = 300 # 5 min windows

# Bin labeling
BIN_EDGES = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
BUCKET_SPLITS = [float("-inf")] + [round(e + 0.05, 2) for e in BIN_EDGES] + [float("inf")]
N_CLASSES = len(BIN_EDGES) + 1

# Evaluation
EVAL_THRESHOLDS = BIN_EDGES
EXTRA_THRESHOLDS = [3.0, 15.0, 30.0]

TRAIN_YEARS = (2017, 2022)
VAL_YEAR = 2023
TEST_YEAR = 2024
SEED = 42

def get_spark(app_name, cores, memory):
    """Start (or reuse) the local Spark session with the project's settings"""
    from pyspark.sql import SparkSession
    spark = (SparkSession.builder
             .master(f"local[{cores}]")
             .appName(app_name)
             .config("spark.driver.memory", memory)
             .config("spark.sql.session.timeZone", "Asia/Singapore")
             .config("spark.sql.shuffle.partitions", "96")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    return spark
