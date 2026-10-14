"""Here we setup configurations for task2, including the path fixing, bin labeling, evaluation thresholds, ..."""
import os
import sys
from pathlib import Path
import tempfile

# find the paths
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "task2" / "data"
LABELS = DATA / "labels"
ASSEMBLED = DATA / "assembled"
STATION_STATIC = DATA / "station_static.csv"
MODELS = ROOT / "task2" / "models"
RESULTS = ROOT / "task2" / "results"
MATRIX = ROOT / "task1_revision" / "data" / "matrix_5min"
STUCK = DATA / "stuck_station_months.csv"
FEAT_T1 = DATA / "feat_t1"
FEAT_T2 = DATA / "feat_t2"
FEAT_T3 = DATA / "feat_t3"

# gauge periods that reported rain that never fell: (station, start, end), local time, end excluded
FAULTY = [("S107", "2022-08-19 22:00", "2022-08-22 06:00"),
          ("S36", "2019-02-01 00:00", "2021-09-01 00:00")]

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

# Tier 3: neighbour summaries LAG_SLOTS slots earlier, and rain upstream along the storm motion of each
# t0_season (0 NE monsoon, 1 inter-monsoon, 2 SW monsoon), from Task 1 Figure 23: (heading towards, degrees; km/h)
LAG_SLOTS = 3
STORM_MOTION = {0: (136.4, 51.4), 1: (118.2, 59.1), 2: (79.3, 37.9)}
UPSTREAM_MINUTES = (15, 30)
UP_SIGMA_KM = 4.0
UP_RADIUS_KM = 10.0

TRAIN_YEARS = (2017, 2022)
VAL_YEAR = 2023
TEST_YEAR = 2024
SEED = 42

# machine settings, read from the environment so a larger machine needs no code changes (see task2/README.md)
SPARK_CORES = int(os.environ.get("RAIN_SPARK_CORES", "8"))
SPARK_MEMORY = os.environ.get("RAIN_SPARK_MEMORY", "10g")
SHUFFLE_PARTITIONS = int(os.environ.get("RAIN_SHUFFLE_PARTITIONS", "96"))
SPARK_LOCAL_DIR = os.environ.get("RAIN_SPARK_LOCAL_DIR")
TRAIN_NAME = os.environ.get("RAIN_TRAIN", "train_frac0.1")
RF_TREES = int(os.environ.get("RAIN_RF_TREES", "150"))
RF_MEMORY_MB = int(os.environ.get("RAIN_RF_MEMORY_MB", "2048"))

def get_spark(app_name, cores=None, memory=None):
    """Start (or reuse) the local Spark session with the project's settings"""
    from pyspark.sql import SparkSession
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    builder = (SparkSession.builder
               .master(f"local[{cores or SPARK_CORES}]")
               .appName(app_name)
               .config("spark.driver.memory", memory or SPARK_MEMORY)
               .config("spark.sql.session.timeZone", "Asia/Singapore")
               .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS)))
    if SPARK_LOCAL_DIR:
        builder = builder.config("spark.local.dir", SPARK_LOCAL_DIR)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.sparkContext.setCheckpointDir((Path(SPARK_LOCAL_DIR or tempfile.gettempdir()) / "rain-checkpoints").as_posix())
    return spark
