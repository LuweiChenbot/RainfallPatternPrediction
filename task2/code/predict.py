"""P(rain over the next 30 minutes > θ) for any station and time of 2017-2024, from the saved final models.

    python task2/code/predict.py S07 "2024-04-14 15:50" 5
    python task2/code/predict.py S07 "2024-04-14 15:50" 5 --model LR
"""
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

import config as C
import features as FT
import mapper as M

ORIGIN = datetime(2017, 1, 1)  # slot 0, Singapore time
STATIC = ["t0_x_km", "t0_y_km", "t0_wet_ratio", "t0_heavy_ratio"]
_loaded = {}


def slot_of(time):
    """Slot of the latest five-minute reading at or before a Singapore local time."""
    return int((pd.Timestamp(time).to_pydatetime() - ORIGIN).total_seconds() // C.SLOT_SECONDS)


def time_of(slot):
    return ORIGIN + timedelta(seconds=int(slot) * C.SLOT_SECONDS)


def load(kind):
    """The saved pipeline of one model ("RF" or "LR") and the threshold mapper, loaded once."""
    if kind not in _loaded:
        _loaded[kind] = PipelineModel.load(str(C.MODELS / kind.lower()))
    if "mapper" not in _loaded:
        _loaded["mapper"] = M.load(C.MODELS / "mapper.json")
    return _loaded[kind], _loaded["mapper"]


def feature_rows(spark, pairs):
    """Model inputs for (station_id, slot) pairs: station statistics, the clock and the stored Tier 1-3 features."""
    years = sorted({time_of(slot).year for _, slot in pairs})
    slots = sorted({slot for _, slot in pairs})
    static = spark.createDataFrame(pd.read_csv(C.STATION_STATIC)[["station_id", *STATIC]])
    rows = spark.createDataFrame(pairs, "station_id string, slot int").join(F.broadcast(static), "station_id")
    for path in (C.FEAT_T1, C.FEAT_T2, C.FEAT_T3):
        tier = spark.read.parquet(str(path)).filter(F.col("year").isin(years) & F.col("slot").isin(slots))
        rows = rows.join(tier.drop("year"), ["station_id", "slot"])
    rows = rows.withColumn("ts", F.timestamp_seconds(F.lit(C.ORIGIN_EPOCH) + F.col("slot") * C.SLOT_SECONDS))
    return FT.add_time_features(rows)


def predict_many(queries, thetas, kind="RF", spark=None):
    """P(Y > θ) for (station_id, time) pairs: one row per pair, one column per threshold θ in mm.

    Y is the rain in the 30 minutes after the latest reading at or before `time`. A station that reported nothing
    at that time gets no forecast (NaN), as in the training data.
    """
    spark = spark or C.get_spark("task2-predict")
    model, mapper = load(kind)
    pairs = [(station, slot_of(time)) for station, time in queries]
    probs = (model.transform(feature_rows(spark, pairs))
                  .select("station_id", "slot", vector_to_array("probability").alias("p")).toPandas())
    out = pd.DataFrame(pairs, columns=["station_id", "slot"]).merge(probs, how="left")
    out.insert(2, "from", [time_of(s) for s in out.slot])
    for theta in thetas:
        out[f"P(Y > {theta:g})"] = [M.exceedance(mapper, list(p), theta) if isinstance(p, (list, np.ndarray)) else np.nan
                                    for p in out.p]
    return out.drop(columns="p")


def predict(station, time, theta, kind="RF", spark=None):
    """P(rain at `station` over the 30 minutes after `time` (Singapore time) exceeds `theta` mm)."""
    return float(predict_many([(station, time)], [theta], kind, spark).iloc[0, -1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probability that the next 30 minutes of rain at a station exceed θ mm.")
    parser.add_argument("station", help="station id, e.g. S07")
    parser.add_argument("time", help='Singapore local time, e.g. "2024-04-14 15:50"')
    parser.add_argument("theta", type=float, help="threshold in mm")
    parser.add_argument("--model", default="RF", choices=["RF", "LR"])
    args = parser.parse_args()

    spark = C.get_spark("task2-predict")
    row = predict_many([(args.station, args.time)], [args.theta], args.model, spark).iloc[0]
    start, p = row["from"], row.iloc[-1]
    window = f"{start:%Y-%m-%d %H:%M} to {start + timedelta(minutes=30):%H:%M}"
    if np.isnan(p):
        print(f"{args.station} reported no reading at {start:%Y-%m-%d %H:%M}, so there is no forecast.")
    else:
        print(f"P(more than {args.theta:g} mm at {args.station}, {window}) = {p:.3f}   ({args.model})")
        observed = (spark.read.parquet(str(C.LABELS))
                         .filter((F.col("year") == start.year) & (F.col("station_id") == args.station)
                                 & (F.col("slot") == int(row.slot))).select("y_mm").first())
        if observed is not None:
            print(f"observed: {observed.y_mm:g} mm")
