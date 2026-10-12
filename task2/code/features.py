import math

import pandas as pd
from pyspark.sql import Window, functions as F

import config as C


def add_time_features(df, ts_col="ts"):
    ts = F.col(ts_col)
    h = F.hour(ts) + F.minute(ts) / 60.0
    year = F.year(ts)
    days_in_year = F.datediff(F.make_date(year + 1, F.lit(1), F.lit(1)),
                              F.make_date(year, F.lit(1), F.lit(1)))
    d = (F.dayofyear(ts) - 1 + h / 24.0) / days_in_year

    for k in (1, 2, 3):
        angle = 2 * math.pi * k * h / 24                     
        df = (df.withColumn(f"t0_hod_s{k}", F.sin(angle))
                .withColumn(f"t0_hod_c{k}", F.cos(angle)))
    for k in (1, 2):
        angle = 2 * math.pi * k * d                      
        df = (df.withColumn(f"t0_doy_s{k}", F.sin(angle))
                .withColumn(f"t0_doy_c{k}", F.cos(angle)))

    month = F.month(ts)
    df = df.withColumn("t0_season",
                       F.when(month.isin(12, 1, 2, 3), 0.0)
                        .when(month.isin(6, 7, 8, 9), 2.0)
                        .otherwise(1.0))
    return df


def clean_grid(spark):
    """Five-minute readings of every station, one row per slot, stuck-gauge months set to null."""
    wide = spark.read.option("pathGlobFilter", "*.parquet").parquet(str(C.MATRIX))
    ids = pd.read_csv(C.MATRIX / "stations.csv").station_id.tolist()
    stuck = (spark.createDataFrame(pd.read_csv(C.STUCK)[["station_id", "ym"]])
                  .withColumn("stuck", F.lit(True)))
    return (wide.unpivot("slot", ids, "station_id", "value")
                .withColumn("ts", F.timestamp_seconds(F.lit(C.ORIGIN_EPOCH) + F.col("slot") * C.SLOT_SECONDS))
                .withColumn("ym", F.date_format("ts", "yyyy-MM"))
                .join(F.broadcast(stuck), ["station_id", "ym"], "left")
                .withColumn("value", F.when(F.col("stuck"), F.lit(None)).otherwise(F.col("value")))
                .drop("stuck", "ym"))


def add_tier1(grid):
    """Rain at the station itself up to and including the prediction time."""
    w = Window.partitionBy("station_id").orderBy("slot")
    v = F.col("value")

    def last(n):
        return w.rowsBetween(-(n - 1), 0)

    out = grid.withColumn("t1_rain_5", F.round(v, 1))
    # rain over the last 5, 15, 30, 60 and 120 minutes, and 24 hours
    for minutes in (15, 30, 60, 120, 1440):
        out = out.withColumn(f"t1_rain_{minutes}", F.round(F.sum(v).over(last(minutes // 5)), 1))
    last_wet = F.max(F.when(v > 0, F.col("slot"))).over(w.rowsBetween(Window.unboundedPreceding, 0))
    return (out.withColumn("t1_max5_30", F.round(F.max(v).over(last(6)), 1))                             # the strongest single 5-minute reading in the last 30 minutes
               .withColumn("t1_wet_slots_60", F.sum((v > 0).cast("int")).over(last(12)))                 # how many of the last 12 readings were wet
               .withColumn("t1_trend_30", F.round(2 * F.col("t1_rain_15") - F.col("t1_rain_30"), 1))     # the trend (is the rain building or easing?)
               .withColumn("t1_min_since_rain",
                           F.least(F.coalesce((F.col("slot") - last_wet) * 5, F.lit(1440)), F.lit(1440)))# minutes since it last rained (capped at 24 hours)
               .withColumn("t1_missing_120", 24 - F.count(v).over(last(24)))                             # how many readings are missing in the last 2 hours
               .filter(v.isNotNull())
               .withColumn("year", F.year("ts"))
               .drop("value", "ts"))

