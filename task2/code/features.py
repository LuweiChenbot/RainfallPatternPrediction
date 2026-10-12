import math

import numpy as np
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


def mask_faulty(df):
    """Set readings from the known faulty gauge periods in config.FAULTY to null."""
    bad = F.lit(False)
    for station, start, end in C.FAULTY:
        bad = bad | ((F.col("station_id") == station)
                     & (F.col("ts") >= F.lit(start).cast("timestamp")) & (F.col("ts") < F.lit(end).cast("timestamp")))
    return df.withColumn("value", F.when(bad, F.lit(None)).otherwise(F.col("value")))


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
                .drop("stuck", "ym")
                .transform(mask_faulty))


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


def station_geometry():
    """Station ids, pairwise distances (km) and bearings (degrees, 0 = north) from each station to every other."""
    st = pd.read_csv(C.MATRIX / "stations.csv").sort_values("station_id")
    x = (st.location_longitude.to_numpy() - 103.82) * 111.32 * np.cos(np.radians(1.35))
    y = (st.location_latitude.to_numpy() - 1.35) * 110.57
    dx, dy = x[None, :] - x[:, None], y[None, :] - y[:, None]
    return st.station_id.tolist(), np.hypot(dx, dy), np.degrees(np.arctan2(dx, dy)) % 360


def tier2_weights(dist, bearing):
    """Neighbour weight matrices (row = target station, column = neighbour), the station itself excluded."""
    other = dist > 0
    idw = 1 / np.maximum(dist, 0.5) ** 2
    quadrant = ((bearing + 45) % 360) // 90
    w = {"idw": np.where(other & (dist <= 10), idw, 0), "far": np.where((dist > 10) & (dist <= 25), idw, 0),
         "5km": (other & (dist <= 5)).astype(float), "10km": (other & (dist <= 10)).astype(float)}
    for q, name in enumerate("nesw"):
        w[name] = (other & (dist <= 25) & (quadrant == q)).astype(float)
    return w


def add_tier2(spark, feat_t1):
    """Rain at the neighbouring stations up to and including the prediction time."""
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")  # rows hold all 91 stations; large batches exhaust macOS socket buffers
    ids, dist, bearing = station_geometry()
    W = tier2_weights(dist, bearing)
    wide = (feat_t1.groupBy("slot", "year").pivot("station_id", ids)
                   .agg(F.first("t1_rain_15").alias("r15"), F.first("t1_rain_30").alias("r30")))
    c15, c30 = [f"{s}_r15" for s in ids], [f"{s}_r30" for s in ids]
    names = ["t2_idw_rain_15", "t2_idw_rain_30", "t2_idw_trend", "t2_far_rain_30", "t2_max_rain_30",
             "t2_wet_frac_5km", "t2_wet_frac_10km", "t2_dist_rain_km",
             "t2_rain_30_n", "t2_rain_30_e", "t2_rain_30_s", "t2_rain_30_w",
             "t2_island_wet_frac", "t2_island_rain_30", "t2_nbrs_10km"]

    def compute(batches):
        for pdf in batches:
            r30 = pdf[c30].to_numpy(dtype=float)
            r15 = np.nan_to_num(pdf[c15].to_numpy(dtype=float))
            have = ~np.isnan(r30)
            r30 = np.nan_to_num(r30)
            wet30, wet15 = (r30 > 0) & have, (r15 > 0) & have

            def mean(x, w):
                den = have @ w.T
                return np.where(den > 0, (x @ w.T) / np.maximum(den, 1e-12), 0.0)

            f = {"t2_idw_rain_15": mean(r15, W["idw"]), "t2_idw_rain_30": mean(r30, W["idw"]),
                 "t2_far_rain_30": mean(r30, W["far"]),
                 "t2_wet_frac_5km": mean(wet30, W["5km"]), "t2_wet_frac_10km": mean(wet30, W["10km"])}
            f["t2_idw_trend"] = 2 * f["t2_idw_rain_15"] - f["t2_idw_rain_30"]
            for q in "nesw":
                f[f"t2_rain_30_{q}"] = mean(r30, W[q])
            near = W["10km"] > 0
            f["t2_max_rain_30"] = np.stack([np.where(near[j], r30, 0).max(axis=1) for j in range(len(ids))], axis=1)
            f["t2_dist_rain_km"] = np.stack([np.where(wet15 & (dist[j] > 0), dist[j], 30.0).min(axis=1)
                                             for j in range(len(ids))], axis=1).clip(max=30.0)
            n_have = np.maximum(have.sum(axis=1, keepdims=True), 1)
            f["t2_island_wet_frac"] = np.repeat(wet30.sum(axis=1, keepdims=True) / n_have, len(ids), axis=1)
            f["t2_island_rain_30"] = np.repeat(r30.sum(axis=1, keepdims=True) / n_have, len(ids), axis=1)
            f["t2_nbrs_10km"] = (have @ W["10km"].T).astype(float)
            rows, cols = np.nonzero(have)
            out = {"station_id": np.array(ids)[cols], "slot": pdf["slot"].to_numpy()[rows],
                   "year": pdf["year"].to_numpy()[rows]}
            out.update({n: np.round(f[n][rows, cols], 4) for n in names})
            yield pd.DataFrame(out)

    schema = "station_id string, slot int, year int, " + ", ".join(f"{n} double" for n in names)
    return wide.mapInPandas(compute, schema)

