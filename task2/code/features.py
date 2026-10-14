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


def station_xy():
    """Station ids (sorted) and positions in km east and north of the island centre."""
    st = pd.read_csv(C.MATRIX / "stations.csv").sort_values("station_id")
    x = (st.location_longitude.to_numpy() - 103.82) * 111.32 * np.cos(np.radians(1.35))
    y = (st.location_latitude.to_numpy() - 1.35) * 110.57
    return st.station_id.tolist(), x, y


def station_geometry():
    """Station ids, pairwise distances (km) and bearings (degrees, 0 = north) from each station to every other."""
    ids, x, y = station_xy()
    dx, dy = x[None, :] - x[:, None], y[None, :] - y[:, None]
    return ids, np.hypot(dx, dy), np.degrees(np.arctan2(dx, dy)) % 360


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


def _wide(feat_t1, ids):
    """One row per slot holding every station's rain of the last 15 and 30 minutes."""
    return (feat_t1.groupBy("slot", "year").pivot("station_id", ids)
                   .agg(F.first("t1_rain_15").alias("r15"), F.first("t1_rain_30").alias("r30")))


def _readings(pdf, c15, c30):
    """Rain of the last 15 and 30 minutes (0 where missing) and whether each station reported."""
    r30 = pdf[c30].to_numpy(dtype=float)
    have = ~np.isnan(r30)
    return np.nan_to_num(pdf[c15].to_numpy(dtype=float)), np.nan_to_num(r30), have


def _wmean(x, have, w):
    """Mean of x over the reporting stations, weighted by w (row = target station); 0 where none reports."""
    den = have @ w.T
    return np.where(den > 0, (x @ w.T) / np.maximum(den, 1e-12), 0.0)


def _neighbour_summaries(r15, r30, have, W, dist):
    """The Tier 2 summaries for a block of slots (rows) and every station (columns)."""
    wet30, wet15 = (r30 > 0) & have, (r15 > 0) & have
    f = {"t2_idw_rain_15": _wmean(r15, have, W["idw"]), "t2_idw_rain_30": _wmean(r30, have, W["idw"]),
         "t2_far_rain_30": _wmean(r30, have, W["far"]),
         "t2_wet_frac_5km": _wmean(wet30, have, W["5km"]), "t2_wet_frac_10km": _wmean(wet30, have, W["10km"])}
    f["t2_idw_trend"] = 2 * f["t2_idw_rain_15"] - f["t2_idw_rain_30"]
    for q in "nesw":
        f[f"t2_rain_30_{q}"] = _wmean(r30, have, W[q])
    near = W["10km"] > 0
    n = dist.shape[0]
    f["t2_max_rain_30"] = np.stack([np.where(near[j], r30, 0).max(axis=1) for j in range(n)], axis=1)
    f["t2_dist_rain_km"] = np.stack([np.where(wet15 & (dist[j] > 0), dist[j], 30.0).min(axis=1)
                                     for j in range(n)], axis=1).clip(max=30.0)
    n_have = np.maximum(have.sum(axis=1, keepdims=True), 1)
    f["t2_island_wet_frac"] = np.repeat(wet30.sum(axis=1, keepdims=True) / n_have, n, axis=1)
    f["t2_island_rain_30"] = np.repeat(r30.sum(axis=1, keepdims=True) / n_have, n, axis=1)
    f["t2_nbrs_10km"] = (have @ W["10km"].T).astype(float)
    return f


def _rows(pdf, ids, have, f, names):
    """One output row per station that reported in the slot."""
    rows, cols = np.nonzero(have)
    out = {"station_id": np.array(ids)[cols], "slot": pdf["slot"].to_numpy()[rows],
           "year": pdf["year"].to_numpy()[rows]}
    out.update({n: np.round(f[n][rows, cols], 4) for n in names})
    return pd.DataFrame(out)


def _ship(spark):
    """Send this module and config to the Python workers, which run the mapInPandas steps below."""
    for path in (C.__file__, __file__):
        spark.sparkContext.addPyFile(path)


def add_tier2(spark, feat_t1):
    """Rain at the neighbouring stations up to and including the prediction time."""
    _ship(spark)
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")  # rows hold all 91 stations; large batches exhaust macOS socket buffers
    ids, dist, bearing = station_geometry()
    W = tier2_weights(dist, bearing)
    c15, c30 = [f"{s}_r15" for s in ids], [f"{s}_r30" for s in ids]
    names = ["t2_idw_rain_15", "t2_idw_rain_30", "t2_idw_trend", "t2_far_rain_30", "t2_max_rain_30",
             "t2_wet_frac_5km", "t2_wet_frac_10km", "t2_dist_rain_km",
             "t2_rain_30_n", "t2_rain_30_e", "t2_rain_30_s", "t2_rain_30_w",
             "t2_island_wet_frac", "t2_island_rain_30", "t2_nbrs_10km"]

    def compute(batches):
        for pdf in batches:
            r15, r30, have = _readings(pdf, c15, c30)
            yield _rows(pdf, ids, have, _neighbour_summaries(r15, r30, have, W, dist), names)

    schema = "station_id string, slot int, year int, " + ", ".join(f"{n} double" for n in names)
    return _wide(feat_t1, ids).mapInPandas(compute, schema)


T3_CHANGED = ("t2_idw_rain_30", "t2_far_rain_30", "t2_max_rain_30", "t2_wet_frac_5km", "t2_wet_frac_10km",
              "t2_dist_rain_km", "t2_rain_30_n", "t2_rain_30_e", "t2_rain_30_s", "t2_rain_30_w",
              "t2_island_wet_frac", "t2_island_rain_30")


def season_of(slots):
    """Season code of each slot, as t0_season: 0 NE monsoon, 1 inter-monsoon, 2 SW monsoon."""
    ts = pd.to_datetime(C.ORIGIN_EPOCH + np.asarray(slots, dtype="int64") * C.SLOT_SECONDS, unit="s", utc=True)
    month = ts.tz_convert("Asia/Singapore").month.to_numpy()
    return np.select([np.isin(month, [12, 1, 2, 3]), np.isin(month, [6, 7, 8, 9])], [0, 2], 1)


def upstream_kernels(minutes):
    """Per season: Gaussian weights around the point the storm moves from in `minutes` (row = target station)."""
    _, x, y = station_xy()
    kernels = {}
    for season, (heading, speed) in C.STORM_MOTION.items():
        step = speed * minutes / 60
        px, py = x - step * np.sin(np.radians(heading)), y - step * np.cos(np.radians(heading))
        d = np.hypot(x[None, :] - px[:, None], y[None, :] - py[:, None])
        k = np.where(d <= C.UP_RADIUS_KM, np.exp(-0.5 * (d / C.UP_SIGMA_KM) ** 2), 0.0)
        np.fill_diagonal(k, 0.0)
        kernels[season] = k
    return kernels


def add_tier3(spark, feat_t1):
    """How the neighbour summaries changed over the last LAG_SLOTS slots, and the rain upstream along the storm motion."""
    _ship(spark)
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "1000")
    ids, dist, bearing = station_geometry()
    W = tier2_weights(dist, bearing)
    up = {m: upstream_kernels(m) for m in C.UPSTREAM_MINUTES}
    far, lag = max(C.UPSTREAM_MINUTES), C.LAG_SLOTS * 5
    c15, c30 = [f"{s}_r15" for s in ids], [f"{s}_r30" for s in ids]
    names = ([f"t3_d{lag}_{n.removeprefix('t2_')}" for n in T3_CHANGED]
             + [f"t3_up{m}_{k}" for m in C.UPSTREAM_MINUTES for k in ("rain", "wet")] + [f"t3_up{far}_nbrs"])

    wide = _wide(feat_t1, ids)
    earlier = wide.select((F.col("slot") + C.LAG_SLOTS).alias("slot"), *[F.col(c).alias(f"lag_{c}") for c in c15 + c30])

    def compute(batches):
        for pdf in batches:
            r15, r30, have = _readings(pdf, c15, c30)
            now = _neighbour_summaries(r15, r30, have, W, dist)
            then = _neighbour_summaries(*_readings(pdf, [f"lag_{c}" for c in c15], [f"lag_{c}" for c in c30]), W, dist)
            f = {f"t3_d{lag}_{n.removeprefix('t2_')}": now[n] - then[n] for n in T3_CHANGED}
            season, wet15 = season_of(pdf["slot"]), ((r15 > 0) & have).astype(float)
            for m, kernels in up.items():
                rain, wet, nbrs = (np.zeros(r15.shape) for _ in range(3))
                for code, k in kernels.items():
                    rows = season == code
                    rain[rows] = _wmean(r15[rows], have[rows], k)
                    wet[rows] = _wmean(wet15[rows], have[rows], k)
                    nbrs[rows] = have[rows].astype(float) @ (k > 0).T.astype(float)
                f[f"t3_up{m}_rain"], f[f"t3_up{m}_wet"] = rain, wet
                if m == far:
                    f[f"t3_up{m}_nbrs"] = nbrs
            yield _rows(pdf, ids, have, f, names)

    schema = "station_id string, slot int, year int, " + ", ".join(f"{n} double" for n in names)
    return wide.join(earlier, "slot", "left").mapInPandas(compute, schema)
