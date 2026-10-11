"""Reference forecasts: climatology and persistence."""
from pyspark.sql import functions as F
from pyspark.ml.functions import array_to_vector

import config as C

K = list(range(C.N_CLASSES))


def _counts(df, keys):
    return (df.withColumn("k", F.col("label").cast("int"))
              .groupBy(*keys).pivot("k", K).agg(F.sum("w")).na.fill(0.0))


def _shares(counts, keys, prefix):
    total = sum(F.col(str(k)) for k in K)
    return counts.select(*keys, *[(F.col(str(k)) / total).alias(f"{prefix}{k}") for k in K])


def _to_vector(df, prefix):
    cols = [f"{prefix}{k}" for k in K]
    return df.withColumn("probability", array_to_vector(F.array(*cols))).drop(*cols)


def _month_hour(df):
    return df.withColumn("month", F.month("ts")).withColumn("hour", F.hour("ts"))


def fit_climatology(train, m):
    """Class shares per station, month and hour, shrunk towards the network-wide month and hour."""
    t = _month_hour(train)
    keys = ["station_id", "month", "hour"]
    parent = _shares(_counts(t, ["month", "hour"]), ["month", "hour"], "q")
    n = sum(F.col(str(k)) for k in K)
    table = (_counts(t, keys).join(parent, ["month", "hour"])
             .select(*keys, *[((F.col(str(k)) + m * F.col(f"q{k}")) / (n + m)).alias(f"c{k}") for k in K]))
    return table.cache(), parent.cache()


def predict_climatology(model, df):
    table, parent = model
    out = (_month_hour(df)
           .join(F.broadcast(table), ["station_id", "month", "hour"], "left")
           .join(F.broadcast(parent), ["month", "hour"], "left"))
    out = out.select(*df.columns, *[F.coalesce(f"c{k}", f"q{k}").alias(f"c{k}") for k in K])
    return _to_vector(out, "c")


def now_class(col="past30_mm"):
    """Class of the rain in the last 30 minutes, -1 when that window has missing readings."""
    return F.when(F.col(col).isNull(), F.lit(-1)).otherwise(sum((F.col(col) > e).cast("int") for e in C.BIN_EDGES))


def fit_persistence(train):
    """Class shares of the next 30 minutes given the class of the last 30 minutes."""
    return _shares(_counts(train.withColumn("now", now_class()), ["now"]), ["now"], "s").cache()


def predict_persistence(table, df):
    out = df.withColumn("now", now_class()).join(F.broadcast(table), "now", "left").drop("now")
    return _to_vector(out, "s")
