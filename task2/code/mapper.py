"""Threshold mapper: P(Y > theta) for any theta from the 9 class probabilities."""
import bisect
import json

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

import config as C
from evaluate import exc_col


def fit(train):
    """Weighted rainfall values inside each wet class of the training table."""
    rows = (train.filter(F.col("label") > 0)
                 .groupBy("label", "y_mm").agg(F.sum("w").alias("w"))
                 .orderBy("label", "y_mm").collect())
    classes = {}
    for r in rows:
        c = classes.setdefault(int(r["label"]), {"y": [], "w": []})
        c["y"].append(float(r["y_mm"]))
        c["w"].append(float(r["w"]))
    return {"bin_edges": list(C.BIN_EDGES), "classes": classes}


def locate(mapper, theta):
    """The class k containing theta and the share q of its training rows above theta."""
    if theta < 0:
        raise ValueError("theta must be >= 0")
    k = bisect.bisect_right(mapper["bin_edges"], theta)
    c = mapper["classes"][k]
    return k, sum(w for y, w in zip(c["y"], c["w"]) if y > theta) / sum(c["w"])


def exceedance(mapper, probs, theta):
    """P(Y > theta) from one list of class probabilities."""
    k, q = locate(mapper, theta)
    return sum(probs[k + 1:]) + probs[k] * q


def add_threshold(df, mapper, theta, prob_col="probability"):
    """Add the column P(Y > theta) to every row."""
    k, q = locate(mapper, theta)
    df = df.withColumn("_p", vector_to_array(F.col(prob_col)))
    above = sum((F.col("_p")[j] for j in range(k + 1, C.N_CLASSES)), F.lit(0.0))
    return df.withColumn(exc_col(theta), above + F.col("_p")[k] * F.lit(q)).drop("_p")


def save(mapper, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapper))


def load(path):
    mapper = json.loads(path.read_text())
    mapper["classes"] = {int(k): v for k, v in mapper["classes"].items()}
    return mapper
