"""Threshold mapper deals with P(Y > theta) for any theta >= 0 from the output 
class probabilities.

For a theta inside class k (theta in [lower edge, upper edge) of that class):
    P(Y > theta) = (sum of the classes above k)  +  p_k * q_k(theta)

similar to linear interpolation
where q_k(theta) is the share of class-k training rows with rainfall above theta. 
The model decides how much probability goes into class k; the training data decides 
how it spreads inside the class. At a bin edge q = 1, the result simply equals the plain 
sum used by evaluate.add_exceedance.
"""
import bisect
import json

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array

import config as C
from evaluate import exc_col


def fit(train):
    """Weighted rainfall values inside each wet class (1-8) of the training table.
    """
    rows = (train.filter(F.col("label") > 0)
                 .groupBy("label", "y_mm").agg(F.sum("w").alias("w"))
                 .orderBy("label", "y_mm")
                 .collect())
    classes = {}
    for r in rows:
        c = classes.setdefault(int(r["label"]), {"y": [], "w": []})
        c["y"].append(float(r["y_mm"]))
        c["w"].append(float(r["w"]))
    assert sorted(classes) == list(range(1, C.N_CLASSES)), "every wet class needs training rows"
    return {"bin_edges": list(C.BIN_EDGES), "classes": classes}


def locate(mapper, theta):
    """Return (k, q): the class k that theta falls in, and q_k(theta) = share of class-k rows with Y > theta."""
    if theta < 0:
        raise ValueError("theta must be >= 0")
    k = bisect.bisect_right(mapper["bin_edges"], theta)      # number of edges <= theta: 1 ... 8
    c = mapper["classes"][k]
    above = sum(w for y, w in zip(c["y"], c["w"]) if y > theta)
    return k, above / sum(c["w"])


def exceedance(mapper, probs, theta):
    """P(Y > theta) from one list of 9 class probabilities (for the prediction interface)."""
    k, q = locate(mapper, theta)
    return sum(probs[k + 1:]) + probs[k] * q


def add_threshold(df, mapper, theta, prob_col="probability"):
    """Add the column P(Y > theta) to every row, named like the bin-edge columns (6.4 -> p_gt_6p4)."""
    k, q = locate(mapper, theta)
    df = df.withColumn("_p", vector_to_array(F.col(prob_col)))
    above = sum((F.col("_p")[j] for j in range(k + 1, C.N_CLASSES)), F.lit(0.0))
    return df.withColumn(exc_col(theta), above + F.col("_p")[k] * F.lit(q)).drop("_p")


def save(mapper, path):
    """Write the mapper as JSON next to the trained models."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapper))


def load(path):
    """Read a mapper written by save()."""
    mapper = json.loads(path.read_text())
    mapper["classes"] = {int(k): v for k, v in mapper["classes"].items()}   # JSON keys come back as strings
    return mapper
