"""This script is responsible for scoring for probability forecasts over the 9 rainfall
classes"""
from pyspark.sql import functions as F
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
import config as C

def exc_col(edge):
    """Column name for P(Y > edge), say 0.2 is coded as p_gt_0p2"""
    return f"p_gt_{edge:g}".replace(".", "p")

def add_exceedance(df, prob_col="probability"):
    """Add one column per bin edge, P(Y > edge) = sum of the class probabilities 
    above that edge"""
    df = df.withColumn("_p", vector_to_array(F.col(prob_col)))
    for j, edge in enumerate(C.BIN_EDGES):
        df = df.withColumn(exc_col(edge), sum(F.col("_p")[k] for k in range(j + 1, C.N_CLASSES)))
    return df.drop("_p")

def brier_rps(df):
    """Weighted Brier score at every bin edge and the RPS."""
    aggs = []
    for j, edge in enumerate(C.BIN_EDGES):
        o = (F.col("label") >= j+1).cast("double")
        aggs.append(F.sum(F.col("w") * (F.col(exc_col(edge)) - o) ** 2).alias(f"bs{j}"))
    r = df.agg(F.sum("w").alias("sw"), *aggs).first()            
    brier = {edge: r[f"bs{j}"] / r["sw"] for j, edge in enumerate(C.BIN_EDGES)}
    return brier, sum(brier.values())

def log_loss(df, prob_col="probability"):
    """Weighted multiclass log loss"""
    if "prediction" not in df.columns:
        df = df.withColumn("prediction", F.lit(0.0))
    ev = MulticlassClassificationEvaluator(metricName="logLoss", labelCol="label",
                                           probabilityCol=prob_col, weightCol="w")
    return ev.evaluate(df)

def auc_pr(df):
    """Area under the precision-recall curve of P(Y > edge) for every bin edge.
    Should run add_exceedance first."""
    out = {}
    for j, edge in enumerate(C.BIN_EDGES):
        d = df.withColumn("_o", (F.col("label") >= j+1).cast("double"))
        ev = BinaryClassificationEvaluator(rawPredictionCol=exc_col(edge), labelCol="_o",
                                           metricName="areaUnderPR", weightCol="w")
        out[edge] = ev.evaluate(d)
    return out

def reliability(df, edge, n_bins=10):
    """Mean forecast vs observed frequency of Y > edge, in equal-width bins of the forecast.
    Should run add_exceedance first."""
    j = C.BIN_EDGES.index(edge)
    p = F.col(exc_col(edge))
    o = (F.col("label") >= j + 1).cast("double")
    b = F.least(F.floor(p * n_bins), F.lit(n_bins - 1)).cast("int")     
    return (df.groupBy(b.alias("bin"))
              .agg(F.sum("w").alias("n"),
                   (F.sum(F.col("w") * p) / F.sum("w")).alias("mean_forecast"),
                   (F.sum(F.col("w") * o) / F.sum("w")).alias("observed"))
              .orderBy("bin")
              .toPandas())


def brier_at(df, theta):
    """Weighted Brier score of P(Y > theta) at any threshold, judged on the rainfall itself (y_mm).
    The column comes from add_exceedance (bin edges) or mapper.add_threshold (any theta)."""
    o = (F.col("y_mm") > theta).cast("double")
    r = df.agg((F.sum(F.col("w") * (F.col(exc_col(theta)) - o) ** 2) / F.sum("w")).alias("bs")).first()
    return r["bs"]
