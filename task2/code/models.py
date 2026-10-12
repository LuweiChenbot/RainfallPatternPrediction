"""Spark ML pipelines for the 9-class rainfall forecast."""
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import OneHotEncoder, SQLTransformer, StandardScaler, VectorAssembler

import config as C

SKEWED = ("t1_rain", "t1_max5", "t1_min_since", "t2_idw_rain", "t2_far_rain", "t2_max_rain", "t2_rain_30", "t2_island_rain")
SIGNED = ("t1_trend", "t2_idw_trend")


def feature_columns(df, tiers):
    """Feature columns of the given tiers, e.g. ("t0", "t1")."""
    return [c for c in df.columns if c.startswith(tuple(f"{t}_" for t in tiers))]


def build_lr(features, reg_param=1e-4, elastic_net=0.0, max_iter=100):
    """Multinomial logistic regression on log-scaled rain amounts, a one-hot season and standardised inputs."""
    num = [c for c in features if c != "t0_season"]
    logged = [c for c in num if c.startswith(SKEWED)]
    signed = [c for c in num if c.startswith(SIGNED)]
    exprs = [f"log1p({c}) AS lg_{c}" for c in logged] + [f"sign({c}) * log1p(abs({c})) AS lg_{c}" for c in signed]
    cols = [f"lg_{c}" if c in logged or c in signed else c for c in num]
    stages = [SQLTransformer(statement=f"SELECT *, {', '.join(exprs)} FROM __THIS__")] if exprs else []
    if "t0_season" in features:
        stages.append(OneHotEncoder(inputCols=["t0_season"], outputCols=["season_onehot"]))
        cols.append("season_onehot")
    stages += [VectorAssembler(inputCols=cols, outputCol="raw_features"),
               StandardScaler(inputCol="raw_features", outputCol="features"),
               LogisticRegression(family="multinomial", weightCol="w", regParam=reg_param,
                                  elasticNetParam=elastic_net, maxIter=max_iter, tol=1e-6)]
    return Pipeline(stages=stages)


def build_rf(features, num_trees=50, max_depth=10, min_instances=100, subsampling=0.5):
    """Random forest on the raw features."""
    return Pipeline(stages=[
        VectorAssembler(inputCols=features, outputCol="features"),
        RandomForestClassifier(weightCol="w", numTrees=num_trees, maxDepth=max_depth,
                               minInstancesPerNode=min_instances, subsamplingRate=subsampling,
                               featureSubsetStrategy="sqrt", maxBins=64, cacheNodeIds=True, seed=C.SEED)])
