"""Spark ML pipelines for the 9-class rainfall forecast."""
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.feature import Bucketizer, Interaction, OneHotEncoder, SQLTransformer, StandardScaler, VectorAssembler
import config as C

SKEWED = ("t1_rain", "t1_max5", "t1_min_since", "t2_idw_rain", "t2_far_rain", "t2_max_rain", "t2_rain_30", "t2_island_rain",
          "t3_up15_rain", "t3_up30_rain")
SIGNED = ("t1_trend", "t2_idw_trend", "t3_d15_idw_rain", "t3_d15_far_rain", "t3_d15_max_rain", "t3_d15_rain_30",
          "t3_d15_island_rain")

INF = float("inf")
BIN_SPLITS = {**{c: C.BUCKET_SPLITS for c in ("t1_rain_5", "t1_rain_15", "t1_rain_30", "t1_max5_30",
                                             "t2_idw_rain_15", "t2_idw_rain_30", "t2_max_rain_30")},
              **{c: [-INF, 1e-6, 0.1, 0.25, 0.5, 0.75, INF] for c in ("t2_wet_frac_5km", "t2_wet_frac_10km",
                                                                      "t2_island_wet_frac")},
              "t2_dist_rain_km": [-INF, 2, 5, 10, 20, 29.99, INF],
              "t1_min_since_rain": [-INF, 0.5, 15.5, 30.5, 60.5, 180.5, 720.5, 1439.5, INF]}


def feature_columns(df, tiers):
    """Feature columns of the given tiers, e.g. ("t0", "t1")."""
    return [c for c in df.columns if c.startswith(tuple(f"{t}_" for t in tiers))]


def build_lr(features, reg_param=1e-4, elastic_net=0.0, max_iter=1000, nonlinear=False):
    """Multinomial logistic regression on log-scaled rain amounts, a one-hot season and standardised inputs.

    nonlinear=True adds the main rain features cut into bands (one-hot) and an hour-of-day x season interaction.
    """
    num = [c for c in features if c != "t0_season"]
    logged = [c for c in num if c.startswith(SKEWED)]
    signed = [c for c in num if c.startswith(SIGNED)]
    exprs = [f"log1p({c}) AS lg_{c}" for c in logged] + [f"sign({c}) * log1p(abs({c})) AS lg_{c}" for c in signed]
    cols = [f"lg_{c}" if c in logged or c in signed else c for c in num]
    stages = [SQLTransformer(statement=f"SELECT *, {', '.join(exprs)} FROM __THIS__")] if exprs else []
    if "t0_season" in features:
        stages.append(OneHotEncoder(inputCols=["t0_season"], outputCols=["season_onehot"]))
        cols.append("season_onehot")
    if nonlinear:
        binned = [c for c in BIN_SPLITS if c in features]
        if binned:
            stages += [Bucketizer(splitsArray=[BIN_SPLITS[c] for c in binned], inputCols=binned,
                                  outputCols=[f"band_{c}" for c in binned]),
                       OneHotEncoder(inputCols=[f"band_{c}" for c in binned], outputCols=[f"oh_{c}" for c in binned])]
            cols += [f"oh_{c}" for c in binned]
        hod = [c for c in features if c.startswith("t0_hod")]
        if hod and "t0_season" in features:
            stages += [VectorAssembler(inputCols=hod, outputCol="hod_vec"),
                       Interaction(inputCols=["season_onehot", "hod_vec"], outputCol="hod_x_season")]
            cols.append("hod_x_season")
    stages += [VectorAssembler(inputCols=cols, outputCol="raw_features"),
               StandardScaler(inputCol="raw_features", outputCol="features"),
               LogisticRegression(family="multinomial", weightCol="w", regParam=reg_param,
                                  elasticNetParam=elastic_net, maxIter=max_iter, tol=1e-6)]
    return Pipeline(stages=stages)


def build_rf(features, num_trees=150, max_depth=10, min_instances=100, subsampling=0.5, feature_subset="sqrt",
             max_bins=64, impurity="gini"):
    """Random forest on the raw features."""
    return Pipeline(stages=[
        VectorAssembler(inputCols=features, outputCol="features"),
        RandomForestClassifier(weightCol="w", numTrees=num_trees, maxDepth=max_depth,
                               minInstancesPerNode=min_instances, subsamplingRate=subsampling,
                               featureSubsetStrategy=feature_subset, maxBins=max_bins, impurity=impurity,
                               maxMemoryInMB=C.RF_MEMORY_MB, cacheNodeIds=True, seed=C.SEED)])