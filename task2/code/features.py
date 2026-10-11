import math
from pyspark.sql import functions as F

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