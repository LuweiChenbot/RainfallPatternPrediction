"""Create coverage-aware descriptive rainfall tables with Spark SQL.

Observed rainfall totals are never filled or silently extrapolated. Comparison
rates are explicitly normalized by observed exposure and averaged over stations.
"""

import argparse
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import functions as F

from project_common import (environment, fresh_directory, read_aligned, spark_options,
                            start_spark, stop_spark, validate_observations, write_csv, write_json)


def daily_aggregates(frame):
    return (frame.withColumn("day", F.to_date("time_5min"))
            .groupBy("station_id", "day").agg(
                F.count("reading_value").alias("observations"),
                F.sum("reading_value").alias("observed_rainfall_mm"),
                F.sum((F.col("reading_value") > 0).cast("int")).alias("wet_intervals"),
                F.max("reading_value").alias("maximum_5min_mm"))
            .withColumn("coverage", F.col("observations") / 288.0)
            .withColumn("year", F.year("day"))
            .withColumn("month", F.month("day")))


def monthly_aggregates(daily, stations, years, spark):
    calendar = spark.createDataFrame([(f"{y}-{m:02d}-01",) for y in years for m in range(1, 13)], ["month_start"])
    calendar = (calendar.withColumn("month_start", F.to_date("month_start"))
                .withColumn("year", F.year("month_start")).withColumn("month", F.month("month_start"))
                .withColumn("expected_observations", F.dayofmonth(F.last_day("month_start")) * 288))
    grouped = daily.groupBy("station_id", "year", "month").agg(
        F.sum("observations").alias("observations"),
        F.sum("observed_rainfall_mm").alias("observed_rainfall_mm"),
        F.sum("wet_intervals").alias("wet_intervals"),
        F.max("maximum_5min_mm").alias("maximum_5min_mm"),
        F.sum((F.col("coverage") == 1).cast("int")).alias("complete_days"),
        F.sum(((F.col("coverage") == 1) & (F.col("observed_rainfall_mm") >= 1)).cast("int")).alias("wet_complete_days"))
    # The calendar makes fully missing months visible, including inactive stations.
    return (stations.crossJoin(calendar).join(grouped, ["station_id", "year", "month"], "left")
            .fillna(0, subset=["observations", "complete_days", "wet_complete_days"])
            .withColumn("coverage", F.col("observations") / F.col("expected_observations"))
            .withColumn("mean_daily_equivalent_mm", F.when(F.col("observations") > 0,
                        F.col("observed_rainfall_mm") * 288.0 / F.col("observations")))
            .withColumn("wet_interval_fraction", F.when(F.col("observations") > 0,
                        F.col("wet_intervals") / F.col("observations")))
            .withColumn("wet_day_fraction", F.when(F.col("complete_days") > 0,
                        F.col("wet_complete_days") / F.col("complete_days"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/processed/rainfall_aligned"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/task1"))
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2017, 2025)))
    parser.add_argument("--minimum-coverage", type=float, default=0.90)
    parser.add_argument("--annual-panel-coverage", type=float, default=0.80)
    spark_options(parser)
    args = parser.parse_args()
    if not 0 < args.minimum_coverage <= 1 or not 0 < args.annual_panel_coverage <= 1:
        parser.error("minimum-coverage must be in (0, 1]")
    args.years = sorted(set(args.years))
    output = fresh_directory(args.output_dir)
    spark = start_spark("Task 1: rainfall patterns", args)
    try:
        base = read_aligned(spark, args.input_dir)
        manifests = []
        for year in args.years:
            print(f"Task 1: aggregating {year}", flush=True)
            frame = base.filter(F.col("source_year") == year)
            validate_observations(frame)
            count = frame.count()
            if not count:
                raise ValueError(f"No observations for {year}")
            manifests.append({"year": year, "observations": count})
            daily_aggregates(frame).write.parquet(str(output / "daily" / f"source_year={year}"))
            (frame.withColumn("year", F.year("time_5min")).withColumn("hour", F.hour("time_5min"))
             .groupBy("station_id", "year", "hour").agg(
                 F.count("*").alias("observations"), F.sum("reading_value").alias("observed_rainfall_mm"),
                 F.sum((F.col("reading_value") > 0).cast("int")).alias("wet_intervals"))
             .write.parquet(str(output / "hourly" / f"source_year={year}")))
        daily = spark.read.parquet(str(output / "daily"))
        stations = base.filter(F.col("source_year").isin(args.years)).select("station_id").distinct()
        monthly = monthly_aggregates(daily, stations, args.years, spark).persist(StorageLevel.MEMORY_AND_DISK)
        write_csv(monthly.orderBy("station_id", "year", "month"), output / "station_month.csv")
        qualified = monthly.filter(F.col("coverage") >= args.minimum_coverage)
        calendar_pattern = qualified.groupBy("year", "month").agg(
            F.count("*").alias("eligible_stations"),
            F.avg("mean_daily_equivalent_mm").alias("mean_daily_equivalent_mm"),
            F.avg("wet_interval_fraction").alias("wet_interval_fraction"),
            F.avg("coverage").alias("mean_coverage"))
        calendar_keys = monthly.select("year", "month").distinct()
        calendar_pattern = calendar_keys.join(calendar_pattern, ["year", "month"], "left").fillna(0, subset=["eligible_stations"])
        write_csv(calendar_pattern.orderBy("year", "month"), output / "monthly_pattern.csv")
        write_csv(calendar_pattern.groupBy("month").agg(
            F.avg("mean_daily_equivalent_mm").alias("mean_daily_equivalent_mm"),
            F.stddev("mean_daily_equivalent_mm").alias("between_year_sd"),
            F.count("mean_daily_equivalent_mm").alias("years_present")).orderBy("month"), output / "seasonality.csv")
        annual = monthly.groupBy("station_id", "year").agg(
            F.sum("observations").alias("observations"),
            F.sum("expected_observations").alias("expected_observations"),
            F.sum("observed_rainfall_mm").alias("observed_rainfall_mm"),
            F.sum("wet_intervals").alias("wet_intervals"))
        annual = (annual.withColumn("coverage", F.col("observations") / F.col("expected_observations"))
                  .withColumn("mean_daily_equivalent_mm", F.when(F.col("observations") > 0,
                              F.col("observed_rainfall_mm") * 288.0 / F.col("observations"))))
        write_csv(annual.orderBy("year", "station_id"), output / "station_year.csv")
        yearly = annual.filter(F.col("coverage") >= args.minimum_coverage).groupBy("year").agg(
            F.count("*").alias("eligible_stations"), F.avg("coverage").alias("mean_coverage"),
            F.avg("mean_daily_equivalent_mm").alias("mean_daily_equivalent_mm"))
        yearly = monthly.select("year").distinct().join(yearly, "year", "left").fillna(0, subset=["eligible_stations"])
        write_csv(yearly.orderBy("year"), output / "annual_pattern.csv")
        # Hold the station composition fixed across every year and month.
        panel = qualified.groupBy("station_id").count().filter(F.col("count") == len(args.years) * 12).select("station_id")
        write_csv(panel.orderBy("station_id"), output / "fixed_panel_stations.csv")
        write_csv(qualified.join(panel, "station_id").groupBy("year", "month").agg(
            F.count("*").alias("stations"), F.avg("mean_daily_equivalent_mm").alias("mean_daily_equivalent_mm"))
                  .orderBy("year", "month"), output / "fixed_panel_pattern.csv")
        # A separately labelled annual panel can be informative even when no
        # station qualifies in every month. Never silently lower the main cutoff.
        annual_panel = (annual.filter(F.col("coverage") >= args.annual_panel_coverage)
                        .groupBy("station_id").count().filter(F.col("count") == len(args.years)).select("station_id"))
        write_csv(annual_panel.orderBy("station_id"), output / "fixed_annual_panel_stations.csv")
        write_csv(annual.join(annual_panel, "station_id").groupBy("year").agg(
            F.count("*").alias("stations"), F.avg("coverage").alias("mean_coverage"),
            F.avg("mean_daily_equivalent_mm").alias("mean_daily_equivalent_mm"))
            .withColumn("coverage_cutoff", F.lit(args.annual_panel_coverage)).orderBy("year"), output / "fixed_annual_panel_pattern.csv")
        for cutoff in [0.80, 0.90, 0.95, 1.00]:
            write_csv(annual.filter(F.col("coverage") >= cutoff).groupBy("year").agg(
                F.count("*").alias("eligible_stations"),
                F.avg("mean_daily_equivalent_mm").alias("mean_daily_equivalent_mm"))
                .orderBy("year"), output / f"coverage_sensitivity_{int(cutoff * 100)}.csv")
        hours = spark.read.parquet(str(output / "hourly"))
        hours = (hours.withColumn("days_in_year", F.datediff(F.make_date(F.col("year") + 1, F.lit(1), F.lit(1)),
                                                           F.make_date("year", F.lit(1), F.lit(1))))
                 .withColumn("coverage", F.col("observations") / (F.col("days_in_year") * 12))
                 .withColumn("mean_hourly_equivalent_mm", F.col("observed_rainfall_mm") * 12 / F.col("observations"))
                 .withColumn("wet_interval_fraction", F.col("wet_intervals") / F.col("observations")))
        hour_year = hours.filter(F.col("coverage") >= args.minimum_coverage).groupBy("year", "hour").agg(
            F.avg("mean_hourly_equivalent_mm").alias("mean_hourly_equivalent_mm"),
            F.avg("wet_interval_fraction").alias("wet_interval_fraction"), F.count("*").alias("stations"))
        write_csv(hour_year.orderBy("year", "hour"), output / "diurnal_by_year.csv")
        write_csv(daily.filter("observations = 288").orderBy(F.desc("observed_rainfall_mm"), "station_id", "day").limit(100),
                  output / "extreme_complete_days.csv")
        selected = base.filter(F.col("source_year").isin(args.years))
        write_csv(selected.orderBy(F.desc("reading_value"), "station_id", "time_5min").select(
            "station_id", "time_5min", "reading_value", "location_longitude", "location_latitude").limit(100),
            output / "extreme_5min_readings.csv")
        locations = selected.groupBy("station_id").agg(F.max(F.struct("time_5min", "location_longitude",
                        "location_latitude", "station_name")).alias("latest"))
        spatial = annual.filter(F.col("coverage") >= args.minimum_coverage).groupBy("station_id").agg(
            F.avg("mean_daily_equivalent_mm").alias("mean_daily_equivalent_mm"), F.count("*").alias("eligible_years"))
        write_csv(spatial.join(locations, "station_id").select("station_id", "mean_daily_equivalent_mm", "eligible_years",
                  "latest.location_longitude", "latest.location_latitude", "latest.station_name").orderBy("station_id"),
                  output / "spatial_pattern.csv")
        write_json(output / "manifest.json", {**environment(spark, args), "year_counts": manifests,
                   "fixed_panel_station_count": panel.count(),
                   "fixed_annual_panel_station_count": annual_panel.count(),
                   "interpretation": "Exposure-normalized station averages, not island-wide total rainfall. Missingness may be informative."})
        monthly.unpersist()
        (output / "_SUCCESS").touch()
        print(f"Task 1 tables saved: {output}", flush=True)
    finally:
        stop_spark(spark)


if __name__ == "__main__":
    main()
