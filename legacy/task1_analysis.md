# Task 1: Coverage-Aware Rainfall Pattern Analysis

This addition contains Task 1 only. Existing data-audit and preparation files are unchanged. It contains no prediction model or Chinese manual.

## Files

- `code/task1_analysis.py`: Spark aggregation and descriptive analysis.
- `code/project_common.py`: Spark settings, aligned-input validation, and small-table export helpers.
- `code/plot_task1.py`: eight Task 1 figures, exported as PNG and SVG.
- `requirements-task1.txt`: analysis and plotting dependencies.

## Run

Run these commands from the repository root with the virtual environment activated. Java must be available. The original project environment uses Java 17 and PySpark 4.2.0.

```sh
python -m pip install -r requirements-task1.txt
python code/task1_analysis.py --input-dir data/processed/rainfall_aligned --output-dir outputs/task1 --master 'local[4]' --driver-memory 4g
python code/plot_task1.py --input-dir outputs/task1
```

The input is the aligned Parquet produced by the preparation scripts already in this repository. No raw data or Parquet is included in this upload. Reuse the locally prepared input rather than repeating the audit unnecessarily. The analysis refuses an existing output directory; select a new output path for a rerun. Plotting requires its `_SUCCESS` marker and writes into its `figures` subdirectory.

## Analysis design

1. Validate nonnegative finite rainfall, five-minute alignment, and unique station-time keys. Process one year at a time to limit memory use. All time-of-day calculations use Asia/Singapore.
2. Aggregate station-day rainfall, wet intervals, maximum five-minute rainfall, and observation counts. A complete day has 288 observations. A wet day is a complete day with at least 1 mm.
3. Build a complete station-month calendar, including absent months. Coverage is observed intervals divided by the full calendar's expected intervals. Missing rainfall stays missing, not zero.
4. Compare rainfall rates using `observed rainfall * 288 / observed intervals` (mm/day). This normalizes observed exposure; it does not impute missing observations or estimate an actual complete-calendar total. Station rates receive equal weight.
5. Qualify station-months and station-years using a default 90% coverage cutoff. Analyze month-by-year patterns, within-year seasonality, annual differences, and time-of-day patterns. Hourly rates use 12 expected five-minute intervals per hour.
6. Assess changing station participation using a strict panel that qualifies in every month of every year, a separately labelled fixed annual panel with an 80% cutoff, and annual sensitivity comparisons at 80%, 90%, 95%, and 100%. An empty strict panel is reported rather than silently relaxing the cutoff.
7. Compare station rainfall spatially using latest recorded coordinates and record each station's number of qualifying years. Rank the largest complete station-day totals and five-minute readings.

## Outputs

Small CSV tables include `station_month`, `station_year`, `monthly_pattern`, `seasonality`, `annual_pattern`, fixed-panel tables, coverage-sensitivity tables, `diurnal_by_year`, `spatial_pattern`, and extreme-rainfall tables. Intermediate daily/hourly data is saved as Parquet. `manifest.json` records the environment, configuration, annual observation counts, and panel sizes. `_SUCCESS` is written only after analysis completes.

The plotting script generates monthly rainfall and coverage heatmaps, seasonality, annual change, diurnal and spatial patterns, coverage sensitivity, and extreme-day figures. Only small aggregate tables are loaded into pandas; raw observations stay in Spark.

## Interpretation limits

- These are exposure-normalized station averages, not island-wide rainfall totals.
- Missing periods may be systematically wetter or drier; normalization cannot remove this bias.
- Station coverage uses full calendar periods, so a station inactive for part of a year has lower coverage.
- Station composition can change. Compare the main analysis with fixed panels and sensitivity results.
- Seasonality shading is between-year standard deviation, not a confidence interval.
- Spatial averages can cover different years and use latest coordinates for display.
- Extreme station-days can belong to the same storm; they are not independent storm events.
- Eight years of descriptive differences alone do not establish a long-term climate trend.

## Verification

The analysis was previously run on the prepared 2017–2024 dataset containing 48,538,627 aligned observations. This upload separates the existing Task 1 plotting logic from Task 2. All Python files were syntax-checked and the plotting script was tested against the existing Task 1 aggregate results. AI assistance should be disclosed in the final project report as required by the assignment.
