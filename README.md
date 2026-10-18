# Rainfall Pattern Prediction

Apache Spark project on Singapore's five-minute historical rainfall data, 2017 to 2024 (about 48.5 million
readings from 91 stations).

- **Task 1** explores rainfall patterns within a year and across years, by time of day, and across the island.
- **Task 2** estimates the probability that a station records more than a given amount of rain in the next
  30 minutes, with Spark MLlib: logistic regression and a random forest on 9 rain classes, a threshold mapper that
  turns the class probabilities into P(Y > θ) for any θ, and three baselines.

## What is submitted

| Requirement | Where |
|---|---|
| Source code | `task1/code/` and `task2/code/`, `task2/notebooks/`, `task2/run_pipeline.py` |
| How to run it | this README |
| Trained model | `task2/models/`: `rf/` (random forest, 49 MB), `lr/` (logistic regression), `mapper.json` (threshold mapper) |
| Results | `task2/results/` (score tables), `task1/docs/` and `task1/results/` (data-quality findings and audit tables) |
| Report | the LaTeX source of pdf is in `report/` |

```text
README.md, requirements.txt, .env.example
task1/
  code/       rainfall_audit, prepare_base_data, prepare_temporal_data, fetch_boundary, build_matrix, common, plot_*.py
  docs/       data_quality_findings.md, S107_gauge_fault_2022_08.txt
  results/    data_quality_audit/ (audit tables)
task2/
  code/       config, features, baselines, models, mapper, evaluate, predict, plot_results, build_replay
  notebooks/  the pipeline, 01 to 10
  run_pipeline.py
  models/     final random forest, logistic regression, threshold mapper
  results/    score tables: validation 2023 and test 2024
  demo/       rain_replay.html, an interactive page with every 2024 forecast
  FEATURES.md, MODELS.md   every feature and model written out as formulas
report/       LaTeX source and figures
```

Not included (about 14 GB, generated or downloaded): `data/`, `task1/data/`, `task1/outputs/`, `task2/data/`,
`task2/outputs/`, `audit_results/`.

**Without the data** you can open `task2/demo/rain_replay.html` in a browser (no server or Python needed) and read
`task2/results/`, `task2/FEATURES.md`, `task2/MODELS.md` and `task1/docs/data_quality_findings.md`. Running the code, or
`task2/code/predict.py` with the saved models, needs the data pipeline below.

## Setup

1. Install Java 17 or 21 (for example Eclipse Temurin); check that `java -version` works in a new terminal.
2. Create a Python 3.12 environment from the repository root and install the requirements (PySpark 4.2.0):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows (PowerShell): py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Windows only: Spark needs Hadoop's `winutils.exe` and `hadoop.dll`. Put a build for Hadoop 3.3 or newer (community
   builds, e.g. `cdarlint/winutils` on GitHub) into `C:\hadoop\bin`, set `HADOOP_HOME=C:\hadoop`, add `C:\hadoop\bin` to
   `Path` and open a new terminal. Keep the repository at a short path such as `C:\rain`, because Spark's temporary paths
   can exceed Windows' 260-character limit. Alternatively, run everything in WSL2 (Ubuntu) with the commands above.
4. Copy `.env.example` to `.env` in the repository root and edit it (Task 2 reads it; the Task 1 scripts use
   command-line options instead):
   - `RAIN_SPARK_CORES`, `RAIN_SPARK_MEMORY` (about 70 % of the RAM, e.g. `45g` on 64 GB), `RAIN_SHUFFLE_PARTITIONS` (default 96)
   - `RAIN_TRAIN`: `train_frac0.1` (all rainy rows, 10 % of dry rows, weighted back up) or `train_full` (every row, about 32 GB of RAM or more)
   - `RAIN_RF_TREES`: trees per forest (the final forest uses 50)
   - `RAIN_RF_MEMORY_MB`: memory per random-forest aggregation pass. `2048` is fine for `train_frac0.1`; the full table
     (notebook 09) ran out of memory with it on 200 GB and needed `512`, because every core holds one such buffer
   - `RAIN_SPARK_LOCAL_DIR` (optional): folder for Spark's temporary files

   VS Code passes `.env` to notebook kernels (restart open kernels after editing it); for plain Jupyter Lab, set the
   variables in the terminal first.

## Data

Download the eight annual files from the
[Historical Rainfall across Singapore collection](https://data.gov.sg/collections/2279/view) into `data/` at the
repository root (`HistoricalRainfallacrossSingapore2017.csv` to `...2024.csv`). Task 1 turns them into the
five-minute station matrix `task1/data/matrix_5min/` (6.6 MB), the only input of Task 2. If another machine has that
folder, copy it to the same place instead of running Task 1's preparation.

## Task 1: data quality and rainfall patterns

From the repository root, with the raw files in `data/`:

```sh
python task1/code/rainfall_audit.py --data-dir data --output-dir audit_results            # optional: audit tables
python task1/code/prepare_base_data.py --data-dir data --output-dir data/processed/rainfall_base    # CSV -> Parquet, 48,538,713 readings
python task1/code/prepare_temporal_data.py --input-dir data/processed/rainfall_base --output-dir data/processed/rainfall_aligned   # five-minute grid, 86 duplicates resolved
python task1/code/fetch_boundary.py                                                       # map outline
python task1/code/build_matrix.py                                                         # station matrix (null = no reading)
cd task1/code
python plot_preliminary.py
python plot_time.py
python plot_space.py
python plot_amount.py
```

`build_matrix.py` refuses to overwrite an existing matrix. The plot scripts write figures, captions, tables and
`figures.tex` to `task1/outputs/`; each adds to the caption store, so run all four. Figures are drawn at A4 print size
(6.3 in text width); in LaTeX set `\graphicspath{{task1/outputs/}}` and `\input` the blocks of `figures.tex`.

| Figure | Shows |
|---|---|
| 00_coverage_by_month | which stations reported when |
| 10_seasonal_cycle | monthly cycle and its sine/cosine fit |
| 11_anomalies (appendix) | monthly anomalies and annual means, with the coverage-cutoff check |
| 12_diurnal | daily cycle, split into frequency and intensity |
| 13_month_hour | how the daily cycle changes by month |
| 20_station_map | relative wetness and rainfall regions by station |
| 21_correlation_dynamics | station correlation against distance, 5 minutes to 1 day |
| 23_lead_lag | storm direction and speed by season |
| 31_persistence | probability of a wet day after a wet or dry day |

Definitions:
- Rates are exposure-normalised (rain per observed interval, scaled to a day), with equal weight per station. Monthly
  figures use station-months with at least 90% coverage; annual figures use station-years with at least 80%.
- Wrong readings are set to missing before any rainfall figure (`00` uses the raw readings): 18 stuck station-months and
  the faulty periods in `task2/code/config.py` (S107 19-22 Aug 2022, S36 Feb 2019-Aug 2021), 128,849 readings. Missing
  readings are never treated as zero.
- A wet reading has more than 0 mm; a wet day at least 1 mm (complete days only).
- Seasons: NE monsoon Dec-Mar, SW monsoon Jun-Sep, inter-monsoon Apr-May and Oct-Nov.

## Task 2: predicting rain in the next 30 minutes

Given a station, a prediction time and a threshold θ, the models estimate P(rainfall in the next 30 minutes > θ).
`task2/FEATURES.md` and `task2/MODELS.md` define the label, the 58 features and each model.

**Run** (needs the Task 1 station matrix):

```bash
python task2/run_pipeline.py             # all notebooks, in pipeline order 01, 02, 05, 06, 08, 03, 04, 07, 09, 10
python task2/run_pipeline.py 03 04 07    # selected notebooks (still in pipeline order)
```

On Windows write `task2\run_pipeline.py`. The notebooks can also be run by hand in that order, one at a time (each
starts its own Spark session). On a 16 GB, 8-core laptop with `train_frac0.1`, notebooks 01 to 06 take about 20 minutes
and notebook 07 about an hour. The reported models were fitted on the full table (`RAIN_TRAIN=train_full`,
`RAIN_RF_TREES=50`, `RAIN_RF_MEMORY_MB=512`) on a 32-core cloud machine with 256 GB of RAM; notebook 09 took about 10
hours, almost all of it the random forest. Its `REUSE` setting loads an already fitted model from `task2/models/`
instead of refitting it; the committed run fitted the logistic regression in an earlier, interrupted run of the same
notebook with the same settings and loaded it here. Notebook 03 writes the training table named by `RAIN_TRAIN`;
notebooks 04 and 07 read the same one.

**Using the trained models.** With the feature tables built (notebooks 01 to 08), `predict.py` gives
P(rain over the next 30 minutes > θ mm) for any station and time of 2017–2024:

```bash
python task2/code/predict.py S07 "2024-04-14 15:50" 5
python task2/code/predict.py S07 "2024-04-14 15:50" 5 --model LR
```

From Python: `predict.predict(station, time, theta, kind="RF")` or `predict.predict_many(queries, thetas)`. Times are
Singapore local time; the forecast uses the latest reading at or before the given time, and a station without a reading
then gets no forecast. Check: `python task2/code/predict.py S90 "2024-12-29 16:40" 5` should print 0.941 for the forest
and report 65 mm observed.

**Replay page and figures.** `task2/demo/rain_replay.html` (12 MB) is self-contained. Choose a station, a date and time
in 2024 and a threshold θ; it shows the forest's P(Y > θ) next to persistence's and the rain that fell, with the day's
forecasts, a map of all stations and Brier scores. It stores notebook 10's test-year forecasts and applies the same
threshold mapper as `predict.py`; it does not run the model. A query can be linked, e.g.
`rain_replay.html#S90/2024-12-29/16:40/5`. Rebuild it after notebook 10 with `python task2/code/build_replay.py`.
`python task2/code/plot_results.py` draws the Task 2 report figures from `task2/results/` into `task2/outputs/`.

On Windows, Spark may print "Failed to delete" errors about its temporary folder when it stops; they are harmless.
