# Rainfall Pattern Prediction

Apache Spark project on Singapore's five-minute historical rainfall data, 2017 to 2024 (about 48.5 million
readings from 91 stations).

- **Task 1** explores rainfall patterns within a year and across years, by time of day, and across the island.
- **Task 2** estimates the probability that a station records more than a given amount of rain in the next
  30 minutes, with Spark MLlib: logistic regression and a random forest on 9 rain classes, a threshold mapper that
  turns the class probabilities into P(Y > θ) for any θ, and three baselines.

## Contents

1. [What is submitted](#what-is-submitted)
2. [Repository layout](#repository-layout)
3. [Quick start](#quick-start)
4. [Setup](#setup)
5. [Data](#data)
6. [Task 1: data quality and rainfall patterns](#task-1-data-quality-and-rainfall-patterns)
7. [Task 2: predicting rain in the next 30 minutes](#task-2-predicting-rain-in-the-next-30-minutes)

## What is submitted

| Requirement | Where |
|---|---|
| Source code | `task1/code/` (audit, preparation, analyses, figures) and `task2/code/`, `task2/notebooks/`, `task2/run_pipeline.py` (prediction pipeline) |
| How to run it | this README: [Setup](#setup), [Task 1](#task-1-data-quality-and-rainfall-patterns) and [Task 2](#task-2-predicting-rain-in-the-next-30-minutes) |
| Trained model | `task2/models/`: `rf/` (random forest, 49 MB), `lr/` (logistic regression) and `mapper.json` (threshold mapper) |
| Results behind the report | `task2/results/` (score tables, CSV) and `task1/docs/`, `task1/results/` (data-quality findings and audit tables) |
| Report | the PDF report, with the names and student IDs of all group members, is submitted on Canvas next to this archive; its LaTeX source and figures are in `report/` |

The raw data and the generated tables (about 14 GB) are not included; see [Data](#data).

## Repository layout

```text
README.md             this file
requirements.txt      Python packages for both tasks
.env.example          machine settings for Spark (copy to .env)
task1/
  code/               rainfall_audit.py, prepare_base_data.py, prepare_temporal_data.py, fetch_boundary.py,
                      build_matrix.py, common.py, plot_preliminary.py, plot_time.py, plot_space.py, plot_amount.py
  docs/               data_quality_findings.md, S107_gauge_fault_2022_08.txt
  results/            data_quality_audit/: the audit's summary tables
task2/
  code/               config, features, baselines, models, mapper, evaluate, predict, plot_results, build_replay
  notebooks/          the pipeline, 01 to 10
  run_pipeline.py     runs the notebooks in order
  models/             the final random forest, logistic regression and threshold mapper
  results/            score tables (CSV): validation on 2023 and test year 2024
  demo/               rain_replay.html, an interactive page with every 2024 forecast
  FEATURES.md         every feature written out as formulas
  MODELS.md           every model and baseline written out as formulas
report/               LaTeX source and figures of the report
```

Generated and not in git: `data/` (raw files, and `data/processed/`), `task1/data/`, `task1/outputs/`,
`task2/data/` (Parquet tables, about 6 to 8 GB), `task2/outputs/` and `audit_results/`.

## Quick start

Without rebuilding anything:
- open `task2/demo/rain_replay.html` in a current browser (no server or Python needed). It shows the final forest's
  forecast for every station and five minutes of 2024, for any threshold, next to the rain that fell;
- read the results in `task2/results/`, the definitions in `task2/FEATURES.md` and `task2/MODELS.md`, and the
  data-quality findings in `task1/docs/data_quality_findings.md`.

To run the code you need the data. The full path is: [Setup](#setup), download the [data](#data), run
[Task 1](#task-1-data-quality-and-rainfall-patterns) up to the station matrix, then
[Task 2](#task-2-predicting-rain-in-the-next-30-minutes). Using the saved models with
`task2/code/predict.py` needs the Task 2 feature tables, so notebooks 01 to 08 must have run on the machine first.

## Setup

1. Install Java 17 or 21, for example Eclipse Temurin, and check that `java -version` works in a new terminal.

2. Create a Python 3.12 environment from the repository root and install the requirements (PySpark 4.2.0).

   macOS or Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   Windows (PowerShell):

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Windows only: Spark needs Hadoop's `winutils.exe` and `hadoop.dll` to write files.
   - Put a build for Hadoop 3.3 or newer into `C:\hadoop\bin`. These are community builds; take them from a well-known GitHub repository such as `cdarlint/winutils`.
   - Set the user environment variable `HADOOP_HOME=C:\hadoop` and add `C:\hadoop\bin` to `Path` (Settings → System → About → Advanced system settings → Environment Variables), then open a new terminal.
   - Keep the repository at a short path such as `C:\rain`, because Spark's temporary file paths can exceed Windows' 260-character limit.
   - Alternatively, run everything inside WSL2 (Ubuntu) with the macOS/Linux commands above; this needs none of the Windows steps.

4. Machine settings: copy `.env.example` to `.env` in the repository root and edit it.
   - `RAIN_SPARK_CORES`: number of CPU cores (Task 2)
   - `RAIN_SPARK_MEMORY`: about 70 % of the RAM, e.g. `45g` on a 64 GB machine
   - `RAIN_SHUFFLE_PARTITIONS`: Spark shuffle partitions (default 96)
   - `RAIN_TRAIN`: `train_frac0.1` (all rainy rows, 10 % of dry rows, weighted back up) or `train_full` (every row, about 32 GB of RAM or more)
   - `RAIN_RF_TREES`: trees per random forest (the final forest uses 50)
   - `RAIN_RF_MEMORY_MB`: memory per random-forest aggregation pass. `2048` is fine for `train_frac0.1`; the full
     table (notebook 09) ran out of memory with it on 200 GB and needed `512`, because every core holds one such
     buffer at a time
   - `RAIN_SPARK_LOCAL_DIR` (optional): folder for Spark's temporary files, on a fast disk with 20 GB free

   VS Code passes `.env` to the notebook kernels (restart open kernels after editing it), and `run_pipeline.py` reads it too. For plain Jupyter Lab, set the variables in the terminal before starting it. The Task 1 scripts take their Spark settings from their command-line options instead.

## Data

Download the eight annual files from the
[Historical Rainfall across Singapore collection](https://data.gov.sg/collections/2279/view) and place them in
`data/` at the repository root:

```text
data/
  HistoricalRainfallacrossSingapore2017.csv
  ...
  HistoricalRainfallacrossSingapore2024.csv
```

Task 1 turns these into the five-minute station matrix `task1/data/matrix_5min/` (6.6 MB), which is also the only
input of Task 2. If another machine already has that folder, you can copy it to the same place instead of running
Task 1's preparation.

## Task 1: data quality and rainfall patterns

Audit and preparation of the raw rainfall files, the five-minute station matrix that Task 2 also starts from, and
the nine report figures.

### Run

From the repository root, with the eight raw CSV files in `data/`:

```sh
python task1/code/rainfall_audit.py --data-dir data --output-dir audit_results      # optional
python task1/code/prepare_base_data.py --data-dir data --output-dir data/processed/rainfall_base
python task1/code/prepare_temporal_data.py --input-dir data/processed/rainfall_base --output-dir data/processed/rainfall_aligned
python task1/code/fetch_boundary.py
python task1/code/build_matrix.py
cd task1/code
python plot_preliminary.py
python plot_time.py
python plot_space.py
python plot_amount.py
```

What each step does:
- `rainfall_audit.py`: data-quality audit of the raw CSV files (summary tables; the latest are in `task1/results/data_quality_audit/`)
- `prepare_base_data.py`: raw CSV → typed, year-partitioned Parquet (48,538,713 readings)
- `prepare_temporal_data.py`: aligns the 2017 ":59" stamps to the five-minute grid and resolves 86 duplicates (48,538,627 readings)
- `fetch_boundary.py`: URA region boundary for the map outline
- `build_matrix.py`: aligned Parquet → dense five-minute station matrix (null = no reading); it refuses to overwrite an existing matrix
- `plot_*.py`: the figures, written to `task1/outputs/`; `common.py` holds the paths, the A4 figure style, the caption store, the data cleaning and the correlation helpers. Each plot script adds its figures to the caption store, so run all four to keep `figures.tex` complete.

### Figures

Drawn at A4 print size (6.3 in text width), with the caption stored in `task1/outputs/captions.json` instead of on the
image. In LaTeX, set `\graphicspath{{task1/outputs/}}` and `\input` the blocks of `task1/outputs/figures.tex`.

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

### Definitions

- Rates are exposure-normalised (rain per observed interval, scaled to a day), and stations have equal weight.
  Monthly figures use station-months with at least 90% coverage; annual figures use station-years with at least 80%.
- Readings found to be wrong are set to missing before any rainfall figure (`00` uses the raw readings): 18 stuck
  station-months and the faulty periods in `task2/code/config.py` (S107 19-22 Aug 2022, S36 Feb 2019-Aug 2021),
  128,849 readings in all. Missing readings are never treated as zero.
- A wet reading has more than 0 mm; a wet day has at least 1 mm, and daily sums use complete days only.
- Seasons: NE monsoon Dec-Mar, SW monsoon Jun-Sep, inter-monsoon Apr-May and Oct-Nov.
- Distances use an equirectangular projection from each station's latest coordinates (error well under 1%).
- Station pairs are not independent, so p-values on pair comparisons are indicative.

## Task 2: predicting rain in the next 30 minutes

Given a station, a prediction time and a threshold θ, the models estimate P(rainfall in the next 30 minutes > θ). Labels
are the 30-minute totals binned into 9 classes; logistic regression and a random forest predict the class probabilities,
and a threshold mapper turns them into P(Y > θ) for any θ.

- `task2/code/`: shared modules (`config`, `features`, `evaluate`, `mapper`, `baselines`, `models`), the `predict`
  function, and the scripts that build the replay page (`build_replay`) and the report figures (`plot_results`)
- `task2/notebooks/`: the pipeline, run in the order 01, 02, 05, 06, 08, 03, 04, 07, 09, 10
- `task2/run_pipeline.py`: runs the notebooks in that order and saves their outputs
- `task2/models/`: the final random forest and logistic regression (Spark ML pipelines, trained on 2017–2023 by
  notebook 09) and the threshold mapper `mapper.json`
- `task2/results/`: score tables (CSV): validation on 2023 (`*_val.csv`, `lr_tuning.csv`, `rf_screening.csv`) and the
  test year 2024 (`models_test.csv`, `test_*.csv`)
- `task2/demo/rain_replay.html`: the interactive page described below
- `task2/FEATURES.md`, `task2/MODELS.md`: every feature and every model written out as formulas

### Run

Everything, in pipeline order (needs the station matrix from Task 1):

```bash
python task2/run_pipeline.py
```

Selected notebooks (they still run in pipeline order):

```bash
python task2/run_pipeline.py 03 04 07
```

On Windows write `task2\run_pipeline.py`. The notebooks can also be run by hand, top to bottom, in the order 01, 02, 05, 06, 08, 03, 04, 07, 09, 10.

On a 16 GB, 8-core laptop with `train_frac0.1`, notebooks 01 to 06 take about 20 minutes together and notebook 07 (training and tuning) about an hour. The reported models were fitted on the full table (`RAIN_TRAIN=train_full`, `RAIN_RF_TREES=50`, `RAIN_RF_MEMORY_MB=512`) on a 32-core cloud machine with 256 GB of RAM.

### Using the trained models

`task2/code/predict.py` gives P(rain over the next 30 minutes > θ mm) for any station and time of 2017–2024, from the
saved models and the stored feature tables (so notebooks 01 to 08 must have run on this machine):

```bash
python task2/code/predict.py S07 "2024-04-14 15:50" 5
python task2/code/predict.py S07 "2024-04-14 15:50" 5 --model LR
```

From Python, `predict.predict(station, time, theta, kind="RF")` returns one probability and
`predict.predict_many(queries, thetas)` a table. Times are Singapore local time; the forecast is made from the
latest reading at or before the given time. A station that reported nothing at that time gets no forecast.

As a check, `python task2/code/predict.py S90 "2024-12-29 16:40" 5` should print 0.941 for the forest and report
65 mm observed.

### Replay page and report figures

- `task2/demo/rain_replay.html` (12 MB) is self-contained: open it in a current browser, no server or Python needed.
  Choose a station, a date and time in 2024 and a threshold θ; it shows the final forest's P(Y > θ) and
  persistence's, what fell, the day's forecasts, a map of all stations and Brier scores at θ for the day, the
  station and the year. It stores notebook 10's test-year forecasts (the forest's P(Y > edge) rounded to 1/254,
  gzip-compressed) and applies the same threshold mapper as `predict.py`; it does not run the model. A query can
  be linked directly, e.g. `rain_replay.html#S90/2024-12-29/16:40/5`. Rebuild it after notebook 10 with
  `python task2/code/build_replay.py`.
- `python task2/code/plot_results.py` draws the Task 2 report figures from `task2/results/` into
  `task2/outputs/figures/`, with their captions in `task2/outputs/captions.md` and ready-made LaTeX blocks in
  `task2/outputs/figures.tex`, in the A4 style of the Task 1 figures.

### Notes

- Run one notebook at a time: each starts its own Spark session.
- Notebook 03 writes the training table named by `RAIN_TRAIN`; notebooks 04 and 07 read the same one.
- Notebook 09 trains the final models on every row of 2017–2023 and scores 2024 once. On a 32-core, 256 GB machine
  it took about 10 hours, almost all of it the random forest. Its `REUSE` setting loads an already fitted model from
  `task2/models/` instead of refitting it; the committed run fitted the logistic regression in an earlier, interrupted run
  of the same notebook with the same settings and loaded it here.
- On Windows, Spark may print "Failed to delete" errors about its temporary folder when it stops; they are harmless.
