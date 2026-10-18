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
| How to run it | please read through this readme |
| Trained model | saved as `task2/models/`: `rf/` (random forest, 49 MB), `lr/` (logistic regression), `mapper.json` (threshold mapper) |
| Results | `task2/results/` (performance score tables), `task1/docs/` and `task1/results/` (data-quality findings and audit tables) |
| Report | `report/` |


**Where to start**
- **Just look at the forecasts:** open `task2/demo/rain_replay.html` in a browser. Nothing to install or run.
- **Make a forecast with the saved models:** do [Setup](#setup), then follow Task 2, part B below (build the feature
  tables first, then run `predict.py`).
- **Task 1 needs no run:** its result, the five-minute station matrix, is included in `task1/data/matrix_5min/`, and the
  raw data is only needed to rebuild it. The findings are in `task1/docs/data_quality_findings.md`; the definitions of
  the label, features and models are in `task2/FEATURES.md` and `task2/MODELS.md`; the scores are in `task2/results/`.

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

The station matrix `task1/data/matrix_5min/` (6.6 MB) and the map boundary `task1/data/external/` are included in the
repository; the matrix is the only input of Task 2. To rebuild the matrix from the raw files, download the eight annual
files from the [Historical Rainfall across Singapore collection](https://data.gov.sg/collections/2279/view) into `data/`
at the repository root (`HistoricalRainfallacrossSingapore2017.csv` to `...2024.csv`) and run the Task 1 commands below.

## Task 1: data quality and rainfall patterns

To go straight to Task 2, skip this section: the station matrix, the result of the last command, is already in the
repository. The commands below only rebuild it from the raw files in `data/`. `build_matrix.py` refuses to overwrite an
existing matrix, so delete `task1/data/matrix_5min/` first.

```sh
python task1/code/prepare_base_data.py --data-dir data --output-dir data/processed/rainfall_base    # CSV -> Parquet, 48,538,713 readings
python task1/code/prepare_temporal_data.py --input-dir data/processed/rainfall_base --output-dir data/processed/rainfall_aligned
# optional, needs internet: the map outline is already included in task1/data/external/
python task1/code/fetch_boundary.py                                                  
# station matrix (essential for task2)
python task1/code/build_matrix.py                                                  
```

## Task 2: predicting rain in the next 30 minutes

Given a station, a prediction time and a threshold θ, the models estimate P(rainfall in the next 30 minutes > θ). The
label, the 58 features and each model are defined in `task2/FEATURES.md` and `task2/MODELS.md`.

| You want to | Do | Time |
|---|---|---|
| A. look at the forecasts | open `task2/demo/rain_replay.html` | nothing to run |
| B. make a forecast with the saved models | Step 1, then Step 2 below | tens of minutes, then about a minute per forecast |
| C. rerun the whole pipeline | `python task2/run_pipeline.py` | hours; overwrites the saved models |

### A. Look at the forecasts (nothing to run)

Open `task2/demo/rain_replay.html` (12 MB, self-contained) in a current browser; it loads in a few seconds. Choose a
station, a date and time in 2024 and a threshold θ. It shows the forest's P(Y > θ) next to persistence's and the rain that
fell, with the day's forecasts, a map of all stations and Brier scores. 

### B. Make a forecast with the saved models

`predict.py` reads the saved models in `task2/models/` and the feature tables, so build the tables first. The station
matrix from Task 1 is already included, and the raw data is not needed.

**Step 1.** Build the labels and feature tables with notebooks 01, 02, 05, 06 and 08:

```bash
python task2/run_pipeline.py 01 02 05 06 08
```
**Do not run notebook 09** (a plain `python task2/run_pipeline.py` runs
it): it refits the models and overwrites `task2/models/`.

**Step 2.** Make a forecast, for any station and time of 2017–2024:

```bash
python task2/code/predict.py S90 "2024-12-29 16:40" 5            # random forest
python task2/code/predict.py S90 "2024-12-29 16:40" 5 --model LR # logistic regression
```

From Python, use
`predict.predict(station, time, theta, kind="RF")` or `predict.predict_many(queries, thetas)`. Times are Singapore local
time; the forecast uses the latest reading at or before the given time, and a station without a reading then gets no
forecast.

### C. Rerun the whole pipeline (optional, retrains the models)

```bash
python task2/run_pipeline.py        # all notebooks, in the order 01, 02, 05, 06, 08, 03, 04, 07, 09, 10
```

- Run notebooks one at a time if you run them by hand: each starts its own Spark session.

On Windows, Spark may print "Failed to delete" errors about its temporary folder when it stops; they are harmless.
