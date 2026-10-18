# Task 2: predicting rain in the next 30 minutes

Given a station, a prediction time and a threshold θ, the models estimate P(rainfall in the next 30 minutes > θ). Labels are the 30-minute totals binned into 9 classes; logistic regression and a random forest predict the class probabilities, and a threshold mapper turns them into P(Y > θ) for any θ.

- `code/`: shared modules (`config`, `features`, `evaluate`, `mapper`, `baselines`, `models`), the `predict`
  function, and the scripts that build the replay page (`build_replay`) and the report figures (`plot_results`)
- `notebooks/`: the pipeline, run in the order 01, 02, 05, 06, 08, 03, 04, 07, 09, 10
- `run_pipeline.py`: runs the notebooks in that order and saves their outputs
- `models/`: the final random forest and logistic regression (Spark ML pipelines, trained on 2017–2023 by
  notebook 09) and the threshold mapper `mapper.json`
- `results/`: score tables (CSV): validation on 2023 (`*_val.csv`, `lr_tuning.csv`, `rf_screening.csv`) and the
  test year 2024 (`models_test.csv`, `test_*.csv`)
- `demo/rain_replay.html`: an interactive, offline page with the forest's forecast for every station and five
  minutes of 2024, for any threshold, next to the rain that fell
- `FEATURES.md`, `MODELS.md`: every feature and every model written out as formulas
- Generated, not in git: `data/` (Parquet tables, about 6 to 8 GB) and `outputs/` (report figures and captions)

## Setup on a new machine (macOS, Linux or Windows)

1. Install Java 17 or 21, for example Eclipse Temurin, and check that `java -version` works in a new terminal.

2. Create a Python 3.12 environment from the repository root and install the requirements.

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

4. Copy the input data. Task 2 starts from the five-minute station matrix `task1/data/matrix_5min/` (6.6 MB, not in git): copy that folder from a machine that has it to the same place. Everything else is rebuilt by the notebooks. To start from the raw CSV files instead, follow the root `README.md` and then run `task1/code/build_matrix.py`.

5. Machine settings: copy `.env.example` to `.env` in the repository root and edit it.
   - `RAIN_SPARK_CORES`: number of CPU cores
   - `RAIN_SPARK_MEMORY`: about 70 % of the RAM, e.g. `45g` on a 64 GB machine
   - `RAIN_TRAIN`: `train_frac0.1` (all rainy rows, 10 % of dry rows, weighted back up) or `train_full` (every row, about 32 GB of RAM or more)
   - `RAIN_RF_TREES`: trees per random forest (the final forest uses 50)
   - `RAIN_RF_MEMORY_MB`: memory per random-forest aggregation pass. `2048` is fine for `train_frac0.1`; the full
     table (notebook 09) ran out of memory with it on 200 GB and needed `512`, because every core holds one such
     buffer at a time

   VS Code passes `.env` to the notebook kernels (restart open kernels after editing it), and `run_pipeline.py` reads it too. For plain Jupyter Lab, set the variables in the terminal before starting it.

## Running

Everything, in pipeline order:

```bash
python task2/run_pipeline.py
```

Selected notebooks (they still run in pipeline order):

```bash
python task2/run_pipeline.py 03 04 07
```

On Windows write `task2\run_pipeline.py`. The notebooks can also be run by hand, top to bottom, in the order 01, 02, 05, 06, 08, 03, 04, 07, 09, 10.

On a 16 GB, 8-core laptop with `train_frac0.1`, notebooks 01 to 06 take about 20 minutes together and notebook 07 (training and tuning) about an hour.

## Using the trained models

`predict.py` gives P(rain over the next 30 minutes > θ mm) for any station and time of 2017–2024, from the saved
models and the stored feature tables (so notebooks 01 to 08 must have run on this machine):

```bash
python task2/code/predict.py S07 "2024-04-14 15:50" 5
python task2/code/predict.py S07 "2024-04-14 15:50" 5 --model LR
```

From Python, `predict.predict(station, time, theta, kind="RF")` returns one probability and
`predict.predict_many(queries, thetas)` a table. Times are Singapore local time; the forecast is made from the
latest reading at or before the given time. A station that reported nothing at that time gets no forecast.

## Replay page and report figures

- `demo/rain_replay.html` (12 MB) is self-contained: open it in a current browser, no server or Python needed.
  Choose a station, a date and time in 2024 and a threshold θ; it shows the final forest's P(Y > θ) and
  persistence's, what fell, the day's forecasts, a map of all stations and Brier scores at θ for the day, the
  station and the year. It stores notebook 10's test-year forecasts (the forest's P(Y > edge) rounded to 1/254,
  gzip-compressed) and applies the same threshold mapper as `predict.py`; it does not run the model. A query can
  be linked directly, e.g. `rain_replay.html#S90/2024-12-29/16:40/5`. Rebuild it after notebook 10 with
  `python task2/code/build_replay.py`.
- `python task2/code/plot_results.py` draws the Task 2 report figures from `results/` into `outputs/figures/`, with
  their captions in `outputs/captions.md` and ready-made LaTeX blocks in `outputs/figures.tex`, in the A4 style of
  the Task 1 figures.

## Notes

- Run one notebook at a time: each starts its own Spark session.
- Notebook 03 writes the training table named by `RAIN_TRAIN`; notebooks 04 and 07 read the same one.
- Notebook 09 trains the final models on every row of 2017–2023 and scores 2024 once. On a 32-core, 256 GB machine
  it took about 10 hours, almost all of it the random forest. Its `REUSE` setting loads an already fitted model from
  `models/` instead of refitting it; the committed run fitted the logistic regression in an earlier, interrupted run
  of the same notebook with the same settings and loaded it here.
- On Windows, Spark may print "Failed to delete" errors about its temporary folder when it stops; they are harmless.
