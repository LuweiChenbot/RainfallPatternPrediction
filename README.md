# Rainfall Pattern Prediction

Apache Spark project on Singapore's five-minute historical rainfall data, 2017 to 2024 (about 48.5 million
readings from 91 stations).

- **Task 1** explores rainfall patterns within a year and across years, by time of day, and across the island.
- **Task 2** estimates the probability that a station records more than a given amount of rain in the next
  30 minutes, with Spark MLlib.

## Repository layout

```text
task1/            data audit and preparation, the five-minute station matrix, Task 1 analyses and figures
  README.md       run order, figure list with findings, definitions
task2/            prediction pipeline (notebooks 01-10), trained models, results, replay page
  README.md       setup, run order, using the trained models
report/           report framework
legacy/           first Task 1 pipeline, superseded by task1/ (kept for reference)
requirements.txt  Python packages for both tasks
.env.example      machine settings for Spark (copy to .env)
```

Not in git: the raw data in `data/` and everything generated (`task1/data/`, `task1/outputs/`, `task2/data/`,
`task2/outputs/`). The trained models in `task2/models/` and the replay page `task2/demo/rain_replay.html` are
in git.

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

## Environment

Java 17 or 21, Python 3.12 and PySpark 4.2.0. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then set the cores and memory for this machine
```

`task2/README.md` covers Windows and the settings in `.env`.

## Running

1. **Task 1** (`task1/README.md`): audit and prepare the raw files, build the five-minute station matrix
   `task1/data/matrix_5min/`, then draw the Task 1 figures.
2. **Task 2** (`task2/README.md`): `python task2/run_pipeline.py` runs the notebooks from the matrix to the
   test-year evaluation. The final models are trained in notebook 09 and scored on 2024 in notebook 10.
3. **Using the results**: `python task2/code/predict.py S90 "2024-12-29 16:40" 5` gives one forecast from the
   saved models; `task2/demo/rain_replay.html` opens in a browser and shows every 2024 forecast.
