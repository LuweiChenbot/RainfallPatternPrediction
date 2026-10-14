# Task 2: predicting rain in the next 30 minutes

Given a station, a prediction time and a threshold θ, the models estimate P(rainfall in the next 30 minutes > θ). Labels are the 30-minute totals binned into 9 classes; logistic regression and a random forest predict the class probabilities, and a threshold mapper turns them into P(Y > θ) for any θ.

- `code/`: shared modules (`config`, `features`, `evaluate`, `mapper`, `baselines`, `models`)
- `notebooks/`: the pipeline, run in the order 01, 02, 05, 06, 08, 03, 04, 07
- `run_pipeline.py`: runs the notebooks in that order and saves their outputs
- Generated, not in git: `data/` (Parquet tables, about 4 to 6 GB), `models/` (trained models and `mapper.json`)
- `results/`: score tables (CSV)

## Setup on a new machine (macOS, Linux or Windows)

1. Install Java 17 or 21, for example Eclipse Temurin, and check that `java -version` works in a new terminal.

2. Create a Python 3.12 environment from the repository root and install the requirements.

   macOS or Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r task2/requirements.txt
   ```

   Windows (PowerShell):

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r task2\requirements.txt
   ```

3. Windows only: Spark needs Hadoop's `winutils.exe` and `hadoop.dll` to write files.
   - Put a build for Hadoop 3.3 or newer into `C:\hadoop\bin`. These are community builds; take them from a well-known GitHub repository such as `cdarlint/winutils`.
   - Set the user environment variable `HADOOP_HOME=C:\hadoop` and add `C:\hadoop\bin` to `Path` (Settings → System → About → Advanced system settings → Environment Variables), then open a new terminal.
   - Keep the repository at a short path such as `C:\rain`, because Spark's temporary file paths can exceed Windows' 260-character limit.
   - Alternatively, run everything inside WSL2 (Ubuntu) with the macOS/Linux commands above; this needs none of the Windows steps.

4. Copy the input data. Task 2 starts from the five-minute station matrix `task1_revision/data/matrix_5min/` (6.6 MB, not in git): copy that folder from a machine that has it to the same place. Everything else is rebuilt by the notebooks. To start from the raw CSV files instead, follow the root `README.md` and then run `task1_revision/code/build_matrix.py`.

5. Machine settings: copy `.env.example` to `.env` in the repository root and edit it.
   - `RAIN_SPARK_CORES`: number of CPU cores
   - `RAIN_SPARK_MEMORY`: about 70 % of the RAM, e.g. `45g` on a 64 GB machine
   - `RAIN_TRAIN`: `train_frac0.1` (all rainy rows, 10 % of dry rows, weighted back up) or `train_full` (every row, about 32 GB of RAM or more)
   - `RAIN_RF_TREES`: trees per random forest, e.g. `200` on a large machine

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

On Windows write `task2\run_pipeline.py`. The notebooks can also be run by hand, top to bottom, in the order 01, 02, 05, 06, 08, 03, 04, 07.

On a 16 GB, 8-core laptop with `train_frac0.1`, notebooks 01 to 06 take about 20 minutes together and notebook 07 (training and tuning) about an hour.

## Notes

- Run one notebook at a time: each starts its own Spark session.
- Notebook 03 writes the training table named by `RAIN_TRAIN`; notebooks 04 and 07 read the same one.
- On Windows, Spark may print "Failed to delete" errors about its temporary folder when it stops; they are harmless.
