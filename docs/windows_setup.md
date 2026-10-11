# Windows setup and Task 1 results

This guide records the Windows-specific setup used to run this project successfully in Anaconda PowerShell Prompt. Commands below are PowerShell commands, executed from the repository root unless stated otherwise.

If Task 1 has already completed, keep the working environment. You can go directly to **Section 7** to save and push the results.

## 1. Local folders and paths

The data can remain outside the Git repository:

- `project2/original data/`: the eight source CSV files and generated preprocessing Parquet.
- `project2/RainfallPatternPrediction/`: the Git repository.
- `RainfallPatternPrediction/outputs/`: local Task 1 working outputs.
- `RainfallPatternPrediction/results/task1_gabriel/`: selected small results for GitHub.

Example project location:

```powershell
cd "D:\AppData\NUS\Sem1\5208 Scalable Distributed Computing\project2\RainfallPatternPrediction"
```

The relative data path is `"../original data"`. Quotation marks are required because the folder name contains a space. Pass paths through the scripts' arguments; changing their default data paths in the source code is unnecessary.

PowerShell does not use Bash's trailing backslash for line continuation. The execution commands in this guide are written on one line.

## 2. Python environment, Java and pandas

Activate the existing environment:

```powershell
conda activate RainfallPrediction
python -c "import sys; print(sys.executable); print(sys.version)"
java -version
python -c "import pyspark,pandas; print('PySpark:',pyspark.__version__); print('pandas:',pandas.__version__)"
```

The original README records Java 17, Python 3.14.7 and PySpark 4.2.0. These are README values, rather than an independent measurement of your environment. The completed run's `manifest.json` records the Python, Java and Spark versions actually used. Keep it with the results.

For a new environment, select a Python version supported by the project's dependencies and available in your Conda channels. Install Java before the pip dependencies. If Java is missing from an existing environment, the installation command used during this setup was:

```powershell
conda install -n RainfallPrediction -c conda-forge openjdk=17
conda deactivate
conda activate RainfallPrediction
```

For a fresh setup, install the Python dependencies from the repository:

```powershell
python -m pip install -r requirements.txt
python -m pip install "pandas<3"
```

The pandas restriction addresses the pandas 3.x compatibility warning encountered in this run. Pip automatically replaces an incompatible installed pandas version; a separate uninstall is unnecessary. Installing packages is a one-time operation, not something to repeat for every terminal session.

## 3. Windows Hadoop native components

The audit could run before Windows Hadoop native support was configured, but writing Parquet failed. Python's `requirements.txt` does not install `winutils.exe` or `hadoop.dll`.

Check the Hadoop version bundled with the installed PySpark:

```powershell
python -c "import pathlib,pyspark; print([p.name for p in (pathlib.Path(pyspark.__file__).parent / 'jars').glob('hadoop-client-api-*.jar')])"
```

The output in this setup was:

```text
['hadoop-client-api-3.5.0.jar']
```

For this version, the Windows binaries used came from the community project [notepass/hadoop-native-win-libs, release 3.5.0](https://github.com/notepass/hadoop-native-win-libs/releases/tag/rel%2Frelease-3.5.0). This is a community build, not an Apache-distributed Windows binary package.

Download the binary archive under the release's Assets, rather than the source-code archive. Extract and arrange the files as:

- `D:\Tools\hadoop-3.5.0\bin\winutils.exe`
- `D:\Tools\hadoop-3.5.0\bin\hadoop.dll`

Use binaries matching your installed Hadoop version. If your extraction location differs, replace the Hadoop path in the commands below. `HADOOP_HOME` points to the parent of `bin`, not to `bin` itself.

## 4. Persist configuration in the Conda environment

A command such as `$env:HADOOP_HOME = "..."` changes only the current PowerShell session. Save the settings in the Conda environment so that they are available after each activation.

### 4.1 Save the Python interpreter and Hadoop location

With `RainfallPrediction` activated:

```powershell
$rainfallPython = (python -c "import sys; print(sys.executable)").Trim()
conda env config vars set -n RainfallPrediction "PYSPARK_PYTHON=$rainfallPython" "PYSPARK_DRIVER_PYTHON=$rainfallPython" "HADOOP_HOME=D:\Tools\hadoop-3.5.0"
```

This chooses the active environment's interpreter for both the local PySpark driver and Python workers. It avoids accidentally launching the Windows Store Python alias.

These settings are for the local execution used in this project. A remote cluster requires its own interpreter configuration.

### 4.2 Add the Hadoop bin directory during PowerShell activation

Still in the active environment:

```powershell
New-Item -ItemType Directory -Force -Path "$env:CONDA_PREFIX\etc\conda\activate.d" | Out-Null
New-Item -ItemType Directory -Force -Path "$env:CONDA_PREFIX\etc\conda\deactivate.d" | Out-Null

@'
$rainfallHadoopBin = "$env:HADOOP_HOME\bin"
if (($env:Path -split ';') -notcontains $rainfallHadoopBin) {
    $env:Path = "$rainfallHadoopBin;$env:Path"
    $env:RAINFALL_ADDED_HADOOP_BIN = $rainfallHadoopBin
}
'@ | Set-Content -Encoding UTF8 -Path "$env:CONDA_PREFIX\etc\conda\activate.d\rainfall-hadoop.ps1"

@'
if ($env:RAINFALL_ADDED_HADOOP_BIN) {
    $env:Path = (($env:Path -split ';') | Where-Object { $_ -ne $env:RAINFALL_ADDED_HADOOP_BIN }) -join ';'
    Remove-Item Env:\RAINFALL_ADDED_HADOOP_BIN
}
'@ | Set-Content -Encoding UTF8 -Path "$env:CONDA_PREFIX\etc\conda\deactivate.d\rainfall-hadoop.ps1"
```

The activation script adds Hadoop only if the entry is absent. The deactivation script removes the entry if this hook added it. An entry previously added manually or through Windows settings is left alone.

The scripts are stored inside the local Conda environment. They are not project source files and do not need to be committed to Git. This documentation allows another user to reproduce the setup on their own machine.

Reactivate to apply the configuration:

```powershell
conda deactivate
conda activate RainfallPrediction
conda env config vars list -n RainfallPrediction
$env:PYSPARK_PYTHON
$env:HADOOP_HOME
Get-Command winutils.exe
```

After this one-time setup, a new Anaconda PowerShell Prompt only needs environment activation and navigation to the project:

```powershell
conda activate RainfallPrediction
cd "D:\AppData\NUS\Sem1\5208 Scalable Distributed Computing\project2\RainfallPatternPrediction"
```

## 5. Verify Parquet writing and Python workers

Run these checks when configuring a new machine or diagnosing the corresponding failure. A completed pipeline does not need to be rerun just to repeat these checks.

### Parquet write check

```powershell
python -c "from pyspark.sql import SparkSession; s=SparkSession.builder.master('local[1]').appName('WriteTest').getOrCreate(); s.range(1).write.mode('overwrite').parquet('outputs/spark_write_test'); s.stop(); print('Parquet write OK')"
```

Expected final message: `Parquet write OK`. This overwrites only the dedicated `outputs/spark_write_test` directory.

### Python worker check

```powershell
python -c "from pyspark.sql import SparkSession; s=SparkSession.builder.master('local[1]').appName('WorkerTest').getOrCreate(); print(s.sparkContext.parallelize([1,2,3],1).map(lambda x:x+1).collect()); s.stop()"
```

Expected result: `[2, 3, 4]`.

The Parquet check primarily exercises Spark's JVM execution. The second check explicitly launches a Python worker.

## 6. Pipeline commands

Run from `RainfallPatternPrediction` with the environment active:

```powershell
python code/rainfall_audit.py --data-dir "../original data" --output-dir audit_results
python code/prepare_base_data.py --data-dir "../original data" --output-dir "../original data/processed/rainfall_base"
python code/prepare_temporal_data.py --input-dir "../original data/processed/rainfall_base" --output-dir "../original data/processed/rainfall_aligned"
python code/task1_analysis.py --input-dir "../original data/processed/rainfall_aligned" --output-dir outputs/task1_gabriel
python code/plot_task1.py --input-dir outputs/task1_gabriel
```

Run each command after the preceding step finishes successfully. The preparation and analysis scripts require a new output directory; they do not overwrite an existing run. For a retry, choose another output name, for example `outputs/task1_gabriel_retry`, and pass that same name to the plotting command.

If Task 1 alone failed, keep the completed aligned Parquet and retry Task 1. There is no need to repeat the audit and preprocessing for a Python worker configuration error.

Successful Task 1 analysis creates `_SUCCESS` in its output root. Plotting checks for that file and writes PNG and SVG figures under `figures/`.

### Messages encountered during Windows setup

| Message or symptom | Interpretation and action |
| --- | --- |
| Missing `winutils.exe`, unset `HADOOP_HOME`, or permission errors during Parquet writing | Configure the matching Windows Hadoop binaries and their paths. |
| `Python was not found`, followed by a Python worker connection timeout | Confirm `PYSPARK_PYTHON` points to the active environment's real `python.exe`; run the worker check. |
| pandas 3.x compatibility warning | Install `pandas<3` in the active environment. |
| Existing output-directory error | Use a new output path for the retry. |
| Incubator-module or native-library warning | A warning alone is not proof of failure. Check the final exception, completion message and expected output. |
| The PowerShell prompt has not returned | The process may still be running. Printing a schema or an aggregation year does not confirm completion. |
| Process-termination messages after successful completion | These were seen during Spark cleanup. Once the program has ended, pressing Enter can redraw the prompt. |

## 7. Save selected results on the Gabriel branch

Git branches share the same working directory. Creating or switching a branch does not create a separate folder. Files become part of a branch's history when committed on that branch.

Use the existing branch name exactly: `Gabriel` (capital G).

### 7.1 Confirm the branch

```powershell
git branch --show-current
git status
```

If already on `Gabriel`, continue. Otherwise switch to the existing branch:

```powershell
git switch Gabriel
```

Do not run `git switch -c Gabriel` again: `-c` creates a new branch.

If Git refuses to switch because local changes would be overwritten, keep those changes and inspect `git status`; do not force the switch.

### 7.2 Add this guide and collect the results

Save this guide as `docs/windows_setup.md`.

Choose the output directory that actually completed. The example below uses the retry directory; replace it if your successful run used another name.

```powershell
$task1Output = ".\outputs\task1_gabriel_retry"
if (-not (Test-Path "$task1Output\_SUCCESS")) { throw "Select the successfully completed Task 1 output directory." }

New-Item -ItemType Directory -Force -Path ".\results\task1_gabriel" | Out-Null
Copy-Item -Path "$task1Output\*.csv" -Destination ".\results\task1_gabriel"
Copy-Item -Path "$task1Output\manifest.json" -Destination ".\results\task1_gabriel"
Copy-Item -Path "$task1Output\_SUCCESS" -Destination ".\results\task1_gabriel"
New-Item -ItemType Directory -Force -Path ".\results\task1_gabriel\figures" | Out-Null
Copy-Item -Path "$task1Output\figures\*.png" -Destination ".\results\task1_gabriel\figures"
Copy-Item -Path "$task1Output\figures\*.svg" -Destination ".\results\task1_gabriel\figures"
```

These copies collect the aggregated CSV files, run metadata, completion marker and figures. They leave the original outputs in place and do not copy the `daily/` or `hourly/` Parquet directories.

Use this destination for one selected run. If you later want to preserve a second run separately, choose a new result directory instead of mixing runs.

### 7.3 Stage, review, commit and push

```powershell
git add -- docs/windows_setup.md results/task1_gabriel
git diff --cached --stat
git status
```

Review the staged paths. They should contain the guide, selected CSVs, metadata and figures.

If Git reports that `results/task1_gabriel` is ignored, inspect the matching ignore rule:

```powershell
git check-ignore -v results/task1_gabriel/annual_pattern.csv
```

For this specifically selected small-results directory, it is possible to add the files explicitly:

```powershell
git add -f -- results/task1_gabriel
git diff --cached --stat
```

Avoid force-adding the entire `outputs/` directory.

Once the staged contents are correct:

```powershell
git commit -m "Document Windows setup and add Gabriel Task 1 results"
git push -u origin Gabriel
```

This pushes to the remote `Gabriel` branch. It does not merge the commit into `main`. On GitHub, select `Gabriel` in the branch selector to view the new files.

If there are separate source-code changes you intend to include, inspect their diff and add those exact paths as well. The commands above stage only this guide and the selected result directory.

## References

- [Conda: managing environments and saving environment variables](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [Conda: activation and environment hooks](https://docs.conda.io/projects/conda/en/latest/dev-guide/deep-dives/activation.html)
- [Spark: Python interpreter environment variables](https://spark.apache.org/docs/latest/configuration.html#environment-variables)
- [Apache Hadoop: Windows problems and native components](https://cwiki.apache.org/confluence/display/HADOOP2/WindowsProblems)
- [Community Windows Hadoop binaries used in this setup](https://github.com/notepass/hadoop-native-win-libs/releases/tag/rel%2Frelease-3.5.0)

