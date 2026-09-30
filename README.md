# Rainfall Pattern Prediction

Apache Spark project using Singapore's five-minute historical rainfall data
from 2017 to 2024. The current repository contains the completed data-quality
audit and temporal preparation needed for Task 1.

## Current status

- Validated all eight annual CSV files: 48,538,713 observations in total.
- Audited missing values, duplicate keys, rainfall validity, station metadata,
  update-time consistency, and station coverage.
- Converted the source CSV files to typed, year-partitioned Parquet.
- Standardized the historical 2017 `:59` timestamp convention to a five-minute
  analysis grid while preserving the original timestamp.
- Resolved 86 verified duplicate aligned keys at the timestamp-system
  transition. The aligned dataset contains 48,538,627 observations.
- Task 1 aggregation and visualizations are the next step.

No raw dataset or generated Parquet data is stored in GitHub.

## Repository structure

```text
code/
  rainfall_audit.py          Data-quality audit and small CSV summaries
  prepare_base_data.py       CSV-to-Parquet conversion and validation
  prepare_temporal_data.py   Five-minute alignment and deterministic deduplication
docs/
  data_quality_findings.md   Findings, decisions, and interpretation
results/data_quality_audit/  Latest small audit tables
requirements.txt
```

## Data

Download the annual datasets from the
[Historical Rainfall across Singapore collection](https://data.gov.sg/collections/2279/view)
and place the eight files in a local `data/` directory:

```text
data/
  HistoricalRainfallacrossSingapore2017.csv
  HistoricalRainfallacrossSingapore2018.csv
  ...
  HistoricalRainfallacrossSingapore2024.csv
```

The `data/` directory is ignored by Git.

## Environment

The current local environment uses:

- Java 17
- Python 3.14.7
- PySpark 4.2.0

Create and activate a virtual environment, then install the Python dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the completed work

Run all commands from the repository root.

### 1. Data-quality audit

```bash
python code/rainfall_audit.py \
  --data-dir data \
  --output-dir audit_results
```

This creates a timestamped folder of small audit CSV files. The latest verified
tables are already included under `results/data_quality_audit/`.

### 2. Typed base Parquet

```bash
python code/prepare_base_data.py \
  --data-dir data \
  --output-dir data/processed/rainfall_base
```

Expected validated total: `48,538,713` rows.

### 3. Aligned five-minute Parquet

```bash
python code/prepare_temporal_data.py \
  --input-dir data/processed/rainfall_base \
  --output-dir data/processed/rainfall_aligned
```

Expected validated total: `48,538,627` rows. The script writes one year at a
time to remain memory-safe on a local machine.

## Task 1 next step

Use `data/processed/rainfall_aligned` to build coverage-aware station-month
aggregates, followed by within-year seasonality, annual changes, diurnal
patterns, spatial comparisons, and extreme-rainfall visualizations.
