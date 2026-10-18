# Task 1: data quality and rainfall patterns

Audit and preparation of the raw rainfall files, the five-minute station matrix that Task 2 also starts from, and
the nine report figures. `data/` and `outputs/` are git-ignored.

## Layout

```text
task1/
  code/
    rainfall_audit.py         data-quality audit of the raw CSV files
    prepare_base_data.py      raw CSV -> typed, year-partitioned Parquet (48,538,713 readings)
    prepare_temporal_data.py  aligns the 2017 ":59" stamps to the five-minute grid, resolves 86 duplicates (48,538,627)
    fetch_boundary.py         URA region boundary (map outline)
    build_matrix.py           aligned Parquet -> dense five-minute station matrix (null = no reading)
    common.py                 paths, A4 figure style, caption store, data cleaning, correlation helpers
    plot_preliminary.py       00 coverage
    plot_time.py              10-13 seasonal cycle, anomalies, daily cycle, month x hour
    plot_space.py             20-23 station map and clusters, correlation vs distance, storm motion
    plot_amount.py            31 day-to-day persistence
  docs/                       data_quality_findings.md, S107_gauge_fault_2022_08.txt
  results/data_quality_audit/ the audit's summary tables
  data/                       matrix_5min/, external/ (boundary)
  outputs/                    figures/ (PDF, PNG), captions.json and .md, figures.tex, tables/, results.json
```

## Run

From the repository root, with the eight raw CSV files in `data/` (see the root `README.md`):

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

`build_matrix.py` refuses to overwrite an existing matrix. Each plot script adds its figures to the caption store,
so run all four to keep `figures.tex` complete.

## Figures

Drawn at A4 print size (6.3 in text width), with the caption stored in `outputs/captions.json` instead of on the
image. In LaTeX, set `\graphicspath{{task1/outputs/}}` and `\input` the blocks of `outputs/figures.tex`.

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

## Definitions

- Rates are exposure-normalised (rain per observed interval, scaled to a day), and stations have equal weight.
  Monthly figures use station-months with at least 90% coverage; annual figures use station-years with at least 80%.
- Readings found to be wrong are set to missing before any rainfall figure (`00` uses the raw readings): 18 stuck
  station-months and the faulty periods in `task2/code/config.py` (S107 19-22 Aug 2022, S36 Feb 2019-Aug 2021),
  128,849 readings in all. Missing readings are never treated as zero.
- A wet reading has more than 0 mm; a wet day has at least 1 mm, and daily sums use complete days only.
- Seasons: NE monsoon Dec-Mar, SW monsoon Jun-Sep, inter-monsoon Apr-May and Oct-Nov.
- Distances use an equirectangular projection from each station's latest coordinates (error well under 1%).
- Station pairs are not independent, so p-values on pair comparisons are indicative.
