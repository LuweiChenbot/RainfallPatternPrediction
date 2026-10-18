# Appendix drafts (A, B, C, D)

## Appendix A. Detailed Data-Quality Results

**Table A.1 Dataset overview by year** The eight annual files hold 48,538,713 observations from 91 station IDs. Every
record is a five-minute rainfall total in mm. No missing timestamp, station, rainfall, coordinate or update-time value, no
negative or non-finite rainfall, and no duplicate original (station, timestamp) key was found in any year.

| Year | Rows | Stations | First observation | Last observation |
|---:|---:|---:|---|---|
| 2017 | 5,256,106 | 62 | 2017-01-01 08:04:59 | 2017-12-31 23:55:00 |
| 2018 | 4,594,246 | 56 | 2018-01-01 00:00:00 | 2018-12-31 23:55:00 |
| 2019 | 5,026,862 | 54 | 2019-01-01 00:00:00 | 2019-12-31 23:55:00 |
| 2020 | 6,334,473 | 80 | 2020-01-01 00:00:00 | 2020-12-31 23:55:00 |
| 2021 | 7,044,646 | 76 | 2021-01-01 00:00:00 | 2021-12-31 23:55:00 |
| 2022 | 7,133,230 | 72 | 2022-01-01 00:00:00 | 2022-12-31 23:55:00 |
| 2023 | 6,742,890 | 74 | 2023-01-01 00:00:00 | 2023-12-31 23:55:00 |
| 2024 | 6,406,260 | 70 | 2024-01-01 00:00:00 | 2024-12-31 23:55:00 |
| **Total** | **48,538,713** | **91 unique** | | |

**Table A.2 Station coverage by year** Coverage is the share of the expected five-minute records that a station reported
between its own first and last observation in that year, so a station that joined or left during the year is not penalised.
It differs from the calendar-based coverage of Section 3.3 and Figure 1, which counts against the full month or year. The
audit coverage is diagnostic: it was not used to delete any station.

| Year | Stations | Mean coverage | Median coverage | Stations below 80% | Stations below 50% |
|---:|---:|---:|---:|---:|---:|
| 2017 | 62 | 0.8972 | 0.9497 | 5 | 3 |
| 2018 | 56 | 0.8110 | 0.8179 | 17 | 0 |
| 2019 | 54 | 0.9030 | 0.9490 | 8 | 2 |
| 2020 | 80 | 0.9144 | 0.9481 | 5 | 2 |
| 2021 | 76 | 0.9563 | 0.9842 | 3 | 1 |
| 2022 | 72 | 0.9775 | 0.9975 | 2 | 1 |
| 2023 | 74 | 0.9331 | 0.9677 | 2 | 2 |
| 2024 | 70 | 0.9608 | 0.9835 | 2 | 0 |

Source: `task1/results/data_quality_audit/yearly_quality.csv` and `coverage_by_year.csv`, written by
`task1/code/rainfall_audit.py`; discussion in `task1/docs/data_quality_findings.md`.

---

## Appendix B. Full Feature Definitions 

**Table B.1 Features**

| Tier | Feature(s) | No. | How it is constructed | Data used |
|---|---|---|---|---|
| T0 | `t0_x_km`, `t0_y_km` | 2 | Station position in km east and north of the island centre (103.82°E, 1.35°N), by an equirectangular projection | Latest station coordinates |
| T0 | `t0_wet_ratio` | 1 | (O + 500) / (E + 500): rainy 30-minute windows observed at the station (Y > 0), over the number expected if it followed the network at the same times; shrunk towards 1 | Labels 2017–2022, constant per station |
| T0 | `t0_heavy_ratio` | 1 | The same ratio for heavy windows (Y > 5 mm) | Labels 2017–2022, constant per station |
| T0 | `t0_hod_s1`–`s3`, `t0_hod_c1`–`c3` | 6 | sin and cos of 2πk·h/24 for k = 1, 2, 3, where h is the local hour of the prediction time (with minutes) | Prediction time |
| T0 | `t0_doy_s1`, `s2`, `t0_doy_c1`, `c2` | 4 | sin and cos of 2πk·δ for k = 1, 2, where δ is the fraction of the year elapsed | Prediction time |
| T0 | `t0_season` | 1 | Monsoon code: 0 for Dec–Mar, 2 for Jun–Sep, 1 otherwise | Prediction time |
| T1 | `t1_rain_5`, `t1_rain_15`, `t1_rain_30`, `t1_rain_60`, `t1_rain_120`, `t1_rain_1440` | 6 | Rain total at the station over the last 5, 15, 30, 60, 120 and 1,440 minutes (`t1_rain_5` is the current reading) | Station's own readings, t and earlier |
| T1 | `t1_max5_30` | 1 | Largest five-minute reading in the last 30 minutes | Same |
| T1 | `t1_wet_slots_60` | 1 | Number of wet readings among the last 12 | Same |
| T1 | `t1_trend_30` | 1 | 2·R¹⁵ − R³⁰: the last 15 minutes minus the 15 minutes before, positive when rain is intensifying | Same |
| T1 | `t1_min_since_rain` | 1 | Minutes since the last wet reading, capped at 1,440 | Same |
| T1 | `t1_missing_120` | 1 | Number of missing readings among the last 24 | Same |
| T2 | `t2_idw_rain_15`, `t2_idw_rain_30` | 2 | Mean 15- and 30-minute rain of the other stations within 10 km, weighted by 1/max(d, 0.5)², reporting stations only (0 if none) | Other stations, time t |
| T2 | `t2_idw_trend` | 1 | 2 × `t2_idw_rain_15` − `t2_idw_rain_30`, the neighbours' version of the trend | Same |
| T2 | `t2_far_rain_30` | 1 | The same weighted mean of 30-minute rain for stations 10 to 25 km away | Same |
| T2 | `t2_max_rain_30` | 1 | Largest 30-minute rain at any reporting station within 10 km | Same |
| T2 | `t2_wet_frac_5km`, `t2_wet_frac_10km` | 2 | Share of reporting stations within 5 km and 10 km with rain in the last 30 minutes | Same |
| T2 | `t2_dist_rain_km` | 1 | Distance to the nearest station with rain in the last 15 minutes, capped at 30 km | Same |
| T2 | `t2_rain_30_n`, `_e`, `_s`, `_w` | 4 | Mean 30-minute rain of the reporting stations within 25 km in the north, east, south and west quadrants | Same |
| T2 | `t2_island_wet_frac`, `t2_island_rain_30` | 2 | Share of all reporting stations that are wet, and their mean 30-minute rain | All stations, time t |
| T2 | `t2_nbrs_10km` | 1 | Number of reporting stations within 10 km (tells the model how many stations the averages rest on) | Other stations, time t |
| T3 | `t3_d15_idw_rain_30`, `_far_rain_30`, `_max_rain_30`, `_wet_frac_5km`, `_wet_frac_10km`, `_dist_rain_km`, `_rain_30_n`, `_e`, `_s`, `_w`, `_island_wet_frac`, `_island_rain_30` | 12 | Each of these twelve Tier 2 summaries now, minus the same summary recomputed from the network readings 15 minutes earlier | Network at t and at t − 15 min |
| T3 | `t3_up15_rain`, `t3_up15_wet` | 2 | Gaussian-weighted mean (σ = 4 km, cut-off 10 km) of 15-minute rain, and of the wet indicator, around the point 15 minutes upstream: the station moved backwards along the seasonal storm motion (Table C.2) | Network at t |
| T3 | `t3_up30_rain`, `t3_up30_wet` | 2 | The same around the point 30 minutes upstream | Network at t |
| T3 | `t3_up30_nbrs` | 1 | Number of stations near the 30-minute upstream point (it separates "dry" from "no gauge there") | Network at t |

## Appendix C. Hyperparameter Tuning and Test Performance

**Table C.1 Logistic regression on the 2023**

| Tiers | λ (regParam) | Non-linear inputs | Validation RPS | Iterations | Fit (min) |
|---|---|---|---|---|---|
| T0+T1+T2 | 1e-05 | no | 0.0961 | 429 | 4.7 |
| T0+T1+T2 | 1e-06 | yes | 0.0935 | 659 | 10.7 |
| T0+T1+T2 | 1e-05 | yes | 0.0936 | 609 | 9.9 |
| T0+T1+T2 | 0.0001 | yes | 0.0937 | 530 | 8.8 |
| T0+T1+T2+T3 | 1e-06 | yes | 0.0922 | 655 | 12.9 |

The first row is the plain model; the last row adds Tier 3 and is the final specification.

**Table C.2 Random forest screening on 2023** Rows are in the
order tried; the last row is the chosen setting before Tier 3 was added.

| maxDepth | minInstancesPerNode | featureSubsetStrategy | maxBins | Impurity | Validation RPS | Fit (min) | Nodes |
|---|---|---|---|---|---|---|---|
| 14 | 100 | sqrt | 64 | gini | 0.0927 | 2.7 | 224,268 |
| 17 | 100 | sqrt | 64 | gini | 0.0923 | 4.5 | 379,530 |
| 20 | 100 | sqrt | 64 | gini | 0.0923 | 6.0 | 435,878 |
| 24 | 100 | sqrt | 64 | gini | 0.0923 | 7.3 | 444,966 |
| 24 | 20 | sqrt | 64 | gini | 0.0930 | 21.5 | 2,675,066 |
| 24 | 50 | sqrt | 64 | gini | 0.0920 | 11.4 | 984,782 |
| 24 | 300 | sqrt | 64 | gini | 0.0935 | 4.0 | 122,922 |
| 24 | 50 | onethird | 64 | gini | 0.0919 | 20.3 | 1,014,978 |
| 24 | 50 | 0.5 | 64 | gini | 0.0919 | 27.8 | 1,029,336 |
| 24 | 50 | onethird | 256 | gini | 0.0912 | 56.1 | 1,030,260 |
| 24 | 50 | onethird | 256 | entropy | 0.0911 | 63.8 | 1,044,176 |

**Table C.3 Test year 2024** RPS: ranked probability score; Brier >θ: Brier score of
P(Y > θ) in mm; AUC-PR: area under the precision-recall curve. Lower is better for RPS, log loss and Brier; higher for
AUC-PR.

| Subset | Forecast | RPS | log loss | Brier >0 | Brier >1 | Brier >5 | Brier >20 | AUC-PR >0 | AUC-PR >1 | AUC-PR >5 | AUC-PR >20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| All forecasts | Constant | 0.1903 | 0.3545 | 0.0596 | 0.0244 | 0.00825 | 0.00103 | 0.0636 | 0.0250 | 0.0083 | 0.0010 |
| All forecasts | Climatology | 0.1864 | 0.3437 | 0.0578 | 0.0240 | 0.00820 | 0.00102 | 0.1254 | 0.0540 | 0.0234 | 0.0043 |
| All forecasts | Persistence | 0.1246 | 0.2577 | 0.0355 | 0.0170 | 0.00701 | 0.00100 | 0.6406 | 0.4731 | 0.2500 | 0.0495 |
| All forecasts | Logistic regression | 0.0922 | 0.1933 | 0.0259 | 0.0126 | 0.00547 | 0.00087 | 0.7850 | 0.6888 | 0.5066 | 0.1956 |
| All forecasts | Random forest | 0.0887 | 0.1863 | 0.0248 | 0.0121 | 0.00539 | 0.00088 | 0.8015 | 0.7117 | 0.5246 | 0.2285 |
| Dry now | Constant | 0.0708 | 0.1760 | 0.0260 | 0.0081 | 0.00275 | 0.00026 | 0.0253 | 0.0079 | 0.0027 | 0.0003 |
| Dry now | Climatology | 0.0710 | 0.1705 | 0.0259 | 0.0082 | 0.00277 | 0.00026 | 0.0522 | 0.0208 | 0.0084 | 0.0009 |
| Dry now | Persistence | 0.0679 | 0.1611 | 0.0247 | 0.0079 | 0.00272 | 0.00026 | 0.0253 | 0.0079 | 0.0027 | 0.0003 |
| Dry now | Logistic regression | 0.0544 | 0.1142 | 0.0187 | 0.0066 | 0.00243 | 0.00025 | 0.3962 | 0.2608 | 0.1616 | 0.0410 |
| Dry now | Random forest | 0.0530 | 0.1111 | 0.0182 | 0.0065 | 0.00241 | 0.00025 | 0.4298 | 0.2959 | 0.1903 | 0.0553 |
| Raining now | Constant | 1.9499 | 2.9833 | 0.5544 | 0.2635 | 0.08926 | 0.01232 | 0.6275 | 0.2759 | 0.0907 | 0.0123 |
| Raining now | Climatology | 1.8859 | 2.8948 | 0.5278 | 0.2578 | 0.08825 | 0.01228 | 0.6403 | 0.2977 | 0.1204 | 0.0236 |
| Raining now | Persistence | 0.9580 | 1.6771 | 0.1935 | 0.1515 | 0.07024 | 0.01185 | 0.8230 | 0.6128 | 0.3401 | 0.0628 |
| Raining now | Logistic regression | 0.6487 | 1.3581 | 0.1324 | 0.1004 | 0.05017 | 0.01001 | 0.9324 | 0.8222 | 0.6352 | 0.2974 |
| Raining now | Random forest | 0.6140 | 1.2924 | 0.1219 | 0.0952 | 0.04934 | 0.01009 | 0.9427 | 0.8418 | 0.6466 | 0.2885 |


## Appendix D. Reproduction Instructions and Model Files

The code, documentation, trained models and small result tables are in the repository
(https://github.com/LuweiChenbot/RainfallPatternPrediction). Raw data and the generated Parquet tables are not included.

### D.1 Environment

Java 17 or 21, Python 3.12, PySpark 4.2.0. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # set cores and memory for the machine
```

### D.2 Run Steps

1. Download the eight annual files of the *Historical Rainfall across Singapore* collection (data.gov.sg) into `data/`.
2. Task 1: run, in order, `task1/code/rainfall_audit.py` (optional), `prepare_base_data.py`, `prepare_temporal_data.py`,
   `fetch_boundary.py` and `build_matrix.py`, then the four `plot_*.py` scripts (commands in `task1/README.md`). This
   produces the five-minute station matrix `task1/data/matrix_5min/` and the Task 1 figures.
3. Task 2: `python task2/run_pipeline.py` runs notebooks 01, 02, 05, 06, 08, 03, 04, 07, 09 and 10 in that order.
   Settings of the final run: `RAIN_TRAIN=train_full`, `RAIN_RF_TREES=50`, `RAIN_RF_MEMORY_MB=512`.
4. Figures and the replay page: `python task2/code/plot_results.py` and `python task2/code/build_replay.py`.

### D.3 Submitted model files

| Path | Content | Size |
|---|---|---|
| `task2/models/lr/` | Spark ML pipeline: logistic regression, with its input transformations | 0.1 MB |
| `task2/models/rf/` | Spark ML pipeline: random forest, 50 trees | 49 MB |
| `task2/models/mapper.json` | threshold mapper: how the rain of each class is spread, used to turn class probabilities into P(Y > θ) | 8 KB |

To use them, with the feature tables of `task2/data/` in place:

```bash
python task2/code/predict.py S90 "2024-12-29 16:40" 5            
python task2/code/predict.py S90 "2024-12-29 16:40" 5 --model LR
```
