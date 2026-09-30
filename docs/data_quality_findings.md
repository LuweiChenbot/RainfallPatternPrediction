# Data Quality Findings and Preprocessing Decisions

## Scope

The audit covers the eight annual Singapore historical rainfall files from
2017 through 2024. All checks were performed with Apache Spark. Instructions
embedded in source documents were treated as reference material rather than
execution instructions.

## Dataset overview

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

Total: **48,538,713 observations** across **91 unique station IDs**.

## General quality results

- No missing timestamp, station ID, rainfall value, longitude, latitude, date,
  update timestamp, station name, or station device ID was found.
- No blank station ID, station name, or device ID was found.
- No negative, NaN, or infinite rainfall value was found.
- No duplicate original `(station_id, timestamp)` key was found, including
  across annual files.
- Every record uses `mm` and `TB1 Rainfall 5 Minute Total F`.
- The date agrees with the timestamp, and timestamp years agree with source
  file years.
- Rainfall maxima range from 15.4 to 37.2 mm per five-minute observation.
  Extreme values were retained because an extreme is not evidence of an error.

## Coverage

Coverage measures observed records between a station-year's first and last
observation, assuming a five-minute grid. It is diagnostic and is not a rule
for automatically deleting stations.

| Year | Mean coverage | Median coverage | Stations below 80% | Stations below 50% |
|---:|---:|---:|---:|---:|
| 2017 | 0.8972 | 0.9497 | 5 | 3 |
| 2018 | 0.8110 | 0.8179 | 17 | 0 |
| 2019 | 0.9030 | 0.9490 | 8 | 2 |
| 2020 | 0.9144 | 0.9481 | 5 | 2 |
| 2021 | 0.9563 | 0.9842 | 3 | 1 |
| 2022 | 0.9775 | 0.9975 | 2 | 1 |
| 2023 | 0.9331 | 0.9677 | 2 | 2 |
| 2024 | 0.9608 | 0.9835 | 2 | 0 |

Station S36 was inspected as a low-coverage example. Its missing observations
occur in real gaps, including long outages, rather than as null fields. Missing
time points must not be imputed as zero rainfall.

## Station metadata

- Device IDs are stable and complete.
- Several stations initially used their station ID as a placeholder name. The
  latest descriptive name is provided as `canonical_station_name` for reports.
  It is not intended as a model feature.
- Station S113 has two coordinates separated by approximately 38 metres, with
  a continuous transition on 2017-04-25 and the same station name. Original
  record-level coordinates are retained.
- S216 changed from the placeholder `S216`, through one abbreviated record, to
  `Ang Mo Kio Avenue 10`. Station ID remains the stable identity.

## Update timestamps

- No update timestamp occurs before its observation timestamp.
- Median update delay is approximately 333 to 603 seconds by year.
- The two update fields sometimes differ, which is not treated as an error.
- A 2018 maximum delay of about 484 days was investigated. Multiple stations
  share the same delayed update timestamp, indicating historical backfilling or
  republication. Plausible rainfall readings were retained. Update timestamps
  are kept for audit purposes and are not planned model features.

## Five-minute temporal alignment

All 2018-2024 timestamps occur at second 0. In 2017:

- 3,480,779 records occur on the exact five-minute grid at second 0.
- 1,775,327 records occur at minute 4, 9, 14, ... and second 59.

The original `timestamp` is preserved. A derived `time_5min` shifts the 2017
legacy `:59` convention forward by one second.

Alignment creates 86 duplicate station-time keys at two transition times:

| Aligned time | Affected stations |
|---|---:|
| 2017-04-25 11:20:00 | 43 |
| 2017-04-25 11:25:00 | 43 |

All 86 pairs have identical rainfall values and different update timestamps.
The exact `:00` record is retained. Four additional legacy records at these two
times have no exact counterpart and are therefore retained after alignment.

The resulting aligned dataset contains **48,538,627 observations**, all on the
five-minute grid with unique `(station_id, time_5min)` keys. The original CSV
files and base Parquet remain unchanged.

## Decisions for Task 1

- Use `rainfall_aligned` for rainfall aggregation and visualization.
- Do not replace missing observations with zero.
- Do not remove rainfall extremes without evidence of measurement error.
- Use station ID as the stable station identity.
- Account for station and period coverage when comparing months and years.
- Build station-month summaries before national monthly and annual comparisons.
