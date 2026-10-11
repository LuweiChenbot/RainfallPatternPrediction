# Task 1 revision

This folder revises the Task 1 figures following the review in `comments on task1.rtf`, and adds analyses
that motivate Task 2 features. It is isolated from the original pipeline: it reads
`outputs/task1/` and `data/processed/rainfall_aligned/` and writes only inside `task1_revision/`.
The `data/` and `outputs/` subfolders are git-ignored, like the rest of the project's data.

## Layout

```text
task1_revision/
  code/
    fetch_boundary.py     URA Master Plan 2019 region boundary from data.gov.sg (map outline and regions)
    build_matrix.py       Spark: aligned Parquet -> dense five-minute station matrix (null = no reading)
    common.py             paths, palette, plotting style, matrix and correlation helpers
    plot_preliminary.py   section 0: data reliability
    plot_time.py          section 1: when it rains
    plot_space.py         section 2: where it rains and how stations relate
    plot_amount.py        section 3: how much it rains and simple models
  data/                   matrix_5min/ (6.6 MB Parquet), external/ (boundary GeoJSON)
  outputs/
    figures/              PNG and SVG
    tables/               small CSV tables behind the figures
    results.json          headline numbers quoted in the figure captions and the change log
```

## Run

From the repository root, after the original preparation and `task1_analysis.py` have run:

```sh
python task1_revision/code/fetch_boundary.py
python task1_revision/code/build_matrix.py          # a few minutes, Spark local[4]
cd task1_revision/code
python plot_preliminary.py
python plot_time.py                                  # about 15 s
python plot_space.py                                 # about 2 minutes (lagged correlations)
python plot_amount.py                                # about 10 s
```

`build_matrix.py` refuses to overwrite an existing matrix. Spark performs all row-level work; numpy
then uses the dense matrix for lagged correlations and rain-event segmentation, which are matrix
products that pairwise self-joins in Spark SQL could not do efficiently on a laptop.

## Figures

Twelve figures, numbered by section (first digit) and order within the section.

| Figure | Question | Finding | Task 2 implication |
|---|---|---|---|
| 00_coverage_by_month | Which stations reported when? | 11 months have no station at >= 90% coverage; these are the grey cells of the old heatmap | Weight or filter by coverage; never treat gaps as zero |
| 10_seasonal_cycle | Is there real seasonality, and can a regression describe it? (idea 2) | Month explains 28% of monthly variation (p = 0.006); only November is significant; two sine/cosine pairs capture the cycle | Encode day-of-year as sin/cos pairs; season is weak at monthly scale |
| 11_anomalies | Which years were unusual? | 2019 is driest (-2.1 mm/day), borderline after multiple-testing correction; annual means move <= 0.16 mm/day across 80/90/95% cutoffs | Year-level effects are small; do not overfit them |
| 12_diurnal | What drives the afternoon peak? (idea 2 for the daily cycle) | Frequency (5.1x) far more than intensity (1.6x); three sine/cosine pairs give R² 0.82 | Encode hour as sin/cos pairs; hour mainly predicts whether it rains |
| 13_month_hour | Does timing change with season? | Morning rain is 32-39% of the total in Jun-Sep vs 8-9% in Feb, Mar, Dec | Let hour interact with season |
| 20_station_map | Where is it wetter? | West 1.11 vs East 0.80 of the network mean | Station location or region as a feature |
| 21_correlation_dynamics | Idea 1: how do neighbours relate? | Correlation reach grows from 10 km (5 min) to 32 km (1 day) | Neighbour rainfall; radius grows with the forecast window |
| 22_rainfall_regions | Are there rainfall regions? | West / Central / East clusters, stable across odd and even years | Cluster label or cluster-mean rainfall |
| 23_lead_lag | Do storms move? | SW monsoon storms move east at about 38 km/h | Lagged upstream (western) rainfall |
| 30_concentration | How concentrated is rain? | Half of all rain falls in 0.3% of 5-minute slots; 45% of days are dry | Separate occurrence and amount, or use a Tweedie loss |
| 31_persistence | Does yesterday matter? | P(wet after wet) 53% vs 38% after dry; memory beyond one day | Lagged indicators and rolling totals |
| 32_compound_poisson_gamma | Does a simple model fit? | Events x gamma sizes fit once the rate follows the day's weather; Tweedie p of about 1.7 | Spark GLR with family = tweedie |

Review on 2026-10-08 removed two earlier figures. `01_coverage_sensitivity` was a whole figure for
one sentence; the number now sits in the `11_anomalies` caption. It also had a bug: it drew only the
80% and 90% lines, so its "<= 0.1 mm/day" understated the spread, which is 0.16 mm/day with 95%
included. `11_harmonic_regression` repeated the data of the seasonal and daily-cycle figures; its
fits now sit inside `10_seasonal_cycle` and `12_diurnal`. The autocorrelation panel was dropped
because the permutation test answers the same question more directly.

## Definitions and caveats

- Rates are exposure-normalised and stations get equal weight, as in Task 1. Monthly figures use
  station-months with >= 90% coverage; annual figures use station-years with >= 80% coverage.
- A wet day has >= 1 mm, a wet hour >= 0.2 mm (one bucket tip). Daily and hourly sums use complete
  periods only.
- A rain event is a run of wet five-minute slots in which dry gaps of up to 30 minutes are bridged.
  It is split at midnight so that a day's total equals the sum of its events.
- Seasons: NE monsoon Dec-Mar, SW monsoon Jun-Sep, inter-monsoon Apr-May and Oct-Nov.
- Coordinates are each station's latest recorded location. Distances use an equirectangular
  projection, which is accurate to well under 1% across Singapore.
- Station elevation is not in the dataset, so no terrain analysis was attempted.
- Correlations between station pairs are not independent, so p-values on pair comparisons are
  indicative. The cluster evidence avoids circularity by building clusters on odd years and testing
  them on even years.
