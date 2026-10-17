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
    common.py             paths, palette, A4 style, caption store, data cleaning, correlation helpers
    plot_preliminary.py   section 0: data reliability
    plot_time.py          section 1: when it rains
    plot_space.py         section 2: where it rains and how stations relate
    plot_amount.py        section 3: day-to-day persistence
  data/                   matrix_5min/ (6.6 MB Parquet), external/ (boundary GeoJSON)
  outputs/
    figures/              PDF (for LaTeX) and 300 dpi PNG of each figure
    captions.json         caption, appendix body text and Task 2 link of each figure, kept off the image
    captions.md           the same, for reading
    figures.tex           LaTeX figure blocks at print width, main text first, then appendix
    tables/               small CSV tables behind the figures
    results.json          headline numbers quoted in the captions
```

## Run

From the repository root, after the original preparation and `task1_analysis.py` have run:

```sh
python task1_revision/code/fetch_boundary.py
python task1_revision/code/build_matrix.py          # a few minutes, Spark local[4]
cd task1_revision/code
python plot_preliminary.py
python plot_time.py                                  # about 45 s
python plot_space.py                                 # about 3 minutes, 5 GB of memory
python plot_amount.py                                # about 10 s
```

`build_matrix.py` refuses to overwrite an existing matrix. Spark performs all row-level work; numpy
then uses the dense matrix for lagged correlations, which are matrix products that pairwise
self-joins in Spark SQL could not do efficiently on a laptop. Each plotting script adds its figures to
the caption store, so run all four after a change to keep `figures.tex` complete.

## Using the figures in the report

Every figure is drawn at its print size for an A4 page with 2.5 cm margins (16 cm, or 6.3 in, of text
width): titles and axis labels at 8 pt, tick labels at 7 pt, light lines, and no text under the plot.
The explanation that used to sit under each figure is now its caption, generated with the figure from
the same numbers. In LaTeX, set `\graphicspath{{task1_revision/outputs/}}` and `\input` the blocks
from `outputs/figures.tex`, or copy single blocks; keep the stored widths so the lettering prints at
7-8 pt next to 10 pt captions. `11_anomalies` is meant for the appendix, and its two-sentence summary
for the main text is stored as its body text.

## Figures

Nine figures, numbered by section (first digit) and order within the section.

| Figure | Question | Finding | Task 2 implication |
|---|---|---|---|
| 00_coverage_by_month | Which stations reported when? | 11 months have no station at >= 90% coverage | Never treat gaps as zero; Task 2 keeps only complete 30-minute windows instead of filtering by coverage |
| 10_seasonal_cycle | Is there real seasonality, and can a regression describe it? (idea 2) | Month explains 28% of monthly variation (p = 0.006); only November is significant; two sine/cosine pairs capture the cycle | Encode day-of-year as sin/cos pairs; season is weak at monthly scale |
| 11_anomalies (appendix) | Which years were unusual? | 2019 is driest (-2.1 mm/day), borderline after multiple-testing correction; annual means move <= 0.16 mm/day across 80/90/95% cutoffs | Year-level effects are small; do not overfit them |
| 12_diurnal | What drives the afternoon peak? (idea 2 for the daily cycle) | Frequency (5.1x) far more than intensity (1.6x); three sine/cosine pairs give R² 0.82 | Encode hour as sin/cos pairs; hour mainly predicts whether it rains |
| 13_month_hour | Does timing change with season? | Morning rain is 32-39% of the total in Jun-Sep vs 8-9% in Feb, Mar, Dec | Let hour interact with season |
| 20_station_map | Where is it wetter, and which stations rain together? | Colour: Jurong West Street 73 gets 1.31 times the network rate, Changi East Close 0.61. Shape: West / Central / East clusters, stable across odd and even years; the West cluster averages 1.11, Central and East 0.94 | Station wetness ratio, position or cluster label |
| 21_correlation_dynamics | Idea 1: how do neighbours relate? | Correlation reach grows from 9 km (5 min) to 32 km (1 day) | Neighbour rainfall; radius grows with the forecast window |
| 23_lead_lag | Do storms move? | SW monsoon storms move east at about 38 km/h | Lagged upstream (western) rainfall |
| 31_persistence | Does yesterday matter? | P(wet after wet) 53% vs 38% after dry; the gap is widest in January and vanishes in June | Lagged indicators and rolling totals |

Review on 2026-10-08 removed two earlier figures. `01_coverage_sensitivity` was a whole figure for
one sentence; the number now sits in the `11_anomalies` caption. It also had a bug: it drew only the
80% and 90% lines, so its "<= 0.1 mm/day" understated the spread, which is 0.16 mm/day with 95%
included. `11_harmonic_regression` repeated the data of the seasonal and daily-cycle figures; its
fits now sit inside `10_seasonal_cycle` and `12_diurnal`. The autocorrelation panel was dropped
because the permutation test answers the same question more directly.

Review on 2026-10-12 made the figures report-ready:

- Style: figures are drawn at A4 print size with lighter lettering (no bold, thinner lines), and the
  text under each figure moved into the stored captions.
- Content: `10_seasonal_cycle` keeps its first two panels, and its test results moved to the caption.
  `11_anomalies` moves to the appendix without the year-effect lines. `13_month_hour` has its colour
  scale on the left. `20_station_map` absorbs the old `22_rainfall_regions`: marker shape is the
  cluster, colour the relative wetness. `21_correlation_dynamics` and `31_persistence` keep only
  their first panel. `23_lead_lag` is laid out 2 x 2 to stay legible at A4 width.
  `30_concentration` and `32_compound_poisson_gamma` were dropped.
- Data: the rainfall figures now drop the readings Task 2 found to be wrong (see below), so both tasks
  use the same data. This raises the network mean of January-March 2017 and March 2021 by 0.10-0.16
  mm/day and lowers August 2022 by 0.53 mm/day. It also moves 8 of the 69 core stations between clusters
  (seven south-western stations from Central to West, Marine Parade from Central to East), all because
  removing three days of spurious rain at S107 changes its correlations (the other masks move no
  station), and agreement with the official regions drops from 0.49 to 0.42. Correlation reach changes
  by at most 0.2 km, storm speed by at most 0.2 km/h and storm direction by at most 0.4 degrees; the
  wettest and driest stations stay the same.

## Definitions and caveats

- Rates are exposure-normalised and stations get equal weight, as in Task 1. Monthly figures use
  station-months with >= 90% coverage; annual figures use station-years with >= 80% coverage.
- Readings Task 2 found to be wrong are set to missing before any rainfall figure (`00` shows the raw
  coverage): 18 station-months in which a gauge recorded no wet reading although, given how often the
  network was wet while it reported, at least 30 were expected (a stuck gauge), and the faulty periods
  in `task2/code/config.py` (S107 19-22 Aug 2022, S36 Feb 2019-Aug 2021). Together that is 128,849
  readings.
- A wet five-minute reading has > 0 mm (at least one 0.2 mm bucket tip); a wet day has >= 1 mm.
  Daily sums use complete days only.
- Seasons: NE monsoon Dec-Mar, SW monsoon Jun-Sep, inter-monsoon Apr-May and Oct-Nov.
- Coordinates are each station's latest recorded location. Distances use an equirectangular
  projection, which is accurate to well under 1% across Singapore.
- Station elevation is not in the dataset, so no terrain analysis was attempted.
- Correlations between station pairs are not independent, so p-values on pair comparisons are
  indicative. The cluster evidence avoids circularity by building clusters on odd years and testing
  them on even years.
