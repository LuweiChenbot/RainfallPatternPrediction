"""Section 0: data reliability, drawn before any rainfall result.

00_coverage_by_month  station x month coverage, with qualifying-station counts

The coverage-cutoff sensitivity check is reported in the caption of 11_anomalies
(annual means move by at most 0.1 mm/day), so it no longer has its own figure.
"""

import numpy as np
import pandas as pd

from common import BLUE, INK2, ORANGE, SEQUENTIAL, TASK1, caption, plt, record, save, style, write_table


def coverage_by_month():
    month = pd.read_csv(TASK1 / "station_month.csv", parse_dates=["month_start"])
    grid = month.pivot(index="station_id", columns="month_start", values="coverage")
    active = month[month.observations > 0]
    first = active.groupby("station_id").month_start.min()
    grid = grid.loc[first.sort_values(kind="stable").index.union(grid.index.difference(first.index), sort=False)]
    qualifying = (grid >= 0.9).sum()
    network = active.groupby("month_start").coverage.median()
    dates = grid.columns
    # Months in which no station reaches 90%: these become the grey (missing)
    # cells of every later monthly figure. Consecutive months form one span.
    empty = [i for i, d in enumerate(dates) if qualifying.loc[d] == 0]
    spans = []
    for i in empty:
        if spans and spans[-1][1] == i - 1:
            spans[-1][1] = i
        else:
            spans.append([i, i])

    fig, (top, ax) = plt.subplots(2, 1, figsize=(12, 11), sharex=False,
                                  gridspec_kw={"height_ratios": [1, 6], "hspace": 0.06})
    positions = np.arange(len(dates))
    top.bar(positions, qualifying.values, width=0.8, color=BLUE)
    for first_i, last_i in spans:
        top.axvspan(first_i - 0.5, last_i + 0.5, color=ORANGE, alpha=0.18, linewidth=0)
        label = dates[first_i].strftime("%b %Y") if first_i == last_i else \
            f"{dates[first_i].strftime('%b')}-{dates[last_i].strftime('%b %Y')}"
        top.text((first_i + last_i) / 2, 4, label, rotation=90, ha="center", va="bottom", fontsize=6.5, color=INK2)
    top.set(ylabel="Stations\n>= 90% coverage", xlim=(-0.5, len(dates) - 0.5))
    top.set_title("Data reliability: which stations reported, month by month", pad=10)
    top.set_xticks([i for i, d in enumerate(dates) if d.month == 1])
    top.tick_params(labelbottom=False)

    image = ax.imshow(grid.values, aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=1, interpolation="nearest")
    years = [i for i, d in enumerate(dates) if d.month == 1]
    ax.set_xticks(years, [dates[i].year for i in years])
    ax.set_yticks(range(len(grid)), grid.index, fontsize=5.2)
    ax.set_ylabel("Station (sorted by first month with data)")
    ax.grid(False)
    for i in years[1:]:
        ax.axvline(i - 0.5, color="white", linewidth=1.2)
    bar = fig.colorbar(image, ax=[top, ax], fraction=0.025, pad=0.01)
    bar.set_label("Observed / expected five-minute readings in the month")
    bar.outline.set_visible(False)
    caption(fig, f"Orange spans: the {len(empty)} months in which no station reached 90% coverage (network-wide "
                 "outages). They are exactly the grey 'no data' cells of the old month-by-year heatmap, "
                 "so those gaps are a recording problem, not dry weather.", y=0.075)
    save(fig, "00_coverage_by_month")

    table = pd.DataFrame({"month_start": dates.date, "stations_ge_90pct": qualifying.values,
                          "median_active_coverage": network.reindex(dates).values})
    write_table(table, "coverage_by_month")
    record("00_coverage", {"months_without_qualifying_station": [dates[i].strftime("%Y-%m") for i in empty],
                           "qualifying_stations_range_other_months": [int(qualifying[qualifying > 0].min()),
                                                                      int(qualifying.max())]})


if __name__ == "__main__":
    style()
    coverage_by_month()
