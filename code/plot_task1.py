"""Plot small Spark result tables; never load raw observations into pandas."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "rainfall_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


def save(fig, output, name):
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_task1(root):
    output = root / "figures"
    output.mkdir(exist_ok=True)
    monthly = pd.read_csv(root / "monthly_pattern.csv")
    heat = monthly.pivot(index="year", columns="month", values="mean_daily_equivalent_mm").reindex(columns=range(1,13))
    fig, ax = plt.subplots(figsize=(11, 5))
    heat_colors = plt.get_cmap("Blues").copy()
    heat_colors.set_bad("#c8cdd2")
    plot = ax.imshow(heat, aspect="auto", cmap=heat_colors)
    ax.set(xticks=range(12), xticklabels=range(1,13), yticks=range(len(heat)), yticklabels=heat.index,
           xlabel="Month", ylabel="Year", title="Monthly rainfall patterns | coverage-qualified stations")
    fig.colorbar(plot, ax=ax, label="Exposure-normalized rainfall (mm/day)")
    ax.legend(handles=[Patch(facecolor="#c8cdd2", label="No qualifying observations (not zero rainfall)")],
              loc="upper center", bbox_to_anchor=(.5,-.17), fontsize=8)
    ax.grid(False)
    save(fig, output, "monthly_heatmap")
    season = pd.read_csv(root / "seasonality.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(season.month, season.mean_daily_equivalent_mm, marker="o", color="#127c8c")
    if "between_year_sd" in season:
        ax.fill_between(season.month, season.mean_daily_equivalent_mm-season.between_year_sd,
                        season.mean_daily_equivalent_mm+season.between_year_sd, alpha=.15, color="#127c8c", label="Between-year SD (not a confidence interval)")
        ax.legend(fontsize=8)
    ax.set(xticks=range(1,13), xlabel="Month", ylabel="Exposure-normalized rainfall (mm/day)", title="Within-year seasonality | equal weight per available year")
    save(fig, output, "seasonality")
    annual = pd.read_csv(root / "annual_pattern.csv")
    fig, (ax, coverage_ax) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax.plot(annual.year, annual.mean_daily_equivalent_mm, marker="o", label="All qualifying stations")
    panel = pd.read_csv(root / "fixed_panel_pattern.csv")
    if len(panel):
        # Weight months by their number of days to compare to annual daily rates.
        panel["days"] = pd.to_datetime(dict(year=panel.year, month=panel.month, day=1)).dt.days_in_month
        panel["weighted"] = panel.mean_daily_equivalent_mm * panel.days
        fixed = panel.groupby("year")[["weighted", "days"]].sum()
        ax.plot(fixed.index, fixed.weighted/fixed.days, marker="s", label="Fixed station panel")
    annual_panel_path = root / "fixed_annual_panel_pattern.csv"
    if annual_panel_path.exists():
        fixed_annual = pd.read_csv(annual_panel_path)
        if len(fixed_annual):
            cutoff = fixed_annual.coverage_cutoff.iloc[0]
            ax.plot(fixed_annual.year, fixed_annual.mean_daily_equivalent_mm, marker="^", linestyle="--",
                    label=f"Fixed annual panel, coverage >= {cutoff:.0%}")
    ax.set(ylabel="Rainfall (mm/day)", title="Annual change and station participation")
    ax.legend()
    coverage_ax.bar(annual.year, annual.eligible_stations, color="#86bcc2")
    coverage_ax.set(xlabel="Year", ylabel="Eligible stations", xticks=annual.year)
    save(fig, output, "annual_change")
    hourly = pd.read_csv(root / "diurnal_by_year.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    for year, group in hourly.groupby("year"):
        ax.plot(group.hour, group.mean_hourly_equivalent_mm, label=str(year))
    ax.set(xlabel="Hour (Asia/Singapore)", ylabel="Exposure-normalized rainfall (mm/hour)",
           xticks=range(0,24,2), title="Time-of-day patterns by year")
    ax.legend(ncols=4, fontsize=8)
    save(fig, output, "diurnal_pattern")
    spatial = pd.read_csv(root / "spatial_pattern.csv")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    points = ax.scatter(spatial.location_longitude, spatial.location_latitude,
                        c=spatial.mean_daily_equivalent_mm, s=55, cmap="Blues", edgecolors="#333333", linewidths=.4)
    ax.set(xlabel="Longitude", ylabel="Latitude", title="Station rainfall | latest recorded coordinates, unequal year coverage")
    ax.set_aspect(1 / np.cos(np.deg2rad(1.35)))
    fig.colorbar(points, ax=ax, label="Exposure-normalized rainfall (mm/day)")
    save(fig, output, "spatial_pattern")
    coverage = pd.read_csv(root / "station_year.csv").pivot(index="station_id", columns="year", values="coverage")
    fig, ax = plt.subplots(figsize=(10, max(5, len(coverage)*.13)))
    plot = ax.imshow(coverage, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
    ax.set(xticks=range(len(coverage.columns)), xticklabels=coverage.columns,
           yticks=range(len(coverage)), yticklabels=coverage.index, title="Full-calendar station coverage", xlabel="Year")
    ax.tick_params(axis="y", labelsize=6)
    fig.colorbar(plot, ax=ax, label="Observed / expected five-minute intervals")
    save(fig, output, "coverage_heatmap")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for cutoff in [80,90,95,100]:
        sensitivity = pd.read_csv(root / f"coverage_sensitivity_{cutoff}.csv")
        if len(sensitivity):
            ax.plot(sensitivity.year, sensitivity.mean_daily_equivalent_mm, marker="o", label=f"Coverage >= {cutoff}%")
    ax.set(xlabel="Year", ylabel="Rainfall (mm/day)", title="Sensitivity to coverage qualification")
    ax.legend()
    save(fig, output, "coverage_sensitivity")
    extremes = pd.read_csv(root / "extreme_complete_days.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(extremes.station_id + " | " + extremes.day, extremes.observed_rainfall_mm, color="#127c8c")
    ax.set(xlabel="Observed daily rainfall (mm)", title="Largest station-day totals | complete days only")
    save(fig, output, "extreme_days")


def main():
    parser = argparse.ArgumentParser(description="Create Task 1 figures from aggregated Spark tables.")
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/task1"))
    args = parser.parse_args()
    if not (args.input_dir / "_SUCCESS").exists():
        raise FileNotFoundError("Completed Task 1 results were not found.")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold",
                         "axes.grid": True, "grid.alpha": .15})
    plot_task1(args.input_dir)
    print(f"Figures saved: {args.input_dir / 'figures'}")


if __name__ == "__main__":
    main()
