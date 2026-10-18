"""Section 1: when does it rain?

10_seasonal_cycle   year curves, mean with 95% CI and a sine/cosine regression, split into its two waves
11_anomalies        deseasonalised monthly anomalies, annual means at the 80% cutoff (appendix)
12_diurnal          daily cycle with its sine/cosine regression, split into how often vs how hard
13_month_hour       season x time of day: the timing of rain changes with the monsoon
"""

import calendar

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import (AQUA, AXIS, BLUE, INK, INK2, MUTED, ORANGE, SEQUENTIAL, TEXT_WIDTH, load_matrix,
                    network_monthly, plt, record, save, station_periods, style, write_table)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
rng = np.random.default_rng(5105)


def monthly_values(X, stations, times):
    """Network-mean rate (mm/day) for every month of 2017-2024; NaN where no station reached 90% coverage."""
    full = pd.DataFrame([(y, m) for y in range(2017, 2025) for m in range(1, 13)], columns=["year", "month"])
    return full.merge(network_monthly(X, stations, times).rename(columns={"rate": "value"}), how="left")


def sum_coded_effects(fit, factor, levels):
    """Every level's deviation from the average level, with a t-test each."""
    names = [n for n in fit.params.index if n.startswith(f"C({factor}, Sum)")]
    rows = []
    for k, level in enumerate(levels):
        contrast = np.zeros(len(fit.params))
        if k < len(names):
            contrast[fit.params.index.get_loc(names[k])] = 1
        else:  # the omitted level equals minus the sum of the others
            for name in names:
                contrast[fit.params.index.get_loc(name)] = -1
        test = fit.t_test(contrast)
        rows.append((level, float(np.squeeze(test.effect)), float(np.squeeze(test.sd)),
                     float(np.squeeze(test.pvalue))))
    table = pd.DataFrame(rows, columns=[factor, "effect", "se", "p"])
    table["p_holm"] = multipletests(table.p, method="holm")[1]
    return table


def harmonics(phase, K):
    return np.column_stack([f(2 * np.pi * k * phase) for k in range(1, K + 1) for f in (np.cos, np.sin)])


def fit_harmonics(y, phase, K):
    return sm.OLS(y, sm.add_constant(harmonics(phase, K))).fit()


def predict_harmonics(fit, phase, K):
    return sm.add_constant(harmonics(phase, K), has_constant="add") @ fit.params


def clock(hour):
    return f"{int(hour):02d}:{int(round(hour % 1 * 60)) % 60:02d}"


# ------------------------------------------------------------ 10 ----------

def seasonal_cycle(monthly):
    data = monthly.dropna(subset=["value"]).copy()
    fit = smf.ols("value ~ C(month, Sum) + C(year, Sum)", data).fit()
    anova = sm.stats.anova_lm(fit, typ=2)
    share = anova.sum_sq / anova.sum_sq.sum()

    def amplitude(frame):
        return frame.groupby("month").value.mean().var()
    observed = amplitude(data)
    shuffled = data.copy()
    null = []
    for _ in range(10000):  # shuffle months within each year: keeps year effects, breaks seasonality
        shuffled["month"] = data.groupby("year").month.transform(lambda m: rng.permutation(m.values))
        null.append(amplitude(shuffled))
    p_perm = (np.sum(np.array(null) >= observed) + 1) / (len(null) + 1)
    effects = sum_coded_effects(fit, "month", range(1, 13))

    y, phase = data.value.to_numpy(), (data.month.to_numpy() - 0.5) / 12
    h1, h2 = fit_harmonics(y, phase, 1), fit_harmonics(y, phase, 2)
    dummies = smf.ols("value ~ C(month)", data).fit()
    p_second = h2.compare_f_test(h1)[1]
    p_rest = dummies.compare_f_test(h2)[1]
    base, a1, b1, a2, b2 = h2.params
    smooth = np.linspace(0, 1, 361)
    month_axis = smooth * 12 + 0.5
    annual = a1 * np.cos(2 * np.pi * smooth) + b1 * np.sin(2 * np.pi * smooth)
    semi = a2 * np.cos(4 * np.pi * smooth) + b2 * np.sin(4 * np.pi * smooth)

    grid = monthly.pivot(index="year", columns="month", values="value")
    n = grid.notna().sum()
    mean, sd = grid.mean(), grid.std()
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)

    fig, (ax, comp) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.7), layout="constrained",
                                   gridspec_kw={"width_ratios": [1.45, 1]})
    for i, (_, row) in enumerate(grid.iterrows()):
        ax.plot(range(1, 13), row.values, color=AXIS, linewidth=0.7, zorder=1,
                label="Single year" if i == 0 else None)
    ax.fill_between(range(1, 13), mean - half, mean + half, color=BLUE, alpha=0.12, linewidth=0,
                    label="95% CI of the mean", zorder=2)
    ax.plot(range(1, 13), mean.values, color=BLUE, marker="o", markeredgecolor="white", markeredgewidth=0.6,
            label="Mean of available years", zorder=3)
    ax.plot(month_axis, predict_harmonics(h2, smooth, 2), color=INK, linewidth=0.9, zorder=4,
            label=f"Two sine/cosine pairs (R² = {h2.rsquared:.2f})")
    for _, row in effects[effects.p_holm < 0.05].iterrows():
        m = int(row.month)
        ax.annotate(f"{MONTHS[m - 1]}: {row.effect:+.1f} mm/day\n(Holm p = {row.p_holm:.3f})", (m, mean[m]),
                    xytext=(-50, 42), textcoords="offset points", ha="left", fontsize=6.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.5))
    top = np.nanmax(grid.values) * 1.42
    ax.set(xticks=range(1, 13), xlabel="Month", ylabel="Rainfall rate (mm/day)", ylim=(0, top), xlim=(0.5, 12.5))
    ax.set_xticklabels([m[0] for m in MONTHS])
    ax.set_title("(a) Monthly rate by year")
    ax.legend(loc="upper left", fontsize=6.5)

    comp.axhline(0, color=AXIS, linewidth=0.6)
    comp.plot(month_axis, annual, color=ORANGE, label=f"Annual wave ({np.hypot(a1, b1):.2f})")
    comp.plot(month_axis, semi, color=AQUA, label=f"Half-year wave ({np.hypot(a2, b2):.2f})")
    comp.plot(month_axis, annual + semi, color=INK, linewidth=0.9, label="Sum (fitted curve)")
    month_means = data.groupby("month").value.mean() - base
    comp.plot(month_means.index, month_means.values, "o", color=BLUE, markeredgecolor="white",
              markeredgewidth=0.6, label="Observed month means")
    low, high = month_means.min(), month_means.max()
    comp.set(xticks=range(1, 13), xlabel="Month", ylabel="Deviation from yearly mean (mm/day)", xlim=(0.5, 12.5),
             ylim=(low - 0.6, high + (high - low) * 0.9))
    comp.set_xticklabels([m[0] for m in MONTHS])
    comp.set_title("(b) Two waves (amplitude in mm/day)")
    comp.legend(loc="upper left", fontsize=6.5)

    significant = effects[effects.p_holm < 0.05]
    named = "; ".join(f"{calendar.month_name[int(r.month)]} {r.effect:+.1f} mm/day, Holm-adjusted p = {r.p_holm:.3f}"
                      for _, r in significant.iterrows())
    save(fig, "10_seasonal_cycle",
         "Seasonal cycle of the network-mean rainfall rate. (a) Rate in each month of each year (grey), the mean "
         "of the available years with its 95% confidence interval (blue), and a regression on two sine/cosine "
         "pairs (black). (b) The fitted curve as the sum of an annual and a half-year wave; dots are the observed "
         f"month means relative to the yearly mean. In a two-way ANOVA, month of year explains {share['C(month, Sum)']:.0%} "
         f"of the monthly variation (p = {anova.loc['C(month, Sum)', 'PR(>F)']:.3f}), year {share['C(year, Sum)']:.0%} "
         f"(p = {anova.loc['C(year, Sum)', 'PR(>F)']:.2f}), and {share['Residual']:.0%} is left to weather specific to "
         f"one month of one year; a permutation test that shuffles months within years confirms the seasonality "
         f"(p = {p_perm:.3f}). The half-year wave is stronger than the annual one, which gives the two "
         f"inter-monsoon peaks; adding it improves the fit (R² from {h1.rsquared:.2f} to {h2.rsquared:.2f}, "
         f"p = {p_second:.3f}), while twelve month dummies add nothing further (R² = {dummies.rsquared:.2f}, "
         f"p = {p_rest:.2f}). Only one month differs significantly from an average month ({named}).",
         task2="Encode the day of year as two sine/cosine pairs rather than a month number.")
    write_table(effects.round(4), "month_effects")
    record("10_seasonal", {
        "variance_share": {k: round(float(v), 3) for k, v in share.items()},
        "anova_p_month": float(anova.loc["C(month, Sum)", "PR(>F)"]),
        "anova_p_year": float(anova.loc["C(year, Sum)", "PR(>F)"]),
        "permutation_p_seasonality": float(p_perm),
        "significant_months_holm": {MONTHS[int(r.month) - 1]: [round(r.effect, 2), round(r.p_holm, 4)]
                                    for _, r in significant.iterrows()},
        "harmonic_r2": {"1_wave": round(h1.rsquared, 3), "2_waves": round(h2.rsquared, 3),
                        "12_dummies": round(dummies.rsquared, 3)},
        "p_second_wave": float(p_second), "p_dummies_beyond_2_waves": float(p_rest),
        "amplitude_annual": round(float(np.hypot(a1, b1)), 3), "amplitude_half_year": round(float(np.hypot(a2, b2)), 3),
    })


# ------------------------------------------------------------ 11 ----------

def anomalies(monthly, X, stations, times):
    data = monthly.copy()
    climate = data.groupby("month").value.mean()
    data["anomaly"] = data.value - data.month.map(climate)
    data["date"] = pd.to_datetime(dict(year=data.year, month=data.month, day=15))
    fit = smf.ols("value ~ C(month, Sum) + C(year, Sum)", data.dropna(subset=["value"])).fit()
    years = sum_coded_effects(fit, "year", range(2017, 2025))
    station_years = station_periods(X, stations, times, "year")
    cutoffs = pd.concat({c: station_years[station_years.coverage >= c / 100].groupby("year").rate.mean()
                         for c in (80, 90, 95)}, axis=1).dropna()
    cutoff_spread = float((cutoffs.max(axis=1) - cutoffs.min(axis=1)).max())
    annual80 = (station_years[station_years.coverage >= 0.8].groupby("year")
                .agg(rate=("rate", "mean"), stations=("rate", "size")).reset_index())

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, (ax, rate_ax, n_ax) = plt.subplots(3, 1, figsize=(TEXT_WIDTH, 4.4), sharex=True, layout="constrained",
                                            gridspec_kw={"height_ratios": [2.4, 1.3, 0.9]})
    ok = data.dropna(subset=["anomaly"])
    ax.bar(ok.date, ok.anomaly, width=25, color=np.where(ok.anomaly >= 0, BLUE, ORANGE), edgecolor="none")
    missing = data[data.anomaly.isna()]
    ax.plot(missing.date, np.zeros(len(missing)), "x", color=MUTED, markersize=3.5, markeredgewidth=0.8)
    handles = [Patch(color=BLUE, label="Wetter than usual"), Patch(color=ORANGE, label="Drier than usual"),
               Line2D([], [], color=MUTED, marker="x", linestyle="none", label="No station at 90% coverage")]
    ax.axhline(0, color=AXIS, linewidth=0.6)
    reach = np.nanmax(np.abs(ok.anomaly))
    ax.set(ylabel="Anomaly (mm/day)", ylim=(ok.anomaly.min() * 1.15, ok.anomaly.max() * 1.3))
    ax.set_title("(a) Monthly anomaly: network-mean rate minus that month's long-run mean")
    ax.legend(handles=handles, loc="upper left", ncols=3, fontsize=6.5)

    mid = pd.to_datetime(dict(year=annual80.year, month=7, day=1))
    rate_ax.plot(mid, annual80.rate, color=BLUE, marker="o", markeredgecolor="white", markeredgewidth=0.6)
    for when, value in zip(mid, annual80.rate):
        rate_ax.annotate(f"{value:.1f}", (when, value), xytext=(0, 4), textcoords="offset points", ha="center",
                         fontsize=6.5, color=INK2)
    rate_ax.set(ylabel="mm/day", ylim=(0, annual80.rate.max() * 1.3))
    rate_ax.set_title("(b) Annual mean rate, station-years with at least 80% coverage")
    n_ax.bar(mid, annual80.stations, width=120, color=BLUE)
    for when, value in zip(mid, annual80.stations):
        n_ax.annotate(str(value), (when, value), xytext=(0, 1.5), textcoords="offset points", ha="center",
                      fontsize=6.5, color=INK2)
    n_ax.set(ylabel="Stations", ylim=(0, annual80.stations.max() * 1.35))
    n_ax.set_title("(c) Stations in (b)")

    driest = years.loc[years.effect.idxmin()]
    others = annual80[annual80.year != int(driest.year)].rate
    save(fig, "11_anomalies",
         "Year-to-year variation of rainfall. (a) Monthly anomalies of the network-mean rate (blue wetter, orange "
         "drier than the long-run mean of the same month); crosses mark months in which no station reached 90% "
         "coverage. (b) Annual mean rate over station-years with at least 80% coverage, and (c) the number of such "
         f"station-years. Annual means move by at most {cutoff_spread:.2f} mm/day when the coverage cutoff changes "
         "between 80%, 90% and 95%.",
         placement="appendix",
         body=f"Rainfall varies far more from month to month than from year to year: monthly anomalies swing by up "
              f"to {reach:.0f} mm/day either way, but after removing each month's own mean only {int(driest.year)} "
              f"stands out, as a dry year ({driest.effect:+.1f} mm/day against an average year, raw p = {driest.p:.3f}), "
              f"and even this is borderline once all eight years are tested (Holm-adjusted p = {driest.p_holm:.3f}). "
              f"The other years' annual means lie between {others.min():.1f} and {others.max():.1f} mm/day, and the "
              f"coverage cutoff changes them by at most {cutoff_spread:.2f} mm/day (appendix figure).")
    write_table(years.round(4), "year_effects")
    write_table(cutoffs.reset_index().round(3), "annual_mean_by_cutoff")
    record("11_anomalies", {"year_effects": {int(r.year): [round(r.effect, 2), round(r.p, 3), round(r.p_holm, 3)]
                                             for _, r in years.iterrows()},
                            "annual_rate_80pct": annual80.set_index("year").rate.round(2).to_dict(),
                            "stations_80pct": annual80.set_index("year").stations.to_dict(),
                            "max_spread_between_cutoffs": round(cutoff_spread, 3)})


# ------------------------------------------------------------ 12/13 -------

def group_sums(X, groups, n_groups):
    S = X.shape[1]
    rain, wet, count = (np.zeros((n_groups, S)) for _ in range(3))
    for s in range(S):
        ok = ~np.isnan(X[:, s])
        g, v = groups[ok], X[ok, s]
        rain[:, s] = np.bincount(g, weights=v, minlength=n_groups)
        wet[:, s] = np.bincount(g, weights=(v > 0), minlength=n_groups)
        count[:, s] = np.bincount(g, minlength=n_groups)
    return rain, wet, count


def diurnal(X, times):
    hour = times.hour.to_numpy()
    year = times.year.to_numpy() - 2017
    expected = np.bincount(year * 24 + hour, minlength=8 * 24)[:, None]
    rain, wet, count = group_sums(X, year * 24 + hour, 8 * 24)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(count >= 0.8 * expected, rain / count * 12, np.nan)
    years = pd.DataFrame({"year": np.repeat(range(2017, 2025), 24), "hour": np.tile(range(24), 8),
                          "rate": np.nanmean(rate, axis=1)})

    r, w, c = (a.reshape(8, 24, -1).sum(axis=0) for a in (rain, wet, count))
    keep = c >= 365 * 12  # at least a year's worth of readings in that hour
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = pd.DataFrame({
            "hour": range(24),
            "rate": np.nanmean(np.where(keep, r / c * 12, np.nan), axis=1),
            "frequency": np.nanmean(np.where(keep, w / c, np.nan), axis=1),
            "intensity": np.nanmean(np.where(keep & (w > 0), r / w, np.nan), axis=1)})
    share = pooled.rate / pooled.rate.sum()
    afternoon = share[(pooled.hour >= 13) & (pooled.hour <= 18)].sum()

    hy, hphase = years.rate.to_numpy(), (years.hour.to_numpy() + 0.5) / 24
    hfits = {K: fit_harmonics(hy, hphase, K) for K in (1, 2, 3)}
    hdummies = smf.ols("rate ~ C(hour)", years).fit()
    smooth = np.linspace(0, 1, 481)
    hour_axis = smooth * 24 - 0.5
    curve = predict_harmonics(hfits[3], smooth, 3)
    peak_time = hour_axis[np.argmax(curve)]
    spread = years.pivot(index="hour", columns="year", values="rate")
    peak_by_year = spread.idxmax().astype(int)

    fig, (ax, idx_ax) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.6), layout="constrained")
    for year in spread.columns:
        ax.plot(spread.index, spread[year], color=AXIS, linewidth=0.7, zorder=1)
    ax.plot([], [], color=AXIS, linewidth=0.7, label="Single year")
    ax.plot(pooled.hour, pooled.rate, color=BLUE, marker="o", markeredgecolor="white", markeredgewidth=0.6,
            label="All years pooled", zorder=3)
    ax.plot(hour_axis, curve, color=INK, linewidth=0.9, zorder=4,
            label=f"Three sine/cosine pairs (R² = {hfits[3].rsquared:.2f})")
    ax.axvspan(12.5, 18.5, color=BLUE, alpha=0.07, linewidth=0)
    top = np.nanmax(spread.values) * 1.3
    ax.text(18.8, np.nanmax(spread.values) * 0.95, f"13:00-18:59:\n{afternoon:.0%} of the rain", ha="left",
            va="top", fontsize=6.5, color=INK2)
    ax.set(xlabel="Hour of day", ylabel="Rainfall rate (mm/hour)", xticks=range(0, 24, 3),
           xlim=(-0.5, 23.5), ylim=(0, top))
    ax.set_title("(a) Rainfall rate by hour of day")
    ax.legend(loc="upper left", fontsize=6.5)

    for column, color, label, width in (("rate", MUTED, "Rate", 0.9), ("frequency", BLUE, "Frequency", 1.1),
                                        ("intensity", ORANGE, "Intensity", 1.1)):
        idx_ax.plot(pooled.hour, pooled[column] / pooled[column].mean(), color=color, label=label, linewidth=width)
    idx_ax.axhline(1, color=AXIS, linewidth=0.6)
    idx_ax.set(xlabel="Hour of day", ylabel="Relative to 24-hour mean", xticks=range(0, 24, 3), xlim=(-0.5, 23.5))
    idx_ax.set_title("(b) How often and how hard it rains")
    idx_ax.legend(loc="upper left", fontsize=6.5)
    peak, trough = pooled.loc[pooled.rate.idxmax()], pooled.loc[pooled.rate.idxmin()]
    save(fig, "12_diurnal",
         "Daily cycle of rainfall, Singapore time. (a) Mean rate per hour in each year (grey), pooled over all "
         f"years (blue), and a regression on three sine/cosine pairs (black; R² = {hfits[3].rsquared:.2f} against "
         f"{hdummies.rsquared:.2f} for 24 hour dummies) that peaks at about {clock(peak_time)}. {afternoon:.0%} of the "
         f"daily rain falls between 13:00 and 18:59, and the peak hour is {peak_by_year.min()}:00 or "
         f"{peak_by_year.max()}:00 in every year. (b) The rate split into how often it rains (share of wet "
         "five-minute readings) and how hard (rain per wet reading), each relative to its 24-hour mean: at "
         f"{int(peak.hour):02d}:00 rain is {peak.frequency / trough.frequency:.1f} times as frequent as at "
         f"{int(trough.hour):02d}:00 but only {peak.intensity / trough.intensity:.1f} times as heavy.",
         task2="Encode the hour as sine/cosine pairs; the hour mainly predicts whether it rains, not how much.")
    write_table(pooled.round(5), "diurnal_pooled")
    record("12_diurnal", {"afternoon_13_19_share": round(float(afternoon), 3),
                          "peak_hour": int(peak.hour), "trough_hour": int(trough.hour),
                          "frequency_ratio_peak_vs_trough": round(float(peak.frequency / trough.frequency), 2),
                          "intensity_ratio_peak_vs_trough": round(float(peak.intensity / trough.intensity), 2),
                          "peak_hour_by_year": peak_by_year.to_dict(),
                          "harmonic_r2": {f"{K}_pairs": round(hfits[K].rsquared, 3) for K in (1, 2, 3)}
                          | {"24_dummies": round(hdummies.rsquared, 3)},
                          "fitted_peak_time": clock(peak_time)})


def month_hour(X, times):
    groups = (times.month.to_numpy() - 1) * 24 + times.hour.to_numpy()
    rain, wet, count = group_sums(X, groups, 12 * 24)
    keep = count >= 60 * 12  # at least two months' worth of readings in the cell
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.nanmean(np.where(keep, rain / count * 12, np.nan), axis=1).reshape(12, 24)
    share = rate / rate.sum(axis=1, keepdims=True)
    morning = share[:, 4:10].sum(axis=1)
    afternoon = share[:, 13:19].sum(axis=1)

    fig = plt.figure(figsize=(TEXT_WIDTH, 2.7), layout="constrained")
    layout = fig.add_gridspec(1, 3, width_ratios=[0.05, 2.2, 1])
    bar_ax, ax, side = (fig.add_subplot(layout[i]) for i in range(3))
    image = ax.imshow(rate, aspect="auto", cmap=SEQUENTIAL, interpolation="nearest")
    ax.set(xticks=range(0, 24, 3), xlabel="Hour of day", yticks=range(12))
    ax.set_yticklabels(MONTHS)
    ax.grid(False)
    ax.set_title("(a) Rainfall rate by month and hour")
    bar = fig.colorbar(image, cax=bar_ax)
    bar_ax.yaxis.set_ticks_position("left")
    bar_ax.yaxis.set_label_position("left")
    bar.set_label("Rainfall rate (mm/hour)")
    bar.outline.set_visible(False)

    side.plot(afternoon, range(12), color=BLUE, marker="o", markeredgecolor="white", markeredgewidth=0.6,
              label="13:00-18:59")
    side.plot(morning, range(12), color=ORANGE, marker="o", markeredgecolor="white", markeredgewidth=0.6,
              label="04:00-09:59")
    side.set(yticks=range(12), yticklabels=[], ylim=(11.5, -0.5), xlabel="Share", xlim=(0, 0.8))
    side.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    side.set_title("(b) Share of the month's rain")
    side.legend(loc="center right", bbox_to_anchor=(1, 0.375), fontsize=6.5)
    wettest = int(np.argmax(morning))
    driest = int(np.argmin(morning))
    full = calendar.month_name[1:]
    flipped = [full[i] for i in range(12) if morning[i] >= afternoon[i]]
    flipped = flipped[0] if len(flipped) == 1 else f"{', '.join(flipped[:-1])} and {flipped[-1]}"
    save(fig, "13_month_hour",
         "Timing of rain through the year, Singapore time. (a) Mean rainfall rate by month and hour of day; the "
         "colour scale is on the left. (b) Share of each month's rain that falls in the afternoon (13:00-18:59) and "
         f"in the early morning (04:00-09:59). Early-morning rain peaks in {full[wettest]} ({morning[wettest]:.0%} of "
         f"that month's rain) against {morning[driest]:.0%} in {full[driest]}, and in {flipped} the morning brings at "
         "least as much rain as the afternoon, consistent with the pre-dawn Sumatra squalls of the southwest monsoon.",
         task2="Let the hour interact with the season.")
    write_table(pd.DataFrame(rate, index=MONTHS, columns=range(24)).reset_index(names="month").round(4), "month_hour_rate")
    record("13_month_hour", {"morning_share_by_month": dict(zip(MONTHS, np.round(morning, 3))),
                             "afternoon_share_by_month": dict(zip(MONTHS, np.round(afternoon, 3)))})


if __name__ == "__main__":
    style()
    X, stations, times = load_matrix()
    monthly = monthly_values(X, stations, times)
    seasonal_cycle(monthly)
    anomalies(monthly, X, stations, times)
    diurnal(X, times)
    month_hour(X, times)
