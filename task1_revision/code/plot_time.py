"""Section 1: when does it rain?

10_seasonal_cycle   year curves, mean with 95% CI, sine/cosine regression (idea 2), variance split
11_anomalies        deseasonalised monthly anomalies, annual means at the 80% cutoff
12_diurnal          daily cycle with its sine/cosine regression, split into how often vs how hard
13_month_hour       season x time of day: the timing of rain changes with the monsoon
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import (AQUA, AXIS, BLUE, INK, INK2, MUTED, ORANGE, SEQUENTIAL, TASK1, caption, load_matrix, plt,
                    record, save, style, write_table)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
rng = np.random.default_rng(5105)


def monthly_values():
    frame = pd.read_csv(TASK1 / "monthly_pattern.csv")
    return frame.rename(columns={"mean_daily_equivalent_mm": "value"})


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

    # Idea 2: the seasonal cycle as a regression on sine/cosine terms.
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

    fig = plt.figure(figsize=(16, 5.4))
    layout = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.15, 0.95], wspace=0.3)
    ax, comp, side = (fig.add_subplot(layout[i]) for i in range(3))
    for i, (_, row) in enumerate(grid.iterrows()):
        ax.plot(range(1, 13), row.values, color=AXIS, linewidth=1.1, zorder=1,
                label="Single year (8 years)" if i == 0 else None)
    ax.fill_between(range(1, 13), mean - half, mean + half, color=BLUE, alpha=0.12, linewidth=0,
                    label="95% CI of the mean", zorder=2)
    ax.plot(range(1, 13), mean.values, color=BLUE, marker="o", markersize=5, markeredgecolor="white",
            markeredgewidth=1.5, label="Mean of available years", zorder=3)
    ax.plot(month_axis, predict_harmonics(h2, smooth, 2), color=INK, linewidth=1.6, zorder=4,
            label=f"Regression on 2 sine/cosine pairs (R²={h2.rsquared:.2f})")
    for _, row in effects[effects.p_holm < 0.05].iterrows():
        m = int(row.month)
        ax.annotate(f"{MONTHS[m - 1]}: {row.effect:+.1f} mm/day vs an average month\n(Holm-adjusted p={row.p_holm:.3f})",
                    (m, mean[m]), xytext=(-125, 42), textcoords="offset points", ha="left", fontsize=7.5,
                    color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set(xticks=range(1, 13), xlabel="Month", ylabel="Rainfall rate (mm/day)", ylim=(0, None), xlim=(0.5, 12.5))
    ax.set_xticklabels(MONTHS)
    ax.set_title("Seasonal cycle: real, but weak next to year-to-year noise")
    ax.legend(loc="upper left", fontsize=7.8)

    comp.axhline(0, color=AXIS, linewidth=0.8)
    comp.plot(month_axis, annual, color=ORANGE, label=f"Annual wave (amplitude {np.hypot(a1, b1):.2f})")
    comp.plot(month_axis, semi, color=AQUA, label=f"Half-year wave (amplitude {np.hypot(a2, b2):.2f})")
    comp.plot(month_axis, annual + semi, color=INK, linewidth=1.6, label="Sum = the fitted curve")
    month_means = data.groupby("month").value.mean() - base
    comp.plot(month_means.index, month_means.values, "o", color=BLUE, markersize=5, markeredgecolor="white",
              label="Observed month means")
    comp.set(xticks=range(1, 13), xlabel="Month", ylabel="Deviation from the yearly mean (mm/day)", xlim=(0.5, 12.5))
    comp.set_xticklabels([m[0] for m in MONTHS])
    comp.set_title("Two waves make the two wet peaks", fontsize=10.5)
    comp.legend(loc="upper left", fontsize=7.4)

    parts = [("Month of year", share["C(month, Sum)"], BLUE, anova.loc["C(month, Sum)", "PR(>F)"]),
             ("Year", share["C(year, Sum)"], BLUE, anova.loc["C(year, Sum)", "PR(>F)"]),
             ("Unexplained\n(that month,\nthat year)", share["Residual"], AXIS, None)]
    for i, (name, value, color, p) in enumerate(parts):
        side.barh(i, value, color=color, height=0.55)
        side.text(value + 0.02, i, f"{value:.0%}" + (f"\np={p:.3f}" if p is not None else ""), va="center",
                  fontsize=8, color=INK2)
    side.set(xlim=(0, 1), yticks=range(3), yticklabels=[p[0] for p in parts], ylim=(4.9, -0.6))
    side.tick_params(axis="y", labelsize=8)
    side.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    side.grid(axis="y", visible=False)
    side.set_title("Where monthly variation comes from", fontsize=10.5)
    side.text(0.02, 0.03, f"Tests\n• seasonality (permutation): p={p_perm:.3f}\n• 2nd wave adds: p={p_second:.3f}\n"
                          f"• 12 month dummies beyond 2 waves: p={p_rest:.2f}\n"
                          f"• R²: 1 wave {h1.rsquared:.2f}, 2 waves {h2.rsquared:.2f}, dummies {dummies.rsquared:.2f}",
              transform=side.transAxes, va="bottom", fontsize=7.8, color=INK2)
    caption(fig, f"Month of year explains {share['C(month, Sum)']:.0%} of the monthly variation, year {share['C(year, Sum)']:.0%} "
                 f"(not significant), and {share['Residual']:.0%} is weather specific to one month of one year, which is why "
                 "the old month-by-year heatmap showed no pattern. Two sine/cosine pairs capture the cycle; the half-year wave is "
                 "stronger than the annual one, giving the two inter-monsoon peaks (Apr, Nov). For Task 2: encode day-of-year "
                 "as two sin/cos pairs instead of a month integer.", y=-0.05)
    save(fig, "10_seasonal_cycle")
    write_table(effects.round(4), "month_effects")
    record("10_seasonal", {
        "variance_share": {k: round(float(v), 3) for k, v in share.items()},
        "anova_p_month": float(anova.loc["C(month, Sum)", "PR(>F)"]),
        "anova_p_year": float(anova.loc["C(year, Sum)", "PR(>F)"]),
        "permutation_p_seasonality": float(p_perm),
        "significant_months_holm": {MONTHS[int(r.month) - 1]: [round(r.effect, 2), round(r.p_holm, 4)]
                                    for _, r in effects[effects.p_holm < 0.05].iterrows()},
        "harmonic_r2": {"1_wave": round(h1.rsquared, 3), "2_waves": round(h2.rsquared, 3),
                        "12_dummies": round(dummies.rsquared, 3)},
        "p_second_wave": float(p_second), "p_dummies_beyond_2_waves": float(p_rest),
        "amplitude_annual": round(float(np.hypot(a1, b1)), 3), "amplitude_half_year": round(float(np.hypot(a2, b2)), 3),
    })


# ------------------------------------------------------------ 11 ----------

def anomalies(monthly):
    data = monthly.copy()
    climate = data.groupby("month").value.mean()
    data["anomaly"] = data.value - data.month.map(climate)
    data["date"] = pd.to_datetime(dict(year=data.year, month=data.month, day=15))
    fit = smf.ols("value ~ C(month, Sum) + C(year, Sum)", data.dropna(subset=["value"])).fit()
    years = sum_coded_effects(fit, "year", range(2017, 2025))
    annual80 = pd.read_csv(TASK1 / "coverage_sensitivity_80.csv")
    cutoffs = pd.concat({c: pd.read_csv(TASK1 / f"coverage_sensitivity_{c}.csv").set_index("year").mean_daily_equivalent_mm
                         for c in (80, 90, 95)}, axis=1).dropna()
    cutoff_spread = float((cutoffs.max(axis=1) - cutoffs.min(axis=1)).max())

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig, (ax, rate_ax, n_ax) = plt.subplots(3, 1, figsize=(12.5, 8.2), sharex=True,
                                            gridspec_kw={"height_ratios": [2.4, 1.3, 0.8], "hspace": 0.22})
    ok = data.dropna(subset=["anomaly"])
    ax.bar(ok.date, ok.anomaly, width=25, color=np.where(ok.anomaly >= 0, BLUE, ORANGE), edgecolor="none")
    missing = data[data.anomaly.isna()]
    ax.plot(missing.date, np.zeros(len(missing)), "x", color=MUTED, markersize=5)
    for _, row in years.iterrows():
        start, end = pd.Timestamp(int(row.year), 1, 1), pd.Timestamp(int(row.year), 12, 31)
        ax.hlines(row.effect, start, end, color=INK, linewidth=1.6)
        star = "*" if row.p_holm < 0.05 else ""
        ax.text(start + pd.Timedelta(days=182), 8.3, f"{row.effect:+.1f}{star}\np={row.p:.3f}", ha="center",
                va="bottom", fontsize=7.5, color=INK2)
    handles = [Patch(color=BLUE, label="Wetter than usual for that month"),
               Patch(color=ORANGE, label="Drier than usual for that month"),
               Line2D([], [], color=INK, linewidth=1.6, label="Year effect, seasonally balanced (label: value, raw p)"),
               Line2D([], [], color=MUTED, marker="x", linestyle="none", label="No qualifying station")]
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set(ylabel="Anomaly (mm/day)", ylim=(-6.5, 10.2))
    ax.set_title("Monthly anomalies: each month minus its own long-run average", pad=8)
    ax.legend(handles=handles, loc="lower left", ncols=4, fontsize=7.5)

    mid = pd.to_datetime(dict(year=annual80.year, month=7, day=1))
    rate_ax.plot(mid, annual80.mean_daily_equivalent_mm, color=BLUE, marker="o", markersize=5,
                 markeredgecolor="white", markeredgewidth=1.5)
    for when, value in zip(mid, annual80.mean_daily_equivalent_mm):
        rate_ax.annotate(f"{value:.1f}", (when, value), xytext=(0, 6), textcoords="offset points",
                         ha="center", fontsize=7.5, color=INK2)
    rate_ax.set(ylabel="mm/day", ylim=(0, 10))
    rate_ax.set_title("Annual mean rate, station-years with >= 80% coverage (2018 now included)", fontsize=10)
    n_ax.bar(mid, annual80.eligible_stations, width=120, color=BLUE)
    for when, value in zip(mid, annual80.eligible_stations):
        n_ax.annotate(str(value), (when, value), xytext=(0, 2), textcoords="offset points", ha="center",
                      fontsize=7.5, color=INK2)
    n_ax.set(ylabel="Stations", ylim=(0, 85))
    driest = years.loc[years.effect.idxmin()]
    caption(fig, f"Year effects are deviations from the average year after accounting for month (two-way regression); "
                 f"* would mark Holm-adjusted p < 0.05 across 8 years. {int(driest.year)} is the driest ({driest.effect:+.1f} "
                 f"mm/day, raw p={driest.p:.3f}, Holm p={driest.p_holm:.3f}): dry in most months, but borderline once 8 years "
                 f"are tested. Annual means change by at most {cutoff_spread:.2f} mm/day between the 80%, 90% and 95% "
                 "coverage cutoffs, so the 80% choice does not drive any conclusion.", y=0.045)
    save(fig, "11_anomalies")
    write_table(years.round(4), "year_effects")
    write_table(cutoffs.reset_index().round(3), "annual_mean_by_cutoff")
    record("11_anomalies", {"year_effects": {int(r.year): [round(r.effect, 2), round(r.p, 3), round(r.p_holm, 3)]
                                             for _, r in years.iterrows()},
                            "annual_rate_80pct": annual80.set_index("year").mean_daily_equivalent_mm.round(2).to_dict(),
                            "stations_80pct": annual80.set_index("year").eligible_stations.to_dict(),
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

    # Idea 2 for the daily cycle: three sine/cosine pairs on the year x hour values.
    hy, hphase = years.rate.to_numpy(), (years.hour.to_numpy() + 0.5) / 24
    hfits = {K: fit_harmonics(hy, hphase, K) for K in (1, 2, 3)}
    hdummies = smf.ols("rate ~ C(hour)", years).fit()
    smooth = np.linspace(0, 1, 481)
    hour_axis = smooth * 24 - 0.5
    curve = predict_harmonics(hfits[3], smooth, 3)
    peak_time = hour_axis[np.argmax(curve)]
    spread = years.pivot(index="hour", columns="year", values="rate")
    peak_by_year = spread.idxmax().astype(int)

    fig, (ax, idx_ax) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"wspace": 0.22})
    for year in spread.columns:
        ax.plot(spread.index, spread[year], color=AXIS, linewidth=1, zorder=1)
    ax.plot([], [], color=AXIS, linewidth=1, label="Single year (2017-2024)")
    ax.plot(pooled.hour, pooled.rate, color=BLUE, marker="o", markersize=4, markeredgecolor="white",
            label="All years pooled", zorder=3)
    ax.plot(hour_axis, curve, color=INK, linewidth=1.6, zorder=4,
            label=f"Regression on 3 sine/cosine pairs (R²={hfits[3].rsquared:.2f})")
    ax.axvspan(12.5, 18.5, color=BLUE, alpha=0.07, linewidth=0)
    top = np.nanmax(spread.values) * 1.08
    ax.text(15.5, top * 0.99, f"13:00-18:59\n{afternoon:.0%} of daily rain\nin 25% of the day", ha="center",
            va="top", fontsize=7.5, color=INK2)
    ax.set(xlabel="Hour of day (Asia/Singapore)", ylabel="Rainfall rate (mm/hour)", xticks=range(0, 24, 3),
           xlim=(-0.5, 23.5), ylim=(0, top))
    ax.set_title("Rain is an afternoon event, every year")
    ax.legend(loc="upper left", fontsize=7.5)

    for column, color, label, width in (("rate", MUTED, "Rainfall rate (= frequency x intensity)", 1.4),
                                        ("frequency", BLUE, "How often it rains (wet share of 5-min slots)", 2),
                                        ("intensity", ORANGE, "How hard it rains (mm per wet 5-min slot)", 2)):
        idx_ax.plot(pooled.hour, pooled[column] / pooled[column].mean(), color=color, label=label, linewidth=width)
    idx_ax.axhline(1, color=AXIS, linewidth=0.8)
    idx_ax.set(xlabel="Hour of day (Asia/Singapore)", ylabel="Relative to the 24-hour average (= 1)",
               xticks=range(0, 24, 3))
    idx_ax.set_title("The afternoon peak is about frequency, not intensity")
    idx_ax.legend(loc="upper left", fontsize=7.5)
    peak, trough = pooled.loc[pooled.rate.idxmax()], pooled.loc[pooled.rate.idxmin()]
    caption(fig, f"Three sine/cosine pairs reproduce the daily cycle (R²={hfits[3].rsquared:.2f}, versus {hdummies.rsquared:.2f} "
                 f"for 24 hour dummies), peaking at about {clock(peak_time)}; the peak hour is {peak_by_year.min()}:00-"
                 f"{peak_by_year.max()}:00 in all 8 years. At {int(peak.hour):02d}:00 rain is {peak.frequency / trough.frequency:.1f}x "
                 f"as frequent as at {int(trough.hour):02d}:00 but only {peak.intensity / trough.intensity:.1f}x as heavy per wet "
                 "slot. For Task 2: encode hour as sin/cos pairs; hour mainly predicts whether it rains.")
    save(fig, "12_diurnal")
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

    fig = plt.figure(figsize=(13.5, 4.8))
    layout = fig.add_gridspec(1, 3, width_ratios=[2.3, 0.06, 1], wspace=0.08)
    ax, bar_ax, side = fig.add_subplot(layout[0]), fig.add_subplot(layout[1]), fig.add_subplot(layout[2])
    image = ax.imshow(rate, aspect="auto", cmap=SEQUENTIAL, interpolation="nearest")
    ax.set(xticks=range(0, 24, 3), xlabel="Hour of day (Asia/Singapore)", yticks=range(12), ylabel="Month")
    ax.set_yticklabels(MONTHS)
    ax.grid(False)
    bar = fig.colorbar(image, cax=bar_ax)
    bar.set_label("Rainfall rate (mm/hour)")
    bar.outline.set_visible(False)
    side.set_position(side.get_position().translated(0.045, 0))
    ax.set_title("Season x time of day: the timing of rain flips with the monsoon")

    side.plot(afternoon, range(12), color=BLUE, marker="o", markersize=5, markeredgecolor="white",
              label="13:00-18:59 (afternoon)")
    side.plot(morning, range(12), color=ORANGE, marker="o", markersize=5, markeredgecolor="white",
              label="04:00-09:59 (early morning)")
    side.set(yticks=range(12), yticklabels=[], ylim=(11.5, -0.5), xlabel="Share of the month's rain", xlim=(0, 0.75))
    side.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    side.set_title("When in the day the rain falls", fontsize=10)
    side.legend(loc="lower right", fontsize=7.5)
    wettest_morning = int(np.argmax(morning))
    flipped = [MONTHS[i] for i in range(12) if morning[i] >= afternoon[i]]
    caption(fig, f"Early-morning rain peaks in {MONTHS[wettest_morning]} ({morning[wettest_morning]:.0%} of that month's rain) "
                 f"versus {morning.min():.0%} in {MONTHS[int(np.argmin(morning))]}; in {', '.join(flipped)} the morning carries at "
                 "least as much rain as the afternoon. This fits the Sumatra squalls of the southwest monsoon (pre-dawn/morning "
                 "storm lines from the west), which 23_lead_lag tests. For Task 2: let hour and season interact.")
    save(fig, "13_month_hour")
    write_table(pd.DataFrame(rate, index=MONTHS, columns=range(24)).reset_index(names="month").round(4), "month_hour_rate")
    record("13_month_hour", {"morning_share_by_month": dict(zip(MONTHS, np.round(morning, 3))),
                             "afternoon_share_by_month": dict(zip(MONTHS, np.round(afternoon, 3)))})


if __name__ == "__main__":
    style()
    monthly = monthly_values()
    seasonal_cycle(monthly)
    anomalies(monthly)
    X, stations, times = load_matrix()
    diurnal(X, times)
    month_hour(X, times)
