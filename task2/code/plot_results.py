"""Report figures for Task 2, drawn from task2/results/ in the A4 style of the Task 1 figures.

40_test_scores          test-year error with bootstrap intervals, and the gain over persistence by current state
41_skill_by_threshold   Brier skill of P(Y > θ) from 0 to 30 mm
42_reliability          calibration at four thresholds
43_tiers                what each feature tier adds (validation) and how much the final models rely on it (test)
44_station_skill        skill over persistence by station
45_case_study           the largest 30-minute total of 2024, forecast by forecast
46_skill_by_time        skill by hour and month (appendix)
47_rf_screening         random forest screening on 2023 (appendix)

    python task2/code/plot_results.py
"""
import calendar
import sys
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

TASK2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK2.parent / "task1" / "code"))
import common as K  # noqa: E402
from common import AXIS, BLUE, INK, INK2, MUTED, ORANGE, SURFACE, TEXT_WIDTH, plt  # noqa: E402

RES = TASK2 / "results"
COLORS = {"RF": BLUE, "LR": ORANGE, "persistence": MUTED, "climatology": AXIS, "constant": AXIS}
NAMES = {"RF": "Random forest", "LR": "Logistic regression", "persistence": "Persistence",
         "climatology": "Climatology", "constant": "Constant"}
PERCENT = plt.FuncFormatter(lambda v, _: f"{v:.0%}")


def mark(ax, x, y, color, **kw):
    ax.plot(x, y, color=color, marker="o", ms=3.5, mec=SURFACE, mew=0.8, **kw)


def test_scores():
    boot = pd.read_csv(RES / "test_bootstrap.csv", index_col=0)
    reg = pd.read_csv(RES / "test_rps_by_regime.csv", index_col="subset")
    order = ["constant", "climatology", "persistence", "LR", "RF"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.0), layout="constrained", gridspec_kw={"width_ratios": [1.1, 1]})
    for i, m in enumerate(order):
        lo, hi, rps = boot.loc[m, "2.5 %"], boot.loc[m, "97.5 %"], reg.loc["all", m]
        a.plot([lo, hi], [i, i], color=COLORS[m], lw=1.6, solid_capstyle="butt")
        mark(a, rps, i, COLORS[m])
        a.text(hi + 0.005, i, f"{rps:.4f}", va="center", fontsize=7, color=INK2)
    a.set(yticks=range(len(order)), yticklabels=[NAMES[m] for m in order], xlim=(0, 0.26),
          xlabel="RPS on 2024 (lower is better)")
    a.grid(axis="y", visible=False)
    a.set_title("(a) Error, with 95% interval")
    subsets = [("all", "All forecasts"), ("dry_now", "Dry now"), ("wet_now", "Raining now")]
    y = np.arange(len(subsets))
    skill = {m: [1 - reg.loc[s, m] / reg.loc[s, "persistence"] for s, _ in subsets] for m in ("LR", "RF")}
    for j, m in enumerate(("LR", "RF")):
        pos = y + (j - 0.5) * 0.34
        b.barh(pos, skill[m], height=0.3, color=COLORS[m], label=NAMES[m])
        for p, v in zip(pos, skill[m]):
            b.text(v + 0.008, p, f"{v:.0%}", va="center", fontsize=7, color=INK2)
    b.set(yticks=y, yticklabels=[label for _, label in subsets], xlim=(0, 0.46), xlabel="Skill over persistence")
    b.xaxis.set_major_formatter(PERCENT)
    b.invert_yaxis()
    b.grid(axis="y", visible=False)
    fig.legend(*b.get_legend_handles_labels(), loc="outside upper right", ncols=2)
    b.set_title("(b) Gain over persistence")
    diff = boot.loc["RF − LR"]
    K.save(fig, "40_test_scores",
           "Forecast error on the test year 2024 (6,374,476 forecasts at 70 stations, unseen in training and tuning). "
           "(a) Ranked probability score over the nine rain classes; whiskers are 95% intervals from 2,000 bootstrap "
           f"samples of whole days. The random forest ({reg.loc['all', 'RF']:.4f}) and the logistic regression "
           f"({reg.loc['all', 'LR']:.4f}) remove {1 - reg.loc['all', 'RF'] / reg.loc['all', 'climatology']:.0%} and "
           f"{1 - reg.loc['all', 'LR'] / reg.loc['all', 'climatology']:.0%} of the error of climatology; the forest's "
           f"RPS is lower than the logistic regression's on every resample, by {-diff.estimate:.4f} (95% interval "
           f"{-diff['97.5 %']:.4f} to {-diff['2.5 %']:.4f}). (b) Skill over persistence, 1 − RPS / RPS of persistence, "
           f"on all forecasts and split by whether it rained at the station in the previous 30 minutes. The gain is "
           f"largest while it rains ({skill['RF'][2]:.0%} for the forest) and smaller, but real, when the station is "
           f"dry ({skill['RF'][1]:.0%}), where persistence can only forecast the average chance of rain.")


def skill_by_threshold():
    c = pd.read_csv(RES / "test_skill_by_threshold.csv")
    x = np.arange(len(c))
    fig, (a, b) = plt.subplots(2, 1, figsize=(TEXT_WIDTH, 2.9), layout="constrained", sharex=True,
                               gridspec_kw={"height_ratios": [2.4, 1]})
    for m in ("persistence", "LR", "RF"):
        mark(a, x, c[f"BSS {m}"], COLORS[m], label=NAMES[m])
    a.set(ylim=(0, 0.62), ylabel="Brier skill over climatology")
    a.yaxis.set_major_formatter(PERCENT)
    a.legend(loc="upper right")
    a.set_title("(a) Skill of the forecast probability P(Y > θ)")
    b.bar(x, c.events, width=0.45, color=AXIS)
    b.set_yscale("log")
    b.set(ylabel="Events", xlabel="Threshold θ (mm)", ylim=(1e3, 1e6))
    b.set_xticks(x, [f"{t:g}" for t in c.threshold])
    b.set_title("(b) Number of 30-minute totals above θ in 2024")
    t = c.set_index("threshold")
    cross = t.index[(t["BSS LR"] >= t["BSS RF"]) & (t.index > 0)].min()
    K.save(fig, "41_skill_by_threshold",
           "Skill at any threshold. The forecast probability that the next 30 minutes bring more than θ mm comes from "
           "the nine class probabilities through the threshold mapper, so thresholds inside a class (3, 7.5, 15, 25 "
           "and 30 mm) are scored the same way as the class edges. (a) Brier skill score against station-month-hour "
           f"climatology on 2024. The forest's skill falls from {t.loc[0, 'BSS RF']:.0%} for any rain to "
           f"{t.loc[5, 'BSS RF']:.0%} above 5 mm and {t.loc[20, 'BSS RF']:.0%} above 20 mm, while persistence falls "
           f"from {t.loc[0, 'BSS persistence']:.0%} to {t.loc[20, 'BSS persistence']:.0%}. From {cross:g} mm upwards "
           "the logistic regression is as skilful as the forest or slightly more. (b) Events become rare: "
           f"{int(t.loc[20, 'events']):,} totals above 20 mm and {int(t.loc[30, 'events']):,} above 30 mm in the "
           "whole year, so the scores at the highest thresholds are the least certain.")


def reliability(min_n=100):
    r = pd.read_csv(RES / "test_reliability.csv")
    dec = pd.read_csv(RES / "test_brier_decomposition.csv").set_index(["threshold", "forecast"])
    thetas = [0.0, 1.0, 5.0, 20.0]
    fig = plt.figure(figsize=(TEXT_WIDTH, 2.9), layout="constrained")
    grid = fig.add_gridspec(2, 4, height_ratios=[3, 1.1])
    for j, th in enumerate(thetas):
        a, h = fig.add_subplot(grid[0, j]), fig.add_subplot(grid[1, j])
        a.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
        for k, m in enumerate(("LR", "RF")):
            t = r[(r.threshold == th) & (r.forecast == m)].sort_values("bin")
            shown = t[t.n >= min_n]
            mark(a, shown.mean_forecast, shown.observed, COLORS[m], label=NAMES[m])
            h.bar(t.bin / 10 + 0.05 + (k - 0.5) * 0.04, t.n, width=0.036, color=COLORS[m])
        a.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xticks=[0, 0.5, 1], yticks=[0, 0.5, 1])
        a.tick_params(labelbottom=False)
        h.set(xlim=(0, 1), xticks=[0, 0.5, 1], yscale="log", ylim=(1, 1e7))
        h.set_yticks([1e1, 1e3, 1e5])
        if j == 0:
            a.set_ylabel("Observed frequency")
            h.set_ylabel("Forecasts")
        else:
            a.tick_params(labelleft=False)
            h.tick_params(labelleft=False)
        h.set_xlabel("Forecast probability")
        a.set_title(f"({'abcd'[j]}) {'any rain' if th == 0 else f'more than {th:g} mm'}")
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="outside upper center", ncols=2)
    rel = dec.xs("RF", level="forecast")
    rf0, lr0 = (r[(r.threshold == 0) & (r.forecast == m) & (r.bin == 8)].iloc[0] for m in ("RF", "LR"))
    rf5 = r[(r.threshold == 5) & (r.forecast == "RF") & (r.bin == 8)].iloc[0]
    K.save(fig, "42_reliability",
           "Calibration on 2024. Top: observed frequency of the event against the mean forecast probability, in ten "
           f"equal-width bins (bins with fewer than {min_n} forecasts are left out); a calibrated forecast lies on the "
           "diagonal. Bottom: number of forecasts in each bin, on a log scale; most forecasts are close to zero because "
           "rain is rare. Both models are close to calibrated at every threshold: the calibration term of the Brier "
           f"score is at most {rel.reliability.max():.5f} for the forest, against a resolution term of "
           f"{rel.loc[0.0, 'resolution']:.3f} for any rain. The forest is slightly under-confident at high probabilities "
           f"(forecasts of {rf0.mean_forecast:.2f} for any rain came true {rf0.observed:.0%} of the time, "
           f"{rf5.mean_forecast:.2f} above 5 mm came true {rf5.observed:.0%}), the logistic regression slightly "
           f"over-confident ({lr0.mean_forecast:.2f} came true {lr0.observed:.0%}).")


def tiers():
    v = pd.read_csv(RES / "models_val.csv").set_index(["subset", "model"])
    perm = pd.read_csv(RES / "test_tier_permutation.csv").set_index("tier shuffled")
    steps, labels = ["T0", "T0+T1", "T0+T1+T2", "T0+T1+T2+T3"], ["T0", "+T1", "+T2", "+T3"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.25), layout="constrained")
    x = np.arange(len(steps))
    rps = {m: [v.loc[("all", f"{m} {s}"), "RPS"] for s in steps] for m in ("LR", "RF")}
    auc = {m: [v.loc[("dry_now", f"{m} {s}"), "AUC-PR 0"] for s in steps] for m in ("LR", "RF")}
    for m in ("LR", "RF"):
        mark(a, x, rps[m], COLORS[m], label=NAMES[m])
        mark(b, x, auc[m], COLORS[m])
    for ref in ("climatology", "persistence"):
        level = v.loc[("all", ref), "RPS"]
        a.axhline(level, color=AXIS, lw=0.8, zorder=0)
        a.text(3, level + 0.003, NAMES[ref], ha="right", va="bottom", fontsize=6.5, color=INK2)
    rate = v.loc[("dry_now", "constant"), "AUC-PR 0"]
    b.axhline(rate, color=AXIS, lw=0.8, zorder=0)
    b.text(3, rate + 0.012, "No skill", fontsize=6.5, color=INK2, va="bottom", ha="right")
    for ax, title in ((a, "(a) RPS on 2023"), (b, "(b) Dry-now onsets, AUC-PR")):
        ax.set_xticks(x, labels)
        ax.set_title(title)
    a.set(ylim=(0.08, 0.215))
    b.set(ylim=(0, 0.5))
    b.legend(*a.get_legend_handles_labels(), loc="upper left")
    order = ["T0", "T1", "T2", "T3"]
    y = np.arange(len(order))
    for j, m in enumerate(("LR", "RF")):
        pos = y + (j - 0.5) * 0.34
        vals = [perm.loc[t, f"RPS rise {m}"] for t in order]
        c.barh(pos, vals, height=0.3, color=COLORS[m])
        for p, val in zip(pos, vals):
            c.text(val + 0.002, p, f"{val:.3f}", va="center", fontsize=6.5, color=INK2)
    c.set(yticks=y, yticklabels=order, xlim=(0, 0.11), xlabel="Rise in RPS when shuffled")
    c.invert_yaxis()
    c.grid(axis="y", visible=False)
    c.set_title("(c) Reliance on 2024")
    K.save(fig, "43_tiers",
           "What the feature tiers contribute. (a) Validation RPS (2023) of models trained on growing sets of tiers: "
           "station and clock features (T0), the station's own recent rain (T1), rain at neighbouring stations (T2) "
           "and how the neighbourhood is changing plus rain upstream (T3). On their own, station and clock "
           f"features do no better than climatology (forest {rps['RF'][0]:.4f}, logistic regression {rps['LR'][0]:.4f}, "
           f"climatology {v.loc[('all', 'climatology'), 'RPS']:.4f}); the station's recent rain brings the largest step "
           f"({rps['RF'][1]:.4f}), neighbours the next ({rps['RF'][2]:.4f}) and Tier 3 a small further gain "
           f"({rps['RF'][3]:.4f}). (b) For rows where the station is dry, the area under the precision-recall curve "
           f"for rain in the next 30 minutes rises from {auc['RF'][1]:.2f} to {auc['RF'][2]:.2f} with neighbours and "
           f"{auc['RF'][3]:.2f} with Tier 3; no skill is {rate:.3f}, the share of dry rows followed by rain. (c) On "
           "2024, shuffling one tier's columns across rows breaks their link to the outcome; the rise in RPS measures "
           f"how much the final models rely on it. Both rely mostly on neighbours (forest +{perm.loc['T2', 'RPS rise RF']:.3f}) "
           f"and the station's own rain (+{perm.loc['T1', 'RPS rise RF']:.3f}); tiers overlap in what they know, so "
           "these are reliance, not unique contributions.")


def station_skill(full_year=20_000):
    s = pd.read_csv(RES / "test_rps_by_station.csv")
    names = pd.read_csv(K.MATRIX / "stations.csv").set_index("station_id").station_name
    s["name"] = s.station_id.map(names)
    x, y = K.local_xy(s)
    d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    s["neighbours"] = ((d > 0) & (d <= 10)).sum(axis=1)
    rho, p = spearmanr(s.neighbours, s["RPSS RF vs persistence"])
    skill = s["RPSS RF vs persistence"]
    norm = TwoSlopeNorm(0, -0.4, 0.4)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.5), layout="constrained")
    K.draw_boundary(ax)
    full = s.n >= full_year
    points = ax.scatter(s.location_longitude[full], s.location_latitude[full], c=skill[full], cmap=K.DIVERGING,
                        norm=norm, s=34, edgecolors=K.INK2, linewidths=0.4, zorder=3)
    ax.scatter(s.location_longitude[~full], s.location_latitude[~full], c=skill[~full], cmap=K.DIVERGING, norm=norm,
               s=14, edgecolors=K.INK2, linewidths=0.4, zorder=3)
    worst, best = s.nsmallest(1, "RPSS RF vs persistence"), s[full].nlargest(1, "RPSS RF vs persistence")
    for rows, offset, ha, va in ((worst, (7, 0), "left", "center"), (best, (0, -7), "center", "top")):
        row = rows.iloc[0]
        ax.annotate(f"{row['name']} ({row['RPSS RF vs persistence']:+.0%})".replace("-", "−"), (row.location_longitude, row.location_latitude),
                    xytext=offset, textcoords="offset points", ha=ha, va=va, fontsize=6.5, color=INK2, zorder=4)
    handles = [Line2D([], [], marker="o", linestyle="none", markersize=5, markerfacecolor=K.NEUTRAL, markeredgecolor=INK2,
                      markeredgewidth=0.4, label=f"{full.sum()} stations with a full record"),
               Line2D([], [], marker="o", linestyle="none", markersize=3.2, markerfacecolor=K.NEUTRAL, markeredgecolor=INK2,
                      markeredgewidth=0.4, label=f"{(~full).sum()} with fewer than {full_year:,} forecasts")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0), ncols=2)
    bar = fig.colorbar(points, ax=ax, fraction=0.03, pad=0.01, shrink=0.85, format=PERCENT)
    bar.set_label("Skill over persistence on 2024 (RPS)")
    bar.outline.set_visible(False)
    weak = s.nsmallest(3, "RPSS RF vs persistence")
    K.save(fig, "44_station_skill",
           "Where the forest helps most. Colour: the forest's skill over persistence at each station on 2024, "
           f"1 − RPS / RPS of persistence (median {skill.median():.0%}, middle half "
           f"{skill.quantile(0.25):.0%} to {skill.quantile(0.75):.0%}). Skill is lowest at the edge of the network, "
           f"where a station has few neighbours: {', '.join(f'{n} {v:+.0%}'.replace('-', '−') for n, v in zip(weak['name'], weak['RPSS RF vs persistence']))}; "
           "the offshore gauge on Semakau Landfill reported for part of the year only and is the one station where "
           "persistence did better. Across stations, skill rises with the number of other stations within 10 km "
           f"(Spearman ρ = {rho:.2f}), the neighbours that Tiers 2 and 3 draw on.")


def case_study():
    c = pd.read_csv(RES / "test_case_study.csv", parse_dates=["time"])
    peak = c.loc[c.y_mm.idxmax()]
    fig, (a, b, d) = plt.subplots(3, 1, figsize=(TEXT_WIDTH, 3.7), layout="constrained", sharex=True)
    step = pd.Timedelta(minutes=5)
    for ax, th, title in ((a, 0, "(a) Chance of any rain in the next 30 minutes"),
                          (b, 5, "(b) Chance of more than 5 mm in the next 30 minutes")):
        for t in c.time[c.y_mm > th]:
            ax.axvspan(t - step / 2, t + step / 2, color=K.NEUTRAL, lw=0, zorder=0)
        for m in ("persistence", "LR", "RF"):
            ax.plot(c.time, c[f"P(Y>{th}) {m}"], color=COLORS[m], label=NAMES[m], lw=1.2)
        ax.set(ylim=(0, 1.02), yticks=[0, 0.5, 1])
        ax.yaxis.set_major_formatter(PERCENT)
        ax.set_title(title)
    fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncols=3)
    d.bar(c.time, c.y_mm, width=step * 0.7, color=INK2)
    d.axhline(5, color=AXIS, lw=0.8)
    d.text(c.time.iloc[-1], 6, "5 mm", ha="right", va="bottom", fontsize=6.5, color=INK2)
    d.set(ylabel="mm")
    d.set_title("(c) Rain that fell in the 30 minutes after each forecast")
    d.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    first = {m: c.time[c[f"P(Y>5) {m}"] >= 0.5].min() for m in ("RF", "LR", "persistence")}
    onset = c.time[c.y_mm > 5].min()
    late = c[c.time >= peak.time - pd.Timedelta(minutes=20)].head(9)
    K.save(fig, "45_case_study",
           f"The largest 30-minute total of 2024: {peak.y_mm:g} mm at Bukit Timah Road (S90) after "
           f"{peak.time:%H:%M} on {peak.time:%-d %B %Y}, with the forecasts issued every five minutes from "
           f"{c.time.iloc[0]:%H:%M} to {c.time.iloc[-1]:%H:%M}. Shading marks the forecasts that were followed by the "
           f"event. More than 5 mm first followed the forecast of {onset:%H:%M}; the forest first gave it a chance of "
           f"50% or more at {first['RF']:%H:%M}, the logistic regression at {first['LR']:%H:%M} and persistence only at "
           f"{first['persistence']:%H:%M}, after rain had fallen at the station itself. Persistence also cannot tell a "
           f"heavy burst from light rain: its chance of more than 5 mm never exceeds "
           f"{c['P(Y>5) persistence'].max():.0%}, while in the run-up to the peak both models reach "
           f"{min(late['P(Y>5) RF'].max(), late['P(Y>5) LR'].max()):.0%} or more.")


def skill_by_time():
    hour = pd.read_csv(RES / "test_rps_by_hour.csv")
    month = pd.read_csv(RES / "test_rps_by_month.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.2), layout="constrained", sharey=True,
                               gridspec_kw={"width_ratios": [1.6, 1]})
    for m in ("persistence", "LR", "RF"):
        mark(a, hour.hour, 1 - hour[m] / hour.climatology, COLORS[m], label=NAMES[m])
        mark(b, month.month, 1 - month[m] / month.climatology, COLORS[m])
    a.set(xticks=range(0, 24, 3), xlabel="Hour of the forecast", ylabel="Skill over climatology", ylim=(0, 0.75))
    a.yaxis.set_major_formatter(PERCENT)
    b.set(xticks=range(1, 13), xlabel="Month")
    b.set_xticklabels(list("JFMAMJJASOND"))
    a.legend(loc="lower left", ncols=3)
    a.set_title("(a) By hour of day")
    b.set_title("(b) By month")
    hs = 1 - hour.RF / hour.climatology
    ms = 1 - month.RF / month.climatology
    K.save(fig, "46_skill_by_time",
           "Skill over climatology (RPS) on 2024 by the hour of the forecast and by month. The forest's skill ranges "
           f"from {hs.min():.0%} at {int(hour.hour[hs.idxmin()]):02d}:00 to {hs.max():.0%} at "
           f"{int(hour.hour[hs.idxmax()]):02d}:00, and from {ms.min():.0%} in {calendar.month_name[int(month.month[ms.idxmin()])]} "
           f"to {ms.max():.0%} in {calendar.month_name[int(month.month[ms.idxmax()])]}; the ranking of the three forecasts is the same in "
           "every hour and every month.", placement="appendix")


def rf_screening():
    sc = pd.read_csv(RES / "rf_screening.csv")
    lr = pd.read_csv(RES / "lr_tuning.csv")
    feats = {"sqrt": "√ of features", "onethird": "⅓ of features", "0.5": "½ of features"}
    label = [f"depth {r.max_depth}, leaf {r.min_instances}, {feats[str(r.feature_subset)]}, {r.max_bins} bins, {r.impurity}"
             for r in sc.itertuples()]
    final = sc.iloc[[sc["val RPS"].idxmin()]].iloc[0]
    path = [3, 5, 7, 9, 10]
    y = np.arange(len(sc))
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.8), layout="constrained", sharey=True,
                               gridspec_kw={"width_ratios": [1.4, 1]})
    colors = [BLUE if i in path else MUTED for i in range(len(sc))]
    for i in y:
        mark(a, sc["val RPS"].iloc[i], i, colors[i])
        b.barh(i, sc["fit min"].iloc[i], height=0.55, color=colors[i])
        b.text(sc["fit min"].iloc[i] + 1, i, f"{sc['fit min'].iloc[i]:.0f}", va="center", fontsize=6.5, color=INK2)
    a.set(yticks=y, yticklabels=label, xlabel="RPS on 2023")
    a.invert_yaxis()
    a.grid(axis="y", visible=False)
    a.set_title("(a) Validation error")
    b.set(xlabel="Fit time (minutes)", xlim=(0, sc["fit min"].max() * 1.18))
    b.grid(axis="y", visible=False)
    b.set_title("(b) Cost")
    plain, best = lr.iloc[0], lr[lr.tiers == "T0+T1+T2"].sort_values("val RPS").iloc[0]
    K.save(fig, "47_rf_screening",
           "Tuning on 2023 with the 10% training sample and Tiers 0 to 2. (a) The random forest was screened one setting "
           "at a time (depth, then minimum rows per leaf, features tried per split, bins and split criterion), keeping the "
           "best value of each before moving on; the kept settings are shown in blue. RPS fell from "
           f"{sc['val RPS'].iloc[0]:.4f} to {final['val RPS']:.4f}, mostly through smaller leaves, more features per "
           f"split and finer bins, while (b) the fit time grew from {sc['fit min'].iloc[0]:.0f} to "
           f"{final['fit min']:.0f} minutes for 50 trees. The logistic regression gained more from its inputs than from "
           f"tuning: non-linear inputs lowered RPS from {plain['val RPS']:.4f} to {best['val RPS']:.4f}, and the L2 "
           f"penalty changed it by less than {np.ptp(lr[lr.nonlinear & (lr.tiers == 'T0+T1+T2')]['val RPS'].to_numpy()):.4f} "
           "once every fit converged.", placement="appendix")


if __name__ == "__main__":
    K.style()
    K.use_outputs(TASK2 / "outputs", "Task 2 figures")
    for draw in (test_scores, skill_by_threshold, reliability, tiers, station_skill, case_study, skill_by_time,
                 rf_screening):
        draw()
