"""Section 3: how much does it rain, and can simple models describe it?

30_concentration           most rain falls in very little time (replaces the top-10 extreme days)
31_persistence             wet/dry day persistence and a Markov-chain check
32_compound_poisson_gamma  a day's rain as N events of random size: the Tweedie model, checked on data
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import (AQUA, AXIS, BLUE, INK, INK2, MUTED, ORANGE, RAMP, SURFACE, block_sum, caption, load_matrix,
                    plt, record, save, style, write_table)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SCALES = {"5 min": (1, 12 * 24 * 365), "1 hour": (12, 24 * 365), "1 day": (288, 365)}  # (slots, units per year)
GAP_SLOTS = 6  # dry gaps of up to 30 minutes are bridged inside one rain event
rng = np.random.default_rng(5105)


def complete_days(X):
    """Station-day matrix of daily totals (NaN unless all 288 slots were observed)."""
    return block_sum(X, 288)


# ------------------------------------------------------------ 30 ----------

def concentration(X):
    fig, (ax, err_ax) = plt.subplots(1, 2, figsize=(14, 4.9), gridspec_kw={"wspace": 0.25})
    summary = {}
    for (name, (k, per_year)), color in zip(SCALES.items(), [RAMP[1], RAMP[2], RAMP[4]]):
        Y = block_sum(X, k)
        values = np.sort(Y[np.isfinite(Y)])[::-1]
        share_time = np.arange(1, len(values) + 1) / len(values)
        rain = np.cumsum(values) / values.sum()
        squared = (values - values.mean()) ** 2
        error = np.cumsum(squared) / squared.sum()
        half = share_time[np.searchsorted(rain, 0.5)]
        wet = np.mean(values > 0)
        step = np.unique(np.logspace(0, np.log10(len(values)), 3000).astype(int)) - 1  # log-spaced, drawing only
        per_station_year = half * per_year * k * 5 / 60  # hours of a station-year that carry half its rain
        amount = f"{per_station_year / 24:.0f} days" if name == "1 day" else f"{per_station_year:.0f} hours"
        ax.plot(share_time[step], rain[step], color=color, label=f"{name}: wet {wet:.1%} of the time")
        ax.plot(half, 0.5, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        unit = {"5 min": "slots", "1 hour": "hours", "1 day": "days"}[name]
        ax.annotate(f"{half:.2%} of {unit}\n≈ {amount} a year", (half, 0.5),
                    xytext=(-8, 6) if name == "5 min" else (7, -26), ha="right" if name == "5 min" else "left",
                    textcoords="offset points", fontsize=7.5, color=INK2)
        err_ax.plot(share_time[step], error[step], color=color, label=name)
        top1 = error[int(0.01 * len(values)) - 1]
        err_ax.plot(0.01, top1, "o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        err_ax.annotate(f"{top1:.0%}", (0.01, top1), xytext=(7, -3), textcoords="offset points", fontsize=7.5, color=INK2)
        summary[name] = dict(wet_share=round(float(wet), 4), half_rain_in_share=round(float(half), 5),
                             half_rain_units_per_year=round(float(half * per_year), 1),
                             top1pct_rain_share=round(float(rain[int(0.01 * len(values)) - 1]), 3),
                             top1pct_squared_error_share=round(float(top1), 3), zero_share=round(float(1 - wet), 4))
    for axis in (ax, err_ax):
        axis.set_xscale("log")
        axis.set(xlim=(1e-5, 1), ylim=(0, 1.02), xlabel="Share of time units, heaviest first (log scale)")
        axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.3%}".rstrip("0").rstrip(".") if v < 0.01
                                                         else f"{v:.0%}"))
        axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.axvline(0.01, color=AXIS, linewidth=0.8)
    err_ax.axvline(0.01, color=AXIS, linewidth=0.8)
    ax.axhline(0.5, color=AXIS, linewidth=0.8)
    ax.set(ylabel="Cumulative share of all rain")
    ax.set_title("Most rain falls in very little time (dots: half of all rain)")
    ax.legend(loc="lower right", fontsize=7.8)
    err_ax.set(ylabel="Cumulative share of squared error\naround the mean")
    err_ax.set_title("Why least squares struggles: the top 1% dominate the error")
    err_ax.legend(loc="lower right", fontsize=7.8)
    caption(fig, f"Sorting every station's 5-minute slots, hours and days from wettest to driest: half of all rain falls in "
                 f"{summary['5 min']['half_rain_in_share']:.2%} of 5-minute slots and {summary['1 day']['half_rain_in_share']:.0%} "
                 f"of days, and {summary['1 day']['zero_share']:.0%} of days have no rain at all. A least-squares model is judged "
                 f"mostly on the heaviest 1% of units (right), so it chases rare storms while predicting light rain on dry days. "
                 "For Task 2: separate 'will it rain' from 'how much', or use a Tweedie loss (32_compound_poisson_gamma).")
    save(fig, "30_concentration")
    write_table(pd.DataFrame(summary).T.reset_index(names="scale"), "rain_concentration")
    record("30_concentration", summary)


# ------------------------------------------------------------ 31 ----------

def spells(states):
    """Lengths of uncensored runs of 1s and of 0s in a 1-D array with NaN breaks."""
    wet_runs, dry_runs = [], []
    finite = np.isfinite(states)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], finite.astype(int), [0]])))
    for start, stop in zip(edges[::2], edges[1::2]):
        seg = states[start:stop]
        change = np.flatnonzero(np.diff(seg)) + 1
        bounds = np.concatenate([[0], change, [len(seg)]])
        for a, b in zip(bounds[1:-2], bounds[2:-1]):  # drop the first and last run: they may be censored
            (wet_runs if seg[a] == 1 else dry_runs).append(b - a)
    return wet_runs, dry_runs


def persistence(X, times):
    days = times[::288]
    totals = complete_days(X)
    wet = np.where(np.isfinite(totals), (totals >= 1 - 1e-9).astype(float), np.nan)
    month = days.month.to_numpy()
    today, yesterday, before = wet[2:], wet[1:-1], wet[:-2]
    m2 = month[2:]
    rows = []
    for mth in range(1, 13):
        sel = (m2 == mth)[:, None] & np.isfinite(today) & np.isfinite(yesterday)
        p_ww = np.mean(today[sel & (yesterday == 1)] == 1)
        p_wd = np.mean(today[sel & (yesterday == 0)] == 1)
        rows.append(dict(month=mth, p_wet=np.mean(today[sel] == 1), p_wet_after_wet=p_ww, p_wet_after_dry=p_wd))
    by_month = pd.DataFrame(rows)

    ok2 = np.isfinite(today) & np.isfinite(yesterday) & np.isfinite(before)
    history = []
    for b, y in ((1, 1), (0, 1), (1, 0), (0, 0)):
        # stratify by month so the comparison is not a seasonal mixture effect
        probs, weights = [], []
        for mth in range(1, 13):
            sel = ok2 & (m2 == mth)[:, None] & (before == b) & (yesterday == y)
            if sel.sum():
                probs.append(np.mean(today[sel] == 1))
                weights.append(sel.sum())
        history.append(dict(two_days_ago=b, yesterday=y, p_wet=np.average(probs, weights=weights), n=int(sum(weights))))
    history = pd.DataFrame(history)

    wet_runs, dry_runs = [], []
    for s in range(wet.shape[1]):
        w, d = spells(wet[:, s])
        wet_runs += w
        dry_runs += d
    sel = np.isfinite(today) & np.isfinite(yesterday)
    p11 = np.mean(today[sel & (yesterday == 1)] == 1)
    p00 = np.mean(today[sel & (yesterday == 0)] == 0)
    # Season-aware Markov chain: simulate with each month's own transition probabilities.
    sim_wet, sim_dry = [], []
    lookup = by_month.set_index("month")
    for _ in range(40):
        state = np.empty(len(days))
        state[0] = rng.random() < lookup.p_wet[month[0]]
        draws = rng.random(len(days))
        for t in range(1, len(days)):
            p = lookup.p_wet_after_wet[month[t]] if state[t - 1] == 1 else lookup.p_wet_after_dry[month[t]]
            state[t] = draws[t] < p
        w, d = spells(state)
        sim_wet += w
        sim_dry += d

    fig, (ax, spell_ax, hist_ax) = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"wspace": 0.3,
                                                                                      "width_ratios": [1.2, 1.2, 1]})
    ax.plot(by_month.month, by_month.p_wet_after_wet, color=BLUE, marker="o", markersize=5, markeredgecolor=SURFACE,
            label="P(wet | yesterday wet)")
    ax.plot(by_month.month, by_month.p_wet, color=MUTED, linewidth=1.4, label="P(wet), any day")
    ax.plot(by_month.month, by_month.p_wet_after_dry, color=ORANGE, marker="o", markersize=5, markeredgecolor=SURFACE,
            label="P(wet | yesterday dry)")
    ax.set(xticks=range(1, 13), xlabel="Month", ylabel="Probability of a wet day (>= 1 mm)", ylim=(0, 0.8))
    ax.set_xticklabels([m[0] for m in MONTHS])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Yesterday matters in every month")
    ax.legend(loc="lower left", fontsize=7.8)

    for runs, sims, p_stay, color, label in ((dry_runs, sim_dry, p00, ORANGE, "Dry spells"),
                                             (wet_runs, sim_wet, p11, BLUE, "Wet spells")):
        k = np.arange(1, 31)
        observed = [np.mean(np.asarray(runs) >= v) for v in k]
        simulated = [np.mean(np.asarray(sims) >= v) for v in k]
        spell_ax.plot(k, observed, "o", color=color, markersize=4.5, markeredgecolor=SURFACE,
                      label=f"{label}: observed")
        spell_ax.plot(k, simulated, color=color, linewidth=1.6, label=f"{label}: month-aware Markov chain")
        spell_ax.plot(k, p_stay ** (k - 1), color=color, linewidth=1, alpha=0.45)
    spell_ax.plot([], [], color=MUTED, linewidth=1, alpha=0.6, label="Faint: single pooled Markov chain")
    spell_ax.set_yscale("log")
    spell_ax.set(xlabel="Spell length (days)", ylabel="P(spell lasts at least this long)", ylim=(1e-4, 1.2), xlim=(0, 30))
    long_obs = np.mean(np.asarray(dry_runs) >= 15)
    long_sim = np.mean(np.asarray(sim_dry) >= 15)
    spell_ax.set_title("A Markov chain gets typical spells right")
    spell_ax.legend(loc="lower left", fontsize=7.2)

    labels = ["wet, wet", "dry, wet", "wet, dry", "dry, dry"]
    colors = [BLUE, BLUE, ORANGE, ORANGE]
    hist_ax.barh(range(4), history.p_wet, color=colors, height=0.55)
    for i, value in enumerate(history.p_wet):
        hist_ax.text(value + 0.01, i, f"{value:.0%}", va="center", fontsize=8, color=INK2)
    hist_ax.set(yticks=range(4), yticklabels=[f"{l}" for l in labels], xlim=(0, 0.75), ylim=(3.6, -0.6),
                xlabel="P(wet today), averaged within months")
    hist_ax.set_ylabel("Two days ago, yesterday")
    hist_ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    hist_ax.grid(axis="y", visible=False)
    hist_ax.set_title("Two days back still matters", fontsize=10.5)
    gap_w = history.p_wet[0] - history.p_wet[1]
    gap_d = history.p_wet[2] - history.p_wet[3]
    caption(fig, f"A wet day is followed by another wet day {p11:.0%} of the time, a dry day only {1 - p00:.0%}. A Markov chain with "
                 "monthly transition probabilities reproduces spells of up to about ten days, but dry spells of 15+ days are "
                 f"{long_obs / long_sim:.1f}x as common as it predicts ({long_obs:.2%} vs {long_sim:.2%} of dry spells), and the day "
                 f"before yesterday still shifts the odds by {gap_w:+.0%} (after a wet day) and {gap_d:+.0%} (after a dry day). "
                 "For Task 2: use lagged wet/dry indicators plus 3-7 day rolling totals, not just yesterday.")
    save(fig, "31_persistence")
    write_table(by_month.round(4), "transition_by_month")
    write_table(history.round(4), "two_day_history")
    record("31_persistence", {"p_wet_after_wet": round(float(p11), 3), "p_wet_after_dry": round(float(1 - p00), 3),
                              "dry_spells_15plus_observed_vs_markov": [round(float(long_obs), 4), round(float(long_sim), 4)],
                              "two_day_history": history.round(3).to_dict("records")})


# ------------------------------------------------------------ 32 ----------

def rain_events(X):
    """Split each complete station-day into events; return depths and events per day."""
    S = X.shape[1]
    per_day = X.reshape(-1, 288, S).transpose(0, 2, 1).reshape(-1, 288)  # rows: (day, station)
    complete = np.isfinite(per_day).all(axis=1)
    rows, cols = np.nonzero(np.where(complete[:, None], per_day, 0) > 0)
    values = per_day[rows, cols]
    starts = np.ones(len(rows), bool)
    starts[1:] = (rows[1:] != rows[:-1]) | (cols[1:] - cols[:-1] > GAP_SLOTS + 1)
    event = np.cumsum(starts) - 1
    depth = np.bincount(event, weights=values)
    counts = np.bincount(rows[starts], minlength=len(per_day))
    day_index = np.repeat(np.arange(per_day.shape[0] // S), S)
    return depth, counts[complete], np.nansum(per_day[complete], axis=1), day_index[complete]


def compound_poisson_gamma(X, times):
    depth, counts, totals, day_index = rain_events(X)
    days = times[::288]
    group = (days.year.to_numpy() * 12 + days.month.to_numpy())[day_index]  # year-month of each station-day
    lam_all = counts.mean()
    lam_month = pd.Series(counts).groupby(group).transform("mean").to_numpy()
    # Rate from the same day at the other stations (leave-one-out): a feature a forecast could use.
    day_total = np.bincount(day_index, weights=counts)
    day_n = np.bincount(day_index)
    others = day_n[day_index] - 1
    lam_day = np.where(others > 0, (day_total[day_index] - counts) / np.maximum(others, 1), lam_all)
    shape, _, scale = stats.gamma.fit(depth, floc=0)
    power = (shape + 2) / (shape + 1)

    k = np.arange(0, 9)
    observed = np.array([np.mean(counts == v) for v in k])
    single = stats.poisson.pmf(k, lam_all)
    mixed = np.array([np.mean(stats.poisson.pmf(v, lam_month)) for v in k])
    daily = np.array([np.mean(stats.poisson.pmf(v, lam_day)) for v in k])

    def simulate(lam):
        n = rng.poisson(lam)
        out = np.zeros(len(n))
        wet = n > 0
        # A sum of n gamma(shape) draws is gamma(n * shape).
        out[wet] = rng.gamma(n[wet] * shape, scale)
        return out
    sim_single, sim_mixed, sim_daily = (simulate(np.full(len(counts), lam_all)), simulate(lam_month),
                                        simulate(lam_day))

    fig, (n_ax, d_ax, t_ax) = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw={"wspace": 0.3})
    n_ax.bar(k, observed, color=AXIS, width=0.6, label="Observed")
    n_ax.plot(k - 0.18, single, "o", color=ORANGE, markersize=6, markeredgecolor=SURFACE,
              label=f"Poisson, one rate ({lam_all:.2f}/day)")
    n_ax.plot(k, mixed, "o", color=AQUA, markersize=6, markeredgecolor=SURFACE, label="Poisson, rate per month")
    n_ax.plot(k + 0.18, daily, "o", color=BLUE, markersize=6, markeredgecolor=SURFACE,
              label="Poisson, rate from other stations that day")
    n_ax.set(xticks=k, xlabel="Rain events in a station-day", ylabel="Share of station-days")
    n_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    n_ax.set_title("1. How many events? (N)")
    n_ax.legend(loc="upper right", fontsize=7.8)

    bins = np.logspace(np.log10(0.15), np.log10(depth.max() * 1.1), 40)
    hist, edges = np.histogram(depth, bins=bins, density=True)
    centres = np.sqrt(edges[:-1] * edges[1:])
    d_ax.step(centres, hist * centres, where="mid", color=INK2, linewidth=1.4, label="Observed event depths")
    grid = np.logspace(np.log10(0.15), np.log10(depth.max()), 200)
    d_ax.plot(grid, stats.gamma.pdf(grid, shape, scale=scale) * grid, color=BLUE,
              label=f"Gamma fit (shape {shape:.2f}, mean {shape * scale:.1f} mm)")
    d_ax.set_xscale("log")
    d_ax.set(xlabel="Rain in one event (mm, log scale)", ylabel="Share of events per log-width of size")
    d_ax.set_title("2. How big is each event? (gamma)")
    d_ax.legend(loc="upper right", fontsize=7.8)
    single_tip = np.mean(np.isclose(depth, 0.2, atol=0.011))
    d_ax.text(0.97, 0.80, f"{single_tip:.0%} of events are a single\n~0.2 mm bucket tip (the spike):\nsensor resolution, not weather",
              transform=d_ax.transAxes, ha="right", va="top", fontsize=7.5, color=INK2)

    thresholds = np.logspace(-1, np.log10(300), 120)
    for data, color, label, width in ((totals, INK, "Observed daily totals", 2.2),
                                      (sim_daily, BLUE, "Simulated: rate from other stations", 1.6),
                                      (sim_mixed, AQUA, "Simulated: rate per month", 1.6),
                                      (sim_single, ORANGE, "Simulated: one rate", 1.6)):
        t_ax.plot(thresholds, [np.mean(data > v) for v in thresholds], color=color, linewidth=width,
                  label=f"{label} (dry: {np.mean(data == 0):.0%})")
    t_ax.set_xscale("log")
    t_ax.set_yscale("log")
    t_ax.set(xlabel="Daily rainfall (mm, log scale)", ylabel="P(daily total exceeds this)", ylim=(1e-5, 1))
    t_ax.set_title("3. Their sum: the daily total")
    t_ax.legend(loc="lower left", fontsize=7.6)
    t_ax.text(0.97, 0.97, f"implied Tweedie power\np = (a+2)/(a+1) = {power:.2f}", transform=t_ax.transAxes, ha="right",
              va="top", fontsize=8, color=INK)
    over = np.var(counts) / np.mean(counts)
    dry = {name: np.mean(np.exp(-lam)) for name, lam in (("one", np.full(1, lam_all)), ("month", lam_month),
                                                         ("day", lam_day))}
    caption(fig, f"A day's rain is a random number of events (dry gaps up to 30 min bridged), each of random size: the compound "
                 f"Poisson-gamma model, i.e. the Tweedie family with 1 < p < 2. With one fixed rate, the counts are over-dispersed "
                 f"(variance/mean {over:.1f}) and dry days are underpredicted ({dry['one']:.0%} vs {np.mean(counts == 0):.0%}). A "
                 f"rate per month barely helps ({dry['month']:.0%}); a rate that follows the day's weather, here how many events "
                 f"the other stations had, lands within {abs(dry['day'] - np.mean(counts == 0)) * 100:.0f} points "
                 f"({dry['day']:.0%}). So the model form fits once the rate depends on "
                 "same-day information, which is what Task 2 features provide. The gamma tail is too light for the biggest "
                 f"storms. Spark: GeneralizedLinearRegression(family='tweedie', variancePower≈{power:.1f}, log link).", y=0.0)
    save(fig, "32_compound_poisson_gamma")
    record("32_tweedie", {"events": int(len(depth)), "events_per_day_mean": round(float(lam_all), 3),
                          "events_per_day_var_over_mean": round(float(over), 3),
                          "dry_share": {"observed": round(float(np.mean(totals == 0)), 3),
                                        "poisson_one_rate": round(float(np.exp(-lam_all)), 3),
                                        "poisson_monthly_rate": round(float(np.mean(np.exp(-lam_month))), 3),
                                        "poisson_same_day_other_stations": round(float(np.mean(np.exp(-lam_day))), 3)},
                          "gamma_shape": round(float(shape), 3), "gamma_scale": round(float(scale), 3),
                          "tweedie_power": round(float(power), 3), "single_tip_share": round(float(single_tip), 3),
                          "depth_quantiles_obs_vs_gamma": {q: [round(float(np.quantile(depth, q)), 1),
                                                               round(float(stats.gamma.ppf(q, shape, scale=scale)), 1)]
                                                           for q in (0.5, 0.9, 0.99, 0.999)}})


if __name__ == "__main__":
    style()
    X, stations, times = load_matrix()
    concentration(X)
    persistence(X, times)
    compound_poisson_gamma(X, times)
