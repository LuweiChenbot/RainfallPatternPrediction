"""Section 3: does yesterday's rain tell us about today's?

31_persistence  probability of a wet day after a wet or a dry day, by month
"""

import calendar

import numpy as np
import pandas as pd

from common import BLUE, MUTED, ORANGE, SURFACE, block_sum, load_matrix, plt, record, save, style, write_table

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def persistence(X, times):
    days = times[::288]
    totals = block_sum(X, 288)  # NaN unless all 288 readings of the day were observed
    wet = np.where(np.isfinite(totals), (totals >= 1 - 1e-9).astype(float), np.nan)
    today, yesterday = wet[1:], wet[:-1]
    month = days.month.to_numpy()[1:]
    rows = []
    for mth in range(1, 13):
        sel = (month == mth)[:, None] & np.isfinite(today) & np.isfinite(yesterday)
        rows.append(dict(month=mth, p_wet=np.mean(today[sel] == 1),
                         p_wet_after_wet=np.mean(today[sel & (yesterday == 1)] == 1),
                         p_wet_after_dry=np.mean(today[sel & (yesterday == 0)] == 1)))
    by_month = pd.DataFrame(rows)
    sel = np.isfinite(today) & np.isfinite(yesterday)
    p_ww = np.mean(today[sel & (yesterday == 1)] == 1)
    p_wd = np.mean(today[sel & (yesterday == 0)] == 1)

    fig, ax = plt.subplots(figsize=(4.2, 2.6), layout="constrained")
    ax.plot(by_month.month, by_month.p_wet_after_wet, color=BLUE, marker="o", markeredgecolor=SURFACE,
            markeredgewidth=0.6, label="After a wet day")
    ax.plot(by_month.month, by_month.p_wet, color=MUTED, linewidth=0.9, label="Any day")
    ax.plot(by_month.month, by_month.p_wet_after_dry, color=ORANGE, marker="o", markeredgecolor=SURFACE,
            markeredgewidth=0.6, label="After a dry day")
    ax.set(xticks=range(1, 13), xlabel="Month", ylabel="P(wet day)", ylim=(0, 0.8), xlim=(0.5, 12.5))
    ax.set_xticklabels([m[0] for m in MONTHS])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="lower left", fontsize=6.5)
    gap = by_month.p_wet_after_wet - by_month.p_wet_after_dry
    widest, narrowest = by_month.loc[gap.idxmax()], by_month.loc[gap.idxmin()]
    save(fig, "31_persistence",
         "Day-to-day persistence of rain. Probability that a station-day is wet (at least 1 mm) after a wet day "
         "(blue), after a dry day (orange) and on any day (grey), by month; only complete days are used. Over the "
         f"year a wet day follows a wet day {p_ww:.0%} of the time and a dry day {p_wd:.0%} of the time. The gap is "
         f"widest in {calendar.month_name[int(widest.month)]} ({widest.p_wet_after_wet:.0%} against "
         f"{widest.p_wet_after_dry:.0%}) and almost vanishes in {calendar.month_name[int(narrowest.month)]} "
         f"({narrowest.p_wet_after_wet:.0%} against {narrowest.p_wet_after_dry:.0%}), consistent with the "
         "short-lived storms of the southwest monsoon.",
         task2="Recent rain at the station is informative; give the model rain totals over several past windows.")
    write_table(by_month.round(4), "transition_by_month")
    record("31_persistence", {"p_wet_after_wet": round(float(p_ww), 3), "p_wet_after_dry": round(float(p_wd), 3),
                              "gap_by_month": dict(zip(MONTHS, gap.round(3)))})


if __name__ == "__main__":
    style()
    X, stations, times = load_matrix()
    persistence(X, times)
