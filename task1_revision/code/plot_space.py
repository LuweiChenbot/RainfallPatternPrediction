"""Section 2: where does it rain, and how do neighbouring stations relate?

20_station_map            relative wetness of each station on the Singapore map
21_correlation_dynamics   idea 1: neighbour correlation vs distance at 5 min ... 1 day
22_rainfall_regions       clustering stations on hourly co-variation, checked on held-out years
23_lead_lag               storm motion: which station gets the rain first, by monsoon season
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import curve_fit
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu
from sklearn.metrics import adjusted_rand_score

from common import (AQUA, AXIS, BLUE, DIVERGING, INK, INK2, MUTED, ORANGE, RAMP, RESOLUTIONS,
                    SURFACE, TASK1, LaggedCorrelation, block_sum, caption, distance_matrix, draw_boundary,
                    load_matrix, local_xy, plt, record, region_of, save, style, write_table)

SEASONS = {"NE monsoon (Dec-Mar)": [12, 1, 2, 3], "SW monsoon (Jun-Sep)": [6, 7, 8, 9],
           "Inter-monsoon (Apr-May, Oct-Nov)": [4, 5, 10, 11]}
SEASON_COLORS = [BLUE, ORANGE, AQUA]
MIN_SHARED = {1: 105_120, 3: 35_040, 12: 8_760, 36: 2_920, 288: 365}  # about one year of shared data
BINS = np.array([0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50])


def exp_decay(d, c0, length):
    return c0 * np.exp(-d / length)


def upper(matrix):
    return matrix[np.triu_indices(matrix.shape[0], 1)]


# ------------------------------------------------------------ 20 ----------

def station_map(stations):
    month = pd.read_csv(TASK1 / "station_month.csv")
    network = pd.read_csv(TASK1 / "monthly_pattern.csv")[["year", "month", "mean_daily_equivalent_mm"]]
    month = month[month.coverage >= 0.9].merge(network.rename(columns={"mean_daily_equivalent_mm": "network"}),
                                                on=["year", "month"])
    # Ratio to the network in the same month removes the effect of which years a station was active.
    month["ratio"] = month.mean_daily_equivalent_mm / month.network
    index = month.groupby("station_id").agg(relative=("ratio", "mean"), months=("ratio", "size")).reset_index()
    data = stations.merge(index, on="station_id")
    data = data[data.months >= 12].copy()
    data["region"] = region_of(data)

    fig, ax = plt.subplots(figsize=(11, 6.4))
    draw_boundary(ax)
    spread = max(abs(data.relative - 1).max(), 0.05)
    points = ax.scatter(data.location_longitude, data.location_latitude, c=data.relative, s=70, cmap=DIVERGING,
                        norm=TwoSlopeNorm(1, 1 - spread, 1 + spread), edgecolors=SURFACE, linewidths=1.5, zorder=3)
    for _, row in pd.concat([data.nlargest(1, "relative"), data.nsmallest(1, "relative")]).iterrows():
        right_edge = row.location_longitude > 103.95
        ax.annotate(f"{row.station_name} ({row.relative:.2f})", (row.location_longitude, row.location_latitude),
                    xytext=(-8 if right_edge else 8, -12), textcoords="offset points", fontsize=7.5, color=INK2,
                    ha="right" if right_edge else "left", zorder=4)
    bar = fig.colorbar(points, ax=ax, fraction=0.03, pad=0.01)
    bar.set_label("Rainfall relative to all stations in the same months (1 = network average)")
    bar.outline.set_visible(False)
    by_region = data.groupby("region").relative.agg(["mean", "size"]).sort_values("mean")
    ax.set_title("Where it rains more: the west and centre are wetter, the east coast drier")
    summary = ", ".join(f"{r} {v['mean']:.2f}" for r, v in by_region.iloc[::-1].iterrows())
    caption(fig, f"Mean relative wetness by official region: {summary}. Each station is compared with the whole network in "
                 f"the same months (>= 90% coverage, >= 12 months; {len(data)} stations), so stations active in a dry year "
                 "such as 2019 are not penalised. Station elevation is not in the dataset and was not used.", y=0.04)
    save(fig, "20_station_map")
    write_table(data[["station_id", "station_name", "location_longitude", "location_latitude", "region", "relative",
                      "months"]].round(4), "station_relative_wetness")
    record("20_map", {"region_mean_relative": by_region["mean"].round(3).to_dict(),
                      "wettest": data.nlargest(3, "relative")[["station_name", "relative"]].round(3).values.tolist(),
                      "driest": data.nsmallest(3, "relative")[["station_name", "relative"]].round(3).values.tolist()})


# ------------------------------------------------------------ 21 ----------

def correlation_dynamics(X, stations):
    D = distance_matrix(stations)
    d_all = upper(D)
    curves, fits, hit = {}, {}, {}
    hourly = None
    for (name, k), color in zip(RESOLUTIONS.items(), RAMP):
        Y = block_sum(X, k)
        corr, n = LaggedCorrelation(Y).at(0)
        ok = (upper(n) >= MIN_SHARED[k]) & np.isfinite(upper(corr))
        d, r = d_all[ok], upper(corr)[ok]
        (c0, length), _ = curve_fit(exp_decay, d, r, p0=(0.8, 10), maxfev=10000)
        which = np.digitize(d, BINS) - 1
        table = pd.DataFrame({"bin": which, "d": d, "r": r}).groupby("bin").agg(
            distance=("d", "median"), median=("r", "median"), q25=("r", lambda v: v.quantile(.25)),
            q75=("r", lambda v: v.quantile(.75)), pairs=("r", "size"))
        curves[name], fits[name] = table, (c0, length, color, k)
        if k == 12:
            hourly = (corr, n)
        if k in (12, 288):
            wet = np.where(np.isnan(Y), np.nan, (Y >= (1.0 if k == 288 else 0.2) - 1e-9).astype(float))
            W, M = np.nan_to_num(wet).astype(np.float32), (~np.isnan(wet)).astype(np.float32)
            with np.errstate(invalid="ignore", divide="ignore"):
                conditional = (W.T @ W) / (W.T @ M)  # P(j wet | i wet), both observed
            observed = M.sum(0) > 0
            base = np.mean(W.sum(0)[observed] / M.sum(0)[observed])
            okc = ok & np.isfinite(upper(conditional))
            both = np.concatenate([upper(conditional)[okc], upper(conditional.T)[okc]])
            dist2 = np.concatenate([d_all[okc], d_all[okc]])
            which2 = np.digitize(dist2, BINS) - 1
            hit[name] = (pd.DataFrame({"bin": which2, "d": dist2, "p": both}).groupby("bin").agg(
                distance=("d", "median"), median=("p", "median")), base)

    fig, (ax, len_ax, hit_ax) = plt.subplots(1, 3, figsize=(15.5, 4.9), gridspec_kw={"width_ratios": [1.5, 1, 1.2],
                                                                                     "wspace": 0.3})
    grid = np.linspace(0, 50, 200)
    for name, table in curves.items():
        c0, length, color, _ = fits[name]
        ax.plot(grid, exp_decay(grid, c0, length), color=color, linewidth=1.6)
        ax.plot(table.distance, table["median"], "o", color=color, markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=f"{name}: reach {length:.0f} km")
    ax.set(xlabel="Distance between the two stations (km)", ylabel="Correlation of rainfall", ylim=(0, 1), xlim=(0, 50))
    ax.set_title("Neighbours agree, and agree more over longer windows")
    ax.legend(loc="upper right", fontsize=7.5, title="Dots: median per distance band\nLines: c·exp(-d / reach)",
              title_fontsize=7.5)

    minutes = [5 * k for *_, k in fits.values()]
    lengths = [fits[n][1] for n in fits]
    len_ax.plot(minutes, lengths, color=BLUE, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
    for name, m, L in zip(fits, minutes, lengths):
        len_ax.annotate(f"{L:.0f} km", (m, L), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8,
                        color=INK2)
    len_ax.set_xscale("log")
    len_ax.set(xticks=minutes, xticklabels=["5 min", "15 min", "1 h", "3 h", "1 day"], xlabel="Time window summed",
               ylabel="Correlation reach (km)", ylim=(0, max(lengths) * 1.25))
    len_ax.minorticks_off()
    len_ax.set_title("Reach of rain grows with time scale")

    for (name, (table, base)), color in zip(hit.items(), [RAMP[2], RAMP[4]]):
        label = "hour >= 0.2 mm" if name == "1 hour" else "day >= 1 mm"
        hit_ax.plot(table.distance, table["median"], color=color, marker="o", markersize=5, markeredgecolor=SURFACE,
                    label=f"P(neighbour wet | station wet), {label}")
        hit_ax.axhline(base, color=color, linewidth=0.9, alpha=0.6)
        hit_ax.text(49, base + 0.012, f"chance level {base:.0%}", ha="right", fontsize=7.5, color=INK2)
    hit_ax.set(xlabel="Distance between the two stations (km)", ylabel="Probability", ylim=(0, 1), xlim=(0, 50))
    hit_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    hit_ax.set_title("If one station is wet, is its neighbour?")
    hit_ax.legend(loc="upper right", fontsize=7.2)
    caption(fig, f"Within 2 km, five-minute rainfall correlates at {curves['5 min']['median'].iloc[0]:.2f} and daily totals at "
                 f"{curves['1 day']['median'].iloc[0]:.2f}. The distance over which correlation falls by a factor e grows from "
                 f"{fits['5 min'][1]:.0f} km (5 min) to {fits['1 day'][1]:.0f} km (1 day): storms are local, rainy days are "
                 "island-wide. For Task 2: neighbour rainfall is a strong feature (and a way to fill gaps), and the useful "
                 "neighbourhood radius should grow with the forecast window.", y=0.0)
    save(fig, "21_correlation_dynamics")
    rows = [dict(resolution=n, c0=fits[n][0], reach_km=fits[n][1]) for n in fits]
    write_table(pd.DataFrame(rows).round(4), "correlation_reach")
    record("21_correlation", {"reach_km": {n: round(fits[n][1], 1) for n in fits},
                              "median_r_within_2km": {n: round(float(curves[n]["median"].iloc[0]), 3) for n in curves},
                              "hit_rate_nearest_band": {n: round(float(t["median"].iloc[0]), 3) for n, (t, _) in hit.items()},
                              "chance_level": {n: round(float(b), 3) for n, (_, b) in hit.items()}})
    return hourly


# ------------------------------------------------------------ 22 ----------

def core_stations(n, min_shared):
    keep = np.ones(n.shape[0], bool)
    while True:
        bad = (n < min_shared) & keep[:, None] & keep[None, :]
        np.fill_diagonal(bad, False)
        if not bad.any():
            return np.flatnonzero(keep)
        keep[np.argmax(bad.sum(1))] = False


def ward_clusters(corr, core, k):
    dist = 1 - corr[np.ix_(core, core)]
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    return fcluster(linkage(squareform(dist, checks=False), "ward"), k, "maxclust")


def name_by_longitude(labels, x):
    order = np.argsort([x[labels == c].mean() for c in np.unique(labels)])
    names = {c: n for c, n in zip(np.unique(labels)[order], ["West", "Central", "East"])}
    return np.array([names[c] for c in labels])


def rainfall_regions(X, stations, hourly, times):
    corr, n = hourly
    x, y = local_xy(stations)
    D = distance_matrix(stations)
    core = core_stations(n, MIN_SHARED[12])
    labels = name_by_longitude(ward_clusters(corr, core, 3), x[core])
    assigned = {}
    for s in np.setdiff1d(np.arange(len(stations)), core):
        shared = n[s, core] >= MIN_SHARED[12]
        if shared.sum() >= 5:
            scores = {c: np.nanmean(corr[s, core][shared & (labels == c)]) for c in np.unique(labels)
                      if np.any(shared & (labels == c))}
            assigned[s] = max(scores, key=scores.get)

    # Held-out check: clusters from odd years, judged on even-year correlations.
    hours = times[::12]
    Y = block_sum(X, 12)
    halves = {}
    for parity in (1, 0):
        Yh = Y.copy()
        Yh[(hours.year.to_numpy() % 2) != parity] = np.nan
        halves[parity] = LaggedCorrelation(Yh).at(0)
    c_odd, n_odd = halves[1]
    c_even, n_even = halves[0]
    both = np.intersect1d(core_stations(n_odd, MIN_SHARED[12] // 2), core_stations(n_even, MIN_SHARED[12] // 2))
    lab_odd = ward_clusters(c_odd, both, 3)
    lab_even = ward_clusters(c_even, both, 3)
    stability = adjusted_rand_score(lab_odd, lab_even)
    same = lab_odd[:, None] == lab_odd[None, :]
    d_b, r_b, s_b = upper(D[np.ix_(both, both)]), upper(c_even[np.ix_(both, both)]), upper(same)
    bands = [(3, 8), (8, 13), (13, 20)]
    evidence = []
    for lo, hi in bands:
        sel = (d_b >= lo) & (d_b < hi) & np.isfinite(r_b)
        within, across = r_b[sel & s_b], r_b[sel & ~s_b]
        p = mannwhitneyu(within, across, alternative="greater").pvalue if len(within) and len(across) else np.nan
        evidence.append(dict(band=f"{lo}-{hi} km", within=np.median(within), across=np.median(across),
                             n_within=len(within), n_across=len(across), p=p))
    evidence = pd.DataFrame(evidence)
    official = region_of(stations.iloc[core])
    ari_official = adjusted_rand_score(official, labels)

    fig, (ax, ev_ax) = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.12})
    draw_boundary(ax, label_color=AXIS)
    colors = dict(zip(["West", "Central", "East"], [BLUE, ORANGE, AQUA]))
    for name, color in colors.items():
        pick = core[labels == name]
        ax.scatter(stations.location_longitude.values[pick], stations.location_latitude.values[pick], s=70,
                   color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3, label=f"{name} ({len(pick)} stations)")
        extra = [s for s, c in assigned.items() if c == name]
        ax.scatter(stations.location_longitude.values[extra], stations.location_latitude.values[extra], s=55,
                   facecolors=SURFACE, edgecolors=color, linewidths=1.8, zorder=3)
    ax.scatter([], [], s=55, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.8,
               label="Short record: assigned to the closest cluster")
    ax.legend(loc="lower left", fontsize=8, title="Clusters from hourly co-variation", title_fontsize=8)
    ax.set_title("Rainfall regions found from the data alone")

    pos = np.arange(len(evidence))
    ev_ax.plot(evidence.across, pos, "o", color=MUTED, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
               label="Different clusters", zorder=3)
    ev_ax.plot(evidence.within, pos, "o", color=BLUE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5,
               label="Same cluster", zorder=4)
    for i, row in evidence.iterrows():
        ev_ax.plot([row.across, row.within], [i, i], color=AXIS, linewidth=1.5, zorder=0)
        ev_ax.annotate(f"p={row.p:.1g}", (max(row.within, row.across), i), xytext=(8, -3), textcoords="offset points",
                       fontsize=7.5, color=INK2)
    ev_ax.set(yticks=pos, yticklabels=[f"pairs {b} apart" for b in evidence.band], ylim=(len(evidence) - 0.4, -0.6),
              xlabel="Median hourly correlation in held-out (even) years")
    ev_ax.set_title("Same distance, same cluster: more alike?", fontsize=10)
    ev_ax.legend(loc="lower right", fontsize=7.5)
    split = evidence[evidence.p < 0.05].band.tolist()
    caption(fig, f"Ward clustering on 1 - hourly correlation ({len(core)} stations with long overlapping records; k = 3). Built on odd "
                 f"years only, the clusters still separate equally distant pairs in the even years at {', '.join(split)} "
                 "(right), but not further apart, so the regions are real yet soft. Odd- and even-year clusterings agree "
                 f"(adjusted Rand index {stability:.2f}); agreement with the five official URA regions is ARI {ari_official:.2f}. "
                 "For Task 2: a cluster label or cluster-mean rainfall is a compact spatial feature.", y=0.03)
    save(fig, "22_rainfall_regions")
    table = stations[["station_id", "station_name"]].copy()
    table["cluster"] = None
    table.loc[core, "cluster"] = labels
    for s, c in assigned.items():
        table.loc[s, "cluster"] = c + " (assigned)"
    write_table(table, "station_clusters")
    write_table(evidence.round(4), "cluster_heldout_evidence")
    record("22_regions", {"core_stations": int(len(core)), "assigned": int(len(assigned)),
                          "sizes": pd.Series(labels).value_counts().to_dict(), "ari_odd_vs_even": round(stability, 3),
                          "ari_vs_official_regions": round(ari_official, 3),
                          "heldout_evidence": evidence.round(4).to_dict("records")})


# ------------------------------------------------------------ 23 ----------

def storm_motion(lc, pairs, dx, dy, lags):
    curves = np.stack([lc.at(L)[0][pairs] for L in lags])
    best = np.argmax(np.nan_to_num(curves, nan=-9), axis=0)
    cols = np.arange(curves.shape[1])
    peak = curves[best, cols]
    inner = (best > 0) & (best < len(lags) - 1) & (peak > 0.1)
    b, c = best[inner], cols[inner]
    lo, mid, hi = curves[b - 1, c], curves[b, c], curves[b + 1, c]
    lag_min = (np.asarray(lags)[b] + 0.5 * (lo - hi) / (lo - 2 * mid + hi)) * 5
    A = np.column_stack([dx[inner], dy[inner]])
    slowness, *_ = np.linalg.lstsq(A, lag_min, rcond=None)
    fitted = A @ slowness
    r2 = 1 - np.sum((lag_min - fitted) ** 2) / np.sum((lag_min - lag_min.mean()) ** 2)
    speed = 60 / np.linalg.norm(slowness)
    heading = np.degrees(np.arctan2(slowness[0], slowness[1])) % 360
    along = A @ (slowness / np.linalg.norm(slowness))
    return dict(speed=speed, heading=heading, r2=r2, along=along, lag=lag_min, pairs=int(inner.sum()))


def compass(degrees):
    names = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return names[int((degrees + 11.25) // 22.5) % 16]


def lead_lag(X, stations, times):
    D = distance_matrix(stations)
    x, y = local_xy(stations)
    iu = np.triu_indices(len(stations), 1)
    sel = (D[iu] >= 5) & (D[iu] <= 30)
    pairs = (iu[0][sel], iu[1][sel])
    dx, dy = x[pairs[1]] - x[pairs[0]], y[pairs[1]] - y[pairs[0]]
    lags = list(range(-12, 13))
    months, years = times.month.to_numpy(), times.year.to_numpy()
    results, yearly = {}, []
    for season, members in SEASONS.items():
        in_season = np.isin(months, members)
        lc = LaggedCorrelation(X, in_season)
        shared = lc.at(0)[1][pairs] >= 50_000
        results[season] = storm_motion(lc, tuple(p[shared] for p in pairs), dx[shared], dy[shared], lags)
        for year in range(2017, 2025):
            rows = years == year  # lags that cross New Year are dropped; negligible
            lc_year = LaggedCorrelation(X[rows], in_season[rows])
            enough = lc_year.at(0)[1][pairs] >= 5_000
            if enough.sum() >= 100:
                fit = storm_motion(lc_year, tuple(p[enough] for p in pairs), dx[enough], dy[enough], lags)
                yearly.append(dict(season=season, year=year, heading=fit["heading"], speed=fit["speed"],
                                   r2=fit["r2"], pairs=fit["pairs"]))
        print(f"  {season}: towards {results[season]['heading']:.0f} deg at {results[season]['speed']:.0f} km/h", flush=True)
    yearly = pd.DataFrame(yearly)

    fig = plt.figure(figsize=(18, 5.6))
    layout = fig.add_gridspec(1, 4, width_ratios=[1.3, 1, 1, 1], wspace=0.38)
    ax = fig.add_subplot(layout[0], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    cap = 80
    for (season, fit), color in zip(results.items(), SEASON_COLORS):
        for _, row in yearly[yearly.season == season].iterrows():
            ax.annotate("", (np.radians(row.heading), min(row.speed, cap)), (0, 0), annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1,
                                        alpha=float(np.clip(row.r2, 0.05, 0.7))))
        ax.annotate("", (np.radians(fit["heading"]), fit["speed"]), (0, 0), annotation_clip=False, zorder=5,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.45,head_length=0.9", color=color, linewidth=3.2))
    ax.set_rlim(0, cap)
    ax.set_rticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80 km/h"], fontsize=7, color=INK2)
    ax.set_rlabel_position(200)
    ax.set_xticks(np.radians(range(0, 360, 45)), ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    ax.grid(color=AXIS, linewidth=0.6)
    ax.spines["polar"].set_color(AXIS)
    ax.set_title("Direction storms move towards", pad=18)
    handles = [Line2D([], [], color=c, linewidth=3,
                      label=f"{s.split(' (')[0]}: to {compass(f['heading'])}, {f['speed']:.0f} km/h, R²={f['r2']:.2f}")
               for (s, f), c in zip(results.items(), SEASON_COLORS)]
    handles.append(Line2D([], [], color=MUTED, linewidth=1, alpha=0.6,
                          label="Thin arrows: single years (fainter = weaker fit)"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.05, -0.06), fontsize=7.6, frameon=False)
    fig.suptitle("Storms cross the island from the west, most consistently in the southwest monsoon",
                 x=0.06, ha="left", fontweight="bold", fontsize=12, y=1.04)

    for i, ((season, fit), color) in enumerate(zip(results.items(), SEASON_COLORS)):
        sub = fig.add_subplot(layout[i + 1])
        sub.scatter(fit["along"], fit["lag"], s=5, color=color, alpha=0.18, linewidths=0)
        span = np.linspace(fit["along"].min(), fit["along"].max(), 10)
        sub.plot(span, span / fit["speed"] * 60, color=INK, linewidth=1.5, label=f"{fit['speed']:.0f} km/h")
        sub.axhline(0, color=AXIS, linewidth=0.8)
        sub.axvline(0, color=AXIS, linewidth=0.8)
        sub.set(xlabel="Distance along the storm direction (km)", ylim=(-62, 62))
        if i == 0:
            sub.set_ylabel("Delay of best correlation (min)")
        sub.set_title(season, fontsize=10)
        sub.legend(loc="upper left", fontsize=7.5)
    sw = yearly[(yearly.season == "SW monsoon (Jun-Sep)") & (yearly.r2 >= 0.3)]
    caption(fig, "For every station pair 5-30 km apart, the five-minute series are shifted against each other by up to "
                 "+/-60 minutes; the shift with the highest correlation is the travel time, and a plane fit of travel time "
                 f"against displacement gives the storm velocity. Southwest-monsoon years with a clear fit (R² >= 0.3, "
                 f"{len(sw)} of 8) all point between {sw.heading.min():.0f}° and {sw.heading.max():.0f}°; the other seasons "
                 "are less organised. For Task 2: in the southwest monsoon a station 15 km upstream sees the rain about "
                 f"{15 / results['SW monsoon (Jun-Sep)']['speed'] * 60:.0f} minutes earlier, so lagged upstream rainfall is a "
                 "forecast feature.", y=-0.04)
    save(fig, "23_lead_lag")
    write_table(yearly.round(3), "storm_motion_by_year")
    record("23_lead_lag", {s: {"heading_deg": round(f["heading"], 1), "towards": compass(f["heading"]),
                               "speed_kmh": round(f["speed"], 1), "r2": round(f["r2"], 3), "pairs": f["pairs"]}
                           for s, f in results.items()}
           | {"yearly_clear_fits_r2_ge_0.3": yearly[yearly.r2 >= 0.3].groupby("season").heading
                .agg(["count", "min", "max"]).round(1).to_dict("index")})


if __name__ == "__main__":
    style()
    X, stations, times = load_matrix()
    station_map(stations)
    hourly = correlation_dynamics(X, stations)
    rainfall_regions(X, stations, hourly, times)
    lead_lag(X, stations, times)
