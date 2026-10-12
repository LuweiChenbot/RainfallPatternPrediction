"""Section 2: where does it rain, and how do neighbouring stations relate?

20_station_map            relative wetness of each station (colour) and its data-derived rainfall region (shape)
21_correlation_dynamics   idea 1: neighbour correlation vs distance at 5 min ... 1 day
23_lead_lag               storm motion: which station gets the rain first, by monsoon season
"""

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import curve_fit
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu
from sklearn.metrics import adjusted_rand_score

from common import (AQUA, AXIS, BLUE, DIVERGING, INK, INK2, MUTED, NEUTRAL, ORANGE, RAMP, RESOLUTIONS, SURFACE,
                    TEXT_WIDTH, LaggedCorrelation, block_sum, distance_matrix, draw_boundary, load_matrix, local_xy,
                    network_monthly, plt, record, region_of, save, station_periods, style, write_table)

SEASONS = {"NE monsoon (Dec-Mar)": [12, 1, 2, 3], "SW monsoon (Jun-Sep)": [6, 7, 8, 9],
           "Inter-monsoon (Apr-May, Oct-Nov)": [4, 5, 10, 11]}
SEASON_COLORS = [BLUE, ORANGE, AQUA]
MIN_SHARED = {1: 105_120, 3: 35_040, 12: 8_760, 36: 2_920, 288: 365}  # about one year of shared data
BINS = np.array([0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50])
SHAPES = {"West": "o", "Central": "s", "East": "^"}


def exp_decay(d, c0, length):
    return c0 * np.exp(-d / length)


def upper(matrix):
    return matrix[np.triu_indices(matrix.shape[0], 1)]


# ------------------------------------------------------------ 21 ----------

def correlation_dynamics(X, stations):
    D = distance_matrix(stations)
    d_all = upper(D)
    curves, fits = {}, {}
    hourly = None
    for (name, k), color in zip(RESOLUTIONS.items(), RAMP):
        Y = block_sum(X, k)
        corr, n = LaggedCorrelation(Y).at(0)
        ok = (upper(n) >= MIN_SHARED[k]) & np.isfinite(upper(corr))
        d, r = d_all[ok], upper(corr)[ok]
        (c0, length), _ = curve_fit(exp_decay, d, r, p0=(0.8, 10), maxfev=10000)
        which = np.digitize(d, BINS) - 1
        table = pd.DataFrame({"bin": which, "d": d, "r": r}).groupby("bin").agg(
            distance=("d", "median"), median=("r", "median"), pairs=("r", "size"))
        curves[name], fits[name] = table, (c0, length, color)
        if k == 12:
            hourly = (corr, n)

    fig, ax = plt.subplots(figsize=(4.2, 2.8), layout="constrained")
    grid = np.linspace(0, 50, 200)
    for name, table in curves.items():
        c0, length, color = fits[name]
        ax.plot(grid, exp_decay(grid, c0, length), color=color, linewidth=0.9)
        ax.plot(table.distance, table["median"], "o", color=color, markeredgecolor=SURFACE, markeredgewidth=0.6,
                label=f"{name}: reach {length:.0f} km")
    ax.set(xlabel="Distance between the two stations (km)", ylabel="Correlation of rainfall", ylim=(0, 1), xlim=(0, 50))
    ax.legend(loc="upper right", fontsize=6.5)
    near = {n: curves[n]["median"].iloc[0] for n in curves}
    save(fig, "21_correlation_dynamics",
         "Correlation of rainfall between pairs of stations against their distance, for rain totals over five "
         "minutes to one day. Dots are medians per distance band; lines are fitted exponential decays "
         "c·exp(-d / reach). The reach grows with the time window, from "
         f"{fits['5 min'][1]:.0f} km for five-minute totals to {fits['1 day'][1]:.0f} km for daily totals, and "
         f"within 2 km the correlation rises from {near['5 min']:.2f} to {near['1 day']:.2f}: individual storms are "
         "local, while rainy days are island-wide.",
         task2="Rain at neighbouring stations is a strong feature; the useful radius grows with the forecast window.")
    rows = [dict(resolution=n, c0=fits[n][0], reach_km=fits[n][1]) for n in fits]
    write_table(pd.DataFrame(rows).round(4), "correlation_reach")
    record("21_correlation", {"reach_km": {n: round(fits[n][1], 1) for n in fits},
                              "median_r_within_2km": {n: round(float(v), 3) for n, v in near.items()}})
    return hourly


# ------------------------------------------------------------ regions -----

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
    """Ward clusters on hourly co-variation, short-record stations assigned, checked on held-out years."""
    corr, n = hourly
    x, _ = local_xy(stations)
    D = distance_matrix(stations)
    core = core_stations(n, MIN_SHARED[12])
    labels = name_by_longitude(ward_clusters(corr, core, 3), x[core])
    region = pd.Series(None, index=stations.index, dtype=object)
    region.iloc[core] = labels
    assigned = []
    for s in np.setdiff1d(np.arange(len(stations)), core):
        shared = n[s, core] >= MIN_SHARED[12]
        if shared.sum() >= 5:
            scores = {c: np.nanmean(corr[s, core][shared & (labels == c)]) for c in np.unique(labels)
                      if np.any(shared & (labels == c))}
            region.iloc[s] = max(scores, key=scores.get)
            assigned.append(s)

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
    stability = adjusted_rand_score(lab_odd, ward_clusters(c_even, both, 3))
    same = lab_odd[:, None] == lab_odd[None, :]
    d_b, r_b, s_b = upper(D[np.ix_(both, both)]), upper(c_even[np.ix_(both, both)]), upper(same)
    evidence = []
    for lo, hi in [(3, 8), (8, 13), (13, 20)]:
        sel = (d_b >= lo) & (d_b < hi) & np.isfinite(r_b)
        within, across = r_b[sel & s_b], r_b[sel & ~s_b]
        p = mannwhitneyu(within, across, alternative="greater").pvalue if len(within) and len(across) else np.nan
        evidence.append(dict(band=f"{lo}-{hi} km", within=np.median(within), across=np.median(across),
                             n_within=len(within), n_across=len(across), p=p))
    evidence = pd.DataFrame(evidence)
    ari_official = adjusted_rand_score(region_of(stations.iloc[core]), labels)

    table = stations[["station_id", "station_name"]].copy()
    table["cluster"] = region.values
    table.loc[assigned, "cluster"] = table.loc[assigned, "cluster"] + " (assigned)"
    write_table(table, "station_clusters")
    write_table(evidence.round(4), "cluster_heldout_evidence")
    stats = {"core_stations": int(len(core)), "assigned": int(len(assigned)),
             "sizes": pd.Series(labels).value_counts().to_dict(), "ari_odd_vs_even": round(stability, 3),
             "ari_vs_official_regions": round(ari_official, 3), "heldout_evidence": evidence.round(4).to_dict("records")}
    record("22_regions", stats)
    return region, stats, evidence


# ------------------------------------------------------------ 20 ----------

def relative_wetness(X, stations, times):
    """Each station's rate divided by the network rate of the same month, averaged over its good months."""
    months = station_periods(X, stations, times, "month")
    months = months[months.coverage >= 0.9].merge(network_monthly(X, stations, times).rename(columns={"rate": "network"}),
                                                  on=["year", "month"])
    months["ratio"] = months.rate / months.network
    index = months.groupby("station_id").agg(relative=("ratio", "mean"), months=("ratio", "size")).reset_index()
    return index[index.months >= 12]


def station_map(stations, wetness, region, stats, evidence):
    data = stations.assign(region=region.values).merge(wetness, on="station_id", how="left")
    spread = max(abs(data.relative - 1).max(), 0.05)
    norm = TwoSlopeNorm(1, 1 - spread, 1 + spread)

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, 3.5), layout="constrained")
    draw_boundary(ax)
    points = None
    for name, marker in SHAPES.items():
        sub = data[(data.region == name) & data.relative.notna()]
        points = ax.scatter(sub.location_longitude, sub.location_latitude, c=sub.relative, cmap=DIVERGING, norm=norm,
                            marker=marker, s=26, edgecolors=INK2, linewidths=0.4, zorder=3)
        empty = data[(data.region == name) & data.relative.isna()]
        ax.scatter(empty.location_longitude, empty.location_latitude, marker=marker, s=26, facecolors=SURFACE,
                   edgecolors=MUTED, linewidths=0.6, zorder=3)
    loose = data[data.region.isna() & data.relative.notna()]
    ax.scatter(loose.location_longitude, loose.location_latitude, c=loose.relative, cmap=DIVERGING, norm=norm,
               marker="D", s=16, edgecolors=INK2, linewidths=0.4, zorder=3)
    for _, row in pd.concat([data.nlargest(1, "relative"), data.nsmallest(1, "relative")]).iterrows():
        right_edge = row.location_longitude > 103.95
        ax.annotate(f"{row.station_name} ({row.relative:.2f})", (row.location_longitude, row.location_latitude),
                    xytext=(-6 if right_edge else 6, -9), textcoords="offset points", fontsize=6, color=INK2,
                    ha="right" if right_edge else "left", zorder=4)
    sizes = data.region.value_counts()
    handles = [Line2D([], [], marker=m, linestyle="none", markersize=4.5, markerfacecolor=NEUTRAL,
                      markeredgecolor=INK2, markeredgewidth=0.4, label=f"{name} ({sizes.get(name, 0)} stations)")
               for name, m in SHAPES.items()]
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=4.5, markerfacecolor=SURFACE,
                          markeredgecolor=MUTED, markeredgewidth=0.6, label="Hollow: fewer than 12 good months"))
    if len(loose):
        handles.append(Line2D([], [], marker="D", linestyle="none", markersize=3.5, markerfacecolor=NEUTRAL,
                              markeredgecolor=INK2, markeredgewidth=0.4, label="Not clustered"))
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, title="Rainfall region from clustering (shape)", title_fontsize=6.5,
              alignment="left")
    bar = fig.colorbar(points, ax=ax, fraction=0.03, pad=0.01, shrink=0.85)
    bar.set_label("Rainfall relative to the network (1 = average)")
    bar.outline.set_visible(False)

    by_region = data.dropna(subset=["relative", "region"]).groupby("region").relative.mean()
    wettest, driest = data.loc[data.relative.idxmax()], data.loc[data.relative.idxmin()]
    split = evidence[evidence.p < 0.05].band.tolist()
    save(fig, "20_station_map",
         "Where it rains more, and which stations rain together. Colour: each station's rainfall rate relative to "
         "the whole network in the same months (station-months with at least 90% coverage, stations with at least "
         f"12 such months; 1 = network average), from {wettest.relative:.2f} at {wettest.station_name} to "
         f"{driest.relative:.2f} at {driest.station_name}. Shape: rainfall regions found by Ward clustering of "
         f"1 - hourly correlation ({stats['core_stations']} stations with long overlapping records, "
         f"{stats['assigned']} short-record stations assigned to the closest cluster). Mean relative wetness is "
         f"{by_region.get('West', np.nan):.2f} in the West region, {by_region.get('Central', np.nan):.2f} in the "
         f"Central and {by_region.get('East', np.nan):.2f} in the East. Clusters built on odd years agree with "
         f"those built on even years (adjusted Rand index {stats['ari_odd_vs_even']:.2f}) and, in the held-out "
         f"years, pairs in the same cluster correlate more than equally distant pairs in different clusters at "
         f"{' and '.join(split)} (one-sided Mann-Whitney p < 0.05) but not further apart, so the regions are real "
         f"but soft; agreement with the five official URA planning regions (grey outlines and labels) is "
         f"{stats['ari_vs_official_regions']:.2f}. Station elevation is not in the dataset.",
         task2="Station position, a wetness ratio or a region label are compact station features.")
    write_table(data[["station_id", "station_name", "location_longitude", "location_latitude", "region", "relative",
                      "months"]].round(4), "station_relative_wetness")
    record("20_map", {"region_mean_relative": by_region.round(3).to_dict(),
                      "wettest": data.nlargest(3, "relative")[["station_name", "relative"]].round(3).values.tolist(),
                      "driest": data.nsmallest(3, "relative")[["station_name", "relative"]].round(3).values.tolist()})


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

    fig = plt.figure(figsize=(TEXT_WIDTH, 5.6), layout="constrained")
    layout = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(layout[0, 0], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    cap = 80
    for (season, fit), color in zip(results.items(), SEASON_COLORS):
        for _, row in yearly[yearly.season == season].iterrows():
            ax.annotate("", (np.radians(row.heading), min(row.speed, cap)), (0, 0), annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>,head_width=0.15,head_length=0.3", color=color,
                                        linewidth=0.6, alpha=float(np.clip(row.r2, 0.05, 0.7))))
        ax.annotate("", (np.radians(fit["heading"]), fit["speed"]), (0, 0), annotation_clip=False, zorder=5,
                    arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.6", color=color, linewidth=1.8))
    ax.set_rlim(0, cap)
    ax.set_rticks([20, 40, 60, 80])
    ax.set_yticklabels(["20", "40", "60", "80 km/h"], fontsize=6, color=INK2)
    ax.set_rlabel_position(200)
    ax.set_xticks(np.radians(range(0, 360, 45)), ["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=6.5)
    ax.grid(color=AXIS, linewidth=0.4)
    ax.spines["polar"].set_color(AXIS)
    ax.spines["polar"].set_linewidth(0.6)
    ax.set_title("(a) Direction storms move towards", pad=10)
    handles = [Line2D([], [], color=c, linewidth=1.8,
                      label=f"{s.split(' (')[0]}: to {compass(f['heading'])}, {f['speed']:.0f} km/h, R² = {f['r2']:.2f}")
               for (s, f), c in zip(results.items(), SEASON_COLORS)]
    handles.append(Line2D([], [], color=MUTED, linewidth=0.6, alpha=0.6, label="Thin: single years (fainter = weaker fit)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=6.3, frameon=False)

    for i, ((season, fit), color) in enumerate(zip(results.items(), SEASON_COLORS)):
        sub = fig.add_subplot(layout[(i + 1) // 2, (i + 1) % 2])
        sub.scatter(fit["along"], fit["lag"], s=1.5, color=color, alpha=0.2, linewidths=0)
        span = np.linspace(fit["along"].min(), fit["along"].max(), 10)
        sub.plot(span, span / fit["speed"] * 60, color=INK, linewidth=0.9, label=f"{fit['speed']:.0f} km/h")
        sub.axhline(0, color=AXIS, linewidth=0.6)
        sub.axvline(0, color=AXIS, linewidth=0.6)
        sub.set(xlabel="Distance along the storm direction (km)", ylabel="Delay of best correlation (min)",
                ylim=(-62, 62))
        sub.set_title(f"({'bcd'[i]}) {season}")
        sub.legend(loc="upper left", fontsize=6.5)
    sw = yearly[(yearly.season == "SW monsoon (Jun-Sep)") & (yearly.r2 >= 0.3)]
    sw_fit = results["SW monsoon (Jun-Sep)"]
    save(fig, "23_lead_lag",
         "Storm motion by monsoon season. For every station pair 5-30 km apart, the five-minute series are shifted "
         "against each other by up to 60 minutes either way; the shift with the highest correlation is taken as the "
         "travel time, and a plane fit of travel time against the pairs' displacement gives the storm velocity. "
         "(a) Direction and speed of storm motion per season (thick arrows) and per single year (thin arrows, "
         "fainter for weaker fits). (b)-(d) Travel time against distance along the fitted direction; the line's slope "
         f"is the inverse speed. In the southwest monsoon storms cross the island towards the {compass(sw_fit['heading'])} "
         f"at about {sw_fit['speed']:.0f} km/h (R² = {sw_fit['r2']:.2f}), and the {len(sw)} years with a clear fit "
         f"(R² ≥ 0.3) all point between {sw.heading.min():.0f}° and {sw.heading.max():.0f}°; the other seasons are "
         "less organised.",
         task2=f"In the southwest monsoon a station 15 km upstream sees the rain about {15 / sw_fit['speed'] * 60:.0f} "
               "minutes earlier, so recent rain at upstream (western) stations is a forecast feature.")
    write_table(yearly.round(3), "storm_motion_by_year")
    record("23_lead_lag", {s: {"heading_deg": round(f["heading"], 1), "towards": compass(f["heading"]),
                               "speed_kmh": round(f["speed"], 1), "r2": round(f["r2"], 3), "pairs": f["pairs"]}
                           for s, f in results.items()}
           | {"yearly_clear_fits_r2_ge_0.3": yearly[yearly.r2 >= 0.3].groupby("season").heading
                .agg(["count", "min", "max"]).round(1).to_dict("index")})


if __name__ == "__main__":
    style()
    X, stations, times = load_matrix()
    hourly = correlation_dynamics(X, stations)
    region, stats, evidence = rainfall_regions(X, stations, hourly, times)
    station_map(stations, relative_wetness(X, stations, times), region, stats, evidence)
    lead_lag(X, stations, times)
