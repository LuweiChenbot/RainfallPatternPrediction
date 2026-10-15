"""Shared paths, plotting style, data cleaning and matrix helpers for the Task 1 revision.

Inputs are the five-minute matrix built by build_matrix.py and, for the coverage figure only,
outputs/task1/station_month.csv. Everything this revision writes stays inside task1_revision/.
"""

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.home() / ".cache" / "rainfall_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.patches import Polygon

ROOT = Path(__file__).resolve().parents[2]
TASK1 = ROOT / "outputs" / "task1"
REVISION = ROOT / "task1_revision"
MATRIX = REVISION / "data" / "matrix_5min"
BOUNDARY = REVISION / "data" / "external" / "mp2019_region_boundary.geojson"
FIGURES = REVISION / "outputs" / "figures"
TABLES = REVISION / "outputs" / "tables"
RESULTS = REVISION / "outputs" / "results.json"
CAPTIONS = REVISION / "outputs" / "captions.json"
TITLE = "Task 1 figures"
TASK2_CONFIG = ROOT / "task2" / "code" / "config.py"

SLOTS_PER_DAY = 288
TEXT_WIDTH = 6.3  # inches: A4 with 2.5 cm margins, so figures print at the size they were drawn
LAT0, LON0 = 1.35, 103.82  # map projection centre
RESOLUTIONS = {"5 min": 1, "15 min": 3, "1 hour": 12, "3 hours": 36, "1 day": 288}

# Validated reference palette (dataviz skill): categorical slots in fixed order,
# one-hue blue ramp for magnitude and blue/orange for wet/dry polarity.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal, light to dark
SURFACE, INK, INK2, MUTED, GRID, AXIS, NEUTRAL = (
    "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec")
SEQUENTIAL = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = matplotlib.colors.LinearSegmentedColormap.from_list(
    "dry_wet", ["#b8461b", ORANGE, "#f6b89b", NEUTRAL, "#9ec5f4", BLUE, "#184f95"])


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.titleweight": "normal",
        "axes.titlelocation": "left", "axes.titlepad": 4, "axes.labelsize": 8, "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "grid.linestyle": "-",
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "legend.frameon": False, "legend.fontsize": 7, "legend.title_fontsize": 7, "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3, "legend.labelspacing": 0.3,
        "lines.linewidth": 1.1, "lines.markersize": 3.5,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def use_outputs(root, title):
    """Send figures, tables, captions and results to another folder (Task 2 reuses this module)."""
    global FIGURES, TABLES, RESULTS, CAPTIONS, TITLE
    FIGURES, TABLES, RESULTS, CAPTIONS, TITLE = (root / "figures", root / "tables", root / "results.json",
                                                 root / "captions.json", title)


def save(fig, name, caption, placement="main", body=None, task2=None):
    """Write the figure as PDF and PNG and store its caption, kept off the image, for the report."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    width = fig.get_tightbbox(fig.canvas.get_renderer()).width + 0.04
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    entries = json.loads(CAPTIONS.read_text()) if CAPTIONS.exists() else {}
    entries[name] = {"placement": placement, "width_in": round(width, 2), "caption": caption,
                     "body": body, "task2": task2}
    CAPTIONS.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    write_caption_files(entries)
    print(f"  figure: {name} ({width:.2f} in wide)", flush=True)


def latex(text):
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                     ("_", r"\_"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("×", r"$\times$"), ("±", r"$\pm$"),
                     ("²", r"$^2$"), ("°", r"$^\circ$"), ("≈", r"$\approx$"), ("−", r"$-$"), ("θ", r"$\theta$"),
                     ("ρ", r"$\rho$"), ("σ", r"$\sigma$"), ("κ", r"$\kappa$"), ("λ", r"$\lambda$")]:
        text = text.replace(old, new)
    return text


def write_caption_files(entries):
    """captions.md for reading and figures.tex with one figure block per figure, main text first."""
    order = sorted(entries, key=lambda n: (entries[n]["placement"] != "main", n))
    md = [f"# {TITLE}: captions and text", "",
          "Generated by the plotting scripts. Each figure is drawn at its print size for an A4 page with 2.5 cm "
          "margins (6.3 in text width), so include it at the width given here (figures.tex does this) and the "
          "figure text prints at 7-8 pt.", ""]
    tex = [f"% {TITLE}, generated by the plotting scripts. Figures are drawn at print size: keep the widths below.", ""]
    for name in order:
        e = entries[name]
        if e["placement"] != "main" and "% Appendix" not in tex:
            tex += ["% Appendix", ""]
        md += [f"## {name} ({e['placement']}, {e['width_in']:.2f} in)", "", f"Caption: {e['caption']}", ""]
        if e.get("body"):
            md += [f"Body text: {e['body']}", ""]
            tex += [f"% Body text for {name}: {latex(e['body'])}"]
        if e.get("task2"):
            md += [f"Link to Task 2: {e['task2']}", ""]
        tex += [r"\begin{figure}[htbp]", r"  \centering",
                rf"  \includegraphics[width={e['width_in']:.2f}in]{{figures/{name}.pdf}}",
                rf"  \caption{{{latex(e['caption'])}}}", rf"  \label{{fig:{name}}}", r"\end{figure}", ""]
    (CAPTIONS.parent / "captions.md").write_text("\n".join(md))
    (CAPTIONS.parent / "figures.tex").write_text("\n".join(tex))


def write_table(frame, name):
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / f"{name}.csv", index=False)


def record(section, values):
    """Merge headline numbers into outputs/results.json for the change log."""
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[section] = values
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(data, indent=2, default=float) + "\n")


# ---------------------------------------------------------------- matrix ----

def load_matrix(clean=True):
    """Return X (slots x stations, float64, NaN = no reading), stations, times.

    With clean=True the readings Task 2 found to be wrong are set to NaN: months in which a gauge was stuck
    at zero, and the periods in which a gauge reported rain that never fell (task2/code/config.py, FAULTY).
    """
    if not (MATRIX / "manifest.json").exists():
        raise FileNotFoundError("Run build_matrix.py first.")
    stations = pd.read_csv(MATRIX / "stations.csv")
    ids = stations.station_id.tolist()
    blocks = []
    for year_dir in sorted(MATRIX.glob("source_year=*")):
        table = pq.read_table(year_dir, columns=["slot", *ids])
        slots = table.column("slot").to_numpy()
        values = np.column_stack([table.column(s).to_numpy(zero_copy_only=False).astype(np.float64)
                                  for s in ids])
        blocks.append(values[np.argsort(slots)])
    X = np.vstack(blocks)
    if X.shape[0] % SLOTS_PER_DAY:
        raise ValueError("Matrix does not contain whole days")
    times = pd.date_range("2017-01-01 00:00", periods=X.shape[0], freq="5min")
    if clean:
        X = mask_bad_readings(X, stations, times)
    return X, stations, times


def faulty_periods():
    spec = importlib.util.spec_from_file_location("task2_config", TASK2_CONFIG)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.FAULTY


def stuck_months(X, stations, times):
    """Station-months with no wet reading although the network expected at least 30 (Task 2's rule)."""
    obs = ~np.isnan(X)
    wet = (np.nan_to_num(X) > 0) & obs
    p_net = wet.sum(1) / np.maximum(obs.sum(1), 1)
    ym = (times.year * 100 + times.month).to_numpy()
    found = []
    for k in np.unique(ym):
        rows = ym == k
        O, E = wet[rows].sum(0), (obs[rows] * p_net[rows, None]).sum(0)
        found += [(stations.station_id[j], k) for j in np.flatnonzero((O == 0) & (E >= 30))]
    return found


def mask_bad_readings(X, stations, times):
    ids = stations.station_id.tolist()
    ym = (times.year * 100 + times.month).to_numpy()
    stuck = stuck_months(X, stations, times)
    for station, k in stuck:
        X[ym == k, ids.index(station)] = np.nan
    for station, start, end in faulty_periods():
        X[(times >= start) & (times < end), ids.index(station)] = np.nan
    return X


def station_periods(X, stations, times, period):
    """Readings, rain and coverage per station and calendar month (period="month") or year (period="year")."""
    key = (times.year * 100 + times.month if period == "month" else times.year).to_numpy()
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    observed = np.add.reduceat(~np.isnan(X), starts, axis=0)
    rain = np.add.reduceat(np.nan_to_num(X), starts, axis=0)
    expected = np.diff(np.r_[starts, len(key)])
    frame = pd.DataFrame({"station_id": np.tile(stations.station_id.to_numpy(), len(starts)),
                          "key": np.repeat(key[starts], X.shape[1]), "observations": observed.ravel(),
                          "expected": np.repeat(expected, X.shape[1]), "rain_mm": rain.ravel()})
    frame["coverage"] = frame.observations / frame.expected
    with np.errstate(invalid="ignore", divide="ignore"):
        frame["rate"] = frame.rain_mm * SLOTS_PER_DAY / frame.observations
    if period == "month":
        frame["year"], frame["month"] = frame.key // 100, frame.key % 100
    else:
        frame["year"] = frame.key
    return frame.drop(columns="key")


def network_monthly(X, stations, times, cutoff=0.9):
    """Mean rainfall rate (mm/day) over the station-months with at least the given coverage."""
    months = station_periods(X, stations, times, "month")
    return months[months.coverage >= cutoff].groupby(["year", "month"], as_index=False).rate.mean()


def block_sum(X, k):
    """Sum consecutive blocks of k slots; a block with any missing slot is NaN."""
    T, S = X.shape
    return X[: T - T % k].reshape(-1, k, S).sum(axis=1)


def local_xy(stations):
    """Equirectangular east/north coordinates in km around the island centre."""
    x = (stations.location_longitude.values - LON0) * 111.32 * np.cos(np.radians(LAT0))
    y = (stations.location_latitude.values - LAT0) * 110.57
    return x, y


def distance_matrix(stations):
    x, y = local_xy(stations)
    return np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])


class LaggedCorrelation:
    """Pearson correlation of every station pair over the slots both observed.

    corr(lag)[i, j] = corr(x_i(t), x_j(t + lag)). Missing readings are excluded
    pair by pair (never treated as zero). An optional 0/1 weight on t restricts
    the reference time to a season.
    """

    def __init__(self, X, weight=None):
        # float32 is exact for the 0/1 counts and ample for correlations.
        self.M = (~np.isnan(X)).astype(np.float32)
        self.Z = np.nan_to_num(X).astype(np.float32)
        self.Z2 = self.Z * self.Z
        w = np.ones(X.shape[0], np.float32) if weight is None else weight.astype(np.float32)
        self.Mw, self.Zw, self.Z2w = self.M * w[:, None], self.Z * w[:, None], self.Z2 * w[:, None]
        # Rows where no station has rain add nothing to the rain sums; skipping
        # them makes the lagged products several times faster.
        self.active = np.flatnonzero(self.Z.any(axis=1))

    def at(self, lag):
        if lag < 0:
            corr, n = self.at(-lag)
            return corr.T, n.T
        T = self.Z.shape[0]
        n = (self.Mw[: T - lag].T @ self.M[lag:]).astype(np.float64)
        ra = self.active[self.active < T - lag]          # rain at t on the reference side
        rb = self.active[self.active >= lag] - lag       # rain at t + lag on the lagged side
        sx = self.Zw[ra].T @ self.M[ra + lag]
        sxx = self.Z2w[ra].T @ self.M[ra + lag]
        sxy = self.Zw[ra].T @ self.Z[ra + lag]
        sy = self.Mw[rb].T @ self.Z[rb + lag]
        syy = self.Mw[rb].T @ self.Z2[rb + lag]
        sx, sy, sxx, syy, sxy = (s.astype(np.float64) for s in (sx, sy, sxx, syy, sxy))
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = sxy / n - (sx / n) * (sy / n)
            var = (sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2)
            corr = cov / np.sqrt(var)
        return corr, n


# --------------------------------------------------------------- mapping ----

def load_regions():
    regions = json.loads(BOUNDARY.read_text())["features"]
    shapes = []
    for feature in regions:
        polygons = feature["geometry"]["coordinates"]
        if feature["geometry"]["type"] == "Polygon":
            polygons = [polygons]
        rings = [np.asarray(polygon[0])[:, :2] for polygon in polygons]
        name = feature["properties"]["REGION_N"].title().replace(" Region", "")
        shapes.append((name, rings))
    return shapes


def draw_boundary(ax, labels=True, label_color=MUTED):
    for name, rings in load_regions():
        for ring in rings:
            ax.add_patch(Polygon(ring, closed=True, facecolor=NEUTRAL, edgecolor=AXIS, linewidth=0.4, zorder=0))
        if labels:
            ring = max(rings, key=len)
            nudge = {"North-East": (0.0, 0.035), "West": (0.0, 0.0)}.get(name, (0.0, -0.012))
            x, y = ring[:, 0].mean() + nudge[0], ring[:, 1].mean() + nudge[1]
            ax.text(x, y, name.upper(), ha="center", va="center", fontsize=6, color=label_color, zorder=1)
    ax.set_xlim(103.60, 104.05)
    ax.set_ylim(1.175, 1.48)
    ax.set_aspect(1 / np.cos(np.radians(LAT0)))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def region_of(stations):
    """Official URA region containing each station (nearest region if offshore)."""
    from matplotlib.path import Path as MplPath
    shapes = load_regions()
    points = np.column_stack([stations.location_longitude.values, stations.location_latitude.values])
    names = np.array([None] * len(points), dtype=object)
    for name, rings in shapes:
        for ring in rings:
            inside = MplPath(ring).contains_points(points)
            names[inside & (names == None)] = name  # noqa: E711
    for i in np.where(names == None)[0]:  # noqa: E711
        best = min(((np.min(np.hypot(*(ring - points[i]).T)), name) for name, rings in shapes for ring in rings))
        names[i] = best[1]
    return names
