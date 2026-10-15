"""Build task2/demo/rain_replay.html: the final random forest's forecast for every station and five-minute step of
2024, the test year, with what happened, as one offline page. Run after notebook 10 (task2/data/test_predictions/):

    python task2/code/build_replay.py
"""
import base64
import gzip
import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

import config as C

DEMO = C.ROOT / "task2" / "demo"
BOUNDARY = C.ROOT / "task1_revision" / "data" / "external" / "mp2019_region_boundary.geojson"
STATIONS = C.MATRIX / "stations.csv"
DAY = 288
EXAMPLES = [  # date, title, threshold in mm, story
    ("2024-12-29", "Biggest burst", 5,
     "Afternoon storms that reach the eastern stations last. At Bukit Timah Road 65 mm fell in the 30 minutes "
     "after 16:40, the largest 30-minute total of 2024."),
    ("2024-06-22", "Pre-dawn squall", 1,
     "A southwest-monsoon squall before dawn: rain reaches the western stations first and moves east, and 97% "
     "of the day's rain falls before 10:00."),
    ("2024-01-07", "Monsoon surge", 1,
     "A northeast-monsoon surge: rain reaches every station between 00:20 and 04:00, the eastern ones first, "
     "and returns on and off through the day."),
    ("2024-11-22", "Afternoon storms", 5,
     "Inter-monsoon thunderstorms: 90% of the rain falls between 13:00 and 19:00, including 64 mm in 30 "
     "minutes at Woodlands Drive 62 from 14:50."),
    ("2024-04-16", "A hard day", 0,
     "Light showers scattered over the island from 03:30 into the afternoon, 214 mm in all across 59 stations. "
     "The rainy day of 2024 on which the forest gained least over persistence."),
    ("2024-07-23", "Dry day", 0,
     "Not a drop at any station. A good forecast stays close to zero all day."),
]


def simplify(points, tol):
    """Ramer-Douglas-Peucker on one ring of (lon, lat) points."""
    if len(points) < 3:
        return points
    a, b = points[0], points[-1]
    ab = b - a
    norm = np.hypot(*ab)
    d = (np.abs(np.cross(ab, points - a)) / norm) if norm > 0 else np.hypot(*(points - a).T)
    i = int(np.argmax(d))
    if d[i] <= tol:
        return np.array([a, b])
    return np.vstack([simplify(points[:i + 1], tol)[:-1], simplify(points[i:], tol)])


def outline(tol=0.0004, min_extent=0.004):
    rings = []
    for feature in json.loads(BOUNDARY.read_text())["features"]:
        geom = feature["geometry"]
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for polygon in polygons:
            ring = np.asarray(polygon[0])[:, :2]
            if np.ptp(ring[:, 0]) < min_extent and np.ptp(ring[:, 1]) < min_extent:
                continue
            rings.append(np.round(simplify(ring, tol), 4).tolist())
    return rings


def main():
    table = ds.dataset(str(C.DATA / "test_predictions"), format="parquet").to_table(
        columns=["station_id", "slot", "y_mm", "past30_mm", "RF", "persistence"])
    ids = sorted(set(table.column("station_id").to_pylist()))
    index = {s: i for i, s in enumerate(ids)}
    stations = pd.read_csv(STATIONS).set_index("station_id").loc[ids]
    slot = table.column("slot").to_numpy()
    first = int(slot.min())
    assert first % DAY == 0
    n_slots, n_stations = int(slot.max()) - first + 1, len(ids)
    n = n_stations * n_slots
    station = np.array([index[s] for s in table.column("station_id").to_pylist()])
    pos = station * n_slots + (slot - first)

    edges = np.asarray(C.BIN_EDGES)
    p30 = table.column("past30_mm").to_numpy(zero_copy_only=False).astype(float)
    now_class = np.where(np.isnan(p30), -1, (p30[:, None] > edges).sum(axis=1))
    pers = table.column("persistence").combine_chunks().flatten().to_numpy().reshape(-1, 8)
    persistence = {}
    for c in range(-1, 9):
        rows = pers[now_class == c]
        assert len(rows) and np.ptp(rows, axis=0).max() < 1e-6, c
        persistence[str(c)] = rows[0].astype(float).round(5).tolist()

    rf = np.full((8, n), 255, np.uint8)
    rf[:, pos] = np.round(table.column("RF").combine_chunks().flatten().to_numpy().reshape(-1, 8).T * 254)
    y = np.full(n, 65535, "<u2")
    y[pos] = np.round(table.column("y_mm").to_numpy() * 10)
    past = np.full(n, 65535, "<u2")
    past[pos] = np.where(np.isnan(p30), 65535, np.round(np.nan_to_num(p30) * 10))
    tenths = np.where(past[pos] == 65535, -1, (past[pos][:, None] > np.round(edges * 10)).sum(axis=1))
    assert (tenths == now_class).all()
    blob = gzip.compress(rf.tobytes() + y.tobytes() + past.tobytes(), 9)

    frame = pd.DataFrame({"station": station, "slot": slot - first, "y": table.column("y_mm").to_numpy()})
    examples = []
    for date, title, theta, text in EXAMPLES:
        day = (pd.Timestamp(date) - pd.Timestamp("2024-01-01")).days
        rows = frame[(frame.slot >= day * DAY) & (frame.slot < (day + 1) * DAY)]
        top = rows.loc[rows.y.idxmax()] if rows.y.max() > 0 else rows.iloc[(rows.slot - day * DAY - 180).abs().argmin()]
        examples.append({"title": title, "theta": theta, "text": text, "station": ids[int(top.station)],
                         "slot": int(top.slot)})

    payload = {"edges": C.BIN_EDGES, "n_slots": n_slots, "first_slot": first, "mapper": json.loads(
        (C.MODELS / "mapper.json").read_text())["classes"], "persistence": persistence,
        "stations": [{"id": s, "name": r.station_name, "lon": r.location_longitude, "lat": r.location_latitude}
                     for s, r in stations.iterrows()],
        "outline": outline(), "examples": examples}
    html = ((DEMO / "replay_template.html").read_text()
            .replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
            .replace("__DATA__", base64.b64encode(blob).decode("ascii")))
    (DEMO / "rain_replay.html").write_text(html)
    print(f"{n_stations} stations x {n_slots} slots, {len(pos):,} forecasts; data {len(blob) / 1e6:.1f} MB gzip; "
          f"wrote {DEMO / 'rain_replay.html'} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
