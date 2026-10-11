"""Download the URA Master Plan 2019 region boundary (no sea) from data.gov.sg.

The GeoJSON supplies Singapore's coastline and its five official planning
regions. It is used only for map context and for comparing rainfall clusters
with official regions.
"""

import argparse
import json
import urllib.request
from pathlib import Path

DATASET_ID = "d_bf4d24df9129d5a8ff8cf82e20959ee0"
POLL_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download"
HEADERS = {"User-Agent": "curl/8.7.1", "Accept": "*/*"}


def fetch(url):
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("task1_revision/data/external/mp2019_region_boundary.geojson"))
    args = parser.parse_args()
    if args.output.exists():
        print(f"Boundary already present: {args.output}")
        return
    download_url = json.loads(fetch(POLL_URL))["data"]["url"]
    payload = fetch(download_url)
    regions = json.loads(payload)
    if regions.get("type") != "FeatureCollection" or len(regions["features"]) != 5:
        raise ValueError("Unexpected boundary file: expected five region features")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(f"Saved {len(payload):,} bytes to {args.output}")


if __name__ == "__main__":
    main()
