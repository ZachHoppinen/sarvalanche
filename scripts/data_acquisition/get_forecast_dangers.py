"""
get_dangers.py — Fetch historical avalanche danger forecasts from avalanche.org

CLI usage:
    python get_dangers.py SNFAC --start 2020-11-01 --end 2021-02-23 --out SNFAC_dangers.csv

Script / import usage:
    from get_dangers import get_dangers
    df = get_dangers("SNFAC", start_date="2020-11-01", end_date="2021-02-23")
    df.to_csv("output.csv", index=False)

Public API docs: https://github.com/NationalAvalancheCenter/Avalanche.org-Public-API-Docs
"""

import argparse
import json
import time
from datetime import date, timedelta

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.avalanche.org/v2/public/products"
MAP_URL  = "https://api.avalanche.org/v2/public/products/map-layer/{center_id}"

DEFAULT_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "origin": "https://www.avalanche.org",
    "referer": "https://www.avalanche.org/",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_zones(center_id: str, headers: dict) -> list[dict]:
    """Return a list of zone dicts for the given avalanche center."""
    url = MAP_URL.format(center_id=center_id)
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    geojson = r.json()

    zones = []
    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        zones.append({
            "zone_id":   feature.get("id"),
            "zone_name": props.get("name"),
            "center_id": props.get("center_id"),
            "state":     props.get("state"),
            "timezone":  props.get("timezone"),
        })
    return zones


def _parse_danger(forecast: dict) -> dict:
    """Extract danger levels and metadata from a single forecast dict."""
    danger = {}
    for d in forecast.get("danger", []):
        elev = d.get("valid_day", d.get("position", ""))
        danger[f"danger_above_{elev}"] = d.get("upper")
        danger[f"danger_near_{elev}"]  = d.get("middle")
        danger[f"danger_below_{elev}"] = d.get("lower")

    danger["danger_travel_advice"] = forecast.get("travel_advice")
    danger["bottom_line"]          = forecast.get("bottom_line")
    danger["published_time"]       = forecast.get("published_time")
    danger["expires_time"]         = forecast.get("expires_time")
    return danger


def _fetch_products(
    center_id: str,
    date_start: str,
    date_end: str,
    headers: dict,
    delay: float,
) -> list[dict]:
    """Fetch all forecast products for a center within a date range."""
    params = {
        "avalanche_center_id": center_id,
        "date_start":          date_start,
        "date_end":            date_end,
    }
    r = requests.get(BASE_URL, params=params, headers=headers, timeout=30)
    if not r.ok:
        print(f"  Warning: HTTP {r.status_code} for {center_id} {date_start}→{date_end}")
        return []
    time.sleep(delay)

    products = r.json()
    if not isinstance(products, list):
        products = products.get("results", [])

    return [p for p in products if p.get("product_type") == "forecast"]


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_dangers(
    center_id: str,
    start_date: str | date = "2020-09-01",
    end_date: str | date   = "2025-05-01",
    *,
    headers: dict | None = None,
    delay: float          = 0.1,
    verbose: bool         = True,
) -> pd.DataFrame:
    """
    Fetch historical avalanche danger forecasts for all zones in a center.

    Parameters
    ----------
    center_id   : Avalanche center ID, e.g. "SNFAC", "CAIC", "NWAC"
    start_date  : First date to fetch (str YYYY-MM-DD or date object)
    end_date    : Last date to fetch  (str YYYY-MM-DD or date object)
    headers     : Optional custom HTTP headers (defaults to DEFAULT_HEADERS)
    delay       : Seconds to sleep between API calls (be polite)
    verbose     : Print progress to stdout

    Returns
    -------
    pd.DataFrame with one row per (zone, forecast date)
    """
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)
    if isinstance(end_date, str):
        end_date = date.fromisoformat(end_date)

    headers = headers or DEFAULT_HEADERS

    # ── Discover zones ───────────────────────────────────────────────────────
    if verbose:
        print(f"Fetching zones for {center_id}...")
    zones = _get_zones(center_id, headers)
    if verbose:
        print(f"Found {len(zones)} zone(s):")
        for z in zones:
            print(f"  {str(z['zone_id']):>8}  {z['zone_name']}")

    # ── Fetch forecasts ──────────────────────────────────────────────────────
    # The /products endpoint supports range queries; chunk by zone to keep
    # responses manageable and to attach zone metadata to each row.

    all_rows: list[dict] = []

    zone_iter = tqdm(zones, desc="Zones", unit="zone") if verbose else zones

    for zone in zone_iter:
        zone_id   = zone["zone_id"]
        zone_name = zone["zone_name"]

        forecasts = _fetch_products(
            center_id,
            start_date.isoformat(),
            end_date.isoformat(),
            headers,
            delay,
        )

        fetched = 0
        skipped = 0

        for forecast in forecasts:
            # Only keep rows that belong to this zone (if zone info present)
            f_zone_id = forecast.get("zone_id") or forecast.get("forecast_zone_id")
            if f_zone_id and str(f_zone_id) != str(zone_id):
                continue

            if not forecast.get("danger"):
                skipped += 1
                continue

            # Determine the forecast date
            forecast_date = (
                forecast.get("published_time", "")[:10]
                or forecast.get("date", "")
            )

            row = {
                "date":      forecast_date,
                "zone_id":   zone_id,
                "zone_name": zone_name,
                "center_id": center_id,
            }
            row.update(_parse_danger(forecast))

            problems = forecast.get("forecast_avalanche_problems", [])
            for i, prob in enumerate(problems[:3]):
                row[f"problem_{i+1}_type"]       = prob.get("avalanche_problem_id")
                row[f"problem_{i+1}_likelihood"]  = prob.get("likelihood")
                row[f"problem_{i+1}_size_min"]    = prob.get("size", [None, None])[0]
                row[f"problem_{i+1}_size_max"]    = prob.get("size", [None, None])[-1]

            all_rows.append(row)
            fetched += 1

        if verbose:
            tqdm.write(f"  {zone_name}: {fetched} forecasts, {skipped} skipped")

    df = pd.DataFrame(all_rows)
    if verbose:
        print(f"\nTotal rows: {len(df)}")
        if not df.empty:
            print(df.head())
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="get_dangers",
        description="Fetch historical avalanche danger forecasts from avalanche.org",
    )
    parser.add_argument(
        "center_id",
        help="Avalanche center ID (e.g. SNFAC, CAIC, NWAC)",
    )
    parser.add_argument(
        "--start",
        default="2020-09-01",
        metavar="YYYY-MM-DD",
        help="End date (default: 2020-09-01)",
    )
    parser.add_argument(
        "--end",
        default="2025-05-01",
        metavar="YYYY-MM-DD",
        help="End date (default: 2025-05-01)",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="FILE.csv",
        help="Output CSV path (default: ./<CENTER_ID>_forecast_danger.csv)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds between API calls (default: 0.1)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    df = get_dangers(
        center_id  = args.center_id,
        start_date = args.start,
        end_date   = args.end,
        delay      = args.delay,
        verbose    = not args.quiet,
    )

    out_path = args.out or f"{args.center_id}_forecast_danger.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main()