"""
get_observations.py — Fetch avalanche observations from avalanche.org

CLI usage:
    python get_observations.py --bbox "-115.23172,43.84563,-114.87403,44.25274" \\
                               --start 2020-10-01 --end 2021-05-01 \\
                               --out snfac_obs.csv

Script / import usage:
    from get_observations import get_observations
    df = get_observations(
        bbox="−115.23172,43.84563,−114.87403,44.25274",
        start_date="2020-10-01",
        end_date="2021-05-01",
    )
    df.to_csv("output.csv", index=False)
"""

import argparse
import time

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.avalanche.org/obs/v1/public/avalanche_observation/list/"

DEFAULT_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "origin": "https://www.sawtoothavalanche.com",
    "referer": "https://www.sawtoothavalanche.com/",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/145.0.0.0 Safari/537.36"
    ),
}

# Sawtooth / SNFAC bounding box as a sensible default
DEFAULT_BBOX = "-115.23172,43.84563,-114.87403,44.25274"

# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def get_observations(
    bbox: str               = DEFAULT_BBOX,
    start_date: str         = "2020-10-01",
    end_date: str           = "2021-05-01",
    *,
    page_size: int          = 100,
    sort_by: str            = "date",
    sort_order: str         = "desc",
    headers: dict | None    = None,
    delay: float            = 0.25,
    verbose: bool           = True,
) -> pd.DataFrame:
    """
    Fetch paginated avalanche observations from avalanche.org.

    Parameters
    ----------
    bbox        : Bounding box string "min_lon,min_lat,max_lon,max_lat"
    start_date  : Earliest observation date (str YYYY-MM-DD)
    end_date    : Latest observation date   (str YYYY-MM-DD)
    page_size   : Results per page (max 100)
    sort_by     : Field to sort by (default: "date")
    sort_order  : "asc" or "desc" (default: "desc")
    headers     : Optional custom HTTP headers
    delay       : Seconds to sleep between page requests
    verbose     : Show tqdm progress bar and summary

    Returns
    -------
    pd.DataFrame with one row per observation
    """
    headers = headers or DEFAULT_HEADERS
    all_obs: list[dict] = []
    page = 1
    pbar = None

    while True:
        params = {
            "date[gte]":  start_date,
            "date[lte]":  end_date,
            "page":       page,
            "page_size":  page_size,
            "sort_by":    sort_by,
            "sort_order": sort_order,
            "bbox":       bbox,
        }

        r = requests.get(BASE_URL, params=params, headers=headers, timeout=15)

        if r.status_code != 200:
            print(f"Error {r.status_code}: {r.text[:200]}")
            break

        data    = r.json()
        results = data.get("results", [])
        total   = data.get("total", 0)
        next_url = data.get("next_url")

        if verbose and pbar is None:
            pbar = tqdm(total=total, desc="Fetching observations", unit="obs")

        if not results:
            break

        all_obs.extend(results)

        if verbose and pbar is not None:
            pbar.update(len(results))

        if not next_url:
            break

        page += 1
        time.sleep(delay)

    if verbose and pbar is not None:
        pbar.close()

    df = pd.DataFrame(all_obs)

    if verbose:
        print(f"\nTotal observations collected: {len(df)}")
        if not df.empty:
            print(f"Columns: {df.columns.tolist()}")
            print(df.head(3))

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="get_observations",
        description="Fetch avalanche observations from avalanche.org",
    )
    parser.add_argument(
        "--bbox",
        default=DEFAULT_BBOX,
        metavar="\"min_lon,min_lat,max_lon,max_lat\"",
        help=f'Bounding box (default: SNFAC region "{DEFAULT_BBOX}")',
    )
    parser.add_argument(
        "--start",
        default="2020-10-01",
        metavar="YYYY-MM-DD",
        help="Start date (default: 2020-10-01)",
    )
    parser.add_argument(
        "--end",
        default="2025-05-01",
        metavar="YYYY-MM-DD",
        help="End date (default: 2021-05-01)",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="FILE.csv",
        help="Output CSV path (default: ./{start_date}_{stop_date}_avalanche_observations.csv)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        metavar="N",
        help="Results per page (default: 100)",
    )
    parser.add_argument(
        "--sort-by",
        default="date",
        help="Sort field (default: date)",
    )
    parser.add_argument(
        "--sort-order",
        default="desc",
        choices=["asc", "desc"],
        help="Sort direction (default: desc)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds between page requests (default: 0.25)",
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

    df = get_observations(
        bbox       = args.bbox,
        start_date = args.start,
        end_date   = args.end,
        page_size  = args.page_size,
        sort_by    = args.sort_by,
        sort_order = args.sort_order,
        delay      = args.delay,
        verbose    = not args.quiet,
    )

    out_path = args.out or f"{args.start}_{args.end}_avalanche_observations.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")


if __name__ == "__main__":
    main()