# src/sarvalanche/__main__.py
import argparse
import logging
from shapely.geometry import box
from datetime import datetime

from sarvalanche.detection_pipeline import detect_avalanche_debris

log = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run SARvalanche detection")
    parser.add_argument("--xmin", type=float, required=True)
    parser.add_argument("--ymin", type=float, required=True)
    parser.add_argument("--xmax", type=float, required=True)
    parser.add_argument("--ymax", type=float, required=True)
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()

def validate_args(args):
    # Add any necessary argument validation here
    pass

def main():
    args = parse_args()

    args = validate_args(args)

    aoi = box(args.xmin, args.ymin, args.xmax, args.ymax)
    start = datetime.fromisoformat(args.start_date)
    end = datetime.fromisoformat(args.end_date)

    # Run detection
    debris_mask, products = detect_avalanche_debris(aoi, start, end)

    # Optional: save output
    if args.output:
        debris_mask.to_netcdf(args.output)
    else:
        log.info("Detection complete. No output file provided.")

if __name__ == "__main__":
    main()
