# -*- coding: utf-8 -*-
"""
Utility script to list all recorded timestamps between the start and end
of a HYDRUS simulation folder.

Usage (from repository root):

    python list_simulation_dates.py --folder path/to/simulation_folder

The script loads the folder via `hydrus_parser.HydrusModel`, extracts the
date sequence, and prints every timestamp to the terminal along with
basic summary information (count, first/last date).
"""

import argparse
import sys
import os
from datetime import datetime

try:
    from hydrus_parser import HydrusModel
except ImportError:
    print("Error: hydrus_parser.py not found. Please run this script from the project root.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="List all timestamps from a HYDRUS simulation folder."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to the HYDRUS simulation folder (must contain MESHTRIA.TXT)."
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort timestamps chronologically if they are out of order."
    )
    return parser.parse_args()


def load_dates(folder_path):
    model = HydrusModel(folder_path)
    if not model.load_all_data():
        raise RuntimeError("Failed to load simulation data.")
    dates = model.get_dates()
    if not dates:
        raise RuntimeError("No date entries were found in the simulation data.")
    return dates


def normalize_dates(date_list):
    """Convert date entries (str/datetime) into datetime objects for sorting/display."""
    normalized = []
    for item in date_list:
        if isinstance(item, datetime):
            normalized.append(item)
        else:
            try:
                normalized.append(datetime.fromisoformat(str(item)))
            except ValueError:
                # Fallback: try parsing as common HYDRUS format (YYYY/MM/DD HH:MM:SS)
                try:
                    normalized.append(datetime.strptime(str(item), "%Y/%m/%d %H:%M:%S"))
                except ValueError:
                    raise ValueError(f"Unrecognized date format: {item}")
    return normalized


def main():
    args = parse_args()
    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Error: Folder not found -> {folder}")
        sys.exit(1)

    try:
        dates_raw = load_dates(folder)
        dates_dt = normalize_dates(dates_raw)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if args.sort:
        dates_dt.sort()

    count = len(dates_dt)
    start = dates_dt[0]
    end = dates_dt[-1]

    print(f"\nSimulation folder : {folder}")
    print(f"Total timestamps  : {count}")
    print(f"Start date/time   : {start.isoformat(sep=' ')}")
    print(f"End date/time     : {end.isoformat(sep=' ')}")
    print("\nAll timestamps:")
    for idx, dt in enumerate(dates_dt, 1):
        print(f"{idx:04d}  {dt.isoformat(sep=' ')}")


if __name__ == "__main__":
    main()

