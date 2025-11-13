# -*- coding: utf-8 -*-
"""
Compute SDI (Soil Desiccation Index) for each predefined Y-Z profile
at a specific date across all (or selected) vegetation folders.

This script reuses the same profile extraction logic as Task 5/6/7,
but instead of seasonal averages it evaluates the raw TH snapshot at
the requested date and writes the metrics to CSV files.

Usage:
    python compute_sdi_profiles.py --date 2010-06-15
    python compute_sdi_profiles.py --date "2010-06-15 12:00:00" --folders exoticshrub,naturalgrass

Outputs are saved under analysis_results/ as:
    <date>_<folder>_SDIProfiles.csv
"""

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd
import numpy as np

try:
    from hydrus_parser import HydrusModel
except ImportError:
    print("Error: hydrus_parser.py not found. Run this script from the project root.")
    sys.exit(1)

try:
    import analysis_runner as runner
except ImportError:
    print("Error: analysis_runner.py not found in the current directory.")
    sys.exit(1)

try:
    from analysis_plotting import _calculate_heatmap_data
except ImportError:
    print("Error: analysis_plotting.py not available; required for profile extraction.")
    sys.exit(1)

try:
    from plot_styles import TASK7_HEATMAP_CONFIG, SEASONAL_MASK_CONFIG
except ImportError:
    print("Error: plot_styles.py not found; required for profile configuration.")
    sys.exit(1)

try:
    from analysis_metrics import compute_profile_metrics
except ImportError:
    print("Error: analysis_metrics.py not found; metrics computation unavailable.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SDI profiles for a specific date.")
    parser.add_argument(
        "--date",
        required=True,
        help="Target date/time (ISO format recommended, e.g., 2010-06-15 or 2010-06-15T12:00:00)."
    )
    parser.add_argument(
        "--folders",
        default="all",
        help="Comma-separated list of simulation folder names. Use 'all' (default) to process every folder detected."
    )
    parser.add_argument(
        "--sort-folders",
        action="store_true",
        help="Sort the folder list alphabetically before processing."
    )
    parser.add_argument(
        "--output-date-format",
        default="%Y%m%d",
        help="Strftime format for embedding the date in output filenames (default: %%Y%%m%%d)."
    )
    return parser.parse_args()


def resolve_folders(args) -> List[str]:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    available = runner.find_simulation_folders(project_dir)
    if not available:
        print("No simulation folders (with MESHTRIA.TXT) found.")
        return []

    if args.folders.strip().lower() == "all":
        folders = available
    else:
        requested = [item.strip() for item in args.folders.split(",") if item.strip()]
        invalid = [name for name in requested if name not in available]
        if invalid:
            print(f"Warning: the following folders were not found and will be skipped: {', '.join(invalid)}")
        folders = [name for name in requested if name in available]

    if args.sort_folders:
        folders = sorted(folders)
    return folders


def parse_target_datetime(date_str: str) -> Optional[pd.Timestamp]:
    target = pd.to_datetime(date_str, errors="coerce")
    if pd.isna(target):
        return None
    return target


def find_timestep_index(date_list, target: pd.Timestamp) -> Optional[int]:
    dates = pd.to_datetime(pd.Series(date_list), errors="coerce")
    matches = dates[dates == target]
    if matches.empty:
        return None
    return int(matches.index[0])


def ensure_output_dir(project_dir: str) -> str:
    output_dir = os.path.join(project_dir, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def compute_profiles_for_folder(project_dir: str, folder_name: str, target_dt: pd.Timestamp,
                                output_dir: str, date_fmt: str):
    folder_path = os.path.join(project_dir, folder_name)
    loader = HydrusModel(folder_path)
    if not loader.load_all_data():
        print(f"[{folder_name}] Failed to load data, skipping.")
        return

    mesh = loader.get_mesh()
    th_data = loader.get_data_by_name("TH")
    dates = loader.get_dates()
    if mesh is None or th_data is None or not dates:
        print(f"[{folder_name}] Missing mesh, TH data, or dates. Skipping.")
        return

    idx = find_timestep_index(dates, target_dt)
    if idx is None:
        print(f"[{folder_name}] Target date {target_dt} not found in dataset.")
        return

    snapshot = th_data[idx, :]
    plot_data_list, _, _, _, _ = _calculate_heatmap_data(
        snapshot,
        "TH",
        "TASK7_HEATMAP_CONFIG",
        mesh
    )
    if not plot_data_list:
        print(f"[{folder_name}] No profile data could be interpolated for snapshot.")
        return

    mask_config = SEASONAL_MASK_CONFIG or {}
    threshold = mask_config.get("threshold", 0.12)
    theta_fc = TASK7_HEATMAP_CONFIG.get("theta_fc")
    sdi_sw_s = TASK7_HEATMAP_CONFIG.get("sdi_sw_s", 0.12)
    sdi_sw_h = TASK7_HEATMAP_CONFIG.get("sdi_sw_h", 0.085)

    rows = []
    for idx_profile, profile_data in enumerate(plot_data_list):
        metrics = compute_profile_metrics(
            profile_data,
            None,
            threshold,
            theta_fc=theta_fc,
            sdi_sw_s=sdi_sw_s,
            sdi_sw_h=sdi_sw_h
        )
        if not metrics:
            continue
        rows.append({
            "folder": folder_name,
            "vegetation": folder_name,
            "timestamp": target_dt.isoformat(sep=" "),
            "profile_index": idx_profile,
            "x_profile_m": profile_data.get("x_profile"),
        } | metrics)

    if not rows:
        print(f"[{folder_name}] Metrics could not be computed for any profile.")
        return

    df = pd.DataFrame(rows)
    date_token = target_dt.strftime(date_fmt)
    filename = f"{date_token}_{folder_name}_SDIProfiles.csv"
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"[{folder_name}] Saved {len(df)} profile rows -> {filepath}")


def main():
    args = parse_args()
    folders = resolve_folders(args)
    if not folders:
        print("No folders to process. Exiting.")
        return

    target_dt = parse_target_datetime(args.date)
    if target_dt is None:
        print(f"Error: Unable to parse date '{args.date}'. Use ISO format like 2010-06-15.")
        sys.exit(1)

    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = ensure_output_dir(project_dir)

    for folder in folders:
        compute_profiles_for_folder(project_dir, folder, target_dt, output_dir, args.output_date_format)


if __name__ == "__main__":
    main()

