# -*- coding: utf-8 -*-
"""
Scan the four vegetation simulations and report the date with the lowest
average soil moisture (TH) value.

The script reuses `analysis_runner.find_simulation_folders` to locate valid
HYDRUS output folders and leverages `hydrus_parser.HydrusModel` to load each
dataset, ensuring we stay consistent with the rest of the toolchain.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    from hydrus_parser import HydrusModel
except ImportError as exc:
    print(f"Error importing hydrus_parser: {exc}")
    sys.exit(1)

try:
    import analysis_runner as runner
except ImportError as exc:
    print(f"Error importing analysis_runner: {exc}")
    sys.exit(1)


TARGET_FOLDERS = ("arablecrop", "exoticgrass", "exoticshrub", "naturalgrass")


@dataclass
class FolderResult:
    """Container that stores the moisture statistics for a single folder."""

    folder: str
    min_avg_theta: float
    min_date: object  # datetime from HydrusModel
    timestep_index: int


def locate_target_folders(project_dir: str) -> Dict[str, str]:
    """
    Find the absolute paths for the vegetation folders that actually exist.
    """
    available = set(runner.find_simulation_folders(project_dir))
    folder_map: Dict[str, str] = {}
    missing: List[str] = []

    for name in TARGET_FOLDERS:
        if name in available:
            folder_map[name] = os.path.join(project_dir, name)
        else:
            missing.append(name)

    if missing:
        print(f"Warning: Missing simulation folders skipped: {', '.join(missing)}")

    return folder_map


def analyze_folder(folder_name: str, folder_path: str) -> Optional[FolderResult]:
    """
    Load TH data for the folder and return the minimum-average moisture date.
    """
    print(f"\n[{folder_name}] Loading HYDRUS outputs...")
    loader = HydrusModel(folder_path)

    if not loader.load_all_data():
        print(f"[{folder_name}] Failed to load model data.")
        return None

    th_data = loader.get_data_by_name("TH")
    dates = loader.get_dates()

    if th_data is None or not dates:
        print(f"[{folder_name}] Missing TH data or dates.")
        return None

    th_array = np.asarray(th_data, dtype=float)
    if th_array.ndim != 2:
        print(f"[{folder_name}] Unexpected TH array shape: {th_array.shape}")
        return None

    with np.errstate(invalid="ignore"):
        avg_per_step = np.nanmean(th_array, axis=1)

    valid_mask = np.isfinite(avg_per_step)
    if not np.any(valid_mask):
        print(f"[{folder_name}] No finite averages found.")
        return None

    valid_indices = np.where(valid_mask)[0]
    valid_values = avg_per_step[valid_mask]
    local_min_pos = int(np.argmin(valid_values))
    min_index = int(valid_indices[local_min_pos])
    min_avg_value = float(valid_values[local_min_pos])

    try:
        min_date = dates[min_index]
    except (IndexError, TypeError):
        print(f"[{folder_name}] Date list mismatch with TH data.")
        return None

    print(
        f"[{folder_name}] Lowest mean TH = {min_avg_value:.5f} "
        f"at index {min_index} (date: {min_date})"
    )
    return FolderResult(folder=folder_name, min_avg_theta=min_avg_value, min_date=min_date, timestep_index=min_index)


def format_date(value: object) -> str:
    """Return a readable string for datetime-like objects."""
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    folder_map = locate_target_folders(project_dir)

    if not folder_map:
        print("Error: none of the target vegetation folders are available.")
        return

    results: List[FolderResult] = []
    for folder, path in folder_map.items():
        result = analyze_folder(folder, path)
        if result is not None:
            results.append(result)

    if not results:
        print("No valid results were produced.")
        return

    print("\n--- Minimum Average Moisture by Vegetation Type ---")
    for res in results:
        print(
            f"{res.folder:>12}: {res.min_avg_theta:0.5f} "
            f"at {format_date(res.min_date)} (timestep #{res.timestep_index})"
        )

    overall_best = min(results, key=lambda item: item.min_avg_theta)
    print("\n--- Global Lowest Average Moisture ---")
    print(
        f"{overall_best.folder} on {format_date(overall_best.min_date)} "
        f"(timestep #{overall_best.timestep_index}) "
        f"with TH mean = {overall_best.min_avg_theta:0.5f}"
    )


if __name__ == "__main__":
    main()
