# -*- coding: utf-8 -*-
"""
Compute SDI (Soil Desiccation Index) for each predefined Y-Z profile
at one or more dates across all (or selected) vegetation folders.

This script reuses the same profile extraction logic as Task 5/6/7,
but instead of seasonal averages it evaluates the raw TH snapshot at
the requested dates and writes the metrics to CSV files. When at least
two dates are processed, the script also generates a comparison plot that
shows SDI depth profiles for the four primary vegetation types.

Usage:
    python compute_sdi_profiles.py --date 2010-06-15
    python compute_sdi_profiles.py --date 2010-06-15 --date 2012-04-07 --folders exoticshrub,naturalgrass

Outputs are saved under analysis_results/ as:
    <date>_<folder>_SDIProfiles.csv
    SDI_TwoDateComparison_<date1>_<date2>.png (if >=2 dates)
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

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
    from plot_styles import (
        TASK7_HEATMAP_CONFIG,
        SEASONAL_MASK_CONFIG,
    )
except ImportError:
    print("Error: plot_styles.py not found; required for profile configuration.")
    sys.exit(1)

try:
    from analysis_metrics import compute_profile_metrics
except ImportError:
    print("Error: analysis_metrics.py not found; metrics computation unavailable.")
    sys.exit(1)

try:
    from data_cache import load_cached_dataframe, save_dataframe_with_signature
except ImportError:
    print("Error: data_cache.py not found; required for caching mechanism.")
    sys.exit(1)

SDI_LAYER_BOUNDS_CM = [
    (0, 10),
    (10, 20),
    (20, 40),
    (40, 60),
    (60, 80),
    (80, 100),
    (100, 120),
    (120, 140),
    (140, 160),
    (160, 180),
    (180, 200),
    (200, 220),
    (220, 240),
    (240, 260),
    (260, 280),
    (280, 300),
    (300, 320),
    (320, 340),
    (340, 360),
    (360, 380),
    (380, 400),
]
SDI_LAYER_COLUMNS = [f"sdi_{a}_{b}cm_percent" for (a, b) in SDI_LAYER_BOUNDS_CM]
SDI_LAYER_MIDPOINTS_CM = [0.5 * (a + b) for (a, b) in SDI_LAYER_BOUNDS_CM]
_layer_boundaries = set()
for lower, upper in SDI_LAYER_BOUNDS_CM:
    _layer_boundaries.add(lower)
    _layer_boundaries.add(upper)
SDI_LAYER_BOUNDARIES_CM = sorted(_layer_boundaries)

PLOT_VEGETATION_ORDER = ["arablecrop", "exoticgrass", "exoticshrub", "naturalgrass"]
VEGETATION_DISPLAY_NAMES = {
    "arablecrop": "Arable Crop",
    "exoticgrass": "Exotic Grass",
    "exoticshrub": "Exotic Shrub",
    "naturalgrass": "Natural Grass",
}

# --- Visualization style specific to this script ---
SCI_FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 18
LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 18
MARKER_SIZE = 10
MARKER_EDGE_WIDTH = 1.6

CUSTOM_VEGETATION_COLORS = {
    "arablecrop": "#C0392B",     # deep red
    "exoticgrass": "#164E73",    # navy
    "exoticshrub": "#8E44AD",    # purple
    "naturalgrass": "#27AE60",   # green
}
CUSTOM_MARKERS = {
    "arablecrop": "o",
    "exoticgrass": "s",
    "exoticshrub": "^",
    "naturalgrass": "D",
}

VEGETATION_STYLE_MAP = {}
for veg in PLOT_VEGETATION_ORDER:
    color = CUSTOM_VEGETATION_COLORS.get(veg, "#444444")
    VEGETATION_STYLE_MAP[veg] = {
        "color": color,
        "marker": CUSTOM_MARKERS.get(veg, "o"),
    }
SDI_REFERENCE_LINES = {
    "moisture_boundary": {
        "value": 0.0,
        "color": "#C0392B",
        "label_left": "Not desiccated",
        "label_right": "Desiccation",
    }
}


def _apply_local_plot_style():
    """
    Apply scientific journal style settings locally for this script.
    """
    plt.rcParams.update({
        "font.family": SCI_FONT_FAMILY,
        "font.weight": "bold",
        "font.size": LABEL_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "legend.title_fontsize": LEGEND_FONT_SIZE,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "figure.titlesize": TITLE_FONT_SIZE + 2,
        "figure.titleweight": "bold",
        "savefig.dpi": 600,
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Compute SDI profiles for one or more dates.")
    parser.add_argument(
        "--date",
        dest="dates",
        action="append",
        required=True,
        help=("Target date/time (ISO). Provide multiple --date entries to process multiple timestamps, "
              "e.g., --date 2010-06-15 --date 2011-04-07.")
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

    # --- Align geometry conventions with analysis_runner (units + flatten) ---
    try:
        points = mesh.points
        # Unit conversion (cm -> m) based on Y-extent heuristic
        y_coords = points[:, 1]
        if np.any(y_coords > 2.5) or np.any(y_coords < -2.5):
            mesh.points = points / 100.0
            points = mesh.points

        # Apply 12-degree rotation to flatten slope (same as analysis_runner)
        angle_rad = np.deg2rad(12.0)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        x_coords = points[:, 0]
        z_coords = points[:, 2]

        top_nodes_mask = np.abs(x_coords) < 0.1
        if np.any(top_nodes_mask):
            z_hinge = np.max(z_coords[top_nodes_mask])
        else:
            z_hinge = np.max(z_coords)

        x_orig = np.copy(x_coords)
        z_orig = np.copy(z_coords)
        z_translated = z_orig - z_hinge
        x_rotated = x_orig * cos_a - z_translated * sin_a
        z_rotated = x_orig * sin_a + z_translated * cos_a
        z_new = z_rotated + z_hinge
        mesh.points[:, 0] = x_rotated
        mesh.points[:, 2] = z_new
    except Exception as e:
        print(f"[{folder_name}] Warning: geometry normalization failed: {e}")

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
            metrics = {}

                # --- Compute SDI at specified depth layers (centerline column) ---
        try:
            grid_vel = profile_data['grid_vel']  # TH grid (Nz, Ny)
            grid_y = profile_data['grid_y']
            grid_z = profile_data['grid_z']
            y_min = profile_data['y_min']
            y_max = profile_data['y_max']
            y_center = 0.5 * (y_min + y_max)
            y_cols = grid_y[0, :]
            j_center = int(np.argmin(np.abs(y_cols - y_center)))
            th_col = grid_vel[:, j_center]
            z_col = grid_z[:, j_center]
            z_surface = float(np.nanmax(z_col))
            denom = sdi_sw_s - sdi_sw_h
            if denom <= 0:
                raise ValueError("Invalid SDI parameters: SWs must be greater than SWh.")
            # depth layers in meters (cm -> m)
            sdi_layer_stats = {}
            for (a,b) in SDI_LAYER_BOUNDS_CM:
                dmid = (a + b) / 2.0 / 100.0  # mid-depth in m
                z_target = z_surface - dmid
                idx_z = int(np.argmin(np.abs(z_col - z_target)))
                th_mid = float(th_col[idx_z])
                sdi_val = (sdi_sw_s - th_mid) / denom * 100.0
                key = f"sdi_{a}_{b}cm_percent"
                sdi_layer_stats[key] = float(sdi_val)
        except Exception as e:
            print(f"[{folder_name}] Warning: layered SDI failed on profile {idx_profile}: {e}")
            sdi_layer_stats = {f"sdi_{a}_{b}cm_percent": np.nan for (a,b) in SDI_LAYER_BOUNDS_CM}

        filtered_metrics = {k: v for k, v in metrics.items() if not (k.startswith('sdi_'))}
        row = {
            "folder": folder_name,
            "vegetation": folder_name,
            "timestamp": target_dt.isoformat(sep=" "),
            "profile_index": idx_profile,
            "x_profile_m": profile_data.get("x_profile"),
        }
        row.update(filtered_metrics)
        row.update(sdi_layer_stats)
        rows.append(row)

    if not rows:
        print(f"[{folder_name}] Metrics could not be computed for any profile.")
        return

    df = pd.DataFrame(rows)
    csv_path = _get_sdi_csv_path(output_dir, folder_name, target_dt, date_fmt)
    cache_params = _build_sdi_cache_params(folder_name, target_dt)
    save_dataframe_with_signature(df, csv_path, "compute_sdi_profiles", cache_params)
    print(f"[{folder_name}] Saved {len(df)} profile rows -> {csv_path}")
    return df


def _get_sdi_csv_path(output_dir: str, folder_name: str, target_dt: pd.Timestamp, date_fmt: str) -> str:
    date_token = target_dt.strftime(date_fmt)
    filename = f"{date_token}_{folder_name}_SDIProfiles.csv"
    return os.path.join(output_dir, filename)


def _build_sdi_cache_params(folder_name: str, target_dt: pd.Timestamp) -> Dict[str, Any]:
    heatmap_subset = {}
    if isinstance(TASK7_HEATMAP_CONFIG, dict):
        for key in ("x_profile_m_list", "grid_resolution_y", "grid_resolution_z", "x_tolerance_m"):
            if key in TASK7_HEATMAP_CONFIG:
                heatmap_subset[key] = TASK7_HEATMAP_CONFIG[key]
    mask_config = SEASONAL_MASK_CONFIG or {}
    return {
        "folder": folder_name,
        "target_date": target_dt.isoformat(),
        "heatmap": heatmap_subset,
        "mask": {
            "enabled": mask_config.get("enabled"),
            "threshold": mask_config.get("threshold"),
        },
        "layer_bounds_cm": SDI_LAYER_BOUNDS_CM,
    }


def load_cached_profiles(folder_name: str, target_dt: pd.Timestamp,
                         output_dir: str, date_fmt: str) -> Optional[pd.DataFrame]:
    csv_path = _get_sdi_csv_path(output_dir, folder_name, target_dt, date_fmt)
    cache_params = _build_sdi_cache_params(folder_name, target_dt)
    df = load_cached_dataframe(csv_path, "compute_sdi_profiles", cache_params)
    if df is not None:
        print(f"[{folder_name}] Loaded cached SDI profiles -> {csv_path}")
    return df


def _aggregate_sdi_layers_for_folder(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Collapse per-profile SDI columns into a single vertical profile by averaging each layer.
    """
    if df is None or df.empty:
        return None

    values = []
    for col in SDI_LAYER_COLUMNS:
        if col not in df.columns:
            return None
        col_series = pd.to_numeric(df[col], errors="coerce")
        values.append(col_series.mean(skipna=True))
    return np.array(values, dtype=float)


def _detect_artist_overlap(fig) -> bool:
    """
    Check if any artists within the figure overlap (based on bounding boxes).
    """
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:
        return False

    bboxes = []
    for ax in fig.axes:
        try:
            bbox = ax.get_tightbbox(renderer)
        except Exception:
            bbox = None
        if bbox is not None:
            bboxes.append(bbox)
    for legend in getattr(fig, "legends", []):
        try:
            bboxes.append(legend.get_window_extent(renderer))
        except Exception:
            continue
    for text in fig.texts:
        try:
            bboxes.append(text.get_window_extent(renderer))
        except Exception:
            continue

    for i, bbox_a in enumerate(bboxes):
        for bbox_b in bboxes[i + 1:]:
            try:
                if bbox_a.overlaps(bbox_b):
                    return True
            except Exception:
                continue
    return False


def _apply_layout_with_fallback(fig, rect, extra_bottom=0.0):
    """
    Apply tight_layout first; if overlaps remain, adjust subplot spacing automatically.
    """
    try:
        fig.tight_layout(rect=rect)
    except Exception as exc:
        print(f"Notice: tight_layout failed ({exc}). Applying fallback spacing.")
        left, bottom, right, top = rect
        fig.subplots_adjust(
            left=max(0.01, left + 0.02),
            right=min(0.99, right - 0.02),
            bottom=max(0.01, bottom + 0.02 + extra_bottom),
            top=min(0.99, top - 0.02)
        )
        return

    if _detect_artist_overlap(fig) or extra_bottom > 0:
        print("Notice: Detected overlapping plot elements. Adjusting layout automatically.")
        left, bottom, right, top = rect
        fig.subplots_adjust(
            left=max(0.01, left + 0.02),
            right=min(0.99, right - 0.02),
            bottom=max(0.01, bottom + 0.03 + extra_bottom),
            top=min(0.99, top - 0.02)
        )


def generate_dual_date_plot(date_to_folder_results, plot_dates, output_dir):
    """
    Generates a two-column SDI plot comparing two timestamps for the main vegetation types.
    """
    if len(plot_dates) < 2:
        print("Two dates are required to generate the SDI comparison plot. Skipping plotting.")
        return

    if len(plot_dates) > 2:
        print("Note: Only the first two dates will be used for the comparison plot.")
    selected_dates = plot_dates[:2]

    _apply_local_plot_style()

    plot_ready = {}
    all_series = []

    for dt in selected_dates:
        folder_results = date_to_folder_results.get(dt, {})
        per_folder = {}
        for folder in PLOT_VEGETATION_ORDER:
            df = folder_results.get(folder)
            series = _aggregate_sdi_layers_for_folder(df) if df is not None else None
            if series is not None:
                per_folder[folder] = series
                all_series.append(series)
        plot_ready[dt] = per_folder

    if not all_series:
        print("No SDI data available to plot. Skipping plotting.")
        return

    all_values = np.concatenate(all_series)
    finite_vals = all_values[np.isfinite(all_values)]
    if finite_vals.size == 0:
        finite_vals = np.array([0.0])

    ref_values = [abs(cfg.get("value", 0.0)) for cfg in SDI_REFERENCE_LINES.values()]
    max_ref = max(ref_values) if ref_values else 0.0
    x_half_range = max(np.nanmax(np.abs(finite_vals)), max_ref, 10.0)
    x_margin = max(5.0, x_half_range * 0.15)
    x_min, x_max = -x_half_range - x_margin, x_half_range + x_margin

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 12))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    legend_entries = {}
    depth_ticks = [tick for tick in SDI_LAYER_BOUNDARIES_CM if tick >= 20]
    if not depth_ticks:
        depth_ticks = SDI_LAYER_BOUNDARIES_CM[:]
    depth_min, depth_max = depth_ticks[0], depth_ticks[-1]
    depth_padding_cm = 20.0
    depth_display_max = depth_max + depth_padding_cm

    for ax, dt in zip(axes, selected_dates):
        per_folder = plot_ready.get(dt, {})
        plotted_any = False
        valid_indices = [idx for idx, depth in enumerate(SDI_LAYER_MIDPOINTS_CM) if depth >= 20]
        valid_depths = [SDI_LAYER_MIDPOINTS_CM[idx] for idx in valid_indices]
        tick_step = 50.0
        tick_start = tick_step * np.floor(x_min / tick_step)
        tick_end = tick_step * np.ceil(x_max / tick_step)
        x_min_display, x_max_display = tick_start, tick_end
        extra_x_space = max(tick_step, (x_max_display - x_min_display) * 0.1)
        x_axis_min = x_min_display - extra_x_space
        x_axis_max = x_max_display + extra_x_space

        for folder in PLOT_VEGETATION_ORDER:
            series = per_folder.get(folder)
            if series is None:
                continue
            series_to_plot = series if len(series) == len(valid_depths) else None
            if series_to_plot is None and series is not None:
                try:
                    series_to_plot = np.asarray(series)[valid_indices]
                except Exception:
                    series_to_plot = None
            if series_to_plot is None:
                continue
            style = VEGETATION_STYLE_MAP.get(folder, {})
            label = VEGETATION_DISPLAY_NAMES.get(folder, folder)
            line, = ax.plot(
                series_to_plot,
                valid_depths,
                color=style.get("color", "#333333"),
                marker=style.get("marker", "o"),
                linewidth=2.4,
                markersize=MARKER_SIZE,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=MARKER_EDGE_WIDTH,
                label=label
            )
            legend_entries[label] = line
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "No SDI data", transform=ax.transAxes, ha="center", va="center")

        for ref_cfg in SDI_REFERENCE_LINES.values():
            value = ref_cfg.get("value", 0.0)
            color = ref_cfg.get("color", "red")
            ax.axvline(value, color=color, linestyle="--", linewidth=1.2)
            label_left = ref_cfg.get("label_left")
            label_right = ref_cfg.get("label_right")
            label_font = max(10, TICK_FONT_SIZE - 1)
            text_y = depth_display_max - depth_padding_cm * 0.55
            text_y = max(depth_min, text_y)
            x_for_left = np.clip(-100.0, x_axis_min + 5.0, x_axis_max - 5.0)
            x_for_right = np.clip(100.0, x_axis_min + 5.0, x_axis_max - 5.0)
            if label_left:
                ax.text(x_for_left, text_y, label_left, color=color,
                        ha="center", va="top", fontsize=label_font, weight="bold",
                        transform=ax.transData, clip_on=True)
            if label_right:
                ax.text(x_for_right, text_y, label_right, color=color,
                        ha="center", va="top", fontsize=label_font, weight="bold",
                        transform=ax.transData, clip_on=True)

        ax.set_title(dt.strftime("%Y-%m-%d"), fontsize=TITLE_FONT_SIZE)
        ax.set_xlim(x_axis_min, x_axis_max)
        ax.set_ylim(depth_min, depth_display_max)
        ax.invert_yaxis()
        xtick_start = tick_step * np.floor(x_axis_min / tick_step)
        xtick_end = tick_step * np.ceil(x_axis_max / tick_step)
        xticks = np.arange(xtick_start, xtick_end + 0.01, tick_step)
        ax.set_xticks(xticks)
        ax.set_yticks(depth_ticks)
        ax.set_xlabel("Soil desiccation index (%)", fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    axes[0].set_ylabel("Soil depth (cm)", fontsize=LABEL_FONT_SIZE)

    legend_object = None
    if legend_entries:
        labels = list(legend_entries.keys())
        handles = [legend_entries[label] for label in labels]
        legend_ncol = 2 if len(handles) > 2 else len(handles)
        legend_object = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=legend_ncol,
            frameon=False,
            prop={"size": LEGEND_FONT_SIZE, "weight": "bold"},
            handlelength=2.6,
            columnspacing=1.6,
            handletextpad=0.6,
            borderaxespad=0.2
        )

    fig.suptitle("SDI Vertical Profiles ", fontsize=TITLE_FONT_SIZE + 2)
    extra_bottom = 0.08 if legend_object else 0.0
    _apply_layout_with_fallback(fig, rect=[0.05, 0.08, 0.95, 0.96], extra_bottom=extra_bottom)

    date_tokens = [dt.strftime("%Y%m%d") for dt in selected_dates]
    filename = f"SDI_TwoDateComparison_{date_tokens[0]}_{date_tokens[1]}.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"Two-date SDI plot saved -> {filepath}")


def main():
    args = parse_args()
    folders = resolve_folders(args)
    if not folders:
        print("No folders to process. Exiting.")
        return

    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = ensure_output_dir(project_dir)

    valid_dates = []
    for input_date in args.dates or []:
        dt = parse_target_datetime(input_date)
        if dt is None:
            print(f"Warning: Unable to parse date '{input_date}', skipping.")
            continue
        valid_dates.append(dt)

    if not valid_dates:
        print("Error: No valid dates were provided. Exiting.")
        sys.exit(1)

    date_to_folder_results = {}
    for target_dt in valid_dates:
        per_folder = {}
        for folder in folders:
            df = load_cached_profiles(folder, target_dt, output_dir, args.output_date_format)
            if df is None:
                df = compute_profiles_for_folder(project_dir, folder, target_dt, output_dir, args.output_date_format)
            if df is not None:
                per_folder[folder] = df
        date_to_folder_results[target_dt] = per_folder

    if len(valid_dates) >= 2:
        try:
            generate_dual_date_plot(
                date_to_folder_results,
                valid_dates[:2],
                output_dir
            )
        except Exception as exc:
            print(f"Warning: Failed to generate two-date SDI plot: {exc}")


if __name__ == "__main__":
    main()
