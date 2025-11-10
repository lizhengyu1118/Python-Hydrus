# -*- coding: utf-8 -*-
"""
Visualization Module for HYDRUS Analysis.

This module contains all functions related to plotting charts and heatmaps
based on the results from 'analysis_calculations.py'.

It is called by 'analysis_runner.py'.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

# --- Visualization-only vertical offset (do not change raw data) ---
Z_VISUAL_OFFSET = 1.24  # meters

import matplotlib.patches as patches
from scipy.interpolate import griddata

SEASON_ORDER_DEFAULT = ["winter", "spring", "summer", "autumn"]
SEASON_DISPLAY_NAMES = {
    "winter": "Winter (Dec-Feb)",
    "spring": "Spring (Mar-May)",
    "summer": "Summer (Jun-Aug)",
    "autumn": "Autumn (Sep-Nov)"
}
COLOR_LIMITS_FILENAME = "color_limits.json"
_COLOR_LIMITS_COMPUTED_THIS_SESSION = set()

def _get_color_limits_path(output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass
    return os.path.join(output_dir, COLOR_LIMITS_FILENAME)

def _load_color_limits(output_dir):
    path = _get_color_limits_path(output_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Warning: Failed to load color limits from {path}: {e}")
        return {}

def _save_color_limits(output_dir, color_limits):
    path = _get_color_limits_path(output_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(color_limits, f, indent=2)
        print(f"Saved color limits to {path}")
    except Exception as e:
        print(f"Warning: Failed to save color limits to {path}: {e}")

def _normalize_limits_tuple(value):
    if isinstance(value, (tuple, list)) and len(value) == 2:
        vmin, vmax = value
        if vmin is None or vmax is None:
            return None
        try:
            return (float(vmin), float(vmax))
        except (TypeError, ValueError):
            return None
    return None

# Import styling configuration
try:
    from plot_styles import (
        apply_y_limits, 
        REGION_COLORS, 
        REGION_LINEWIDTHS, 
        PLOT_Y_LIMITS,
        PLOT_FONT_SIZES,
        TASK5_HEATMAP_CONFIG,
        TASK6_HEATMAP_CONFIG,
        TASK7_HEATMAP_CONFIG
    )
    # Create a simple dict for styles to pass around
    PLOT_STYLES = {
        "REGION_COLORS": REGION_COLORS,
        "REGION_LINEWIDTHS": REGION_LINEWIDTHS,
        "PLOT_Y_LIMITS": PLOT_Y_LIMITS,
        "PLOT_FONT_SIZES": PLOT_FONT_SIZES,
        "TASK5_HEATMAP_CONFIG": TASK5_HEATMAP_CONFIG,
        "TASK6_HEATMAP_CONFIG": TASK6_HEATMAP_CONFIG,
        "TASK7_HEATMAP_CONFIG": TASK7_HEATMAP_CONFIG
    }
except ImportError:
    print("Warning: 'plot_styles.py' not found. Using default plot styles.")
    # Define dummy styles if import fails
    PLOT_STYLES = {
        "REGION_COLORS": ['#1f77b4'] * 5,
        "REGION_LINEWIDTHS": [1.5] * 5,
        "PLOT_Y_LIMITS": {},
        "PLOT_FONT_SIZES": {},
        "TASK5_HEATMAP_CONFIG": {},
        "TASK6_HEATMAP_CONFIG": {},
        "TASK7_HEATMAP_CONFIG": {}
    }

# --- I/O Helper ---

def log_save_message(action, filepath):
    """
    Logs a standardized save message to the console.
    """
    try:
        abs_path = os.path.abspath(filepath)
        print(f"{action}: {abs_path}")
    except Exception:
        print(f"{action}: {filepath}")

def save_plot_png(figure, output_dir, base_filename):
    """
    Saves a specific matplotlib figure to a PNG file.
    
    Args:
        figure (matplotlib.figure.Figure): The figure object to save.
    """
    try:
        if not base_filename.endswith(".png"):
            base_filename += ".png"
            
        filepath = os.path.join(output_dir, base_filename)
        
        # Apply tight_layout only to 2D figures (unless explicitly skipped)
        skip_tight = getattr(figure, "_skip_tight_layout", False) or base_filename.endswith("_SeasonalGrid.png")
        if not skip_tight:
            if not base_filename.endswith("_3D.png") and not base_filename.endswith("_Combined.png"):
                 figure.tight_layout()
             
        figure.savefig(filepath)
        plt.close(figure) # Explicitly close the figure object
        
        log_save_message("Plot saved", filepath)
        
    except Exception as e:
        print(f"Error: Failed to save plot to {base_filename}. Reason: {e}")

# --- Task 1 Plotting ---

def plot_task_1(dates, results_volumes, output_dir, base_filename_prefix):
    """
    Generates and saves the plot for Task 1.
    """
    print("Generating plot for Task 1...")
    # MODIFICATION: Add check for valid results
    if dates is None or results_volumes is None or len(results_volumes) == 0:
        print("Warning: Skipping Task 1 plot due to missing calculation data.")
        return
        
    try:
        fig = plt.figure(figsize=(12, 7)) # Get the figure object
        plt.plot(dates, results_volumes, linestyle='-')
        plt.xlabel('Date')
        plt.ylabel('Calculated Volume [m^3]') 
        save_plot_png(fig, output_dir, f"{base_filename_prefix}_plot")
    except Exception as e:
        print(f"An error occurred during plotting Task 1: {e}")

# --- Task 2 Plotting ---

def plot_task_2_charts(dates, results, output_dir, base_filename_prefix):
    """
    Generates and saves all plots for Task 2.
    """
    
    # Get common data
    # MODIFICATION: Use .get() for safety, returns None if key is missing
    delta_dates = results.get("delta_dates") 
    colors = PLOT_STYLES.get('REGION_COLORS', ['#1f77b4'] * 5)
    linewidths = PLOT_STYLES.get('REGION_LINEWIDTHS', [1.5] * 5)
    y_limits_config = PLOT_STYLES.get('PLOT_Y_LIMITS', {})

    # --- PLOT 1: Daily Total Water Storage (Combined) ---
    print("Generating plot for Task 2 (Daily Storage)...")
    # MODIFICATION: Use .get() to check for required data
    volume_timeseries_data = results.get("volume_timeseries")
    if volume_timeseries_data is not None and dates is not None:
        try:
            fig1 = plt.figure(figsize=(12, 7)) # Get the figure object
            ax = fig1.add_subplot(111)
            for i in range(5):
                ax.plot(
                    dates,
                    volume_timeseries_data[:, i],
                    linestyle='-',
                    color=colors[i],
                    linewidth=linewidths[i],
                    label=f'Region {i+1}'
                )
            ax.set_xlabel('Date')
            ax.set_ylabel('Total SWS [m^3]')
            ax.legend(loc='upper left')
            ax.grid(True, axis='y')
            
            # MODIFICATION: Call apply_y_limits
            # We pass [ax] (list with one item) and [limit] (list with one item)
            # This matches the function's expected input (list of axes, list of limits)
            task2_plot1_limits = y_limits_config.get('Task2_DailyStorage')
            apply_y_limits([ax], [task2_plot1_limits])
            # END MODIFICATION
            
            save_plot_png(fig1, output_dir, f"{base_filename_prefix}_daily_storage_plot")
        except Exception as e:
            print(f"An error occurred during plotting Task 2 (Plot 1): {e}")
    else:
        print("Warning: Skipping Task 2 (Plot 1) due to missing 'volume_timeseries' data (calculation may have failed).")


    # --- PLOT 2: Cumulative Positive Moisture Increase (Combined) ---
    print("Generating plot for Task 2 (Cumulative Increase)...")
    # MODIFICATION: Use .get() to check for required data
    cumulative_increase_data = results.get("cumulative_increase")
    if cumulative_increase_data is not None and delta_dates is not None:
        try:
            fig2 = plt.figure(figsize=(12, 7)) # Get the figure object
            ax = fig2.add_subplot(111)
            for i in range(5):
                ax.plot(
                    delta_dates, 
                    cumulative_increase_data[:, i], 
                    linestyle='-',
                    color=colors[i],
                    linewidth=linewidths[i],
                    label=f'Region {i+1}'
                )
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Increase [m^3]')
            ax.legend(loc='upper left')
            ax.grid(True, axis='y')
            
            # MODIFICATION: Call apply_y_limits
            # The config 'Task2_CumulativeIncrease' is now a single tuple (0, 160)
            task2_plot2_limits = y_limits_config.get('Task2_CumulativeIncrease')
            apply_y_limits([ax], [task2_plot2_limits])
            # END MODIFICATION
            
            save_plot_png(fig2, output_dir, f"{base_filename_prefix}_cumulative_increase_plot")
        except Exception as e:
            print(f"An error occurred during plotting Task 2 (Plot 2): {e}")
    else:
        print("Warning: Skipping Task 2 (Plot 2) due to missing 'cumulative_increase' data (calculation may have failed).")

    # --- PLOT 3: Daily Positive Moisture Increase (Rate) (Combined) ---
    print("Generating plot for Task 2 (Daily Rate)...")
    # MODIFICATION: Use .get() to check for required data
    positive_deltas_data = results.get("positive_deltas")
    if positive_deltas_data is not None and delta_dates is not None:
        try:
            fig3 = plt.figure(figsize=(12, 7)) # Get the figure object
            ax = fig3.add_subplot(111)
            for i in range(5):
                ax.plot(
                    delta_dates, 
                    positive_deltas_data[:, i], 
                    linestyle='-',
                    color=colors[i],
                    linewidth=linewidths[i],
                    label=f'Region {i+1}'
                )
            ax.set_xlabel('Date')
            ax.set_ylabel('Daily Increase Rate [m^3/day]')
            ax.legend(loc='upper left')
            ax.grid(True, axis='y')
            
            # MODIFICATION: Call apply_y_limits
            task2_plot3_limits = y_limits_config.get('Task2_DailyRate')
            apply_y_limits([ax], [task2_plot3_limits])
            # END MODIFICATION
            
            save_plot_png(fig3, output_dir, f"{base_filename_prefix}_daily_increase_rate_plot")
        except Exception as e:
            print(f"An error occurred during plotting Task 2 (Plot 3): {e}")
    else:
        print("Warning: Skipping Task 2 (Plot 3) due to missing 'positive_deltas' data (calculation may have failed).")

    # --- MODIFICATION: PLOT 4: Cumulative Increase Difference from Region 3 ---
    print("Generating plot for Task 2 (Cumulative Difference)...")
    
    # MODIFICATION: Use .get() for all keys to prevent any KeyError
    diff_data = results.get("cumulative_difference")
    diff_labels = results.get("column_names_diff")

    if diff_data is not None and diff_labels is not None and delta_dates is not None:
        try:
            fig4 = plt.figure(figsize=(12, 7)) # Get the figure object
            ax = fig4.add_subplot(111)
            
            # Define colors for the 4 difference lines (R1, R2, R4, R5)
            # We skip R3's color (index 2)
            diff_colors = [colors[0], colors[1], colors[3], colors[4]]
            diff_linewidths = [linewidths[0], linewidths[1], linewidths[3], linewidths[4]]

            for i in range(4): # Loop through the 4 difference series
                ax.plot(
                    delta_dates, 
                    diff_data[:, i], 
                    linestyle='-',
                    color=diff_colors[i],
                    linewidth=diff_linewidths[i],
                    label=diff_labels[i]
                )
            
            # Add Y=0 baseline
            ax.axhline(0, color='black', linestyle='--', linewidth=1.0)
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Difference in Cumulative Increase [m^3] (vs Region 3)')
            ax.legend(loc='upper left')
            ax.grid(True, axis='y')
            save_plot_png(fig4, output_dir, f"{base_filename_prefix}_cumulative_difference_plot")
        except Exception as e:
            # This will catch errors during the plotting itself
            print(f"An error occurred during plotting Task 2 (Plot 4 - Difference): {e}")
    else:
        # This warning will now be printed instead of the KeyError
        print("Warning: Skipping Task 2 (Plot 4 - Difference) due to missing 'cumulative_difference' or 'delta_dates' data (calculation may have failed).")
    # --- End of MODIFICATION ---

    # --- MODIFICATION: PLOT 5: Final Cumulative Increase (Bar Chart) ---
    print("Generating plot for Task 2 (Final Cumulative Bar Chart)...")
    # MODIFICATION: Get error bar data
    std_dev_data = results.get("std_dev_deltas")
    if cumulative_increase_data is not None:
        try:
            fig5 = plt.figure(figsize=(10, 7)) # Get the figure object
            ax = fig5.add_subplot(111)
            
            # Get the final value from the last time step
            final_values = cumulative_increase_data[-1, :]
            region_labels = [f'Region {i+1}' for i in range(5)]
            
            # --- MODIFICATION: Add yerr and capsize ---
            ax.bar(
                region_labels, 
                final_values, 
                color=colors
                # yerr=std_dev_data, # Add error bars
                # capsize=5          # Add caps to error bars
            )
            # --- End of MODIFICATION ---
            
            ax.set_ylabel('Delt_S [m^3]')
            ax.grid(True, axis='y')
            
            # --- MODIFICATION: Adjust Y-axis limits based on plot_styles ---
            
            # Get the manual setting from config
            manual_y_min = y_limits_config.get('Task2_FinalBar_YMin', None)
            manual_y_max = y_limits_config.get('Task2_FinalBar_YMax', None) # <-- MODIFICATION: Get YMax
            
            # Calculate max and padding (always needed)
            # MODIFICATION: Account for error bars in max/min calculation
            if std_dev_data is not None:
                max_val_data = np.max(final_values)
                min_val = np.min(final_values)
            else:
                max_val_data = np.max(final_values)
                min_val = np.min(final_values)
            # --- End of MODIFICATION ---
                
            data_range = max_val_data - min_val
            padding = data_range * 0.1
            
            # Handle case where all values are the same
            if data_range < 1e-9: 
                padding = np.abs(max_val_data * 0.1) # Use 10% of the value
                if padding < 1e-9: 
                    padding = 0.1 # Set a default padding
            
            # --- MODIFICATION: Y-Max logic ---
            if manual_y_max is not None:
                y_max_limit = manual_y_max
                print(f"Applying manual Y-max to bar chart: {y_max_limit}")
            else:
                y_max_limit = max_val_data + padding
                print(f"Applying automatic Y-max to bar chart: {y_max_limit:.2f}")
            # --- End MODIFICATION ---
            
            # Check if user wants manual or automatic y_min
            if manual_y_min is not None:
                # User provided a value (e.g., 0)
                y_min_limit = manual_y_min
                print(f"Applying manual Y-min to bar chart: {y_min_limit}")
            else:
                # User left it as None, use auto-zoom logic
                # MODIFICATION: Make auto-zoom less aggressive, ensure it's below min_val
                y_min_limit_auto = min_val - padding
                # Ensure the lower limit doesn't go above 0 if all data is positive
                y_min_limit = min(0, y_min_limit_auto) 
                print(f"Applying automatic Y-min to bar chart: {y_min_limit:.2f}")

            ax.set_ylim(y_min_limit, y_max_limit)
            # --- End of MODIFICATION ---
            
            save_plot_png(fig5, output_dir, f"{base_filename_prefix}_final_cumulative_bar_plot")
        except Exception as e:
            print(f"An error occurred during plotting Task 2 (Plot 5 - Bar Chart): {e}")
    else:
        print("Warning: Skipping Task 2 (Plot 5 - Bar Chart) due to missing 'cumulative_increase' data.")
    # --- End of MODIFICATION ---


# --- Task 4 Plotting ---

def plot_task_4_charts(dates, results, output_dir, base_filename_prefix):
    """
    Generates and saves all plots for Task 4.
    """
    
    # Get common data
    colors = PLOT_STYLES.get('REGION_COLORS', ['#1f77b4'] * 5)
    linewidths = PLOT_STYLES.get('REGION_LINEWIDTHS', [1.5] * 5)
    y_limits_config = PLOT_STYLES.get('PLOT_Y_LIMITS', {})

    # --- PLOT 1: Daily Y-Direction Flux (Rate) (Subplots) ---
    print("Generating plot for Task 4 (Daily Y-Flux Rate)...")
    # MODIFICATION: Use .get() to check for required data
    y_flux_timeseries_data = results.get("y_flux_timeseries")
    if y_flux_timeseries_data is not None and dates is not None:
        try:
            fig1, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True) # Get the figure object
            
            for i in range(5):
                ax = axes[i]
                ax.plot(
                    dates,
                    y_flux_timeseries_data[:, i],
                    linestyle='-',
                    color=colors[i],
                    linewidth=linewidths[i]
                )
                ax.set_ylabel(f'Region {i+1}\nY-Flux Rate')
                ax.grid(True, axis='x') 
                if i < 4:
                    plt.setp(ax.get_xticklabels(), visible=False)

            axes[-1].set_xlabel('Date')
            
            task4_plot1_limits = y_limits_config.get('Task4_DailyRate')
            apply_y_limits(axes, task4_plot1_limits)
            
            save_plot_png(fig1, output_dir, f"{base_filename_prefix}_daily_y_flux_rate_plot")
        except Exception as e:
            print(f"An error occurred during plotting Task 4 (Plot 1): {e}")
    else:
        print("Warning: Skipping Task 4 (Plot 1) due to missing 'y_flux_timeseries' data (calculation may have failed).")


    # --- PLOT 2: Cumulative Y-Direction Flux (Combined) ---
    print("Generating plot for Task 4 (Cumulative Y-Flux Combined)...")
    # MODIFICATION: Use .get() to check for required data
    cumulative_y_flux_data = results.get("cumulative_y_flux")
    if cumulative_y_flux_data is not None and dates is not None:
        try:
            fig2 = plt.figure(figsize=(12, 7)) # Get the figure object
            ax = fig2.add_subplot(111)
            
            for i in range(5):
                ax.plot(
                    dates,
                    cumulative_y_flux_data[:, i], 
                    linestyle='-',
                    color=colors[i],
                    linewidth=linewidths[i],
                    label=f'Region {i+1}'
                )
            
            ax.set_ylabel('Cumulative Y-Flux')
            ax.grid(True, axis='y')
            ax.set_xlabel('Date')
            ax.legend(loc='upper left')
            
            # Apply Y-limits if defined for the *combined* plot
            task4_plot2_limits = y_limits_config.get('Task4_CumulativeFlux')
            if task4_plot2_limits and isinstance(task4_plot2_limits, (tuple, list)) and len(task4_plot2_limits) == 2:
                 ax.set_ylim(task4_plot2_limits)
            
            save_plot_png(fig2, output_dir, f"{base_filename_prefix}_cumulative_y_flux_plot")
        except Exception as e:
            print(f"An error occurred during plotting Task 4 (Plot 2): {e}")
    else:
        print("Warning: Skipping Task 4 (Plot 2) due to missing 'cumulative_y_flux' data (calculation may have failed).")


# --- MODIFICATION: NEW Task 8 Plotting ---
def plot_task_8_combined_bar(global_results_cache, folders_to_plot, plot_order_map, output_dir, base_filename_prefix):
    """
    Generates and saves the combined 2x2 bar plot for Task 8.
    """
    print("Generating combined plot for Task 8...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        colors = PLOT_STYLES.get('REGION_COLORS', ['#1f77b4'] * 5)
        y_limits_config = PLOT_STYLES.get('PLOT_Y_LIMITS', {})
        font_config = PLOT_STYLES.get('PLOT_FONT_SIZES', {})
        subplot_label_size = font_config.get('task8_subplot_label', plt.rcParams.get('axes.labelsize', 12))
        
        # --- MODIFICATION: Add subplot labels ---
        label_map = {
            "exoticshrub": "(a)",
            "exoticgrass": "(b)",
            "arablecrop":  "(c)",
            "naturalgrass": "(d)"
        }
        # --- End MODIFICATION ---
        
        for folder_name in folders_to_plot:
            # Get the results for this folder
            results = global_results_cache.get(folder_name)
            if results is None:
                print(f"Warning: No data for {folder_name} in cache. Skipping subplot.")
                continue
            
            # Get the position for this plot
            pos = plot_order_map.get(folder_name)
            if pos is None:
                print(f"Warning: No position defined for {folder_name}. Skipping subplot.")
                continue
                
            ax = axes[pos[0], pos[1]]
            
            # --- MODIFICATION: Add subplot label text ---
            label = label_map.get(folder_name)
            if label:
                ax.text(0.05, 0.95, label, transform=ax.transAxes, 
                        fontsize=subplot_label_size, fontweight='bold', va='top', ha='left')
            # --- End MODIFICATION ---
            
            # --- This is the plotting logic copied from plot_task_2_charts (PLOT 5) ---
            cumulative_increase_data = results.get("cumulative_increase")
            std_dev_data = results.get("std_dev_deltas")
            
            if cumulative_increase_data is None:
                print(f"Warning: No 'cumulative_increase' data for {folder_name}. Skipping subplot.")
                ax.set_title(f"{folder_name} (No Data)")
                continue

            # Get the final value from the last time step
            final_values = cumulative_increase_data[-1, :]
            region_labels = [f'Region {i+1}' for i in range(5)]
            
            ax.bar(
                region_labels, 
                final_values, 
                color=colors
                # yerr=std_dev_data, # Add error bars
                # capsize=5          # Add caps to error bars
            )
            
            ax.set_ylabel('Delt_S [m^3]')
            ax.grid(True, axis='y')
            #ax.set_title(folder_name)
            
            # --- Apply Y-axis limits logic (copied) ---
            manual_y_min = y_limits_config.get('Task2_FinalBar_YMin', None)
            manual_y_max = y_limits_config.get('Task2_FinalBar_YMax', None)
            
            if std_dev_data is not None:
                max_val_data = np.max(final_values)
                min_val = np.min(final_values)
            else:
                max_val_data = np.max(final_values)
                min_val = np.min(final_values)
                
            data_range = max_val_data - min_val
            padding = data_range * 0.1
            if data_range < 1e-9: 
                padding = np.abs(max_val_data * 0.1)
                if padding < 1e-9: padding = 0.1
            
            if manual_y_max is not None:
                y_max_limit = manual_y_max
            else:
                y_max_limit = max_val_data + padding
            
            if manual_y_min is not None:
                y_min_limit = manual_y_min
            else:
                y_min_limit_auto = min_val - padding
                y_min_limit = min(0, y_min_limit_auto) 
            
            ax.set_ylim(y_min_limit, y_max_limit)
            # --- End of copied logic ---
            
        save_plot_png(fig, output_dir, f"{base_filename_prefix}_plot")

    except Exception as e:
        print(f"An error occurred during plotting Task 8: {e}")
        # Ensure figure is closed on error
        if 'fig' in locals():
            plt.close(fig)


# --- Task 5, 6, 7 Generic Heatmap Plotter ---

# MODIFICATION: This is the new helper function to calculate data
def _calculate_heatmap_data(velocity_data, velocity_name, config_key, mesh, manual_limits=None):
    """
    Pre-calculates all interpolation data for heatmap plots.
    """
    print(f"\n--- Calculating Heatmap Data ({velocity_name}) ---")
    
    if velocity_data is None:
        print(f"Error: No average {velocity_name} data provided. Skipping calculation.")
        return [], None, None, None, {}
        
    # 1. Load configuration
    config = dict(PLOT_STYLES.get(config_key, {}))
    x_profiles = config.get('x_profile_m_list', [15.0, 30.0, 45.0])
    x_tol = config.get('x_tolerance_m', 0.5)
    grid_res_y = config.get('grid_resolution_y', 100)
    grid_res_z = config.get('grid_resolution_z', 100)
    cmap = config.get('cmap', 'coolwarm')
    
    if not x_profiles:
         print(f"Error: 'x_profile_m_list' in {config_key} is empty. Skipping plot.")
         return [], None, None, None, {}
         
    print(f"Config: {len(x_profiles)} Y-Z profile(s) (Tolerance: {x_tol}m)")
    
    # 2. Get node coordinates and bounds
    mesh_points = mesh.points
    node_x_coords = mesh_points[:, 0]
    node_y_coords = mesh_points[:, 1]
    node_z_coords = mesh_points[:, 2]
    
    bounds = {
        'x': [np.min(node_x_coords), np.max(node_x_coords)],
        'y': [np.min(node_y_coords), np.max(node_y_coords)],
        'z': [np.min(node_z_coords), np.max(node_z_coords)]
    }
    
    # 3. Find global min/max for color scaling
    all_profile_data = []
    for x_profile in x_profiles:
        profile_mask = (node_x_coords >= (x_profile - x_tol)) & (node_x_coords <= (x_profile + x_tol))
        if np.sum(profile_mask) > 0:
            all_profile_data.append(velocity_data[profile_mask])
            
    if not all_profile_data:
         print(f"Error: No nodes found for any X-profile slice. Skipping plot.")
         return [], None, None, None, {}
    
    # 4. Create colormap normalizer
    if manual_limits is None:
        config_manual = config.get('manual_color_limits')
        if isinstance(config_manual, (tuple, list)) and len(config_manual) == 2:
            manual_limits = config_manual
    if isinstance(manual_limits, (tuple, list)) and len(manual_limits) == 2:
        v_min, v_max = manual_limits
        if v_min is None or v_max is None:
            manual_limits = None
    else:
        manual_limits = None

    if manual_limits:
        norm = plt.Normalize(manual_limits[0], manual_limits[1])
    elif 'TH' in velocity_name: # Handle TH (sequential)
        v_min = np.nanmin(np.concatenate(all_profile_data))
        v_max = np.nanmax(np.concatenate(all_profile_data))
        if v_min == v_max: v_max += 0.1 # Avoid error
        norm = plt.Normalize(v_min, v_max)
    else: # Handle Vy, Vz (diverging)
        v_abs_max = np.nanmax(np.abs(np.concatenate(all_profile_data)))
        if v_abs_max == 0: v_abs_max = 1.0 # Avoid error
        norm = plt.Normalize(-v_abs_max, v_abs_max)
    
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])

    # 5. Loop, interpolate, and store plot data
    plot_data_list = []
    for i, x_profile in enumerate(x_profiles):
        print(f"Interpolating profile at X = {x_profile}m...")
        
        profile_mask = (node_x_coords >= (x_profile - x_tol)) & (node_x_coords <= (x_profile + x_tol))
        num_nodes_in_slice = np.sum(profile_mask)
        
        if num_nodes_in_slice < 4:
            print(f"Warning: Not enough nodes ({num_nodes_in_slice}) found in slice (X={x_profile} +/- {x_tol}). Skipping subplot.")
            continue
            
        profile_y_coords = node_y_coords[profile_mask]
        profile_z_coords = node_z_coords[profile_mask]
        profile_avg_vel = velocity_data[profile_mask]
        
        y_min, y_max = np.min(profile_y_coords), np.max(profile_y_coords)
        z_min, z_max = np.min(profile_z_coords), np.max(profile_z_coords)
        
        grid_y, grid_z = np.meshgrid(
            np.linspace(y_min, y_max, grid_res_y),
            np.linspace(z_min, z_max, grid_res_z)
        )
        
        points = np.column_stack((profile_y_coords, profile_z_coords))
        values = profile_avg_vel
        
        grid_vel = griddata(points, values, (grid_y, grid_z), method='linear')
        
        nan_mask = np.isnan(grid_vel)
        if np.any(nan_mask):
            grid_vel_nearest = griddata(points, values, (grid_y[nan_mask], grid_z[nan_mask]), method='nearest')
            grid_vel[nan_mask] = grid_vel_nearest

        if np.all(np.isnan(grid_vel)):
             print(f"Warning: Interpolation failed for X-profile slice (X={x_profile}). Skipping.")
             continue
        
        plot_data_list.append({
            'x_profile': x_profile,
            'grid_y': grid_y,
            'grid_z': grid_z,
            'grid_vel': grid_vel,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max
        })

    return plot_data_list, norm, mappable, bounds, config

# MODIFICATION: New helper to plot 2D part
def _plot_heatmap_2d(fig, axes, plot_data_list, mappable, velocity_name, config, add_cbar=True):
    """
    Plots the 2D heatmap data onto a given set of axes.
    """
    # MODIFICATION: Simplified logic. Assume 'axes' is a 1D iterable.
    axes_list = list(axes) # Ensure it's a list

    for i, data in enumerate(plot_data_list):
        if i >= len(axes_list): break # Stop if we run out of axes
        
        ax_2d_current = axes_list[i] # Use the flattened list
        
        c = ax_2d_current.imshow(
            data['grid_vel'], 
            extent=[data['y_min'], data['y_max'], data['z_min'], data['z_max']],
            origin='lower', 
            cmap=mappable.cmap, 
            aspect='auto',
            norm=mappable.norm
        )
        # ax_2d_current.set_xlabel('Y-Coordinate (Width) [m]') # MODIFICATION: Removed
        # ax_2d_current.set_ylabel('Z-Coordinate (Vertical) [m]') # MODIFICATION: Removed

# MODIFICATION: New helper to plot 3D part
def _plot_heatmap_3d(fig, ax, plot_data_list, mappable, bounds, config, velocity_name, add_colorbar=True):
    """
    Plots the 3D heatmap data onto a given 3D axis.
    """
    # 1. Plot 3D Surfaces
    for data in plot_data_list:
        X_surf = np.full_like(data['grid_y'], data['x_profile'])
        colors = mappable.to_rgba(data['grid_vel'])
        
        ax.plot_surface(
            X_surf, 
            data['grid_y'], 
            data['grid_z'], 
            facecolors=colors,
            shade=False,
            rstride=5,
            cstride=5
        )

    # 2. Draw 3D Wireframe
    xb, yb, zb = bounds['x'], bounds['y'], bounds['z']
    box_color = 'grey'
    box_style = '--'
    # Bottom
    ax.plot([xb[0], xb[1]], [yb[0], yb[0]], [zb[0], zb[0]], c=box_color, linestyle=box_style)
    ax.plot([xb[0], xb[1]], [yb[1], yb[1]], [zb[0], zb[0]], c=box_color, linestyle=box_style)
    ax.plot([xb[0], xb[0]], [yb[0], yb[1]], [zb[0], zb[0]], c=box_color, linestyle=box_style)
    ax.plot([xb[1], xb[1]], [yb[0], yb[1]], [zb[0], zb[0]], c=box_color, linestyle=box_style)
    # Top
    ax.plot([xb[0], xb[1]], [yb[0], yb[0]], [zb[1], zb[1]], c=box_color, linestyle=box_style)
    ax.plot([xb[0], xb[1]], [yb[1], yb[1]], [zb[1], zb[1]], c=box_color, linestyle=box_style)
    ax.plot([xb[0], xb[0]], [yb[0], yb[1]], [zb[1], zb[1]], c=box_color, linestyle=box_style)
    ax.plot([xb[1], xb[1]], [yb[0], yb[1]], [zb[1], zb[1]], c=box_color, linestyle=box_style)
    # Verticals
    ax.plot([xb[0], xb[0]], [yb[0], yb[0]], [zb[0], zb[1]], c=box_color, linestyle=box_style)
    ax.plot([xb[1], xb[1]], [yb[0], yb[0]], [zb[0], zb[1]], c=box_color, linestyle=box_style)
    ax.plot([xb[0], xb[0]], [yb[1], yb[1]], [zb[0], zb[1]], c=box_color, linestyle=box_style)
    ax.plot([xb[1], xb[1]], [yb[1], yb[1]], [zb[0], zb[1]], c=box_color, linestyle=box_style)

    # 3. Set 3D labels and view
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.view_init(
        elev=config.get('view_elevation', 25), 
        azim=config.get('view_azimuth', -75)
    ) 
    if add_colorbar:
        fig.colorbar(
            mappable,
            ax=ax,
            shrink=config.get('colorbar_shrink', 0.85),
            aspect=config.get('colorbar_aspect', 25),
            fraction=config.get('colorbar_fraction', 0.04),
            pad=config.get('colorbar_pad', 0.08),
            label=f'Average {velocity_name}'
        )

# MODIFICATION: New function to create the combined plot
def _generate_combined_heatmap_plot(
    plot_data_list, mappable, bounds, config, velocity_name,
    output_dir, base_filename_prefix
):
    """
    Generates the combined plot with three 2D slices and one 3D view
    arranged in a 2x2 layout.
    """
    print("Generating combined 2x2 heatmap plot...")
    try:
        num_profiles = len(plot_data_list)
        if num_profiles == 0:
            print("Warning: No plot data to generate combined plot.")
            return
        if num_profiles > 3:
            print("Warning: More than three profiles supplied; only the first three will be shown in the 2×2 layout.")

        combined_figsize = config.get('combined_figsize') or (16, 10)
        fig_comb = plt.figure(figsize=combined_figsize)

        width_ratios = config.get('combined_width_ratios') or config.get('width_ratios') or [1, 1]
        height_ratios = config.get('combined_height_ratios') or config.get('height_ratios') or [1, 1]
        hspace = config.get('combined_hspace', 0.25)
        wspace = config.get('combined_wspace', 0.3)

        gs = fig_comb.add_gridspec(
            2, 2,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            hspace=hspace,
            wspace=wspace
        )

        # Create axes for the 2D slices (top-left, top-right, bottom-left)
        axes_positions = [(0, 0), (0, 1), (1, 0)]
        axes_2d_comb = [
            fig_comb.add_subplot(gs[r, c])
            for r, c in axes_positions[:num_profiles]
        ]

        # Create axis for the 3D plot (bottom-right)
        ax_3d_comb = fig_comb.add_subplot(gs[1, 1], projection='3d')

        # Resolve subplot label styling from configuration
        show_labels = config.get('show_subplot_labels', False)
        subplot_labels = config.get('subplot_labels', [])
        label_fontsize = config.get('subplot_label_fontsize', plt.rcParams.get('axes.labelsize', 12))
        label_box_props = config.get('subplot_label_box')
        bbox_props = dict(label_box_props) if isinstance(label_box_props, dict) else None
        offset_2d = config.get('subplot_label_offset_2d', (0.03, 0.95))
        align_2d = config.get('subplot_label_alignment_2d', {})
        ha_2d = align_2d.get('ha', 'left')
        va_2d = align_2d.get('va', 'top')
        align_3d = config.get('subplot_label_alignment_3d', {})
        ha_3d = align_3d.get('ha', 'center')
        va_3d = align_3d.get('va', 'bottom')
        z_offset_ratio = config.get('subplot_label_z_offset_ratio', 0.05)
        z_bounds = bounds.get('z') if isinstance(bounds, dict) else None
        z_span = None
        if z_bounds and isinstance(z_bounds, (tuple, list)) and len(z_bounds) == 2:
            z_span = z_bounds[1] - z_bounds[0]
        z_offset = (z_span * z_offset_ratio) if z_span not in (None, 0) else 0.0
        label_limit = min(len(axes_2d_comb), len(subplot_labels))
        # Plot 2D slices (no additional colorbars; use shared one later)
        _plot_heatmap_2d(
            fig_comb, axes_2d_comb,
            plot_data_list[:len(axes_2d_comb)],
            mappable, velocity_name, config,
            add_cbar=False
        )

        # Annotate 2D subplots if enabled
        if show_labels and label_limit > 0:
            if not (isinstance(offset_2d, (tuple, list)) and len(offset_2d) == 2):
                offset_2d = (0.03, 0.95)
            x_off, y_off = offset_2d
            for idx in range(label_limit):
                ax = axes_2d_comb[idx]
                label = subplot_labels[idx]
                ax.text(
                    x_off, y_off, label,
                    transform=ax.transAxes,
                    fontsize=label_fontsize,
                    fontweight='bold',
                    ha=ha_2d,
                    va=va_2d,
                    bbox=dict(bbox_props) if bbox_props else None
                )

        # Plot 3D heatmap
        _plot_heatmap_3d(
            fig_comb, ax_3d_comb,
            plot_data_list, mappable, bounds,
            config, velocity_name, add_colorbar=False
        )

        # Annotate corresponding 3D surfaces if enabled
        if show_labels and label_limit > 0:
            for idx in range(label_limit):
                data = plot_data_list[idx]
                label = subplot_labels[idx]

                x_pos = data.get('x_profile', 0.0)

                y_min = data.get('y_min')
                y_max = data.get('y_max')
                if y_min is not None and y_max is not None:
                    y_pos = 0.5 * (y_min + y_max)
                else:
                    grid_y = data.get('grid_y')
                    y_pos = float(np.nanmean(grid_y)) if grid_y is not None else 0.0

                z_base = data.get('z_max')
                if z_base is None:
                    grid_z = data.get('grid_z')
                    z_base = float(np.nanmax(grid_z)) if grid_z is not None else 0.0
                z_pos = z_base + z_offset

                ax_3d_comb.text(
                    x_pos, y_pos, z_pos,
                    label,
                    fontsize=label_fontsize,
                    fontweight='bold',
                    ha=ha_3d,
                    va=va_3d,
                    bbox=dict(bbox_props) if bbox_props else None
                )

        # Shared axes labels (same as before)
        if axes_2d_comb:
            axes_2d_comb[-1].set_xlabel('Y (Width) [m]')
        fig_comb.text(
            0.05, 0.5, 'Z (Vertical) [m]',
            va='center', rotation='vertical',
            fontweight='bold',
            fontsize=plt.rcParams.get('axes.labelsize', 15)
        )

        # Shared colorbar (placed to the right of the entire grid)
        cbar_left = 0.92
        cbar_bottom = 0.12
        cbar_width = 0.015
        cbar_height = 0.76
        cax = fig_comb.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
        fig_comb.colorbar(mappable, cax=cax, label=f'Average {velocity_name}')

        save_plot_png(fig_comb, output_dir, f"{base_filename_prefix}_plot_Combined")

    except Exception as e:
        print(f"Error generating combined heatmap: {e}")
        if 'fig_comb' in locals():
            plt.close(fig_comb)


# --- Seasonal grid helper ---
def _plot_seasonal_heatmap_grid(
    seasonal_plot_data,
    season_order,
    num_profiles,
    velocity_name,
    mappable,
    output_dir,
    base_filename_prefix,
    suffix="SeasonalGrid"
):
    if not seasonal_plot_data:
        print("No seasonal data available for seasonal grid plot.")
        return

    num_seasons = len(season_order)
    if num_profiles == 0 or num_seasons == 0:
        print("Seasonal grid skipped due to empty profile/season configuration.")
        return

    fig, axes = plt.subplots(
        num_profiles,
        num_seasons,
        figsize=(4.5 * num_seasons, 3.6 * num_profiles),
        squeeze=False
    )
    fig._skip_tight_layout = True
    fig.subplots_adjust(left=0.07, right=0.88, top=0.93, bottom=0.08, wspace=0.25, hspace=0.25)

    for col, season_key in enumerate(season_order):
        season_label = SEASON_DISPLAY_NAMES.get(season_key, season_key.title())
        axes[0, col].set_title(season_label)
        season_data_list = seasonal_plot_data.get(season_key)

        for row in range(num_profiles):
            ax = axes[row, col]
            if season_data_list and row < len(season_data_list):
                data = season_data_list[row]
                ax.imshow(
                    data['grid_vel'],
                    extent=[data['y_min'], data['y_max'], data['z_min'], data['z_max']],
                    origin='lower',
                    cmap=mappable.cmap,
                    aspect='auto',
                    norm=mappable.norm
                )
                if col == 0:
                    ax.set_ylabel('Z (Vertical) [m]')
                if row == num_profiles - 1:
                    ax.set_xlabel('Y (Width) [m]')
            else:
                ax.axis('off')

    scalar_mappable = plt.cm.ScalarMappable(cmap=mappable.cmap, norm=mappable.norm)
    scalar_mappable.set_array([])
    cbar_ax = fig.add_axes([0.9, 0.12, 0.02, 0.76])
    fig.colorbar(
        scalar_mappable,
        cax=cbar_ax,
        label=f'Average {velocity_name}'
    )

    save_plot_png(fig, output_dir, f"{base_filename_prefix}_{suffix}")


# --- Visualization helper: shift Z for 2D/3D heatmaps (only for display) ---
def _shift_z_for_visual(plot_data_list, bounds, z_offset):
    """Shift Z coordinates in plot data (extent and grids) by z_offset (meters)."""
    if not z_offset:
        return plot_data_list, bounds
    try:
        for d in plot_data_list:
            if 'z_min' in d: d['z_min'] += z_offset
            if 'z_max' in d: d['z_max'] += z_offset
            if d.get('grid_z') is not None:
                d['grid_z'] = d['grid_z'] + z_offset
        if isinstance(bounds, dict) and bounds.get('z') is not None:
            zb = bounds['z']
            if isinstance(zb, (list, tuple)) and len(zb) == 2:
                bounds['z'] = (zb[0] + z_offset, zb[1] + z_offset)
    except Exception as e:
        print(f"Warning: failed to apply z-offset {z_offset}: {e}")
    return plot_data_list, bounds


# --- Task-specific wrappers for the generic plotter ---

def plot_task_5_heatmaps(results, mesh, output_dir, base_filename_prefix):
    """
    Task 5 specific wrapper.
    Generates 2D, 3D, and Combined plots for average Vy.
    """
    velocity_name = "Vy [m/day]"
    config_key = "TASK5_HEATMAP_CONFIG"
    if results is None:
        print("Task 5: No calculation results provided. Skipping plots.")
        return

    avg_vy = results.get("overall")
    if avg_vy is None:
        print("Task 5: Missing overall average data. Skipping plots.")
        return
    seasonal_avgs = results.get("seasonal", {})
    season_order = results.get("season_order", SEASON_ORDER_DEFAULT)
    
    # 1. Calculate data
    color_limit_store = _load_color_limits(output_dir)
    task_key = "Task5"
    base_config = PLOT_STYLES.get(config_key, {})
    manual_override = _normalize_limits_tuple(base_config.get('manual_color_limits'))
    saved_entry = None
    if manual_override is None:
        if task_key not in _COLOR_LIMITS_COMPUTED_THIS_SESSION:
            saved_entry = color_limit_store.get(task_key)
        if isinstance(saved_entry, dict):
            manual_override = _normalize_limits_tuple((saved_entry.get("vmin"), saved_entry.get("vmax")))

    (plot_data_list, norm, mappable, bounds, config) = _calculate_heatmap_data(
        avg_vy, velocity_name, config_key, mesh, manual_limits=manual_override
    )
    plot_data_list, bounds = _shift_z_for_visual(plot_data_list, bounds, Z_VISUAL_OFFSET)
    if not plot_data_list: 
        print("Task 5: No data calculated. Skipping all plots.")
        return # Exit if calculation failed

    used_limits = (mappable.norm.vmin, mappable.norm.vmax)
    if manual_override is None:
        saved_entry = color_limit_store.get(task_key)
        if (not saved_entry or task_key in _COLOR_LIMITS_COMPUTED_THIS_SESSION) and None not in used_limits:
            color_limit_store[task_key] = {
                "vmin": float(used_limits[0]),
                "vmax": float(used_limits[1])
            }
            _save_color_limits(output_dir, color_limit_store)
            _COLOR_LIMITS_COMPUTED_THIS_SESSION.add(task_key)

    num_profiles = len(plot_data_list)
    axes_list_flat = [] # To store flattened axes

    # 2. Generate and save individual 2D plot
    try:
        fig_2d, axes_2d = plt.subplots(num_profiles, 1, figsize=(10.5, 8 * num_profiles), squeeze=False)
        axes_list_flat = axes_2d.flatten()
        # MODIFICATION: Call with add_cbar=True
        _plot_heatmap_2d(fig_2d, axes_list_flat, plot_data_list, mappable, velocity_name, config, add_cbar=False)
        scalar_map = plt.cm.ScalarMappable(cmap=mappable.cmap, norm=mappable.norm)
        scalar_map.set_array([])
        fig_2d.colorbar(
            scalar_map,
            ax=axes_list_flat,
            orientation='vertical',
            fraction=0.025,
            pad=0.02,
            label=f'Average {velocity_name}'
        )
        
        # MODIFICATION: Add shared labels for the individual 2D plot
        axes_list_flat[-1].set_xlabel('Y (Width) [m]')
        fig_2d.text(
            0.04, 0.5, 'Z (Vertical) [m]', 
            va='center', 
            rotation='vertical',
            fontweight='bold',
            fontsize=plt.rcParams.get('axes.labelsize', 15)
        )
        
        save_plot_png(fig_2d, output_dir, f"{base_filename_prefix}_plot_2D") # Saves and closes
    except Exception as e:
        print(f"Error generating Task 5 2D plot: {e}")

    # 3. Generate and save individual 3D plot
    try:
        fig_3d = plt.figure(figsize=config.get('figsize', (18, 12)))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        _plot_heatmap_3d(fig_3d, ax_3d, plot_data_list, mappable, bounds, config, velocity_name)
        save_plot_png(fig_3d, output_dir, f"{base_filename_prefix}_plot_3D") # Saves and closes
    except Exception as e:
        print(f"Error generating Task 5 3D plot: {e}")

    # 4. Generate and save combined plot
    _generate_combined_heatmap_plot(
        plot_data_list, mappable, bounds, config, velocity_name,
        output_dir, base_filename_prefix
    )

    # 5. Generate seasonal grid (new visualization)
    if seasonal_avgs:
        seasonal_plot_data = {}
        for season_key, season_avg in seasonal_avgs.items():
            if season_avg is None:
                continue
            season_plot_data, _, _, _, _ = _calculate_heatmap_data(
                season_avg, velocity_name, config_key, mesh,
                manual_limits=manual_override or used_limits
            )
            season_plot_data, _ = _shift_z_for_visual(
                season_plot_data,
                dict(bounds) if isinstance(bounds, dict) else bounds,
                Z_VISUAL_OFFSET
            )
            if season_plot_data:
                seasonal_plot_data[season_key] = season_plot_data

        if seasonal_plot_data:
            _plot_seasonal_heatmap_grid(
                seasonal_plot_data,
                season_order,
                num_profiles,
                velocity_name,
                mappable,
                output_dir,
                base_filename_prefix,
                suffix="SeasonalGrid"
            )

def plot_task_6_heatmaps(results, mesh, output_dir, base_filename_prefix):
    """
    Task 6 specific wrapper.
    Generates 2D, 3D, and Combined plots for average Vz.
    """
    velocity_name = "Vz [m/day]"
    config_key = "TASK6_HEATMAP_CONFIG"
    if results is None:
        print("Task 6: No calculation results provided. Skipping plots.")
        return

    avg_vz = results.get("overall")
    if avg_vz is None:
        print("Task 6: Missing overall average data. Skipping plots.")
        return
    seasonal_avgs = results.get("seasonal", {})
    season_order = results.get("season_order", SEASON_ORDER_DEFAULT)

    # 1. Calculate data
    color_limit_store = _load_color_limits(output_dir)
    task_key = "Task6"
    base_config = PLOT_STYLES.get(config_key, {})
    manual_override = _normalize_limits_tuple(base_config.get('manual_color_limits'))
    saved_entry = None
    if manual_override is None:
        if task_key not in _COLOR_LIMITS_COMPUTED_THIS_SESSION:
            saved_entry = color_limit_store.get(task_key)
        if isinstance(saved_entry, dict):
            manual_override = _normalize_limits_tuple((saved_entry.get("vmin"), saved_entry.get("vmax")))

    (plot_data_list, norm, mappable, bounds, config) = _calculate_heatmap_data(
        avg_vz, velocity_name, config_key, mesh, manual_limits=manual_override
    )
    plot_data_list, bounds = _shift_z_for_visual(plot_data_list, bounds, Z_VISUAL_OFFSET)
    if not plot_data_list: 
        print("Task 6: No data calculated. Skipping all plots.")
        return # Exit if calculation failed

    used_limits = (mappable.norm.vmin, mappable.norm.vmax)
    if manual_override is None:
        saved_entry = color_limit_store.get(task_key)
        if (not saved_entry or task_key in _COLOR_LIMITS_COMPUTED_THIS_SESSION) and None not in used_limits:
            color_limit_store[task_key] = {
                "vmin": float(used_limits[0]),
                "vmax": float(used_limits[1])
            }
            _save_color_limits(output_dir, color_limit_store)
            _COLOR_LIMITS_COMPUTED_THIS_SESSION.add(task_key)

    num_profiles = len(plot_data_list)
    axes_list_flat = []

    # 2. Generate and save individual 2D plot
    try:
        fig_2d, axes_2d = plt.subplots(num_profiles, 1, figsize=(10.5, 8 * num_profiles), squeeze=False)
        axes_list_flat = axes_2d.flatten()
        _plot_heatmap_2d(fig_2d, axes_list_flat, plot_data_list, mappable, velocity_name, config, add_cbar=False)
        scalar_map = plt.cm.ScalarMappable(cmap=mappable.cmap, norm=mappable.norm)
        scalar_map.set_array([])
        fig_2d.colorbar(
            scalar_map,
            ax=axes_list_flat,
            orientation='vertical',
            fraction=0.025,
            pad=0.02,
            label=f'Average {velocity_name}'
        )
        
        # MODIFICATION: Add shared labels for the individual 2D plot
        axes_list_flat[-1].set_xlabel('Y (Width) [m]')
        fig_2d.text(
            0.04, 0.5, 'Z (Vertical) [m]', 
            va='center', 
            rotation='vertical',
            fontweight='bold',
            fontsize=plt.rcParams.get('axes.labelsize', 15)
        )
        
        save_plot_png(fig_2d, output_dir, f"{base_filename_prefix}_plot_2D") # Saves and closes
    except Exception as e:
        print(f"Error generating Task 6 2D plot: {e}")

    # 3. Generate and save individual 3D plot
    try:
        fig_3d = plt.figure(figsize=config.get('figsize', (18, 12)))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        _plot_heatmap_3d(fig_3d, ax_3d, plot_data_list, mappable, bounds, config, velocity_name)
        save_plot_png(fig_3d, output_dir, f"{base_filename_prefix}_plot_3D") # Saves and closes
    except Exception as e:
        print(f"Error generating Task 6 3D plot: {e}")

    # 4. Generate and save combined plot
    _generate_combined_heatmap_plot(
        plot_data_list, mappable, bounds, config, velocity_name,
        output_dir, base_filename_prefix
    )

    if seasonal_avgs:
        seasonal_plot_data = {}
        for season_key, season_avg in seasonal_avgs.items():
            if season_avg is None:
                continue
            season_plot_data, _, _, _, _ = _calculate_heatmap_data(
                season_avg, velocity_name, config_key, mesh,
                manual_limits=manual_override or used_limits
            )
            season_plot_data, _ = _shift_z_for_visual(
                season_plot_data,
                dict(bounds) if isinstance(bounds, dict) else bounds,
                Z_VISUAL_OFFSET
            )
            if season_plot_data:
                seasonal_plot_data[season_key] = season_plot_data

        if seasonal_plot_data:
            _plot_seasonal_heatmap_grid(
                seasonal_plot_data,
                season_order,
                num_profiles,
                velocity_name,
                mappable,
                output_dir,
                base_filename_prefix,
                suffix="SeasonalGrid"
            )

def plot_task_7_heatmaps(results, mesh, output_dir, base_filename_prefix):
    """
    Task 7 specific wrapper.
    Generates 2D, 3D, and Combined plots for average TH.
    """
    velocity_name = "TH" # TH is unitless
    config_key = "TASK7_HEATMAP_CONFIG"
    if results is None:
        print("Task 7: No calculation results provided. Skipping plots.")
        return

    avg_th = results.get("overall")
    if avg_th is None:
        print("Task 7: Missing overall average data. Skipping plots.")
        return
    seasonal_avgs = results.get("seasonal", {})
    season_order = results.get("season_order", SEASON_ORDER_DEFAULT)
    
    # 1. Calculate data
    color_limit_store = _load_color_limits(output_dir)
    task_key = "Task7"
    base_config = PLOT_STYLES.get(config_key, {})
    manual_override = _normalize_limits_tuple(base_config.get('manual_color_limits'))
    saved_entry = None
    if manual_override is None:
        if task_key not in _COLOR_LIMITS_COMPUTED_THIS_SESSION:
            saved_entry = color_limit_store.get(task_key)
        if isinstance(saved_entry, dict):
            manual_override = _normalize_limits_tuple((saved_entry.get("vmin"), saved_entry.get("vmax")))

    (plot_data_list, norm, mappable, bounds, config) = _calculate_heatmap_data(
        avg_th, velocity_name, config_key, mesh, manual_limits=manual_override
    )
    plot_data_list, bounds = _shift_z_for_visual(plot_data_list, bounds, Z_VISUAL_OFFSET)
    if not plot_data_list: 
        print("Task 7: No data calculated. Skipping all plots.")
        return # Exit if calculation failed

    used_limits = (mappable.norm.vmin, mappable.norm.vmax)
    if manual_override is None:
        saved_entry = color_limit_store.get(task_key)
        if (not saved_entry or task_key in _COLOR_LIMITS_COMPUTED_THIS_SESSION) and None not in used_limits:
            color_limit_store[task_key] = {
                "vmin": float(used_limits[0]),
                "vmax": float(used_limits[1])
            }
            _save_color_limits(output_dir, color_limit_store)
            _COLOR_LIMITS_COMPUTED_THIS_SESSION.add(task_key)

    num_profiles = len(plot_data_list)
    axes_list_flat = []

    # 2. Generate and save individual 2D plot
    try:
        fig_2d, axes_2d = plt.subplots(num_profiles, 1, figsize=(10.5, 8 * num_profiles), squeeze=False)
        axes_list_flat = axes_2d.flatten()
        _plot_heatmap_2d(fig_2d, axes_list_flat, plot_data_list, mappable, velocity_name, config, add_cbar=False)
        scalar_map = plt.cm.ScalarMappable(cmap=mappable.cmap, norm=mappable.norm)
        scalar_map.set_array([])
        fig_2d.colorbar(
            scalar_map,
            ax=axes_list_flat,
            orientation='vertical',
            fraction=0.025,
            pad=0.02,
            label=f'Average {velocity_name}'
        )
        
        # MODIFICATION: Add shared labels for the individual 2D plot
        axes_list_flat[-1].set_xlabel('Y (Width) [m]')
        fig_2d.text(
            0.04, 0.5, 'Z (Vertical) [m]', 
            va='center', 
            rotation='vertical',
            fontweight='bold',
            fontsize=plt.rcParams.get('axes.labelsize', 15)
        )
        
        save_plot_png(fig_2d, output_dir, f"{base_filename_prefix}_plot_2D") # Saves and closes
    except Exception as e:
        print(f"Error generating Task 7 2D plot: {e}")

    # 3. Generate and save individual 3D plot
    try:
        fig_3d = plt.figure(figsize=config.get('figsize', (18, 12)))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        _plot_heatmap_3d(fig_3d, ax_3d, plot_data_list, mappable, bounds, config, velocity_name)
        save_plot_png(fig_3d, output_dir, f"{base_filename_prefix}_plot_3D") # Saves and closes
    except Exception as e:
        print(f"Error generating Task 7 3D plot: {e}")

    # 4. Generate and save combined plot
    _generate_combined_heatmap_plot(
        plot_data_list, mappable, bounds, config, velocity_name,
        output_dir, base_filename_prefix
    )

    if seasonal_avgs:
        seasonal_plot_data = {}
        for season_key, season_avg in seasonal_avgs.items():
            if season_avg is None:
                continue
            season_plot_data, _, _, _, _ = _calculate_heatmap_data(
                season_avg, velocity_name, config_key, mesh,
                manual_limits=manual_override or used_limits
            )
            season_plot_data, _ = _shift_z_for_visual(
                season_plot_data,
                dict(bounds) if isinstance(bounds, dict) else bounds,
                Z_VISUAL_OFFSET
            )
            if season_plot_data:
                seasonal_plot_data[season_key] = season_plot_data

        if seasonal_plot_data:
            _plot_seasonal_heatmap_grid(
                seasonal_plot_data,
                season_order,
                len(plot_data_list),
                velocity_name,
                mappable,
                output_dir,
                base_filename_prefix,
                suffix="SeasonalGrid"
            )

