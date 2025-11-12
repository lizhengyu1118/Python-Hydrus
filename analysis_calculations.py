# -*- coding: utf-8 -*-
"""
Calculation Module for HYDRUS Analysis.

This module contains all functions for:
1.  Preparing static geometric data (e.g., element volumes, regions).
2.  Running time-step calculations in parallel using multiprocessing.
3.  Post-processing results (e.g., calculating cumulative sums).
4.  Saving numerical results to CSV or TXT files.

It is called by 'analysis_runner.py'.
"""

import os
import numpy as np
import pandas as pd
import multiprocessing
from datetime import datetime
from collections import OrderedDict

# --- Constants ---
# Seasons order for seasonal averaging (Task 5-7)
SEASON_KEYS = ["winter", "spring", "summer", "autumn"]
MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn"
}

def _calculate_seasonal_node_means(time_series_data, dates):
    """
    Calculates node-wise seasonal averages given a full time-series array.

    Args:
        time_series_data (np.ndarray): Shape (n_timesteps, n_nodes)
        dates (list-like): Sequence of date strings/datetime objects aligned with timesteps

    Returns:
        OrderedDict: season_key -> seasonal average array (n_nodes,)
    """
    seasonal_avgs = OrderedDict()
    if dates is None or len(dates) == 0:
        return seasonal_avgs

    if len(dates) != time_series_data.shape[0]:
        print("Warning: Unable to compute seasonal averages (date count mismatch).")
        return seasonal_avgs

    datetime_index = pd.to_datetime(dates, errors='coerce')
    season_indices = {key: [] for key in SEASON_KEYS}

    for idx, dt in enumerate(datetime_index):
        if pd.isna(dt):
            continue
        season_key = MONTH_TO_SEASON.get(dt.month)
        if season_key:
            season_indices[season_key].append(idx)

    for season_key in SEASON_KEYS:
        indices = season_indices.get(season_key, [])
        if indices:
            seasonal_avgs[season_key] = np.mean(time_series_data[indices, :], axis=0)

    return seasonal_avgs
# (None defined, tasks are self-contained)

# --- I/O Helper Functions ---

def log_save_message(action, filepath):
# ... existing code ...
    """
    Logs a standardized save message to the console.
    """
    try:
        abs_path = os.path.abspath(filepath)
        print(f"{action}: {abs_path}")
    except Exception:
        print(f"{action}: {filepath}")

def save_results_csv(output_dir, base_filename, dates, results_data, column_name, is_multi_region=False):
# ... existing code ...
    """
    Saves the analysis results to a CSV file.
    """
    try:
        if not base_filename.endswith(".csv"):
            base_filename += ".csv"
            
        filepath = os.path.join(output_dir, base_filename)
        
        if is_multi_region:
            df = pd.DataFrame(results_data, index=dates, columns=column_name)
        else:
            df = pd.DataFrame(results_data, index=dates, columns=[column_name])
            
        df.index.name = "Date"
        df.to_csv(filepath)
        log_save_message("Data saved", filepath)
        
    except Exception as e:
        print(f"Error: Failed to save CSV to {base_filename}. Reason: {e}")

def save_task_3_txt(output_filepath, times, dates, results_positive_deltas):
# ... existing code ...
    """
    Saves the results of Task 3 to a HYDRUS-compatible TXT file.
    """
    try:
        with open(output_filepath, 'w') as f:
            for i, date in enumerate(dates):
                time_val = times[i]
                f.write(f"   Time =   {time_val}\n")
                data_slice = results_positive_deltas[i]
                for value in data_slice:
                    f.write(f"  {value:.8E}\n")
        
        print(f"\n--- Task 3 Complete ---")
        log_save_message("Successfully saved", output_filepath)
        print("To view this, re-run 'main_viewer.py' and select this folder.")

    except Exception as e:
        print(f"An error occurred during file writing: {e}")


# --- Task 1: Low Moisture Volume ---

def _setup_task_1_static_data(mesh):
# ... existing code ...
    """
    Pre-calculates static data for Task 1 (on flattened mesh).
    """
    print("Pre-calculating static data for Task 1 (on flattened mesh)...")
    
    points = mesh.points
    z_max = np.max(points[:, 2])
    print(f"Flattened top surface detected at Z={z_max:.2f}m.")
    
    interface_z_level = z_max - 1.0
    print(f"Interface level defined at Z={interface_z_level:.2f}m.")
    
    print("Calculating element centroids...")
    element_centroids = mesh.cell_centers().points
    element_z_coords = element_centroids[:, 2]
    
    elements_below_interface_mask = (element_z_coords < interface_z_level)
    print(f"Found {np.sum(elements_below_interface_mask)} elements below the 1.0m interface.")
    
    print("Calculating element volumes...")
    element_volumes = mesh.compute_cell_sizes().cell_data['Volume']
    
    print("Mapping elements to nodes...")
    try:
        node_indices_for_cells = mesh.cells.reshape(-1, 5)[:, 1:5]
    except AttributeError:
        print("Error: Mesh object has no 'cells' attribute. Mesh parsing may have failed.")
        return None
    
    return node_indices_for_cells, elements_below_interface_mask, element_volumes

def _task_1_calculate_volume_for_timestep(th_data_slice, node_indices_for_cells, elements_below_interface_mask, element_volumes):
# ... existing code ...
    """
    Parallel worker function for Task 1.
    """
    try:
        # 1. Get TH value for each node
        node_th = th_data_slice
        
        # 2. Calculate average TH for each element (avg of its 4 nodes)
        element_th = np.mean(node_th[node_indices_for_cells], axis=1)
        
        # 3. Find elements that meet *both* criteria:
        #    a) Centroid is below the 1m interface (pre-calculated)
        #    b) Average TH is less than 0.12
        is_low_moisture = (element_th < 0.12)
        target_elements_mask = elements_below_interface_mask & is_low_moisture
        
        # 4. Get the volumes of these target elements
        target_volumes = element_volumes[target_elements_mask]
        
        # 5. Sum the volumes
        total_volume = np.sum(target_volumes)
        return total_volume
        
    except Exception as e:
        print(f"Error in Task 1 worker: {e}")
        return 0.0 # Return 0.0 on failure

def run_task_1_calculations(mesh, th_data, dates):
# ... existing code ...
    """
    Main orchestrator for Task 1 calculations.
    """
    task1_statics = _setup_task_1_static_data(mesh)
    if task1_statics is None:
        return np.zeros(len(dates)) # Return empty results if setup failed

    node_indices, interface_mask, volumes = task1_statics

    tasks = []
    for i in range(len(dates)):
        tasks.append((
            th_data[i, :],
            node_indices,
            interface_mask,
            volumes
        ))

    print(f"\nStarting parallel processing of {len(tasks)} time steps for Task 1...")
    start_time = datetime.now()
    
    results_volumes = []
    try:
        with multiprocessing.Pool() as pool:
            results_volumes = pool.starmap(
                _task_1_calculate_volume_for_timestep, 
                tasks
            )
        end_time = datetime.now()
        print(f"Processing complete. Time taken: {end_time - start_time}")
    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")
        return np.zeros(len(dates))

    return np.array(results_volumes)


# --- Task 2: Regional Moisture Storage ---

def _setup_task_2_static_data(mesh):
# ... existing code ...
    """
    Pre-calculates static data for Task 2 / Task 4.
    """
    print("Pre-calculating static data for Task 2 / Task 4...")
    
    points = mesh.points
    y_max = np.max(points[:, 1])
    y_min = np.min(points[:, 1])
    print(f"Mesh Y-axis bounds detected: Max={y_max:.2f}m, Min={y_min:.2f}m")
    
    print("Partitioning Y-axis into 5 equal segments.")
    region_boundaries = np.linspace(y_max, y_min, 6)
    
    print("Calculating element centroids...")
    element_centroids = mesh.cell_centers().points
    element_y_coords = element_centroids[:, 1]
    
    print("Calculating element volumes...")
    element_volumes = mesh.compute_cell_sizes().cell_data['Volume']
    
    print("Mapping elements to nodes...")
    try:
        node_indices_for_cells = mesh.cells.reshape(-1, 5)[:, 1:5]
    except AttributeError:
        print("Error: Mesh object has no 'cells' attribute. Mesh parsing may have failed.")
        return None
    
    bins = np.flip(region_boundaries) 
    bin_indices = np.digitize(element_y_coords, bins)
    remap = np.array([0, 5, 4, 3, 2, 1, 0])
    element_region_map = remap[bin_indices]

    print("Element region mapping complete:")
    for i in range(1, 6):
        count = np.sum(element_region_map == i)
        print(f"  Region {i} (Y: {region_boundaries[i]:.2f}m to {region_boundaries[i-1]:.2f}m): {count} elements")

    return node_indices_for_cells, element_volumes, element_region_map

def _task_2_calculate_water_volume_for_timestep(th_data_slice, node_indices_for_cells, element_volumes, element_region_map):
# ... existing code ...
    """
    Parallel worker function for Task 2.
    """
    try:
        # 1. Get TH value for each node
        node_th = th_data_slice
        
        # 2. Calculate average TH for each element
        element_th = np.mean(node_th[node_indices_for_cells], axis=1)
        
        # 3. Calculate water volume for *each* element
        element_water_volume = element_volumes * element_th
        
        # 4. Sum water volumes by region
        regional_sums = np.zeros(5)
        for i in range(1, 6):
            region_mask = (element_region_map == i)
            regional_sums[i-1] = np.sum(element_water_volume[region_mask])
            
        return regional_sums # Return array [sum_R1, sum_R2, ..., sum_R5]
        
    except Exception as e:
        print(f"Error in Task 2 worker: {e}")
        return np.zeros(5) # Return array of 0.0 on failure

def run_task_2_calculations(mesh, th_data, dates):
    """
    Main orchestrator for Task 2 calculations.
    """
    task2_statics = _setup_task_2_static_data(mesh)
    if task2_statics is None:
        return {} # Return empty results if setup failed

    node_indices, volumes, region_map = task2_statics

    tasks = []
    for i in range(len(dates)):
        tasks.append((
            th_data[i, :],
            node_indices,
            volumes,
            region_map
        ))
        
    print(f"\nStarting parallel processing of {len(tasks)} time steps for Task 2...")
    start_time = datetime.now()

    results_water_volumes = []
    try:
        with multiprocessing.Pool() as pool:
            results_water_volumes = pool.starmap(
                _task_2_calculate_water_volume_for_timestep, 
                tasks
            )
        end_time = datetime.now()
        print(f"Processing complete. Time taken: {end_time - start_time}")
    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")
        return {}
        
    print("Post-processing results for Task 2...")
    try:
        volume_timeseries = np.array(results_water_volumes)
        deltas = np.diff(volume_timeseries, axis=0)
        positive_deltas = np.maximum(0, deltas)
        cumulative_increase = np.cumsum(positive_deltas, axis=0)
        delta_dates = dates[1:]
        region_cols_m3 = [f"Region_{i+1}_m3" for i in range(5)]
        
        # --- MODIFICATION: Calculate Standard Deviation for Error Bars ---
        # Calculate the standard deviation of the *daily positive increase*
        std_dev_deltas = np.std(positive_deltas, axis=0)
        # --- End of MODIFICATION ---
        
        # --- MODIFICATION: Calculate difference from Region 3 ---
        print("Calculating cumulative difference from Region 3...")
        # Get baseline (Region 3 is index 2)
        baseline_r3 = cumulative_increase[:, 2]
        
        # Calculate differences (R1-R3, R2-R3, R4-R3, R5-R3)
        diff_r1_r3 = cumulative_increase[:, 0] - baseline_r3
        diff_r2_r3 = cumulative_increase[:, 1] - baseline_r3
        diff_r4_r3 = cumulative_increase[:, 3] - baseline_r3
        diff_r5_r3 = cumulative_increase[:, 4] - baseline_r3
        
        # Stack them into a new (N_times, 4) array
        cumulative_difference = np.stack(
            [diff_r1_r3, diff_r2_r3, diff_r4_r3, diff_r5_r3], 
            axis=1
        )
        column_names_diff = [
            "Region 1 - Region 3",
            "Region 2 - Region 3",
            "Region 4 - Region 3",
            "Region 5 - Region 3"
        ]
        # --- End of MODIFICATION ---

        return {
            "volume_timeseries": volume_timeseries,
            "positive_deltas": positive_deltas,
            "cumulative_increase": cumulative_increase,
            "delta_dates": delta_dates,
            "column_names": region_cols_m3,
            "cumulative_difference": cumulative_difference, # MODIFICATION: Add to results
            "column_names_diff": column_names_diff,        # MODIFICATION: Add to results
            "std_dev_deltas": std_dev_deltas # MODIFICATION: Add new std_dev data
        }

    except Exception as e:
        print(f"An error occurred during post-processing: {e}")
        return {}


# --- Task 3: Positive TH Delta ---

def _task_3_calculate_positive_th_delta_for_nodes(th_data_slice, th_data_previous_slice):
# ... existing code ...
    """
    Parallel worker function for Task 3.
    """
    try:
        delta_th = th_data_slice - th_data_previous_slice
        positive_delta_th = np.maximum(0, delta_th)
        return positive_delta_th
    except Exception as e:
        print(f"Error in Task 3 worker: {e}")
        # Return zeros matching node count of input
        return np.zeros_like(th_data_slice)

def run_task_3_calculations(th_data):
# ... existing code ...
    """
    Main orchestrator for Task 3 calculations.
    """
    tasks = []
    tasks.append((th_data[0, :], th_data[0, :])) # First time step has 0 delta
    for i in range(1, len(th_data)):
        tasks.append((th_data[i, :], th_data[i-1, :]))
        
    print(f"\nStarting parallel processing of {len(tasks)} time steps for Task 3...")
    start_time = datetime.now()

    results_positive_deltas = []
    try:
        with multiprocessing.Pool() as pool:
            results_positive_deltas = pool.starmap(
                _task_3_calculate_positive_th_delta_for_nodes, 
                tasks
            )
        end_time = datetime.now()
        print(f"Processing complete. Time taken: {end_time - start_time}")
    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")
        return None

    return results_positive_deltas


# --- Task 4: Regional Y-Flux ---

def _task_4_calculate_y_flux_for_timestep(th_data_slice, vy_data_slice, node_indices_for_cells, element_volumes, element_region_map):
# ... existing code ...
    """
    Parallel worker function for Task 4.
    """
    try:
        # 1. Get TH and Vy for each node
        node_th = th_data_slice
        node_vy = vy_data_slice
        
        # 2. Calculate average TH and Vy for each element
        element_th = np.mean(node_th[node_indices_for_cells], axis=1)
        element_vy = np.mean(node_vy[node_indices_for_cells], axis=1)
        
        # 3. Calculate Y-Flux (Water Volume * Vy) for *each* element
        # (element_volumes * element_th) = Water Volume in the element
        # This is (m^3 * m/day) = m^4/day, which represents a flux volume
        element_y_flux_volume = (element_volumes * element_th) * element_vy
        
        # 4. Sum flux volumes by region
        regional_sums = np.zeros(5)
        for i in range(1, 6):
            region_mask = (element_region_map == i)
            regional_sums[i-1] = np.sum(element_y_flux_volume[region_mask])
            
        return regional_sums # Return array [sum_R1, sum_R2, ..., sum_R5]
        
    except Exception as e:
        print(f"Error in Task 4 worker: {e}")
        return np.zeros(5)

def run_task_4_calculations(mesh, th_data, vy_data, dates):
# ... existing code ...
    """
    Main orchestrator for Task 4 calculations.
    """
    # Reuse Task 2's geometric setup
    task4_statics = _setup_task_2_static_data(mesh)
    if task4_statics is None:
        return {} # Return empty results if setup failed

    node_indices, volumes, region_map = task4_statics
    
    tasks = []
    for i in range(len(dates)):
        tasks.append((
            th_data[i, :],
            vy_data[i, :],
            node_indices,
            volumes,
            region_map
        ))
        
    print(f"\nStarting parallel processing of {len(tasks)} time steps for Task 4...")
    start_time = datetime.now()

    results_y_flux = []
    try:
        with multiprocessing.Pool() as pool:
            results_y_flux = pool.starmap(
                _task_4_calculate_y_flux_for_timestep, 
                tasks
            )
        end_time = datetime.now()
        print(f"Processing complete. Time taken: {end_time - start_time}")
    except Exception as e:
        print(f"An error occurred during parallel processing: {e}")
        return {}
        
    print("Post-processing results for Task 4...")
    try:
        y_flux_timeseries = np.array(results_y_flux)
        cumulative_y_flux = np.cumsum(y_flux_timeseries, axis=0)
        
        # NOTE: Unit conversion for 'exoticgrass' is now done in analysis_runner.py
        # *before* this function is called. No conversion is needed here.
        
        region_cols_flux = [f"Region_{i+1}_Y_Flux_Volume" for i in range(5)]
        
        return {
            "y_flux_timeseries": y_flux_timeseries,
            "cumulative_y_flux": cumulative_y_flux,
            "column_names": region_cols_flux
        }
    except Exception as e:
        print(f"An error occurred during post-processing: {e}")
        return {}


# --- Task 5: Vy Heatmap ---

def run_task_5_calculations(vy_data, dates=None):
    """
    Main orchestrator for Task 5 calculations (Vy).
    """
    print("Calculating time-average Vy for all nodes...")
    try:
        avg_vy_all_nodes = np.mean(vy_data, axis=0)
        seasonal_avgs = _calculate_seasonal_node_means(np.asarray(vy_data), dates)
        return {
            "overall": avg_vy_all_nodes,
            "seasonal": seasonal_avgs,
            "season_order": SEASON_KEYS
        }
    except Exception as e:
        print(f"Error calculating time-average Vy: {e}")
        return None

# --- Task 6: Vz Heatmap ---

def run_task_6_calculations(vz_data, dates=None):
    """
    Main orchestrator for Task 6 calculations (Vz).
    """
    print("Calculating time-average Vz for all nodes...")
    try:
        avg_vz_all_nodes = np.mean(vz_data, axis=0)
        seasonal_avgs = _calculate_seasonal_node_means(np.asarray(vz_data), dates)
        return {
            "overall": avg_vz_all_nodes,
            "seasonal": seasonal_avgs,
            "season_order": SEASON_KEYS
        }
    except Exception as e:
        print(f"Error calculating time-average Vz: {e}")
        return None

# --- Task 7: TH Heatmap (NEW) ---

def run_task_7_calculations(th_data, dates=None):
    """
    Main orchestrator for Task 7 calculations (TH).
    """
    print("Calculating time-average TH for all nodes...")
    try:
        avg_th_all_nodes = np.mean(th_data, axis=0)
        seasonal_avgs = _calculate_seasonal_node_means(np.asarray(th_data), dates)
        return {
            "overall": avg_th_all_nodes,
            "seasonal": seasonal_avgs,
            "season_order": SEASON_KEYS
        }
    except Exception as e:
        print(f"Error calculating time-average TH: {e}")
        return None

