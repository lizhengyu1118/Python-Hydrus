# -*- coding: utf-8 -*-
"""
Main script (Orchestrator) to run modular data analysis tasks.

This script is responsible for the overall workflow:
1.  Finds all available simulation folders.
2.  Prompts the user to select which folder(s) to analyze.
3.  Prompts the user to select which task(s) to run.
4.  Loops through each selected folder.
5.  Inside each folder, it loads all data (Mesh, TH, V, etc.) using HydrusModel.
6.  Applies unit conversions (cm->m) and coordinate transformations (slope flattening).
7.  Loops through each selected task.
8.  For each task, it calls the corresponding functions from:
    - analysis_calculations.py (to get numerical results)
    - analysis_plotting.py (to generate plots)
9.  Handles email notifications on success or failure.
"""

import os
import sys
import numpy as np
import multiprocessing
from datetime import datetime

# --- Module Imports ---

# 1. Data Loader
try:
    from hydrus_parser import HydrusModel
except ImportError:
    print("Error: Could not find 'hydrus_parser.py'.")
    print("Please ensure 'hydrus_parser.py' is in the same directory.")
    sys.exit(1)

# 2. Calculation Engine
try:
    import analysis_calculations as calc
except ImportError:
    print("Error: Could not find 'analysis_calculations.py'.")
    print("Please ensure 'analysis_calculations.py' is in the same directory.")
    sys.exit(1)

# 3. Plotting Engine
try:
    import analysis_plotting as plot
except ImportError:
    print("Error: Could not find 'analysis_plotting.py'.")
    print("Please ensure 'analysis_plotting.py' is in the same directory.")
    sys.exit(1)

# 4. Styling & Configuration
try:
    from plot_styles import set_scientific_style
except ImportError:
    print("Warning: 'plot_styles.py' not found.")
    print("Plots will use default matplotlib styling.")
    def set_scientific_style(grid_alpha=0.5):
        print("Using default plot style (plot_styles.py not found).")
        pass

# 5. Email Notifier
try:
    import notify_on_exit
except ImportError:
    print("Warning: 'notify_on_exit.py' not found.")
    print("Email notification on completion will be disabled.")
    notify_on_exit = None

# --- Email Configuration ---
# (This section is hard-coded as requested)

SMTP_CONFIG = {
    "server": "smtp.gmail.com",
    "port": 465,
    "sender_email": "lizhengyu1118@gmail.com",
    "sender_password": "zcup lvjb ayre pxpd" 
}
RECIPIENT_EMAIL = "lizhengyu1118@gmail.com"

# --- Helper Functions ---

def find_simulation_folders(root_dir):
    """
    Finds all subfolders in root_dir that contain a MESHTRIA.TXT file.
    """
    folders = []
    for entry in os.scandir(root_dir):
        if entry.is_dir():
            mesh_file_found = False
            try:
                for file in os.listdir(entry.path):
                    if file.lower() == "meshtria.txt":
                        mesh_file_found = True
                        break
            except OSError:
                continue
            if mesh_file_found:
                 folders.append(entry.name)
    return sorted(folders)

def prompt_for_folder(folders):
    """
    Displays a menu and asks the user to select one or more folders.
    """
    print("\nFound the following simulation folders:")
    for i, folder in enumerate(folders, 1):
        print(f"  [{i}] {folder}")

    while True:
        try:
            choice_str = input(f"\nWhich simulation(s) do you want to load?\n(Enter numbers separated by commas, e.g., '1,2', or 'all'): ")
            choice_str = choice_str.strip().lower()

            if choice_str == 'all':
                return folders # Return all found folders
            
            selected_folders = []
            invalid_folders = []
            parts = choice_str.split(',')
            
            if not parts:
                raise ValueError("Input is empty.")

            for part in parts:
                part = part.strip()
                if not part: continue
                
                choice_index = int(part) - 1
                if 0 <= choice_index < len(folders):
                    folder_name = folders[choice_index]
                    if folder_name not in selected_folders:
                        selected_folders.append(folder_name)
                else:
                    invalid_folders.append(part)
            
            if invalid_folders:
                print(f"Error: Invalid folder number(s) provided: {', '.join(invalid_folders)}. Please try again.")
            elif not selected_folders:
                print("Error: No valid folders selected. Please try again.")
            else:
                return sorted(selected_folders) # Return the list of selected folder names

        except ValueError:
            print("Invalid input. Please enter numbers separated by commas (e.g., '1,2') or 'all'.")
            
def prompt_for_task():
    """
    Displays a menu and asks the user to select one or more analysis tasks.
    """
    print("\nSelect the analysis task(s) to run:")
    print("  [1] Task 1: Low Moisture Volume (TH < 0.12) Below 1m Interface (Plot)")
    print("  [2] Task 2: Cumulative Moisture Increase by Y-Axis Region (Plot)")
    print("  [3] Task 3: Generate Positive TH Delta File (Node-based, saves TXT file)")
    print("  [4] Task 4: Cumulative Y-Direction Flux by Y-Axis Region (Plot)")
    print("  [5] Task 5: Y-Velocity (Vy) Average Heatmap at Y-Z Profile (Plot)")
    print("  [6] Task 6: Z-Velocity (Vz) Average Heatmap at Y-Z Profile (Plot)")
    print("  [7] Task 7: Soil Moisture (TH) Average Heatmap at Y-Z Profile (Plot)") # NEW
    print("  [8] Task 8: Combined Task 2 Bar Plot (exoticshrub, exoticgrass, arablecrop, naturalgrass)") # NEW
    
    while True:
        try:
            choice_str = input("\nWhich task(s) do you want to run?\n(Enter numbers separated by commas, e.g., '1,2,5', or 'all'): ")
            choice_str = choice_str.strip().lower()

            if choice_str == 'all':
                return [1, 2, 3, 4, 5, 6, 7, 8] # NEW
            
            tasks = []
            invalid_tasks = []
            
            parts = choice_str.split(',')
            if not parts:
                raise ValueError("Input is empty.")

            for part in parts:
                part = part.strip()
                if not part: continue
                
                choice_int = int(part)
                if 1 <= choice_int <= 8: # NEW
                    if choice_int not in tasks:
                         tasks.append(choice_int)
                else:
                    invalid_tasks.append(part)
            
            if invalid_tasks:
                print(f"Error: Invalid task number(s) provided: {', '.join(invalid_tasks)}. Please only use numbers 1-8.") # NEW
            elif not tasks:
                 print("Error: No valid tasks entered. Please try again.")
            else:
                return sorted(tasks)
                
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas (e.g., '1,3,4') or 'all'.")

# --- Main Orchestration Function ---

def run_analysis():
    """
    Main orchestration function.
    """
    
    # 1. Find simulation folders
    project_dir = os.path.dirname(os.path.abspath(__file__))
    sim_folders = find_simulation_folders(project_dir)
    
    if not sim_folders:
        print(f"Error: No simulation folders containing MESHTRIA.TXT found in {project_dir}")
        sys.exit(1)
    
    # 2. Create the central output directory
    output_dir = os.path.join(project_dir, "analysis_results")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {os.path.abspath(output_dir)}")
    except Exception as e:
        print(f"Error: Could not create output directory at {output_dir}. Reason: {e}")
        sys.exit(1)
        
    # 3. Prompt user for tasks and folders
    tasks_to_run = prompt_for_task()
    selected_folder_names = prompt_for_folder(sim_folders)
    
    print(f"\n--- Preparing to run {len(tasks_to_run)} task(s) on {len(selected_folder_names)} folder(s) ---")
    
    # --- MODIFICATION: Add cache for Task 8 ---
    global_results_cache = {}
    # --- End MODIFICATION ---
    
    # 4. === MAIN FOLDER LOOP ===
    for selected_folder_name in selected_folder_names:
        
        print(f"\n=======================================================")
        print(f"=== PROCESSING FOLDER: {selected_folder_name} ===")
        print(f"=======================================================\n")
        
        selected_folder_path = os.path.join(project_dir, selected_folder_name)

        # 5. Load all data into memory
        print(f"Loading data from: {selected_folder_path}")
        loader = HydrusModel(selected_folder_path)
        
        if not loader.load_all_data():
            print(f"Data loading failed for {selected_folder_name}. Skipping folder.")
            continue # Skip to the next folder
            
        mesh = loader.get_mesh()
        dates = loader.get_dates()
        
        if mesh is None or not dates:
            print(f"Error: Mesh or Dates were not loaded correctly for {selected_folder_name}. Skipping folder.")
            continue

        # 6. Apply Mesh Unit Conversion (cm -> m)
        print("Checking model coordinate units...")
        points = mesh.points
        y_coords = points[:, 1]
        
        if np.any(y_coords > 2.5) or np.any(y_coords < -2.5):
            print("Unit check: Centimeters detected (Y > 2.5 or < -2.5).")
            print("Converting all mesh coordinates (X, Y, Z) from CM to M...")
            mesh.points = points / 100.0
            print(f"Conversion complete. New Y-bounds: [{np.min(mesh.points[:,1]):.2f}m, {np.max(mesh.points[:,1]):.2f}m]")
        else:
            print("Unit check: Meters assumed.")

        # 7. Apply V.TXT Unit Conversion (cm/day -> m/day)
        # This modifies the data in the loader object *before* tasks are run.
        if selected_folder_name == "exoticgrass":
            print("Applying V.TXT unit conversion for 'exoticgrass' (dividing by 100)...")
            try:
                loader.data_in_memory['Vx'] = loader.data_in_memory['Vx'] / 100.0
                loader.data_in_memory['Vy'] = loader.data_in_memory['Vy'] / 100.0
                loader.data_in_memory['Vz'] = loader.data_in_memory['Vz'] / 100.0
                loader.data_in_memory['V_mag'] = loader.data_in_memory['V_mag'] / 100.0
                print("V.TXT (Vx, Vy, Vz, V_mag) conversion complete.")
            except KeyError:
                print("Warning: Could not apply V.TXT conversion, velocity data missing.")
            except Exception as e:
                print(f"Error during V.TXT conversion: {e}")
        
        # 8. Apply Coordinate Transformation (Slope Flattening)
        print("Applying 12-degree coordinate rotation to flatten geometry...")
        try:
            angle_rad = np.deg2rad(12.0)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            points = mesh.points
            x_coords = points[:, 0]
            z_coords = points[:, 2]
            
            top_nodes_mask = np.abs(x_coords) < 0.1
            z_hinge = 0.0
            
            if np.any(top_nodes_mask):
                z_hinge = np.max(z_coords[top_nodes_mask])
                print(f"Rotation hinge identified at (X~0, Z={z_hinge:.2f}m)")
            else:
                z_hinge = np.max(z_coords)
                print(f"Warning: No nodes found near X=0. Using max Z ({z_hinge:.2f}m) as rotation hinge.")
                
            x_orig = np.copy(x_coords)
            z_orig = np.copy(z_coords)
            z_translated = z_orig - z_hinge
            
            x_rotated = x_orig * cos_a - z_translated * sin_a
            z_rotated = x_orig * sin_a + z_translated * cos_a
            
            z_new = z_rotated + z_hinge
            
            mesh.points[:, 0] = x_rotated
            mesh.points[:, 2] = z_new
            
            print("Coordinate transformation complete.")
            points = mesh.points
            print(f"New X-Bounds: [{np.min(points[:,0]):.2f}m, {np.max(points[:,0]):.2f}m]")
            print(f"New Y-Bounds: [{np.min(points[:,1]):.2f}m, {np.max(points[:,1]):.2f}m]")
            print(f"New Z-Bounds: [{np.min(points[:,2]):.2f}m, {np.max(points[:,2]):.2f}m]")
            
        except Exception as e:
            print(f"Error during coordinate transformation: {e}")
            print("Continuing with original sloped geometry.")

        # 9. Get time-series data
        th_data = loader.get_data_by_name('TH')
        if th_data is None:
            print(f"Error: TH data was not loaded correctly for {selected_folder_name}. Skipping folder.")
            continue
            
        print(f"Successfully loaded {len(dates)} time steps.")

        # 10. === MAIN TASK LOOP ===
        print(f"--- Executing tasks for {selected_folder_name}: {tasks_to_run} ---")
        for task_to_run in tasks_to_run:
        
            # Generate standardized file prefix for this task
            run_date_str = datetime.now().strftime('%Y%m%d')
            base_filename_prefix = f"{run_date_str}_{selected_folder_name}_Task{task_to_run}"
            
            task_name_map = {
                1: "Task1_LowMoisture",
                2: "Task2_RegionStorage",
                3: "Task3_PositiveDelta",
                4: "Task4_RegionYFlux",
                5: "Task5_VyHeatmap",
                6: "Task6_VzHeatmap",
                7: "Task7_THHeatmap", # NEW
                8: "Task8_CombinedBar" # NEW
            }
            task_name = task_name_map.get(task_to_run, f"UnknownTask{task_to_run}")
            base_filename_prefix = f"{run_date_str}_{selected_folder_name}_{task_name}"
            
            # Apply global plot style
            if task_to_run in [1, 2, 4, 5, 6, 7, 8]: # NEW
                set_scientific_style(grid_alpha=0.3)
            
            # --- Task 1 Execution ---
            if task_to_run == 1:
                print("\n--- Running Task 1 ---")
                results = calc.run_task_1_calculations(mesh, th_data, dates)
                calc.save_results_csv(
                    output_dir, 
                    f"{base_filename_prefix}_data", 
                    dates, 
                    results, 
                    "Volume_m3"
                )
                plot.plot_task_1(dates, results, output_dir, base_filename_prefix)

            # --- Task 2 Execution ---
            elif task_to_run == 2:
                print("\n--- Running Task 2 ---")
                results = calc.run_task_2_calculations(mesh, th_data, dates)
                
                # --- MODIFICATION: Store results if Task 8 is also selected ---
                if 8 in tasks_to_run:
                    print(f"Caching Task 2 results for {selected_folder_name} for use in Task 8.")
                    global_results_cache[selected_folder_name] = results
                # --- End MODIFICATION ---
                
                # Save all 3 CSVs
                calc.save_results_csv(output_dir, f"{base_filename_prefix}_daily_storage_data",
                    dates, results["volume_timeseries"], results["column_names"], is_multi_region=True)
                calc.save_results_csv(output_dir, f"{base_filename_prefix}_cumulative_increase_data",
                    results["delta_dates"], results["cumulative_increase"], results["column_names"], is_multi_region=True)
                calc.save_results_csv(output_dir, f"{base_filename_prefix}_daily_increase_rate_data",
                    results["delta_dates"], results["positive_deltas"], results["column_names"], is_multi_region=True)
                
                # Generate all 3 plots
                plot.plot_task_2_charts(dates, results, output_dir, base_filename_prefix)
                    
            # --- Task 3 Execution ---
            elif task_to_run == 3:
                print("\n--- Running Task 3 ---")
                results_positive_deltas = calc.run_task_3_calculations(th_data)
                
                output_filename = f"{base_filename_prefix}.TXT"
                output_filepath = os.path.join(selected_folder_path, output_filename)
                
                print(f"Writing data to source simulation folder:")
                calc.save_task_3_txt(
                    output_filepath, 
                    loader.times, 
                    dates, 
                    results_positive_deltas
                )

            # --- Task 4 Execution ---
            elif task_to_run == 4:
                print("\n--- Running Task 4 ---")
                vy_data = loader.get_data_by_name('Vy')
                if vy_data is None:
                    print("Error: Task 4 requires 'Vy' data. Skipping task.")
                    continue
                
                results = calc.run_task_4_calculations(mesh, th_data, vy_data, dates)
                
                # Save both CSVs
                calc.save_results_csv(output_dir, f"{base_filename_prefix}_daily_y_flux_data",
                    dates, results["y_flux_timeseries"], results["column_names"], is_multi_region=True)
                calc.save_results_csv(output_dir, f"{base_filename_prefix}_cumulative_y_flux_data",
                    dates, results["cumulative_y_flux"], results["column_names"], is_multi_region=True)

                # Generate both plots
                plot.plot_task_4_charts(dates, results, output_dir, base_filename_prefix)

            # --- Task 5 Execution ---
            elif task_to_run == 5:
                print("\n--- Running Task 5 ---")
                vy_data = loader.get_data_by_name('Vy')
                if vy_data is None:
                    print("Error: Task 5 requires 'Vy' data. Skipping task.")
                    continue
                
                task5_results = calc.run_task_5_calculations(vy_data, dates)
                if task5_results is None or task5_results.get("overall") is None:
                    print("Warning: Task 5 calculations returned no data. Skipping plotting.")
                    continue
                plot.plot_task_5_heatmaps(task5_results, mesh, output_dir, base_filename_prefix)

            # --- Task 6 Execution ---
            elif task_to_run == 6:
                print("\n--- Running Task 6 ---")
                vz_data = loader.get_data_by_name('Vz')
                if vz_data is None:
                    print("Error: Task 6 requires 'Vz' data. Skipping task.")
                    continue
                
                task6_results = calc.run_task_6_calculations(vz_data, dates)
                if task6_results is None or task6_results.get("overall") is None:
                    print("Warning: Task 6 calculations returned no data. Skipping plotting.")
                    continue
                plot.plot_task_6_heatmaps(task6_results, mesh, output_dir, base_filename_prefix)

            # --- Task 7 Execution (NEW) ---
            elif task_to_run == 7:
                print("\n--- Running Task 7 ---")
                # TH data is already loaded and checked
                
                task7_results = calc.run_task_7_calculations(th_data, dates)
                if task7_results is not None and task7_results.get("overall") is not None:
                    plot.plot_task_7_heatmaps(task7_results, mesh, output_dir, base_filename_prefix)
                else:
                    print("Warning: Task 7 calculations returned no data. Skipping plotting.")
        
        print(f"\n--- Finished processing folder: {selected_folder_name} ---")

    print("\n--- All selected folders processed. ---")
    
    # --- MODIFICATION: Task 8 Execution (After folder loop) ---
    if 8 in tasks_to_run:
        print("\n--- Running Task 8 ---")
        
        # Define the 4 required folders and their plot order
        folders_to_plot = ["exoticshrub", "exoticgrass", "arablecrop", "naturalgrass"]
        plot_order_map = {
            "exoticshrub": (0, 1), # Top-left
            "exoticgrass": (0, 0), # Top-right
            "arablecrop":  (1, 1), # Bottom-left
            "naturalgrass": (1, 0) # Bottom-right
        }
        
        # Check if we have data for all 4
        missing_folders = [f for f in folders_to_plot if f not in global_results_cache]
        
        if missing_folders:
            print(f"Error: Cannot run Task 8. Missing Task 2 data for: {', '.join(missing_folders)}")
            print("Please re-run and ensure you select Task 2 and all four required folders.")
        else:
            print("Found data for all 4 required folders. Generating combined plot...")
            run_date_str = datetime.now().strftime('%Y%m%d')
            base_filename_prefix = f"{run_date_str}_ALL_Task8_CombinedBar"
            
            # Apply style just before plotting
            set_scientific_style(grid_alpha=0.3)
            
            plot.plot_task_8_combined_bar(
                global_results_cache,
                folders_to_plot,
                plot_order_map,
                output_dir,
                base_filename_prefix
            )
    # --- End MODIFICATION ---

# --- Script Entry Point ---

if __name__ == "__main__":
    # Protects main execution logic when imported by multiprocessing pool.
    multiprocessing.freeze_support()
    
    # Register email notification hook
    if notify_on_exit:
        try:
            notify_on_exit.register_completion_notify(
                recipient_email=RECIPIENT_EMAIL,
                smtp_config=SMTP_CONFIG,
                program_name="analysis_runner.py"
            )
        except Exception as e:
            print(f"Warning: Failed to register email notification. Error: {e}")
            
    print("\n--- Analysis Started ---")
    start_time = datetime.now()
    
    try:
        run_analysis()
        end_time = datetime.now()
        print(f"\n--- Analysis Finished Successfully ---")
        print(f"Total execution time: {end_time - start_time}")
    except Exception as e:
        end_time = datetime.now()
        print(f"\n--- Analysis FAILED ---")
        print(f"Total execution time before failure: {end_time - start_time}")
        print(f"An unexpected error occurred: {e}")
        raise

