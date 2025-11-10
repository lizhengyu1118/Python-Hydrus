# -*- coding: utf-8 -*-
"""
Script to load data from multiple vegetation models and check the min/max
values for H, TH, and V (magnitude) on a specific target date.

Relies on 'hydrus_parser.py'.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Import the main data loader
try:
    from hydrus_parser import HydrusModel
except ImportError:
    print("Error: Could not find 'hydrus_parser.py'.")
    print("Please ensure 'hydrus_parser.py' is in the same directory.")
    sys.exit(1)

# --- Configuration ---

# TODO: Update this list with the exact names of your four vegetation folders
VEGETATION_FOLDERS = [
    "exoticgrass",
    "exoticshrub",
    "naturalgrass",
    "arablecrop"
]

# --- End Configuration ---

def find_nearest_date_index(dates, target_date):
    """
    Finds the index of the date in the list that is closest to the target_date.
    """
    if not dates:
        return None, None
        
    # Convert datetime objects to timestamps for numerical comparison
    target_ts = target_date.timestamp()
    date_timestamps = np.array([d.timestamp() for d in dates])
    
    # Find the index of the minimum absolute difference
    nearest_index = np.argmin(np.abs(date_timestamps - target_ts))
    found_date = dates[nearest_index]
    
    return nearest_index, found_date

def get_data_stats(loader, date_index, data_name):
    """
    Gets the data slice for a given index and returns its min/max stats.
    """
    data_array = loader.get_data_by_name(data_name)
    
    if data_array is None or date_index >= len(data_array):
        return f"Data '{data_name}' not found or index out of bounds."
        
    data_slice = data_array[date_index, :]
    
    # Use nanmin/nanmax to safely handle potential NaN values
    min_val = np.nanmin(data_slice)
    max_val = np.nanmax(data_slice)
    
    return f"Min: {min_val:.4E}, Max: {max_val:.4E}"

def run_data_check():
    """
    Main orchestration function.
    """
    
    # 1. Get Target Date from User
    while True:
        try:
            date_str = input("Please enter the target date (YYYY-MM-DD): ")
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
            print(f"Target date set to: {target_date.strftime('%Y-%m-%d')}")
            break
        except ValueError:
            print("Invalid format. Please use YYYY-MM-DD.")

    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Loop through each vegetation folder
    print("\n--- Starting Data Check ---")
    for folder_name in VEGETATION_FOLDERS:
        
        print(f"\n==========================================")
        print(f"=== Processing Folder: {folder_name} ===")
        print(f"==========================================")
        
        folder_path = os.path.join(project_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found at {folder_path}. Skipping.")
            continue
            
        # 3. Load Data
        print(f"Loading data from: {folder_path}")
        loader = HydrusModel(folder_path)
        
        # Suppress user verification prompt during load_all_data
        # We assume the parser is correct for this automated script
        # Note: This requires modifying hydrus_parser.py to accept a flag,
        # or temporarily redirecting stdin.
        # For simplicity, this script assumes you will manually press 'y'
        # or you have modified the parser to skip verification.
        if not loader.load_all_data():
            print(f"Data loading failed for {folder_name}. Skipping.")
            continue
            
        dates = loader.get_dates()
        if not dates:
            print(f"No dates loaded for {folder_name}. Skipping.")
            continue
            
        # 4. Find Nearest Date
        nearest_index, found_date = find_nearest_date_index(dates, target_date)
        
        if nearest_index is None:
            print(f"Could not find any dates for {folder_name}.")
            continue
            
        # Check if the found date is reasonably close (e.g., within 1 day)
        if abs((found_date - target_date).days) > 1:
            print(f"Warning: Target date {target_date.strftime('%Y-%m-%d')} not found.")
            print(f"Using nearest available date: {found_date.strftime('%Y-%m-%d')} (Index {nearest_index})")
        else:
            print(f"Found matching date: {found_date.strftime('%Y-%m-%d')} (Index {nearest_index})")
            
        # 5. Get and Print Stats
        print(f"--- Statistics for {folder_name} on {found_date.strftime('%Y-%m-%d')} ---")
        
        # H (Pressure Head)
        h_stats = get_data_stats(loader, nearest_index, 'H')
        print(f"  H (Head):      {h_stats}")
        
        # TH (Water Content)
        th_stats = get_data_stats(loader, nearest_index, 'TH')
        print(f"  TH (Content):  {th_stats}")
        
        # V (Velocity Magnitude)
        # Note: hydrus_parser.py calculates this and stores it as 'V_mag'
        v_stats = get_data_stats(loader, nearest_index, 'V_mag')
        print(f"  V (Magnitude): {v_stats}")

    print("\n--- Data Check Complete ---")

if __name__ == "__main__":
    run_data_check()
