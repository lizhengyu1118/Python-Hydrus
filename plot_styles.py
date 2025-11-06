# -*- coding: utf-8 -*-
"""
Reusable module for applying standardized plotting styles.

This module provides functions to set matplotlib rcParams to ensure
a consistent, publication-quality (scientific standard) appearance
for all generated plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns

# --- MODIFICATION: Added global color and Y-limit definitions ---

# Define 5 distinct, readable, cold-toned colors for Regions 1-5
REGION_COLORS = [
    "#202F48",  # Region 1: Blue
    '#DD8452',  # Region 2: Orange
    '#55A868',  # Region 3: Green
    '#C44E52',  # Region 4: Red
    "#806DC1"   # Region 5: Purple
]

# MODIFICATION: Define line widths for each of the 5 regions
REGION_LINEWIDTHS = [
    2.5,  # Region 1
    2.5,  # Region 2
    2.5,  # Region 3
    2.5,  # Region 4
    2.5   # Region 5
]

# --- Task 5 (Vy) Heatmap Configuration ---
TASK5_HEATMAP_CONFIG = {
    # List of X-coordinates (in meters) for each Y-Z profile
    'x_profile_m_list': [-15.0, 0, 15.0],
    
    # Tolerance (in meters) to find nodes near each x_profile
    'x_tolerance_m': 0.05,
    
    # Resolution of the interpolation grid
    'grid_resolution_y': 100,
    'grid_resolution_z': 100,
    
    # Colormap for the heatmap (e.g., 'coolwarm', 'viridis', 'jet')
    'cmap': 'coolwarm',
    
    # 3D Plotting Parameters
    'figsize': (18, 12),
    'view_elevation': 25,
    'view_azimuth': -75,
    'colorbar_shrink': 0.6,
    'colorbar_aspect': 15
}

# --- Task 6 (Vz) Heatmap Configuration ---
TASK6_HEATMAP_CONFIG = {
    'x_profile_m_list': [-15.0, 0, 15.0],
    'x_tolerance_m': 0.5,
    'grid_resolution_y': 100,
    'grid_resolution_z': 100,
    # Different colormap for Vz
    'cmap': 'viridis', 
    'figsize': (18, 12),
    'view_elevation': 25,
    'view_azimuth': -75,
    'colorbar_shrink': 0.6,
    'colorbar_aspect': 15
}

# --- Task 7 (TH) Heatmap Configuration ---
TASK7_HEATMAP_CONFIG = {
    'x_profile_m_list': [-15.0, 0, 15.0],
    'x_tolerance_m': 0.05,
    'grid_resolution_y': 100,
    'grid_resolution_z': 100,
    # Different colormap for TH
    'cmap': 'Blues', 
    'figsize': (18, 12),
    'view_elevation': 25,
    'view_azimuth': -75,
    'colorbar_shrink': 0.6,
    'colorbar_aspect': 15
}

# Define a dictionary to hold all manual Y-axis limits.
PLOT_Y_LIMITS = {
    # Task 2 Plots
    'Task2_DailyStorage': None,
    # MODIFICATION: Changed from list of tuples to a single tuple
    'Task2_CumulativeIncrease': (0, 160),
    'Task2_DailyRate': None,
    # MODIFICATION: Add Y-min setting for the bar chart
    # Set to a number (e.g., 0) to manually set the lower limit
    # Set to None to use automatic "zoom-in"
    'Task2_FinalBar_YMin': 100,
    'Task2_FinalBar_YMax': 200, # Set to a number to manually set the upper limit
    
    # Task 4 Plots
    'Task4_DailyRate': [ None, None, (-0.6,0.6), None, None],
    # MODIFICATION: Changed to a single tuple for the combined plot
    'Task4_CumulativeFlux': (-0.7, 0.7) 
}

# --- End of MODIFICATION ---


# MODIFICATION: Added grid_alpha parameter
def set_scientific_style(grid_alpha=0.5):
    """
    Applies a default scientific plotting style (based on seaborn-white)
    and sets standardized font sizes and weights for professional publications.
    
    This function modifies the global plt.rcParams.
    
    Args:
        grid_alpha (float): The transparency level for grid lines (if added manually).
    """
    
    # MODIFICATION: Use 'white' style to remove background grid by default.
    sns.set_theme(style="white")

    # Define standard font sizes for scientific plots
    BASE_SIZE = 15
    TITLE_SIZE = 15
    LABEL_SIZE = 15
    
    # Apply global rcParams (matplotlib runtime configuration)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        # MODIFICATION: Set Arial as preferred font, bold weight
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.weight': 'bold',
        
        # Base font size
        'font.size': BASE_SIZE,
        
        # Title sizes (will also be bold due to global font.weight)
        'axes.titlesize': TITLE_SIZE,
        'axes.titleweight': 'bold',
        'figure.titlesize': TITLE_SIZE,
        'figure.titleweight': 'bold',
        
        # Axis label sizes (will also be bold)
        'axes.labelsize': LABEL_SIZE,
        'axes.labelweight': 'bold',
        
        # Tick (the numbers on the axis) sizes (will also be bold)
        'xtick.labelsize': BASE_SIZE,
        'ytick.labelsize': BASE_SIZE,
        
        # Legend size (will also be bold)
        'legend.fontsize': BASE_SIZE,
        'legend.title_fontsize': BASE_SIZE,
        
        # MODIFICATION: Set default grid appearance (even if off)
        'axes.grid': False, # Turn off grid by default
        'grid.color': '#b0b0b0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': grid_alpha,
        
        # MODIFICATION: Set savefig DPI
        'savefig.dpi': 300
    })
    
    print("Applied global scientific plotting style (using seaborn-white base, Arial font).")

# --- MODIFICATION: Added new function to apply Y-limits ---
def apply_y_limits(axes, y_limits=[(100,200), (100,200), (100,200), (100,200)]):
    """
    Applies custom Y-axis limits to a list of subplot axes.
    
    This function is designed for the 5-subplot (sharex=True) layout.
    
    Args:
        axes (list): The list of matplotlib axes (e.g., from plt.subplots).
        y_limits (list of tuples or None): A list where each item is:
            - A tuple (min, max) for custom limits (e.g., (0, 100)).
            - None to use matplotlib's automatic scaling for that axis.
            
    Example:
        y_limits = [(0, 1000), (0, 800), None, (0, 500), None]
    """
    
    # If no limits are provided (default), do nothing.
    if y_limits is None:
        return
        
    try:
        # Ensure we have the same number of limits as axes
        if len(y_limits) != len(axes):
            print(f"Warning: y_limits list length ({len(y_limits)}) does not match axes count ({len(axes)}).")
            print("Skipping Y-limit application.")
            return

        # Iterate over each axis and its corresponding limit
        for i, ax in enumerate(axes):
            limits = y_limits[i]
            
            # If the limit for this axis is not None, apply it
            if limits is not None:
                # Check if it's a valid (min, max) tuple or list
                if isinstance(limits, (tuple, list)) and len(limits) == 2:
                    ax.set_ylim(limits)
                else:
                    print(f"Warning: Invalid y_limit format for axis {i}: {limits}. Expected (min, max) or None.")
                    
    except Exception as e:
        print(f"Error applying Y-limits: {e}")
# --- End of MODIFICATION ---

