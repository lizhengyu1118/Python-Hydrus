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

# --- Global font configuration for generated plots ---
PLOT_FONT_SIZES = {
    # Base font size applied globally (fallback for other entries)
    'base': 22,
    # Axis/figure titles
    'title': 22,
    # Axis labels
    'label': 22,
    # Tick labels
    'tick': 22,
    # Legend text
    'legend': 22,
    # Task 8 combined plot subplot corner labels
    'task8_subplot_label': 22
}

# --- Task 5 (Vy) Heatmap Configuration ---
TASK5_HEATMAP_CONFIG = {
    # List of X-coordinates (in meters) for each Y-Z profile
    'x_profile_m_list': [-15.0, 0, 15.0],
    
    # Tolerance (in meters) to find nodes near each x_profile
    'x_tolerance_m': 0.05,
    
    # Resolution of the interpolation grid
    'grid_resolution_y': 200,
    'grid_resolution_z': 200,
    
    # Colormap for the heatmap (e.g., 'coolwarm', 'viridis', 'jet')
    'cmap': 'coolwarm',
    
    # 3D Plotting Parameters
    'figsize': (7.2, 5.5),
    'view_elevation': 25,
    'view_azimuth': -75,
    'colorbar_shrink': 0.6,
    'colorbar_aspect': 15,

    # Combined (2D+3D) figure layout
    'combined_figsize': (27, 20),
    'combined_width_ratios': [1.0, 1.0],
    'combined_height_ratios': [1.0, 1.0],

    # Subplot labeling (2D/3D combined plot)
    'show_subplot_labels': True,
    'subplot_labels': ['(a)', '(b)', '(c)'],
    'subplot_label_fontsize': 22,
    'subplot_label_box': {
        'facecolor': 'white',
        'alpha': 0.6,
        'edgecolor': 'none',
        'pad': 3
    },
    'subplot_label_offset_2d': (0.03, 0.95),
    'subplot_label_alignment_2d': {'ha': 'left', 'va': 'top'},
    'subplot_label_alignment_3d': {'ha': 'center', 'va': 'bottom'},
    'subplot_label_z_offset_ratio': 0.05,

    # Optional manual color scale (vmin, vmax); set to None for automatic
    'manual_color_limits': None
}

# --- Task 6 (Vz) Heatmap Configuration ---
TASK6_HEATMAP_CONFIG = {
    'x_profile_m_list': [-15.0, 0, 15.0],
    'x_tolerance_m': 0.05,
    'grid_resolution_y': 200,
    'grid_resolution_z': 200,
    # Different colormap for Vz
    'cmap': 'viridis', 
    'figsize': (7.2, 5.5),
    'view_elevation': 25,
    'view_azimuth': -75,
    'colorbar_shrink': 0.6,
    'colorbar_aspect': 15,

    'combined_figsize': (27, 20),
    'combined_width_ratios': [1.0, 1.0],
    'combined_height_ratios': [1.0, 1.0],

    'show_subplot_labels': True,
    'subplot_labels': ['(a)', '(b)', '(c)'],
    'subplot_label_fontsize': 22,
    'subplot_label_box': {
        'facecolor': 'white',
        'alpha': 0.6,
        'edgecolor': 'none',
        'pad': 3
    },
    'subplot_label_offset_2d': (0.03, 0.95),
    'subplot_label_alignment_2d': {'ha': 'left', 'va': 'top'},
    'subplot_label_alignment_3d': {'ha': 'center', 'va': 'bottom'},
    'subplot_label_z_offset_ratio': 0.05,

    'manual_color_limits': None
}

# --- Task 7 (TH) Heatmap Configuration ---
TASK7_HEATMAP_CONFIG = {
    'x_profile_m_list': [-15.0, 0, 15.0],
    'x_tolerance_m': 0.05,
    'grid_resolution_y': 200,
    'grid_resolution_z': 200,
    # Different colormap for TH
    'cmap': 'Blues', 
    'figsize': (7.2, 5.5),
    'view_elevation': 25,
    'view_azimuth': -75,
    'colorbar_shrink': 0.6,
    'colorbar_aspect': 15,
    'combined_figsize': (27, 20),
    'combined_width_ratios': [1.0, 1.0],
    'combined_height_ratios': [1.0, 1.0],
    'show_subplot_labels': True,
    'subplot_labels': ['(a)', '(b)', '(c)'],
    'subplot_label_fontsize': 22,
    'subplot_label_box': {
        'facecolor': 'white',
        'alpha': 0.6,
        'edgecolor': 'none',
        'pad': 3
    },
    'subplot_label_offset_2d': (0.03, 0.95),
    'subplot_label_alignment_2d': {'ha': 'left', 'va': 'top'},
    'subplot_label_alignment_3d': {'ha': 'center', 'va': 'bottom'},
    'subplot_label_z_offset_ratio': 0.05,

    'manual_color_limits': None
}

SEASONAL_MASK_CONFIG = {
    'enabled': True,
    'threshold': 0.12,
    'color': '#ff0000',
    'linewidth': 2.0,
    'linestyle': '-',
    'alpha': 0.8
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
    fonts = PLOT_FONT_SIZES if isinstance(PLOT_FONT_SIZES, dict) else {}
    BASE_SIZE = fonts.get('base', 15)
    TITLE_SIZE = fonts.get('title', BASE_SIZE)
    LABEL_SIZE = fonts.get('label', BASE_SIZE)
    TICK_SIZE = fonts.get('tick', BASE_SIZE)
    LEGEND_SIZE = fonts.get('legend', BASE_SIZE)

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
        'xtick.labelsize': TICK_SIZE,
        'ytick.labelsize': TICK_SIZE,
        
        # Legend size (will also be bold)
        'legend.fontsize': LEGEND_SIZE,
        'legend.title_fontsize': LEGEND_SIZE,
        
        # MODIFICATION: Set default grid appearance (even if off)
        'axes.grid': False, # Turn off grid by default
        'grid.color': '#b0b0b0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'grid.alpha': grid_alpha,
        
        # MODIFICATION: Set savefig DPI
        'savefig.dpi': 600
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

