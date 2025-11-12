# -*- coding: utf-8 -*-
"""
Modular functions for parallel data analysis of HYDRUS results.
Each function is designed to be called by 'analysis_runner.py' using
multiprocessing.

All functions are pure (no side effects) and receive all required data
as arguments, including static mesh geometry and the data slice for a
single time step.
"""

import numpy as np

def _get_element_average_th(th_data_slice, node_indices_for_cells):
    """
    Helper function to calculate the average theta (TH) for each element
    by averaging the TH values of its 4 constituent nodes.
    
    Args:
        th_data_slice (np.ndarray): 1D array (N_nodes,) of TH data for one time step.
        node_indices_for_cells (np.ndarray): 2D array (N_elements, 4) mapping
                                             elements to their node indices.
                                             
    Returns:
        np.ndarray: 1D array (N_elements,) of average TH for each element.
    """
    # Use fancy indexing to get the (N_elements, 4) array of TH values
    th_at_nodes = th_data_slice[node_indices_for_cells]
    
    # Calculate the mean along the node axis (axis=1)
    element_average_th = np.mean(th_at_nodes, axis=1)
    return element_average_th

# MODIFICATION: Add helper for Vy
def _get_element_average_vy(vy_data_slice, node_indices_for_cells):
    """
    Helper function to calculate the average Vy (Y-velocity) for each element
    by averaging the Vy values of its 4 constituent nodes.
    
    Args:
        vy_data_slice (np.ndarray): 1D array (N_nodes,) of Vy data for one time step.
        node_indices_for_cells (np.ndarray): 2D array (N_elements, 4) mapping
                                             elements to their node indices.
                                             
    Returns:
        np.ndarray: 1D array (N_elements,) of average Vy for each element.
    """
    # Use fancy indexing to get the (N_elements, 4) array of Vy values
    vy_at_nodes = vy_data_slice[node_indices_for_cells]
    
    # Calculate the mean along the node axis (axis=1)
    element_average_vy = np.mean(vy_at_nodes, axis=1)
    return element_average_vy

# --- Task 1 Function ---

def task_1_calculate_volume_for_timestep(
    th_data_slice, 
    node_indices_for_cells,
    elements_below_interface_mask,
    element_volumes
):
    """
    Calculates the total volume of elements that meet two criteria:
    1. They are below the 1m interface (pre-calculated mask).
    2. Their average moisture (TH) is below 0.12.
    
    Args:
        th_data_slice (np.ndarray): (N_nodes,) array for one time step.
        node_indices_for_cells (np.ndarray): (N_elements, 4) array.
        elements_below_interface_mask (np.ndarray): (N_elements,) boolean array.
        element_volumes (np.ndarray): (N_elements,) array of volumes.
        
    Returns:
        float: The total volume (m^3) for this time step.
    """
    try:
        # 1. Calculate average TH for all elements
        element_average_th = _get_element_average_th(
            th_data_slice, 
            node_indices_for_cells
        )
        
        # 2. Create mask for elements with TH < 0.12
        low_moisture_mask = (element_average_th < 0.12)
        
        # 3. Combine masks: (Below Interface) AND (Low Moisture)
        final_mask = elements_below_interface_mask & low_moisture_mask
        
        # 4. Sum the volumes of elements that match the final mask
        total_volume = np.sum(element_volumes[final_mask])
        
        return total_volume
    except Exception as e:
        # Log error in parallel process (will be raised by pool.starmap)
        raise Exception(f"Error in task_1_calculate_volume: {e}")

# --- Task 2 Function ---

def task_2_calculate_water_volume_for_timestep(
    th_data_slice,
    node_indices_for_cells,
    element_volumes,
    element_region_map
):
    """
    Calculates the total water volume (sum of [TH * Volume]) for each
    of the 5 predefined Y-axis regions.
    
    Args:
        th_data_slice (np.ndarray): (N_nodes,) array for one time step.
        node_indices_for_cells (np.ndarray): (N_elements, 4) array.
        element_volumes (np.ndarray): (N_elements,) array of volumes.
        element_region_map (np.ndarray): (N_elements,) array mapping each
                                         element to a region ID (1-5, or 0).
                                         
    Returns:
        np.ndarray: A 1D array (5,) containing the total water volume
                    for regions 1 through 5.
    """
    try:
        # 1. Calculate average TH for all elements
        element_average_th = _get_element_average_th(
            th_data_slice, 
            node_indices_for_cells
        )
        
        # 2. Calculate water volume for every element
        element_water_volume = element_average_th * element_volumes
        
        # 3. Sum the water volumes for each region (1 to 5)
        # We use np.bincount, which is highly efficient.
        region_totals = np.bincount(
            element_region_map, 
            weights=element_water_volume, 
            minlength=6
        )
        
        # Return only regions 1 through 5
        return region_totals[1:6]
        
    except Exception as e:
        raise Exception(f"Error in task_2_calculate_water_volume: {e}")

# --- Task 3 Function ---

def task_3_calculate_positive_th_delta_for_nodes(
    th_data_slice_t,
    th_data_slice_t_minus_1
):
    """
    Calculates the positive-only change in moisture content (TH)
    for *every node* between two time steps.
    
    Args:
        th_data_slice_t (np.ndarray): (N_nodes,) array for current time step.
        th_data_slice_t_minus_1 (np.ndarray): (N_nodes,) array for previous time step.
        
    Returns:
        np.ndarray: A 1D array (N_nodes,) containing the positive-only
                    moisture content change (delta) for each node.
    """
    try:
        # 1. Calculate delta at nodes
        delta = th_data_slice_t - th_data_slice_t_minus_1
        
        # 2. Apply logical judgment (keep positive-only)
        positive_delta = np.maximum(0, delta)
        
        return positive_delta
        
    except Exception as e:
        raise Exception(f"Error in task_3_calculate_positive_delta_for_nodes: {e}")

# --- Task 4 Function (NEW) ---

def task_4_calculate_y_flux_for_timestep(
    th_data_slice,
    vy_data_slice,
    node_indices_for_cells,
    element_volumes,
    element_region_map
):
    """
    Calculates the total Y-direction flux (q_y = Vy * TH) weighted by
    element volume, summed for each of the 5 predefined Y-axis regions.
    The sign is preserved (positive for +Y, negative for -Y).
    
    Args:
        th_data_slice (np.ndarray): (N_nodes,) array for one time step.
        vy_data_slice (np.ndarray): (N_nodes,) array of Y-velocity for one time step.
        node_indices_for_cells (np.ndarray): (N_elements, 4) array.
        element_volumes (np.ndarray): (N_elements,) array of volumes.
        element_region_map (np.ndarray): (N_elements,) array mapping each
                                         element to a region ID (1-5, or 0).
                                         
    Returns:
        np.ndarray: A 1D array (5,) containing the total flux-volume
                    for regions 1 through 5.
    """
    try:
        # 1. Calculate average TH for all elements
        element_average_th = _get_element_average_th(
            th_data_slice, 
            node_indices_for_cells
        )
        
        # 2. Calculate average Vy for all elements
        element_average_vy = _get_element_average_vy(
            vy_data_slice,
            node_indices_for_cells
        )
        
        # 3. Calculate Y-flux (q_y = Vy * TH) for each element
        # This value is signed
        element_y_flux = element_average_vy * element_average_th
        
        # 4. Calculate flux-volume (q_y * Volume)
        element_y_flux_volume = element_y_flux * element_volumes
        
        # 5. Sum the flux-volumes for each region (1 to 5)
        region_totals = np.bincount(
            element_region_map, 
            weights=element_y_flux_volume, 
            minlength=6
        )
        
        # Return only regions 1 through 5
        return region_totals[1:6]
        
    except Exception as e:
        raise Exception(f"Error in task_4_calculate_y_flux_for_timestep: {e}")

