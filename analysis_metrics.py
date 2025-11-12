# -*- coding: utf-8 -*-
"""
Metric computation helpers for HYDRUS seasonal slice analysis.

These utilities operate on the interpolated 2D Y–Z grids produced by
analysis_plotting._calculate_heatmap_data, so they can be called without
needing to touch the original unstructured mesh. All functions are
pure and return dictionaries of scalar metrics that can be assembled
into a DataFrame by the caller.

Notes
- Grid inputs are expected to be 2D arrays of identical shape (Nz, Ny).
- Depth metrics assume Z increases upward; depths are computed as the
  difference between the local surface level (max Z in a column) and
  the boundary Z within the column. A constant vertical offset (used
  only for visualization) does not affect depth differences.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label


# --- Core helpers ---

def _grid_spacings(y_min: float, y_max: float, z_min: float, z_max: float,
                   grid_shape: Tuple[int, int]) -> Tuple[float, float, float]:
    """Return dy, dz, domain_area for a regular grid extent and shape."""
    nz, ny = grid_shape
    # Guard against degenerate dimensions
    dy = 0.0 if ny <= 1 else (y_max - y_min) / (ny - 1)
    dz = 0.0 if nz <= 1 else (z_max - z_min) / (nz - 1)
    domain_area = max(y_max - y_min, 0.0) * max(z_max - z_min, 0.0)
    return dy, dz, domain_area


def largest_region_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    """Binary mask of the largest connected region where values < threshold.

    Uses 8-neighbour connectivity via a 3x3 structure. Returns a boolean mask
    of the same shape. If no cell is below threshold, returns an all-False mask.
    """
    if values is None:
        return np.zeros((0, 0), dtype=bool)
    mask = np.asarray(values) < threshold
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    structure = np.ones((3, 3), dtype=int)
    labeled, num = label(mask, structure=structure)
    if num == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background label 0
    idx = int(np.argmax(counts))
    return labeled == idx


# --- Geometric metrics ---

def geometric_metrics(grid_y: np.ndarray, grid_z: np.ndarray, mask: np.ndarray,
                      y_min: float, y_max: float, z_min: float, z_max: float) -> Dict[str, float]:
    """Compute geometric metrics of the masked dry region on a regular grid.

    Returns a dict with keys: 'area_m2', 'area_frac', 'depth_mean_m',
    'depth_max_m', 'depth_roughness_m', 'perimeter_m'.
    """
    nz, ny = mask.shape
    dy, dz, domain_area = _grid_spacings(y_min, y_max, z_min, z_max, mask.shape)
    dA = dy * dz

    area = float(mask.sum()) * dA
    area_frac = float(area / domain_area) if domain_area > 0 else 0.0

    # Depth by column: surface at max Z in the column
    depths = []
    for j in range(ny):
        col = mask[:, j]
        if not np.any(col):
            continue
        # boundary Z = max Z where mask is True in this column
        # grid_z increases with row index due to meshgrid(linspace(z_min,z_max))
        z_col = grid_z[:, j]
        z_boundary = float(np.max(z_col[col]))
        z_surface = float(np.max(z_col))
        depths.append(z_surface - z_boundary)
    if len(depths) == 0:
        d_mean = d_max = d_std = 0.0
    else:
        arr = np.asarray(depths, dtype=float)
        d_mean = float(np.mean(arr))
        d_max = float(np.max(arr))
        d_std = float(np.std(arr, ddof=0))

    # Perimeter length using grid-edge counting (4-neighbour)
    # Horizontal edges (between columns): count transitions across Y → length dz
    h_edges = np.logical_xor(mask[:, 1:], mask[:, :-1]).sum()
    # Vertical edges (between rows): transitions across Z → length dy
    v_edges = np.logical_xor(mask[1:, :], mask[:-1, :]).sum()
    perimeter = float(h_edges) * dz + float(v_edges) * dy

    return {
        'area_m2': area,
        'area_frac': area_frac,
        'depth_mean_m': d_mean,
        'depth_max_m': d_max,
        'depth_roughness_m': d_std,
        'perimeter_m': perimeter
    }


# --- Flux/hydrodynamic metrics ---

def flux_metrics(grid_vz: np.ndarray, mask: np.ndarray,
                 y_min: float, y_max: float, z_min: float, z_max: float) -> Dict[str, float]:
    """Compute Vz-based metrics inside/outside the mask and net flux through the mask.

    Qin integrates Vz over the masked area (units: m^3/day per meter in X).
    'fdown_in' is the fraction of the masked area where Vz < 0.
    """
    dy, dz, domain_area = _grid_spacings(y_min, y_max, z_min, z_max, grid_vz.shape)
    dA = dy * dz
    area_in = float(mask.sum()) * dA

    vz = np.asarray(grid_vz, dtype=float)
    if area_in <= 0:
        return {
            'Qin_m3_per_day_per_m': 0.0,
            'Vz_in_mean_m_per_day': float(np.nanmean(vz)),
            'Vz_out_mean_m_per_day': float(np.nanmean(vz)),
            'fdown_in': 0.0,
            'dVz_boundary_m_per_day': 0.0
        }

    # Means inside/outside
    vz_in = vz[mask]
    vz_out = vz[~mask]
    vz_in_mean = float(np.mean(vz_in)) if vz_in.size else 0.0
    vz_out_mean = float(np.mean(vz_out)) if vz_out.size else 0.0

    # Net vertical flux through masked area
    Qin = float(np.sum(vz_in) * dA)

    # Fraction of downward flow (Vz < 0) inside mask
    fdown_in = float(np.sum(vz_in < 0) * dA / area_in) if area_in > 0 else 0.0

    # Boundary rings: one-cell thick inner and outer rings
    structure = np.ones((3, 3), dtype=bool)
    inner_ring = mask & (~binary_erosion(mask, structure=structure, iterations=1))
    outer_ring = binary_dilation(mask, structure=structure, iterations=1) & (~mask)
    vz_inner = vz[inner_ring]
    vz_outer = vz[outer_ring]
    dVz_boundary = float(np.mean(vz_inner) - np.mean(vz_outer)) if vz_inner.size and vz_outer.size else 0.0

    return {
        'Qin_m3_per_day_per_m': Qin,
        'Vz_in_mean_m_per_day': vz_in_mean,
        'Vz_out_mean_m_per_day': vz_out_mean,
        'fdown_in': fdown_in,
        'dVz_boundary_m_per_day': dVz_boundary
    }


# --- Coupling metrics ---

def coupling_metrics(grid_th: np.ndarray, grid_vz: np.ndarray, threshold: float,
                     y_min: float, y_max: float, z_min: float, z_max: float) -> Dict[str, float]:
    """Compute coupling metric J = ∬ (threshold - TH)+ * Vz dA over domain.

    Positive J implies that vertical flow tends to relieve moisture deficit
    (downward recharge in a deficit), while negative implies it tends to
    exacerbate it (upward flow where soil is already dry), given Vz sign
    convention (downward negative in many HYDRUS outputs). Interpret per
    your sign convention.
    """
    if grid_th is None or grid_vz is None:
        return {'J_def_vz_m3_per_day_per_m': 0.0}
    if grid_th.shape != grid_vz.shape:
        # Require identical grids to avoid resampling here
        return {'J_def_vz_m3_per_day_per_m': np.nan}

    dy, dz, _ = _grid_spacings(y_min, y_max, z_min, z_max, grid_th.shape)
    dA = dy * dz
    deficit = np.maximum(0.0, threshold - np.asarray(grid_th, dtype=float))
    J = float(np.sum(deficit * np.asarray(grid_vz, dtype=float)) * dA)
    return {'J_def_vz_m3_per_day_per_m': J}


# --- Convenience aggregator ---

def compute_profile_metrics(th_data: Optional[dict], vz_data: Optional[dict], threshold: float) -> Dict[str, float]:
    """Compute combined metrics for a single profile slice.

    th_data / vz_data are dictionaries with keys:
      'grid_y','grid_z','grid_vel','y_min','y_max','z_min','z_max'
    Any of them can be None if a class of metrics is not desired.
    """
    results: Dict[str, float] = {}

    mask = None
    if th_data is not None:
        th_grid = th_data.get('grid_vel')
        if th_grid is not None and th_grid.size:
            mask = largest_region_mask(th_grid, threshold)
            gm = geometric_metrics(
                th_data['grid_y'], th_data['grid_z'], mask,
                th_data['y_min'], th_data['y_max'], th_data['z_min'], th_data['z_max']
            )
            results.update(gm)

    if vz_data is not None and vz_data.get('grid_vel') is not None and mask is not None:
        fm = flux_metrics(
            vz_data['grid_vel'], mask,
            vz_data['y_min'], vz_data['y_max'], vz_data['z_min'], vz_data['z_max']
        )
        results.update(fm)

        # Coupling on common grid only
        if th_data is not None and th_data.get('grid_vel') is not None:
            cm = coupling_metrics(
                th_data['grid_vel'], vz_data['grid_vel'], threshold,
                vz_data['y_min'], vz_data['y_max'], vz_data['z_min'], vz_data['z_max']
            )
            results.update(cm)

    return results

