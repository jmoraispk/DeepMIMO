"""
Utilities Module for DeepMIMO Dataset Processing.

This module provides utility functions and classes for processing DeepMIMO datasets,
including:
- Unit conversions (dBm to Watts)
- Array steering vector calculations
- Path analysis and feature extraction
- Position sampling and filtering utilities

The module serves as a collection of helper functions used throughout the DeepMIMO
dataset generation process.
"""

# Standard library imports
from typing import List, Optional

# Third-party imports
import numpy as np

################################## For User ###################################

def dbw2watt(val: float | np.ndarray) -> float | np.ndarray:
    """Convert power from dBW to Watts.
    
    This function performs the standard conversion from decibel-watts (dBW)
    to linear power in Watts.

    Args:
        val: Power value(s) in dBW

    Returns:
        Power value(s) in Watts
    """
    return 10**(val/10)

def get_uniform_idxs(n_ue: int, grid_size: np.ndarray, steps: List[int]) -> np.ndarray:
    """Return indices of users at uniform intervals in a grid.
    
    This function generates indices for uniform sampling of users in a grid,
    allowing for different sampling rates in each dimension. It validates the
    grid structure and handles edge cases appropriately.

    Args:
        n_ue: Number of users in the dataset
        grid_size: Array with [x_size, y_size] - number of points in each dimension
        steps: List of sampling steps for each dimension [x_step, y_step]
        
    Returns:
        Array of indices for uniformly sampled users
        
    Raises:
        ValueError: If dataset does not have a valid grid structure and steps != [1,1]
        
    Example:
        >>> rx_pos = np.array([[0,0], [0,1], [1,0], [1,1]])  # 2x2 grid
        >>> grid_size = np.array([2, 2])
        >>> steps = [2, 2]  # Sample every other point
        >>> get_uniform_idxs(rx_pos, grid_size, steps)
        array([0])  # Returns index of first point
    """    
    # Check if dataset has valid grid structure
    if np.prod(grid_size) != n_ue:
        print(f"Warning. Grid_size: {grid_size} = {np.prod(grid_size)} users != {n_ue} users in rx_pos")
        if steps == [1, 1]:
            idxs = np.arange(n_ue)
        else:
            raise ValueError("Dataset does not have a valid grid structure. Cannot perform uniform sampling.")
    else:
        # Get indices of users at uniform intervals
        cols = np.arange(grid_size[0], step=steps[0])
        rows = np.arange(grid_size[1], step=steps[1])
        idxs = np.array([j + i*grid_size[0] for i in rows for j in cols])
    
    return idxs

def uniform_sampling(steps: List[int], n_rows: int, users_per_row: int) -> np.ndarray:
    """Return indices of users at uniform intervals.
    
    This function generates indices for uniform sampling of users in a grid,
    allowing for different sampling rates in each dimension.

    Args:
        steps: List of sampling steps for each dimension
        n_rows: Number of rows in the grid
        users_per_row: Number of users per row

    Returns:
        Array of indices for uniformly sampled users
    """
    cols = np.arange(users_per_row, step=steps[0])
    rows = np.arange(n_rows, step=steps[1])
    uniform_idxs = np.array([j + i*users_per_row for i in rows for j in cols])
    return uniform_idxs


class LinearPath:
    """Class for creating and analyzing linear paths through DeepMIMO datasets.
    
    This class handles the creation of linear sampling paths through a DeepMIMO
    dataset and extracts relevant features along these paths, including path
    loss, delays, and angles.
    
    Attributes:
        rx_pos (np.ndarray): Positions of dataset points
        first_pos (np.ndarray): Starting position of the linear path
        last_pos (np.ndarray): Ending position of the linear path
        n (int): Number of points along the path
        idxs (np.ndarray): Indices of dataset points along the path
        pos (np.ndarray): Positions of points along the path
        feature_names (List[str]): Names of extracted features
    """
    def __init__(self, rx_pos: np.ndarray, first_pos: np.ndarray,
                 last_pos: np.ndarray, res: float = 1, n_steps: Optional[int] = None, 
                 filter_repeated: bool = True) -> None:
        """Initialize a linear path through the dataset.
        
        Args:
            deepmimo_dataset: DeepMIMO dataset or list of datasets
            first_pos: Starting position coordinates
            last_pos: Ending position coordinates
            res: Spatial resolution in meters. Defaults to 1.
            n_steps: Number of steps along path. Defaults to None.
            filter_repeated: Whether to filter repeated positions. Defaults to True.
        """
        if len(first_pos) == 2:  # if not given, assume z-coordinate = 0
            first_pos = np.concatenate((first_pos,[0]))
            last_pos = np.concatenate((last_pos,[0]))
            
        self.first_pos = first_pos
        self.last_pos = last_pos
        
        self._set_idxs_pos_res_steps(rx_pos, res, n_steps, filter_repeated)

    def _set_idxs_pos_res_steps(self, rx_pos: np.ndarray, res: float,
                                n_steps: Optional[int], filter_repeated: bool) -> None:
        """Set path indices, positions, resolution and steps.
        
        Args:
            res: Spatial resolution in meters
            n_steps: Number of steps along path
            filter_repeated: Whether to filter repeated positions
        """
        if not n_steps:
            data_res = np.linalg.norm(rx_pos[0] - rx_pos[1])
            if res < data_res and filter_repeated:
                print(f'Changing resolution to {data_res} to eliminate repeated positions')
                res = data_res
                
            self.n = int(np.linalg.norm(self.first_pos - self.last_pos) / res)
        else:
            self.n = n_steps
        
        xs = np.linspace(self.first_pos[0], self.last_pos[0], self.n).reshape((-1,1))
        ys = np.linspace(self.first_pos[1], self.last_pos[1], self.n).reshape((-1,1))
        zs = np.linspace(self.first_pos[2], self.last_pos[2], self.n).reshape((-1,1))
        
        interpolated_pos = np.hstack((xs,ys,zs))
        idxs = np.array([np.argmin(np.linalg.norm(rx_pos - pos, axis=1)) 
                         for pos in interpolated_pos])
        
        if filter_repeated:
            # soft: removes adjacent repeated only
            idxs = np.concatenate(([idxs[0]], idxs[1:][(idxs[1:]-idxs[:-1]) != 0]))
            
            if filter_repeated == 'hard':
                # hard: removes all repeated
                idxs = np.unique(idxs)
            
            self.n = len(idxs)
    
        self.idxs = idxs

def get_idxs_with_limits(data_pos: np.ndarray, **limits) -> np.ndarray:
    """Get indices of positions within specified coordinate limits.
    
    This function filters position indices based on specified minimum and
    maximum coordinate values in each dimension. Each limit (x_min, x_max, y_min,
    y_max, z_min, z_max) is optional and only applied if provided.

    Args:
        data_pos: Array of positions with shape (n_points, n_dims)
        **limits: Keyword arguments specifying coordinate limits
                 (x_min, x_max, y_min, y_max, z_min, z_max)

    Returns:
        Array of indices for positions within the specified limits
        
    Raises:
        ValueError: If invalid limit keys are provided or dimensions don't match
    """
    valid_limits = {'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'}
    if not all(key in valid_limits for key in limits):
        raise ValueError(f"Invalid limit key. Supported limits are: {valid_limits}")
    
    # Start with all indices as valid
    valid_idxs = np.arange(len(data_pos))
    
    # Apply each limit sequentially
    coord_map = {'x': 0, 'y': 1, 'z': 2}
    for limit_name, limit_value in limits.items():
        coord = limit_name.split('_')[0]  # Extract 'x', 'y', or 'z'
        is_min = limit_name.endswith('min')
        
        if coord_map[coord] >= data_pos.shape[1]:
            raise ValueError(f"Cannot apply {coord} limit to {data_pos.shape[1]}D positions")
            
        if is_min:
            mask = data_pos[valid_idxs, coord_map[coord]] >= limit_value
        else:  # is_max
            mask = data_pos[valid_idxs, coord_map[coord]] <= limit_value
            
        valid_idxs = valid_idxs[mask]
    
    return valid_idxs