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
import time
from typing import List, Tuple, Optional, Dict, Any

# Third-party imports
import numpy as np

################################# Internal ####################################

def safe_print(text: str, stop_dur: float = 0.3) -> None:
    """Print text with a delay for better display compatibility.
    
    This function adds a delay after printing to ensure compatibility with
    progress bars and other dynamic terminal output.

    Args:
        text: The text to print
        stop_dur: Delay duration in seconds. Defaults to 0.3.
    """
    print(text)
    time.sleep(stop_dur)
        
################################## For User ###################################

def dbm2watt(val: float | np.ndarray) -> float | np.ndarray:
    """Convert power from dBm to Watts.
    
    This function performs the standard conversion from decibel-milliwatts (dBm)
    to linear power in Watts.

    Args:
        val: Power value(s) in dBm

    Returns:
        Power value(s) in Watts
    """
    return 10**(val/10 - 3)

def steering_vec(array: Tuple[int, ...] | List[int] | np.ndarray, phi: float = 0, theta: float = 0,
                spacing: float = 0.5) -> np.ndarray:
    """Create array steering vector for uniform arrays.
    
    This function computes the normalized array steering vector for a uniform
    antenna array with specified geometry and steering direction.

    Args:
        array: Array dimensions as tuple/list/array (Mx, My, Mz)
        phi: Azimuth angle in degrees. Defaults to 0.
        theta: Elevation angle in degrees. Defaults to 0.
        spacing: Antenna spacing in wavelengths. Defaults to 0.5.

    Returns:
        Normalized steering vector for the array
    """
    idxs = ant_indices(array)
    resp = array_response(idxs, phi*np.pi/180, theta*np.pi/180 + np.pi/2, 2*np.pi*spacing)
    return resp / np.linalg.norm(resp)


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
        dataset (Dict): DeepMIMO dataset containing path information
        first_pos (np.ndarray): Starting position of the linear path
        last_pos (np.ndarray): Ending position of the linear path
        n (int): Number of points along the path
        idxs (np.ndarray): Indices of dataset points along the path
        pos (np.ndarray): Positions of points along the path
        feature_names (List[str]): Names of extracted features
    """
    def __init__(self, deepmimo_dataset: Dict[str, Any] | List[Dict[str, Any]], first_pos: np.ndarray,
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
        
        self.dataset = deepmimo_dataset if type(deepmimo_dataset) != list else deepmimo_dataset[0]
        self._set_idxs_pos_res_steps(res, n_steps, filter_repeated)
        self._copy_data_from_dataset()
        self._extract_features()

    def _set_idxs_pos_res_steps(self, res: float, n_steps: Optional[int], 
                                filter_repeated: bool) -> None:
        """Set path indices, positions, resolution and steps.
        
        Args:
            res: Spatial resolution in meters
            n_steps: Number of steps along path
            filter_repeated: Whether to filter repeated positions
        """
        dataset_pos = self.dataset['user']['location']
        if not n_steps:
            data_res = np.linalg.norm(dataset_pos[0] - dataset_pos[1])
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
        idxs = np.array([np.argmin(np.linalg.norm(dataset_pos - pos, axis=1)) 
                         for pos in interpolated_pos])
        
        if filter_repeated:
            # soft: removes adjacent repeated only
            idxs = np.concatenate(([idxs[0]], idxs[1:][(idxs[1:]-idxs[:-1]) != 0]))
            
            if filter_repeated == 'hard':
                # hard: removes all repeated
                idxs = np.unique(idxs)
            
            self.n = len(idxs)
    
        self.idxs = idxs
        self.pos = dataset_pos[idxs]

    def _copy_data_from_dataset(self) -> None:
        """Copy relevant data from dataset to class attributes."""
        self.feature_names = ['LoS', 'pathloss', 'distance']
        
        self.LoS = self.dataset['user']['LoS'][self.idxs]
        self.pathloss = self.dataset['user']['pathloss'][self.idxs]
        self.distance = self.dataset['user']['distance'][self.idxs]
        self.paths = self.dataset['user']['paths'][self.idxs]
        self.channel = self.dataset['user']['channel'][self.idxs]

    def _extract_features(self) -> None:
        """Extract path features and compute derived quantities."""
        # Main path features
        self.path_features = ['DoD_phi', 'DoD_theta', 'DoA_phi', 'DoA_theta',
                            'ToA', 'phase', 'power']
        self.feature_names += ['main_path_' + var for var in self.path_features]
        for feat in self.path_features:
            setattr(self, f'main_path_{feat}',
                    np.array([self.paths[i][feat][0] for i in range(self.n)]))
        
        # Other features
        self.feature_names += ['pwr_ratio_main_path', 'total_power']
        self.total_power = np.array([np.sum(self.paths[i]['power']) for i in range(self.n)])
        self.pwr_ratio_main_path = np.array([self.main_path_power[i] / np.sum(self.paths[i]['power'])
                                           if self.LoS[i] != -1 else np.nan for i in range(self.n)])

    def get_feature_names(self) -> List[str]:
        """Get list of available feature names.
        
        Returns:
            List of feature names extracted from the path
        """
        return self.feature_names


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