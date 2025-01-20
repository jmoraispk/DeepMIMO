"""
DeepMIMO Visualization Module.

This module provides visualization utilities for the DeepMIMO dataset, including:
- Coverage map visualization with customizable parameters
- Path characteristics visualization
- Channel properties plotting
- Data export utilities for external visualization tools

The module uses matplotlib for generating plots and supports both 2D and 3D visualizations.
"""

# Standard library imports
import csv
from typing import Optional, Tuple, Union, Dict, Any, List

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar


def plot_coverage(rxs: np.ndarray, cov_map: Union[Tuple[float, ...], List[float], np.ndarray],
                 dpi: int = 300, figsize: Tuple[int, int] = (6,4), cbar_title: Optional[str] = None,
                 title: Union[bool, str] = False, scat_sz: float = 0.5,
                 bs_pos: Optional[np.ndarray] = None, bs_ori: Optional[np.ndarray] = None,
                 legend: bool = False, lims: Optional[Tuple[float, float]] = None,
                 proj_3D: bool = False, equal_aspect: bool = False, tight: bool = True,
                 cmap: str = 'viridis') -> Tuple[Figure, Axes, Colorbar]:
    """Generate coverage map visualization for user positions.
    
    This function creates a customizable plot showing user positions colored by
    coverage values, with optional base station position and orientation indicators.

    Args:
        rxs (np.ndarray): User position array with shape (n_users, 3)
        cov_map (Union[Tuple[float, ...], List[float], np.ndarray]): Coverage map values for coloring
        dpi (int): Plot resolution in dots per inch. Defaults to 300.
        figsize (Tuple[int, int]): Figure dimensions (width, height) in inches. Defaults to (6,4).
        cbar_title (Optional[str]): Title for the colorbar. Defaults to None.
        title (Union[bool, str]): Plot title. Defaults to False.
        scat_sz (float): Size of scatter markers. Defaults to 0.5.
        bs_pos (Optional[np.ndarray]): Base station position coordinates. Defaults to None.
        bs_ori (Optional[np.ndarray]): Base station orientation angles. Defaults to None.
        legend (bool): Whether to show plot legend. Defaults to False.
        lims (Optional[Tuple[float, float]]): Color scale limits (min, max). Defaults to None.
        proj_3D (bool): Whether to create 3D projection. Defaults to False.
        equal_aspect (bool): Whether to maintain equal axis scaling. Defaults to False.
        tight (bool): Whether to set tight axis limits around data points. Defaults to True.
        cmap (str): Matplotlib colormap name. Defaults to 'viridis'.

    Returns:
        Tuple containing:
        - matplotlib Figure object
        - matplotlib Axes object
        - matplotlib Colorbar object
    """
    plt_params = {'cmap': cmap}
    if lims:
        plt_params['vmin'], plt_params['vmax'] = lims[0], lims[1]
    
    n = 3 if proj_3D else 2 # n = coordinates to consider
    
    xyz = {s: rxs[:,i] for s,i in zip(['x', 'y', 'zs'], range(n))}
    
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize,
                           subplot_kw={'projection': '3d'} if proj_3D else {})
    
    im = plt.scatter(**xyz, c=cov_map, s=scat_sz, marker='s', **plt_params)

    cbar = plt.colorbar(im, label='Received Power [dBm]' if not cbar_title else cbar_title)
    
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    
    # TX position
    if bs_pos is not None:
        ax.scatter(*bs_pos[:n], marker='P', c='r', label='TX')
    
    # TX orientation
    if bs_ori is not None and bs_pos is not None:
        r = 30 # ref size of pointing direction
        tx_lookat = np.copy(bs_pos)
        tx_lookat[:2] += r * np.array([np.cos(bs_ori[2]), np.sin(bs_ori[2])]) # azimuth
        tx_lookat[2] -= r / 10 * np.sin(bs_ori[1]) # elevation
        
        line_components = [[bs_pos[i], tx_lookat[i]] for i in range(n)]
        ax.plot(*line_components, c='k', alpha=.5, zorder=3)
        
    if title:
        ax.set_title(title)
    
    if legend:
        plt.legend(loc='upper center', ncols=10, framealpha=.5)
    
    if tight:
        s = 1
        mins, maxs = np.min(rxs, axis=0)-s, np.max(rxs, axis=0)+s
        
        plt.xlim([mins[0], maxs[0]])
        plt.ylim([mins[1], maxs[1]])
        if proj_3D:
            zlims = [mins[2], maxs[2]] if bs_pos is None else [np.min([mins[2], bs_pos[2]]),
                                                               np.max([mins[2], bs_pos[2]])]
            ax.axes.set_zlim3d(zlims)
    
    if equal_aspect: # often disrups the plot if in 3D.
        plt.axis('scaled')
    
    return fig, ax, cbar

def transform_coordinates(coords, lon_max, lon_min, lat_min, lat_max):
    """Transform Cartesian coordinates to geographical coordinates.
    
    This function converts x,y coordinates from a local Cartesian coordinate system
    to latitude/longitude coordinates using linear interpolation between provided bounds.
    
    Args:
        coords (np.ndarray): Array of shape (N,2) or (N,3) containing x,y coordinates
        lon_max (float): Maximum longitude value for output range
        lon_min (float): Minimum longitude value for output range  
        lat_min (float): Minimum latitude value for output range
        lat_max (float): Maximum latitude value for output range
        
    Returns:
        Tuple[List[float], List[float]]: Lists of transformed latitudes and longitudes
    """
    lats = []
    lons = []
    x_min, y_min = np.min(coords, axis=0)[:2]
    x_max, y_max = np.max(coords, axis=0)[:2]
    for (x, y) in zip(coords[:,0], coords[:,1]):
        lons += [lon_min + ((x - x_min) / (x_max - x_min)) * (lon_max - lon_min)]
        lats += [lat_min + ((y - y_min) / (y_max - y_min)) * (lat_max - lat_min)]
    return lats, lons

def export_xyz_csv(data: Dict[str, Any], z_var: np.ndarray, outfile: str = '',
                  google_earth: bool = False, lat_min: float = 33.418081,
                  lat_max: float = 33.420961, lon_min: float = -111.932875,
                  lon_max: float = -111.928567) -> None:
    """Export user locations and values to CSV format.
    
    This function generates a CSV file containing x,y,z coordinates that can be 
    imported into visualization tools like Blender or Google Earth. It supports
    both Cartesian and geographical coordinate formats.

    Args:
        data (Dict[str, Any]): DeepMIMO dataset for one basestation
        z_var (np.ndarray): Values to use for z-coordinate or coloring
        outfile (str): Output CSV file path. Defaults to ''.
        google_earth (bool): Whether to convert coordinates to geographical format. Defaults to False.
        lat_min (float): Minimum latitude for coordinate conversion. Defaults to 33.418081.
        lat_max (float): Maximum latitude for coordinate conversion. Defaults to 33.420961.
        lon_min (float): Minimum longitude for coordinate conversion. Defaults to -111.932875.
        lon_max (float): Maximum longitude for coordinate conversion. Defaults to -111.928567.

    Returns:
        None. Writes data to CSV file.
    """
    user_idxs = np.where(data['user']['LoS'] != -1)[0]
    
    locs = data['user']['location'][user_idxs]
    
    if google_earth:
        lats, lons = transform_coordinates(locs, 
                                           lon_min=lon_min, lon_max=lon_max, 
                                           lat_min=lat_min, lat_max=lat_max)
    else:
        lats, lons = locs[:,0], locs[:,1]
    
    # Transform xy to coords and create data dict
    data_dict = {'latitude'  if google_earth else 'x': lats if google_earth else locs[:,0], 
                 'longitude' if google_earth else 'y': lons if google_earth else locs[:,1], 
                 'z': z_var[user_idxs]}
    
    if not outfile:
        outfile = 'test.csv'
    
    # Equivalent in pandas (opted out to minimize dependencies.)
    # pd.DataFrame.from_dict(data_dict).to_csv(outfile, index=False)
    
    with open(outfile, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(data_dict.keys())
        # Write the data rows
        writer.writerows(zip(*data_dict.values()))






