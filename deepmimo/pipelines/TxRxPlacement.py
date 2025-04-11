import numpy as np
from typing import Dict, Any

from .utils.geo_utils import convert_Gps2RelativeCartesian, convert_GpsBBox2CartesianBBox

def gen_tx_pos(rt_params: Dict[str, Any]) -> np.ndarray:
    """Generate transmitter positions from GPS coordinates.
    
    Args:
        rt_params (Dict[str, Any]): Ray tracing parameters
        Required Parameters:
            - bs_lats (List[float]): Latitude coordinates of base stations
            - bs_lons (List[float]): Longitude coordinates of base stations
            - bs_heights (List[float]): Height coordinates of base stations
            - origin_lat, origin_lon (float): Origin GPS coordinates
        
    Returns:
        List[List[float]]: Transmitter positions in Cartesian coordinates
    """
    num_bs = len(rt_params['bs_lats'])
    print(f"Number of BSs: {num_bs}")
    bs_pos = []
    for bs_idx in range(num_bs):
        bs_cartesian = convert_Gps2RelativeCartesian(rt_params['bs_lats'][bs_idx], 
                                                     rt_params['bs_lons'][bs_idx],
                                                     rt_params['origin_lat'],
                                                     rt_params['origin_lon'])
        bs_pos.append([bs_cartesian[0], bs_cartesian[1], rt_params['bs_heights'][bs_idx]])
    return np.array(bs_pos)


def gen_rx_grid(rt_params: Dict[str, Any]) -> np.ndarray:
    """Generate user grid in Cartesian coordinates.
    
    Args:
        rt_params (Dict[str, Any]): Ray tracing parameters
            Required Parameters:
                - min_lat, min_lon, max_lat, max_lon (float): GPS coordinates of the area
                - origin_lat, origin_lon (float): Origin GPS coordinates
                - grid_spacing (float): Grid spacing in meters
                - ue_height (float): UE height in meters
        
    Returns:
        np.ndarray: User grid positions in Cartesian coordinates
    """
    xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(
        rt_params['min_lat'], rt_params['min_lon'], 
        rt_params['max_lat'], rt_params['max_lon'], 
        rt_params['origin_lat'], rt_params['origin_lon'])
    
    spacing = rt_params['grid_spacing']
    grid_x = np.arange(xmin, xmax + spacing, spacing)
    grid_y = np.arange(ymin, ymax + spacing, spacing)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = np.zeros_like(grid_x) + rt_params['ue_height']
    
    user_grid = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=-1) 
    
    print(f"User grid shape: {user_grid.shape}")
    
    return user_grid
