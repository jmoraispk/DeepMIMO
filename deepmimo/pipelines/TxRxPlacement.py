import os
import numpy as np
from typing import Dict, Any

from .geo_utils import convert_Gps2RelativeCartesian, convert_GpsBBox2CartesianBBox
from .pipeline_consts import POS_PREC

def gen_tx_pos(rt_params: Dict[str, Any]) -> np.ndarray:
    """Generate transmitter positions from GPS coordinates.
    
    Args:
        rt_params (Dict[str, Any]): Ray tracing parameters
        
    Returns:
        List[List[float]]: Transmitter positions in Cartesian coordinates
    """
    num_bs = len(rt_params['bs_lats'])
    print(f"Number of BSs: {num_bs}")
    bs_pos = []
    for bs_lat, bs_lon in zip(rt_params['bs_lats'], rt_params['bs_lons']):
        bs_cartesian = convert_Gps2RelativeCartesian(bs_lat, bs_lon, 
                                                     rt_params['origin_lat'], 
                                                     rt_params['origin_lon'])
        bs_pos.append([bs_cartesian[0], bs_cartesian[1], rt_params['bs_height']])
    return np.round(np.array(bs_pos), POS_PREC)

def gen_rx_pos(rt_params: Dict[str, Any], osm_folder: str) -> np.ndarray:
    """Generate receiver positions from GPS coordinates.
    
    Args:
        row (pd.Series): Row from the parameters DataFrame
        osm_folder (str): Path to the OSM folder
        
    Returns:
        np.ndarray: Receiver positions in Cartesian coordinates
    """
    with open(os.path.join(osm_folder, 'osm_gps_origin.txt'), "r") as f:
        origin_lat, origin_lon = map(float, f.read().split())
    print(f"origin_lat: {origin_lat}, origin_lon: {origin_lon}")

    user_grid = generate_user_grid(rt_params, origin_lat, origin_lon)
    print(f"User grid shape: {user_grid.shape}")
    return np.round(user_grid, POS_PREC) 

def generate_user_grid(rt_params: Dict[str, Any], origin_lat: float, origin_lon: float) -> np.ndarray:
    """Generate user grid in Cartesian coordinates.
    
    Args:
        rt_params (Dict[str, Any]): Ray tracing parameters
        origin_lat (float): Origin latitude in degrees
        origin_lon (float): Origin longitude in degrees
        
    Returns:
        np.ndarray: User grid positions in Cartesian coordinates
    """
    min_lat, min_lon = rt_params['min_lat'], rt_params['min_lon']
    max_lat, max_lon = rt_params['max_lat'], rt_params['max_lon']
    xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(
        min_lat, min_lon, 
        max_lat, max_lon, 
        origin_lat, origin_lon)
    spacing = rt_params['grid_spacing']
    grid_x = np.arange(xmin, xmax + spacing, spacing)
    grid_y = np.arange(ymin, ymax + spacing, spacing)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = np.zeros_like(grid_x) + rt_params['ue_height']
    return np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=-1) 
