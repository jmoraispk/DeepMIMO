import os
import pandas as pd
import numpy as np
from typing import Dict, Any

from .geo_utils import convert_Gps2RelativeCartesian, convert_GpsBBox2CartesianBBox
from .pipeline_consts import UE_HEIGHT, BS_HEIGHT, GRID_SPACING, POS_PREC

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
        bs_pos.append([bs_cartesian[0], bs_cartesian[1], BS_HEIGHT])
    return np.round(np.array(bs_pos), POS_PREC)

def gen_rx_pos(row: pd.Series, osm_folder: str) -> np.ndarray:
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

    user_grid = generate_user_grid(row, origin_lat, origin_lon)
    print(f"User grid shape: {user_grid.shape}")
    return np.round(user_grid, POS_PREC) 

def generate_user_grid(row: pd.Series, origin_lat: float, origin_lon: float) -> np.ndarray:
    """Generate user grid in Cartesian coordinates.
    
    Args:
        row (pd.Series): Row from the parameters DataFrame
        origin_lat (float): Origin latitude in degrees
        origin_lon (float): Origin longitude in degrees
        
    Returns:
        np.ndarray: User grid positions in Cartesian coordinates
    """
    min_lat, min_lon = row['min_lat'], row['min_lon']
    max_lat, max_lon = row['max_lat'], row['max_lon']
    xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(
        min_lat, min_lon, 
        max_lat, max_lon, 
        origin_lat, origin_lon)
    grid_x = np.arange(xmin, xmax + GRID_SPACING, GRID_SPACING)
    grid_y = np.arange(ymin, ymax + GRID_SPACING, GRID_SPACING)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = np.zeros_like(grid_x) + UE_HEIGHT
    return np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=-1) 
