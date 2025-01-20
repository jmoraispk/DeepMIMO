# -*- coding: utf-8 -*-
"""
Wireless Insite Path File Parser Module.

This module provides functionality for parsing Wireless Insite .p2m path files,
which contain detailed ray-tracing information including:
- Path angles (arrival/departure)
- Path delays, powers and phases
- Interaction coordinates and types
- Transmitter/receiver positions

The module handles both single and multi-path scenarios, with support for
various interaction types (reflection, diffraction, scattering, etc).
"""

import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any

from ... import consts as c

# Configuration constants for parsing p2m files
LINE_START = 22 # Skip info lines and n_rxs line. 
MAX_PATHS = 25
MAX_INTER_PER_PATH = 10

# Map interaction types to numeric codes (R=Reflection, D=Diffraction, etc.)
INTERACTIONS_MAP = {'R': 1,   # Reflection
                    'D': 2,   # Diffraction
                    'DS': 3,  # Diffuse Scattering
                    'T': 4,   # Transmission
                    'F': 5,   # Transmission through Leafs/Trees/Vegetation
                    'X': 6,   # Transmission through Leafs/Trees/Vegetation
                    }

def paths_parser(file: str) -> Dict[str, np.ndarray]:
    """Parse a Wireless Insite paths.p2m file to extract path information.
    
    This function reads and processes a .p2m file to extract detailed path information
    for each receiver, including angles, delays, powers, and interaction points.
    
    Args:
        file (str): Path to the .p2m file to parse
        
    Returns:
        Dict[str, np.ndarray]: Dictionary containing path information with keys:
            - aoa_az/el: Angles of arrival (azimuth/elevation)
            - aod_az/el: Angles of departure (azimuth/elevation)
            - toa: Time of arrival
            - power: Received power
            - phase: Path phase
            - inter: Interaction codes
            - inter_pos: Interaction point positions
            
    Note:
        The function limits the maximum number of paths per receiver to MAX_PATHS(25)
        and the maximum interactions per path to MAX_INTER_PER_PATH(10).
    """
    
    # Read file
    print('Reading file...')
    with open(file, 'r') as file:
        lines = file.readlines()
        
    n_rxs = int(lines[LINE_START-1])
    
    # Make data dictionary (to save for each tx set pair)
    data = {
        c.NUM_PATHS_PARAM_NAME: np.zeros((n_rxs), dtype=np.float32),
        c.AOA_AZ_PARAM_NAME: np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        c.AOA_EL_PARAM_NAME: np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        c.AOD_AZ_PARAM_NAME: np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        c.AOD_EL_PARAM_NAME: np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        c.TOA_PARAM_NAME: np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        c.PWR_PARAM_NAME: np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        c.PHASE_PARAM_NAME: np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        c.INTERACTIONS_PARAM_NAME: np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        c.INTERACTIONS_POS_PARAM_NAME: np.zeros((n_rxs, MAX_PATHS, MAX_INTER_PER_PATH, 3), dtype=np.float32) * np.nan,
    }
    
    line_idx = LINE_START
    for rx_i in tqdm(range(n_rxs), desc='Processing paths of each RX'):
        line = lines[line_idx]
        
        # The start of each "path info" is the rx idx and *number of paths*
        rx_n_paths = int(line.split()[1])
        
        if rx_n_paths == 0:
            line_idx += 1
            continue
        
        n_paths_to_read = min(rx_n_paths, MAX_PATHS)
        
        data[c.NUM_PATHS_PARAM_NAME][rx_i] = n_paths_to_read
        line_idx += 2
        
        for path_idx in range(n_paths_to_read):
            # Line 1 (Example: '1 2 -133.1 31.9 1.7e-06 84.0 40.3 90.9 -13.4\n')
            line = lines[line_idx]
            i1, i2, i3, i4, i5, i6, i7, i8, i9 = tuple(line.split())
            # (no need) i1 = <path number>
    		# (no need) i2 = <total interactions for path> (not including Tx and Rx)
            data[c.PWR_PARAM_NAME][rx_i, path_idx]   = np.float32(i3) # i3 = <received power(dBm)>
            data[c.PHASE_PARAM_NAME][rx_i, path_idx] = np.float32(i4) # i4 = <phase(deg)>
            data[c.TOA_PARAM_NAME][rx_i, path_idx]   = np.float32(i5) # i5 = <time of arrival(sec)>
            data[c.AOA_EL_PARAM_NAME][rx_i, path_idx] = np.float32(i6) # i6 = <arrival theta(deg)>
            data[c.AOA_AZ_PARAM_NAME][rx_i, path_idx] = np.float32(i7) # i7 = <arrival phi(deg)>
            data[c.AOD_EL_PARAM_NAME][rx_i, path_idx] = np.float32(i8) # i8 = <departure theta(deg)>
            data[c.AOD_AZ_PARAM_NAME][rx_i, path_idx] = np.float32(i9) # i9 = <departure phi(deg)>
            
            # Line 2 (Example: "Tx-D-R-Rx")
            line = lines[line_idx+1]
            inter_strs = line.split('-')[1:-1] # Example: ['D', 'R']
            # Map to interactions integers ['2', '1'] and join -> '21'
            inter_total_s = ''.join([str(INTERACTIONS_MAP[i_str]) for i_str in inter_strs])
            data[c.INTERACTIONS_PARAM_NAME][rx_i, path_idx] = \
                np.float32(inter_total_s) if inter_total_s else 0
            
            # Line 3-end (Example: "166 104 22")
            n_iteractions = int(i2)
            for inter_idx in range(n_iteractions):
                line = lines[line_idx + 3 + inter_idx] # skip tx and rx in 1st & last lines
                xyz = [np.float32(i) for i in line.split()]
                data[c.INTERACTIONS_POS_PARAM_NAME][rx_i, path_idx, inter_idx] = xyz
            
            line_idx += (4 + n_iteractions) # add number of description lines each path has
    
    # Remove extra paths and bounces
    data_compressed = compress_data(data)
    
    del data[c.NUM_PATHS_PARAM_NAME] # not needed
    
    return data_compressed

def compress_data(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Remove unused paths and interactions to optimize memory usage.
    
    This function compresses the path data by:
    1. Finding the maximum number of actual paths used
    2. Computing maximum number of interactions (bounces)
    3. Trimming arrays to remove unused entries
    
    Args:
        data (Dict[str, np.ndarray]): Dictionary containing path information arrays
        
    Returns:
        Dict[str, np.ndarray]: Compressed data dictionary with unused entries removed
    """
    # Compute max paths
    max_paths = data[c.NUM_PATHS_PARAM_NAME].max().astype(int)
    
    # Compute max bounces
    max_bounces = np.max(comp_next_pwr_10(data[c.INTERACTIONS_PARAM_NAME]))
    
    # Compress arrays to not take more than that space
    for key in data.keys():
        if len(data[key].shape) >= 2:
            data[key] = data[key][:, :max_paths, ...]
        if len(data[key].shape) >= 3:
            data[key] = data[key][:, :max_paths, :max_bounces]
    
    return data

def comp_next_pwr_10(arr: np.ndarray) -> np.ndarray:
    """Calculate number of interactions from interaction codes.
    
    This function computes the number of interactions (bounces) from the
    interaction code array by calculating the number of digits.
    
    Args:
        arr (np.ndarray): Array of interaction codes
        
    Returns:
        np.ndarray: Array containing number of interactions for each path
    """
    # Handle zero separately
    result = np.zeros_like(arr, dtype=int)
    
    # For non-zero values, calculate order
    non_zero = arr > 0
    result[non_zero] = np.floor(np.log10(arr[non_zero])).astype(int) + 1
    
    return result

def get_max_n_paths(arr: Dict[str, np.ndarray]) -> int:
    """Find maximum number of valid paths in the dataset.
    
    This function determines the maximum number of valid paths by finding
    the first path index where all entries are NaN.
    
    Args:
        arr (Dict[str, np.ndarray]): Dictionary containing path information arrays
        
    Returns:
        int: Maximum number of valid paths, or MAX_PATHS if all paths contain data
    """
    # The first path index with all entries at NaN
    all_nans_per_path_idx = np.all(np.isnan(arr[c.AOA_AZ_PARAM_NAME]), axis=0)
    n_max_paths = np.where(all_nans_per_path_idx)[0]
    return n_max_paths[0] if len(n_max_paths) else MAX_PATHS

def extract_tx_pos(filename: str) -> np.ndarray:
    """Extract transmitter position from a paths.p2m file.
    
    This function reads through a .p2m file to find and extract the transmitter
    position coordinates from the first valid path information.
    
    Args:
        filename (str): Path to the .p2m file to read
        
    Returns:
        np.ndarray: Array containing transmitter [x, y, z] coordinates of shape (3,)
        
    Note:
        The function skips receivers with no paths until it finds one with
        valid path information containing the transmitter position.
    """
    
    # Read file
    print('Reading paths file looking for tx position... ', end='')
    
    with open(filename, 'r') as file:
        for _ in range(LINE_START-1):  # Skip the first 20 lines
            next(file)
            
        n_rxs = int(file.readline())
        
        for _ in range(n_rxs):
            
            # The start of each "path info" is the rx idx and *number of paths*
            rx_n_paths = int(file.readline().split()[1])
            
            if rx_n_paths == 0:
                continue
            
            # Found user with paths!
            for _ in range(3):  # Skip 3 lines with other info
                next(file)
            
            # Read position
            tx_pos_line = file.readline()
            tx_pos = np.array([float(i) for i in tx_pos_line.split()], dtype=np.float32)
            break
    
    print('Found it!')
    return tx_pos

if __name__ == '__main__':
    file = './P2Ms/ASU_campus_just_p2m/study_area_asu5/asu_campus.paths.t001_01.r004.p2m'
    file = './P2Ms/simple_street_canyon/study_rays=0.25_res=2m_3ghz/simple_street_canyon_test.paths.t001_01.r002.p2m'
    data = paths_parser(file)
    print(data)