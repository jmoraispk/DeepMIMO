# -*- coding: utf-8 -*-
"""
paths_parser(.) collects all the information inside the paths.p2m file. 
"""

import numpy as np
from tqdm import tqdm

from ... import consts as c

LINE_START = 22 # Skip info lines and n_rxs line. 
MAX_PATHS = 25
MAX_INTER_PER_PATH = 10

INTERACTIONS_MAP = {'R': 1, 'D': 2, 'DS': 3, 'T': 4, 'F': 5, 'X': 6}

def paths_parser(file):
    
    # Read file
    print('Reading file...')
    with open(file, 'r') as file:
        lines = file.readlines()
        
    n_rxs = int(lines[LINE_START-1])
    
    # Make data dictionary (to save for each tx set pair)
    data = dict(
        n_paths = np.zeros((n_rxs), dtype=np.float32),
        aoa_az = np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        aoa_el = np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan, 
        aod_az = np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        aod_el = np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        toa    = np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        power  = np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        phase  = np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        inter  = np.zeros((n_rxs, MAX_PATHS), dtype=np.float32) * np.nan,
        inter_loc = np.zeros((n_rxs, MAX_PATHS, MAX_INTER_PER_PATH, 3),
                              dtype=np.float32) * np.nan,
        )
    
    line_idx = LINE_START
    for rx_i in tqdm(range(n_rxs), desc='Processing paths of each RX'):
        line = lines[line_idx]
        
        # The start of each "path info" is the rx idx and *number of paths*
        rx_n_paths = int(line.split()[1])
        
        if rx_n_paths == 0:
            line_idx += 1
            continue
        
        n_paths_to_read = min(rx_n_paths, MAX_PATHS)
        
        data['n_paths'][rx_i] = n_paths_to_read
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
    
    del data['n_paths'] # not needed
    
    return data_compressed

def compress_data(data):
    """ Remove extra paths and path bounces (interactions with environment)"""
    # Compute max paths
    max_paths = data['n_paths'].max().astype(int)
    
    # Compute max bounces
    max_bounces = np.max(comp_next_pwr_10(data[c.INTERACTIONS_PARAM_NAME]))
    
    # Compress arrays to not take more than that space
    for key in data.keys():
        if len(data[key].shape) >= 2:
            data[key] = data[key][:, :max_paths, ...]
        if len(data[key].shape) >= 3:
            data[key] = data[key][:, :max_paths, :max_bounces]
    
    return data

def comp_next_pwr_10(arr): # = number of bounces
    # Handle zero separately
    result = np.zeros_like(arr, dtype=int)
    
    # For non-zero values, calculate order
    non_zero = arr > 0
    result[non_zero] = np.floor(np.log10(arr[non_zero])).astype(int) + 1
    
    return result


def get_max_n_paths(arr):
    # The first path index with all entries at NaN
    all_nans_per_path_idx = np.all(np.isnan(arr['aoa_azi']), axis=0)
    n_max_paths = np.where(all_nans_per_path_idx)[0]
    return n_max_paths[0] if len(n_max_paths) else MAX_PATHS


if __name__ == '__main__':
    file = './P2Ms/ASU_campus_just_p2m/study_area_asu5/asu_campus.paths.t001_01.r004.p2m'
    file = './P2Ms/simple_street_canyon/study_rays=0.25_res=2m_3ghz/simple_street_canyon_test.paths.t001_01.r002.p2m'
    data = paths_parser(file)
    print(data)