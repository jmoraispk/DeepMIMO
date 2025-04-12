"""
Sionna Ray Tracing Paths Module.

This module handles loading and converting path data from Sionna's format to DeepMIMO's format.
"""

import os
import numpy as np
from tqdm import tqdm
from typing import Dict
from ... import consts as c
from .. import converter_utils as cu

# Interaction Type Map for Sionna
INTERACTIONS_MAP = {
    0:  c.INTERACTION_LOS,           # LoS
    1:  c.INTERACTION_REFLECTION,    # Reflection
    2:  c.INTERACTION_DIFFRACTION,   # Diffraction
    3:  c.INTERACTION_SCATTERING,    # Diffuse Scattering
    4:  None,  # Sionna RIS is not supported yet
}

def _preallocate_data(n_rx: int) -> Dict:
    """Pre-allocate data for path conversion.
    
    Args:
        n_rx: Number of RXs

    Returns:
        data: Dictionary containing pre-allocated data
    """
    data = {
        c.RX_POS_PARAM_NAME: np.zeros((n_rx, 3), dtype=c.FP_TYPE),
        c.TX_POS_PARAM_NAME: np.zeros((1, 3), dtype=c.FP_TYPE),
        c.AOA_AZ_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.AOA_EL_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.AOD_AZ_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.AOD_EL_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.DELAY_PARAM_NAME:  np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.POWER_PARAM_NAME:  np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.PHASE_PARAM_NAME:  np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.INTERACTIONS_PARAM_NAME:  np.zeros((n_rx, c.MAX_PATHS), dtype=c.FP_TYPE) * np.nan,
        c.INTERACTIONS_POS_PARAM_NAME: np.zeros((n_rx, c.MAX_PATHS, c.MAX_INTER_PER_PATH, 3), dtype=c.FP_TYPE) * np.nan,
    }
    
    return data
    
    

def _process_paths_batch(paths_dict: Dict, data: Dict, b: int, 
                          t: int, curr_max_inter: int, last_idx: int, batch_size: int) -> int:
    """Process a batch of paths from Sionna format and store in DeepMIMO format.
    
    Args:
        paths_dict: Dictionary containing Sionna path data
        data: Dictionary to store processed path data
        b: Batch index
        t: Transmitter index in current paths dictionary
        curr_max_inter: Maximum number of interactions per path
        last_idx: Starting index for storing in data arrays
        batch_size: Number of receivers in current batch
        
    Returns:
        int: Number of inactive receivers found in this batch
    """
    # Get amplitude data for current TX
    a = paths_dict['a'][0,:,0,t,0,:,0]  # Get users and paths for this TX
    
    inactive_count = 0
    # Process each receiver in the batch
    for rel_idx in range(batch_size):
        abs_idx = last_idx + rel_idx
        
        path_idxs = np.where(a[rel_idx] != 0)[0][:c.MAX_PATHS]
        n_paths = len(path_idxs)

        if n_paths == 0:
            inactive_count += 1
            continue

        # Power, phase, delay
        data[c.POWER_PARAM_NAME][abs_idx,:n_paths] = 20 * np.log10(np.absolute(a[rel_idx, path_idxs]))
        data[c.PHASE_PARAM_NAME][abs_idx,:n_paths] = np.angle(a[rel_idx, path_idxs], deg=True)
        data[c.DELAY_PARAM_NAME][abs_idx,:n_paths] = paths_dict['tau'][b, rel_idx, t, path_idxs]
        
        # Angles
        rad2deg = lambda x: np.rad2deg(x[b, rel_idx, t, path_idxs])
        data[c.AOA_AZ_PARAM_NAME][abs_idx, :n_paths] = rad2deg(paths_dict['phi_r'])
        data[c.AOD_AZ_PARAM_NAME][abs_idx, :n_paths] = rad2deg(paths_dict['phi_t'])
        data[c.AOA_EL_PARAM_NAME][abs_idx, :n_paths] = rad2deg(paths_dict['theta_r'])
        data[c.AOD_EL_PARAM_NAME][abs_idx, :n_paths] = rad2deg(paths_dict['theta_t'])

        # Interaction positions ([depth, num_rx, num_tx, path, 3(xyz)])
        data[c.INTERACTIONS_POS_PARAM_NAME][abs_idx, :n_paths, :curr_max_inter, :] = \
            np.transpose(paths_dict['vertices'][:curr_max_inter, rel_idx, t, path_idxs, :], (1,0,2))

        # Interactions types
        types = paths_dict['types'][b, path_idxs]
        inter_pos_rx = data[c.INTERACTIONS_POS_PARAM_NAME][abs_idx, :n_paths]
        interactions = get_sionna_interaction_types(types, inter_pos_rx)
        data[c.INTERACTIONS_PARAM_NAME][abs_idx, :n_paths] = interactions
    
    return inactive_count

def read_paths(load_folder: str, save_folder: str, txrx_dict: Dict) -> None:
    """Read and convert path data from Sionna format.
    
    Args:
        load_folder: Path to folder containing Sionna path files
        save_folder: Path to save converted path data
        txrx_dict: Dictionary containing TX/RX set information from read_txrx
        
    Notes:
        - Each path dictionary can contain one or more transmitters
        - Transmitters are identified by their positions across all path dictionaries
        - RX positions maintain their relative order across path dictionaries
    
    -- Information about the Sionna paths (from https://nvlabs.github.io/sionna/api/rt.html#paths) --

    [Amplitude]
    - paths_dict['a'] is the amplitude of the path
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps]

    [Delay]
    - paths_dict['tau'] is the delay of the path
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or 
        [batch_size, num_rx, num_tx, max_num_paths], float

    [Angles]
    - paths_dict['phi_r'] is the azimuth angle of the arrival of the path
    - paths_dict['theta_r'] is the elevation angle of the arrival of the path
    - paths_dict['phi_t'] is the azimuth angle of the departure of the path
    - paths_dict['theta_t'] is the elevation angle of the departure of the path
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or 
        [batch_size, num_rx, num_tx, max_num_paths], float

    [Types]
    - paths_dict['types'] is the type of the path
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or 
        [batch_size, num_rx, num_tx, max_num_paths], float

    [Vertices]
    - paths_dict['vertices'] is the vertices of the path
        [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths] or 
        [batch_size, num_rx, num_tx, max_num_paths], float

    """
    path_dict_list = cu.load_pickle(os.path.join(load_folder, 'sionna_paths.pkl'))

    # Collect all unique TX positions from all path dictionaries
    all_tx_pos = np.unique(np.vstack([paths_dict['sources'] for paths_dict in path_dict_list]), axis=0)
    n_tx = len(all_tx_pos)

    # Collect all RX positions while maintaining order and removing duplicates
    all_rx_pos = np.vstack([paths_dict['targets'] for paths_dict in path_dict_list])
    _, unique_indices = np.unique(all_rx_pos, axis=0, return_index=True)
    rx_pos = all_rx_pos[np.sort(unique_indices)]  # Sort indices to maintain original order
    n_rx = len(rx_pos)

    # Initialize inactive indices list
    rx_inactive_idxs_count = 0
    
    for tx_idx, tx_pos_target in enumerate(all_tx_pos):
        # Pre-allocate matrices
        data = _preallocate_data(n_rx)

        data[c.RX_POS_PARAM_NAME], data[c.TX_POS_PARAM_NAME] = rx_pos, tx_pos_target
        
        # Create progress bar
        pbar = tqdm(total=n_rx, desc=f"Processing receivers for TX {tx_idx}")
        
        b = 0  # batch index 
        last_idx = 0
        bs_bs_paths = False
        # Process each batch of paths
        for path_dict_idx, paths_dict in enumerate(path_dict_list):
            # Find if and where this TX exists in current paths_dict
            tx_idx_in_dict = np.where(np.all(paths_dict['sources'] == tx_pos_target, axis=1))[0]
            if len(tx_idx_in_dict) == 0:
                continue

            # Check if BS-BS paths exist (they are the first paths_dict)
            if path_dict_idx == 0:
                if np.array_equal(paths_dict['sources'], paths_dict['targets']):
                    bs_bs_paths = True
                    continue
                
            t = tx_idx_in_dict[0]  # Get the index of this TX in current paths_dict
            batch_size = paths_dict['a'].shape[1]
            
            # Get max number of interactions per path
            curr_max_inter = min(c.MAX_INTER_PER_PATH, paths_dict['vertices'].shape[0])

            # Process the batch using helper function
            inactive_count = _process_paths_batch(paths_dict, data, b, t, curr_max_inter,
                                                  last_idx, batch_size)
            
            if tx_idx == 0:  # Only count inactive RXs for first TX
                rx_inactive_idxs_count += inactive_count
            
            # Update progress bar for each receiver processed
            pbar.update(batch_size)
            last_idx += batch_size

        pbar.close()

        # Compress data before saving
        data = cu.compress_path_data(data)
        
        # Save each data key
        for key in data.keys():
            cu.save_mat(data[key], key, save_folder, 0, tx_idx, 1)  # Static for Sionna
        
        if bs_bs_paths:
            print(f'BS-BS paths found for TX {tx_idx}')
            
            paths_dict = path_dict_list[0]
            all_bs_pos = paths_dict['sources']
            num_bs = len(all_bs_pos)
            data_bs_bs = _preallocate_data(num_bs)
            data_bs_bs[c.RX_POS_PARAM_NAME] = all_bs_pos
            data_bs_bs[c.TX_POS_PARAM_NAME] = tx_pos_target
            
            # Get max number of interactions per path
            curr_max_inter = min(c.MAX_INTER_PER_PATH, paths_dict['vertices'].shape[0])

            # Process BS-BS paths using helper function
            _process_paths_batch(paths_dict, data_bs_bs, b, t, curr_max_inter, 0, num_bs)
            
            # Compress data before saving
            data_bs_bs = cu.compress_path_data(data_bs_bs)
            
            # Save each data key
            for key in data_bs_bs.keys():
                cu.save_mat(data_bs_bs[key], key, save_folder, 0, tx_idx, 0)  # Same RX & TX set
    
    if bs_bs_paths:
        txrx_dict['txrx_set_0']['is_rx'] = True  # add BS set also as RX

    # Update txrx_dict with tx and rx numbers 
    txrx_dict['txrx_set_0']['num_points'] = n_tx
    txrx_dict['txrx_set_0']['num_active_points'] = n_tx
    
    txrx_dict['txrx_set_1']['num_points'] = n_rx
    txrx_dict['txrx_set_1']['num_active_points'] = n_rx - rx_inactive_idxs_count

def get_sionna_interaction_types(types: np.ndarray, inter_pos: np.ndarray) -> np.ndarray:
    """
    Convert Sionna interaction types to DeepMIMO interaction codes.
    
    Args:
        types: Array of interaction types from Sionna (N_PATHS,)
        inter_pos: Array of interaction positions (N_PATHS x MAX_INTERACTIONS x 3)

    Returns:
        np.ndarray: Array of DeepMIMO interaction codes (N_PATHS,)
    """
    # Ensure types is a numpy array
    types = np.asarray(types)
    if types.ndim == 0:
        types = np.array([types])
    
    # Get number of paths
    n_paths = len(types)
    result = np.zeros(n_paths, dtype=np.float32)
    
    # For each path
    for path_idx in range(n_paths):
        # Skip if no type (nan or 0)
        if np.isnan(types[path_idx]) or types[path_idx] == 0:
            continue
            
        sionna_type = int(types[path_idx])
        
        # Handle LoS case (type 0)
        if sionna_type == 0:
            result[path_idx] = c.INTERACTION_LOS
            continue
            
        # Count number of actual interactions by checking non-nan positions
        if inter_pos.ndim == 2:  # Single path case
            n_interactions = np.nansum(~np.isnan(inter_pos[:, 0]))
        else:  # Multiple paths case
            n_interactions = np.nansum(~np.isnan(inter_pos[path_idx, :, 0]))
            
        if n_interactions == 0:  # Skip if no interactions
            continue
            
        # Handle different Sionna interaction types
        if sionna_type == 1:  # Pure reflection path
            # Create string of '1's with length = number of reflections
            code = '1' * n_interactions
            result[path_idx] = np.float32(code)
            
        elif sionna_type == 2:  # Single diffraction path
            # Always just '2' since Sionna only allows single diffraction
            result[path_idx] = c.INTERACTION_DIFFRACTION
            
        elif sionna_type == 3:  # Scattering path with possible reflections
            # Create string of '1's for reflections + '3' at the end for scattering
            if n_interactions > 1:
                code = '1' * (n_interactions - 1) + '3'
            else:
                code = '3'
            result[path_idx] = np.float32(code)
            
        else:
            if sionna_type == 4:
                raise NotImplementedError('RIS code not supported yet')
            else:
                raise ValueError(f'Unknown Sionna interaction type: {sionna_type}')
    
    return result 