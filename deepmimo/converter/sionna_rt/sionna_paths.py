"""
Sionna Ray Tracing Paths Module.

This module handles loading and converting path data from Sionna's format to DeepMIMO's format.
"""

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

def read_paths(load_folder: str, save_folder: str, txrx_dict: Dict) -> None:
    """Read and convert path data from Sionna format.
    
    Args:
        load_folder: Path to folder containing Sionna path files
        save_folder: Path to save converted path data
        txrx_dict: Dictionary containing TX/RX set information from read_txrx
        
    Notes:
        Current assumptions about TX/RX positions:
        - Multiple TX positions are supported, but all TX positions must be present
          in each paths dictionary (i.e., same set of TXs for different RX batches)
        - RX positions can vary across path dictionaries and are tracked using 
          sequential indices (e.g., first batch has RXs 0-9, second has 10-22, etc.)
    """
    path_dict_list = cu.load_pickle(load_folder + 'sionna_paths.pkl')

    # Get TX positions from first paths dictionary
    # NOTE: All paths dictionaries must contain the same TX positions
    tx_pos = path_dict_list[0]['sources']
    
    # Verify all paths dictionaries have the same TX positions
    for paths_dict in path_dict_list[1:]:
        if not np.array_equal(paths_dict['sources'], tx_pos):
            raise ValueError("Found different TX positions across paths. "
                             "All paths must contain the same set of TX positions.")

    # Concatenate all RX positions in order
    rx_pos = np.vstack([paths_dict['targets'] for paths_dict in path_dict_list])
    n_rx = rx_pos.shape[0]

    # Pre-allocate matrices
    max_iteract = min(c.MAX_INTER_PER_PATH, path_dict_list[0]['vertices'].shape[0])

    data = {
        'rx_pos': rx_pos,
        'tx_pos': tx_pos,
        'aoa_az': np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'aoa_el': np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'aod_az': np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'aod_el': np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'delay':  np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'power':  np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'phase':  np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'inter':  np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'inter_pos': np.zeros((n_rx, c.MAX_PATHS, max_iteract, 3), dtype=np.float32) * np.nan,
    }

    # Squeeze and slice function to be applied to each Sionna array
    ss = lambda array: array.squeeze()[..., :c.MAX_PATHS]

    # Create progress bar
    pbar = tqdm(total=n_rx, desc="Processing receivers")
    
    last_idx = 0
    inactive_idxs = []
    # Process each batch of paths
    for paths_dict in path_dict_list:
        batch_size = paths_dict['a'].shape[1]

        # Get absolute indices for this batch
        abs_idxs = np.arange(batch_size) + last_idx
        last_idx += batch_size
        
        a = ss(paths_dict['a'])
        not_nan_mask = a != 0

        # Process each field with proper masking
        for rel_idx, path_mask in enumerate(not_nan_mask):
            if rel_idx >= batch_size:
                break
                
            abs_idx = abs_idxs[rel_idx]
            n_paths = np.sum(path_mask)

            if n_paths == 0:
                inactive_idxs.append(abs_idx)
                continue

            data['power'][abs_idx,:n_paths] = 20 * np.log10(np.absolute(a[rel_idx][path_mask]))
            data['phase'][abs_idx,:n_paths] = np.angle(a[rel_idx][path_mask], deg=True)
            data['delay'][abs_idx,:n_paths] = ss(paths_dict['tau'])[rel_idx][path_mask]
            data['aoa_az'][abs_idx,:n_paths] = ss(paths_dict['phi_r'])[rel_idx][path_mask] * 180 / np.pi
            data['aod_az'][abs_idx,:n_paths] = ss(paths_dict['phi_t'])[rel_idx][path_mask] * 180 / np.pi
            data['aoa_el'][abs_idx,:n_paths] = ss(paths_dict['theta_r'])[rel_idx][path_mask] * 180 / np.pi
            data['aod_el'][abs_idx,:n_paths] = ss(paths_dict['theta_t'])[rel_idx][path_mask] * 180 / np.pi

            # Handle interactions for this receiver
            types = ss(paths_dict['types'])[path_mask]
            inter_pos_rx = data['inter_pos'][abs_idx, :n_paths]
            interactions = get_sionna_interaction_types(types, inter_pos_rx)
            data['inter'][abs_idx, :n_paths] = interactions

            # Update progress bar for each receiver processed
            pbar.update(1)

        # Handle interaction positions
        inter_pos = paths_dict['vertices'].squeeze()[:max_iteract, :, :c.MAX_PATHS, :]
        data['inter_pos'][abs_idxs, :len(path_mask), :inter_pos.shape[0]] = np.transpose(inter_pos, (1,2,0,3))

    pbar.close()

    # Compress data before saving
    data = cu.compress_path_data(data)
    
    # Save each data key
    for key in data.keys():
        cu.save_mat(data[key], key, save_folder, 1, 1, 2) # Static for Sionna (for now - later use txrx_dict)

    # Update txrx_dict with tx and rx numbers 
    
    txrx_dict['txrx_set_1']['num_points'] = tx_pos.shape[0]
    txrx_dict['txrx_set_1']['inactive_idxs'] = np.array([])
    txrx_dict['txrx_set_1']['num_active_points'] = tx_pos.shape[0]
    
    txrx_dict['txrx_set_2']['num_points'] = n_rx
    txrx_dict['txrx_set_2']['inactive_idxs'] = np.array(inactive_idxs)
    txrx_dict['txrx_set_2']['num_active_points'] = n_rx - len(inactive_idxs)

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