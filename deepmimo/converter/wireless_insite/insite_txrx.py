"""
TX/RX handling for Wireless Insite conversion.

This module provides functionality for parsing TX/RX configurations from Wireless Insite files
and converting them to the base TxRxSet format.
"""

import os
import re
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from .setup_parser import parse_file
from .paths_parser import extract_tx_pos
from ... import consts as c
from ..converter_utils import save_mat


@dataclass
class InsiteTxRxSet:
    """
    TX/RX set class for Wireless Insite
    """
    name: str = ''
    id_orig: int = 0   # Original Wireless Insite ID 
    idx: int = 0  # TxRxSet index for saving after conversion and generation
    # id_orig -> idx example: [3, 5, 7] -> [1, 2, 3]
    is_tx: bool = False
    is_rx: bool = False
    
    num_points: int = 0    # all points
    inactive_idxs: tuple = ()  # list of indices of points with at least one path
    num_active_points: int = 0  # number of points with at least one path
    
    # Antenna elements of tx / rx
    tx_num_ant: int = 1
    rx_num_ant: int = 1
    
    dual_pol: bool = False # if '_dual-pol' in name


def read_txrx(sim_folder: str, p2m_folder: str, output_folder: str) -> Dict:
    """Create TX/RX information from a folder containing Wireless Insite files.
    
    This function:
    1. Reads the .txrx file to get TX/RX set configurations
    2. Creates position matrices for each TX/RX pair
    3. Returns a dictionary with all TX/RX information
    
    Args:
        sim_folder: Path to simulation folder containing .txrx file
        p2m_folder: Path to folder containing .p2m files
        output_folder: Path to folder where .mat files will be saved

    Returns:
        Dictionary containing TX/RX set information and indices
        
    Raises:
        ValueError: If folders don't exist or required files not found
    """
    sim_folder = Path(sim_folder)
    p2m_folder = Path(p2m_folder)
    if not sim_folder.exists():
        raise ValueError(f"Simulation folder does not exist: {sim_folder}")
    if not p2m_folder.exists():
        raise ValueError(f"P2M folder does not exist: {p2m_folder}")
    
    # Find .txrx file
    txrx_files = list(sim_folder.glob("*.txrx"))
    if not txrx_files:
        raise ValueError(f"No .txrx file found in {sim_folder}")
    if len(txrx_files) > 1:
        raise ValueError(f"Multiple .txrx files found in {sim_folder}")
    
    # Parse TX/RX sets
    tx_ids, rx_ids, txrx_dict = read_txrx_file(str(txrx_files[0]))

    # Process each TX/RX pair
    proj_name = sim_folder.name
    for tx_id in tx_ids:
        for rx_id in rx_ids:
            # Generate filenames
            for tx_idx, tx_num in enumerate([1]):  # We assume each TX/RX SET only has one BS
                base_filename = f'{proj_name}.pl.t{tx_num:03}_{tx_id:02}.r{rx_id:03}.p2m'
                pl_p2m_file = p2m_folder / base_filename
                
                # Extract positions and path loss
                rx_pos, _, path_loss = read_pl_p2m_file(str(pl_p2m_file))
                
                # Update TX/RX set information
                rx_set_idx = get_id_to_idx_map(txrx_dict)[rx_id]
                update_txrx_points(txrx_dict, rx_set_idx, rx_pos, path_loss)
                
                # Extract TX positions
                paths_p2m_file = str(pl_p2m_file).replace('.pl.', '.paths.')
                tx_pos = extract_tx_pos(paths_p2m_file)
                
                tx_set_idx = get_id_to_idx_map(txrx_dict)[tx_id]
                save_mat(rx_pos, c.RX_POS_PARAM_NAME, output_folder, tx_set_idx, tx_idx, rx_set_idx)
                save_mat(tx_pos, c.TX_POS_PARAM_NAME, output_folder, tx_set_idx, tx_idx, rx_set_idx)
    
    return txrx_dict


def read_txrx_file(txrx_file: str) -> Tuple[List[int], List[int], Dict]:
    """Parse a Wireless Insite .txrx file.
    
    Args:
        txrx_file: Path to .txrx file
        
    Returns:
        Tuple containing:
        - List of transmitter IDs
        - List of receiver IDs
        - Dictionary containing TX/RX set information
    """
    print(f'Reading txrx file: {os.path.basename(txrx_file)}')
    document = parse_file(txrx_file)
    tx_ids, rx_ids = [], []
    txrx_objs = []
    
    for txrx_set_idx, key in enumerate(document.keys()):
        txrx = document[key]
        txrx_obj = InsiteTxRxSet()
        txrx_obj.name = key
        
        # Insite ID is used during ray tracing
        insite_id = (int(txrx.name[-1]) if txrx.name.startswith('project_id')
                     else txrx.values['project_id'])
        txrx_obj.id_orig = insite_id 
        # TX/RX set ID is used to abstract from the ray tracing configurations
        # (how the DeepMIMO dataset will be saved and generated)
        txrx_obj.idx = txrx_set_idx + 1 # 1-indexed
        
        # Is TX or RX?
        txrx_obj.is_tx = txrx.values['is_transmitter']
        txrx_obj.is_rx = txrx.values['is_receiver']
        
        # Antennas and Power
        if txrx_obj.is_tx:
            tx_ids += [insite_id]
            tx_vals = txrx.values['transmitter']
            assert tx_vals.values['power'] == 0.0, 'Tx power should be 0 dBm!'
            txrx_obj.tx_num_ant = tx_vals['pattern'].values['antenna']
        if txrx_obj.is_rx:
            rx_ids += [insite_id]
            rx_vals = txrx.values['receiver']
            txrx_obj.rx_num_ant = rx_vals['pattern'].values['antenna']
        
        # The number of tx/rx points inside set is updated when reading the p2m
        txrx_objs.append(txrx_obj)
    
    # Create txrx_sets dictionary with idx-based keys (as integers)
    txrx_sets = {}
    for obj in txrx_objs:
        # Remove 'None' from dict (to be saved as .mat)
        obj_dict = {key: val for key, val in asdict(obj).items() if val is not None}
        txrx_sets[f'txrx_set_{obj.idx}'] = obj_dict

    return tx_ids, rx_ids, txrx_sets


def update_txrx_points(txrx_dict: Dict, rx_set_idx: int, rx_pos: np.ndarray, path_loss: np.ndarray) -> None:
    """Update TxRx set information with point counts and inactive indices.
    
    Args:
        txrx_dict: Dictionary containing TxRx set information
        rx_set_idx: Index of the receiver set to update
        rx_pos: Array of receiver positions
        path_loss: Array of path loss values
    """
    # Update number of points
    n_points = rx_pos.shape[0]
    txrx_dict[f'txrx_set_{rx_set_idx}']['num_points'] = n_points
    
    # Find inactive points (those with path loss of 250 dB)
    inactive_idxs = np.where(path_loss == 250.)[0]
    txrx_dict[f'txrx_set_{rx_set_idx}']['inactive_idxs'] = inactive_idxs
    txrx_dict[f'txrx_set_{rx_set_idx}']['num_active_points'] = n_points - len(inactive_idxs)


def get_id_to_idx_map(txrx_dict: Dict) -> Dict[int, int]:
    """Create mapping from Wireless Insite IDs to indices.
    
    Args:
        txrx_dict: Dictionary containing TX/RX set information
        
    Returns:
        Dictionary mapping original IDs to indices
    """
    ids = [txrx_dict[key]['id_orig'] for key in txrx_dict.keys()]
    idxs = [i + 1 for i in range(len(ids))]
    return {key:val for key, val in zip(ids, idxs)}


def read_pl_p2m_file(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read position and path loss data from a .p2m file.
    
    Args:
        filename: Path to the .p2m file to read
        
    Returns:
        Tuple containing:
        - xyz: Array of positions with shape (n_points, 3)
        - dist: Array of distances with shape (n_points, 1)
        - path_loss: Array of path losses with shape (n_points, 1)
    """
    assert filename.endswith('.p2m') # should be a .p2m file
    assert '.pl.' in filename        # should be the pathloss p2m

    # Initialize empty lists for matrices
    xyz_list = []
    dist_list = []
    path_loss_list = []

    # Define (regex) patterns to match numbers (optionally signed floats)
    re_data = r"-?\d+\.?\d*"
    
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    
    for line in lines:
        if line[0] != '#':
            data = re.findall(re_data, line)
            xyz_list.append([float(data[1]), float(data[2]), float(data[3])]) # XYZ (m)
            dist_list.append([float(data[4])])       # distance (m)
            path_loss_list.append([float(data[5])])  # path loss (dB)

    # Convert lists to numpy arrays
    xyz_matrix = np.array(xyz_list, dtype=np.float32)
    dist_matrix = np.array(dist_list, dtype=np.float32)
    path_loss_matrix = np.array(path_loss_list, dtype=np.float32)

    return xyz_matrix, dist_matrix, path_loss_matrix


if __name__ == "__main__":
    # Test directory with TX/RX files
    test_dir = r"./P2Ms/simple_street_canyon_test/"
    
    print(f"\nTesting TX/RX set extraction from: {test_dir}")
    print("-" * 50)
    
    # Find p2m folder
    p2m_folder = None
    for subdir in Path(test_dir).iterdir():
        if subdir.is_dir() and 'p2m' in subdir.name.lower():
            p2m_folder = subdir
            break
    if not p2m_folder:
        print(f"No P2M folder found in {test_dir}")
        exit(1)
    
    # Create TX/RX information
    txrx_dict = read_txrx(test_dir, p2m_folder)
    
    # Print summary
    print("\nSummary:")
    tx_ids = []
    rx_ids = []
    for key, set_info in txrx_dict.items():
        if set_info['is_tx']:
            tx_ids.append(set_info['id_orig'])
        if set_info['is_rx']:
            rx_ids.append(set_info['id_orig'])
            
    print(f"Found {len(tx_ids)} transmitter sets: {tx_ids}")
    print(f"Found {len(rx_ids)} receiver sets: {rx_ids}")
    print("\nTX/RX Sets Details:")
    for set_key, set_info in txrx_dict.items():
        print(f"\n{set_key}:")
        print(f"  Original ID: {set_info['id_orig']}")
        print(f"  Is TX: {set_info.get('is_tx', False)}")
        print(f"  Is RX: {set_info.get('is_rx', False)}")
        if 'tx_num_ant' in set_info:
            print(f"  TX antennas: {set_info['tx_num_ant']}")
        if 'rx_num_ant' in set_info:
            print(f"  RX antennas: {set_info['rx_num_ant']}") 