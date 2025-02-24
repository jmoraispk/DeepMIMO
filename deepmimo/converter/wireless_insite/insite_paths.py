"""
Path data handling for Wireless Insite conversion.

This module provides high-level functionality for processing path data from Wireless Insite
path files (.paths.p2m). It serves as a bridge between the low-level p2m file parsing
and the high-level scenario conversion by:

1. Managing TX/RX point information and mapping
2. Coordinating path data extraction for all TX/RX pairs
3. Saving path data in DeepMIMO format

Dependencies:
- p2m_parser.py: Low-level parsing of .p2m files
- converter_utils.py: Utility functions for data conversion and saving

Main Functions:
    read_paths(): Process and save path data for all TX/RX pairs
    get_id_to_idx_map(): Map Wireless Insite IDs to DeepMIMO indices
    update_txrx_points(): Update TX/RX point information with path data
"""

from pathlib import Path
from typing import Dict
import numpy as np

from .p2m_parser import paths_parser, extract_tx_pos, read_pl_p2m_file
from .. import converter_utils as cu
from ... import consts as c


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
    txrx_dict[f'txrx_set_{rx_set_idx}']['num_active_points'] = n_points - len(inactive_idxs)


def read_paths(rt_folder: str, output_folder: str, txrx_dict: Dict) -> None:
    """Create path data from a folder containing Wireless Insite files.
    
    This function:
    1. Uses provided TX/RX set configurations
    2. Finds all path files for each TX/RX pair
    3. Parses and saves path data for each pair
    4. Extracts and saves position information
    5. Updates TX/RX point information
    
    Args:
        rt_folder: Path to folder containing .setup, .txrx, and material files
        txrx_dict: Dictionary containing TX/RX set information from read_txrx
        output_folder: Path to folder where .mat files will be saved

    Raises:
        ValueError: If folder doesn't exist or required files not found
    """
    p2m_folder = next(p for p in Path(rt_folder).iterdir() if p.is_dir())
    if not p2m_folder.exists():
        raise ValueError(f"Folder does not exist: {p2m_folder}")
    
    # Get TX/RX IDs from dictionary
    tx_ids = []
    rx_ids = []
    for key, set_info in txrx_dict.items():
        if set_info['is_tx']:
            tx_ids.append(set_info['id_orig'])
        if set_info['is_rx']:
            rx_ids.append(set_info['id_orig'])
    
    # Get ID to index mapping
    id_to_idx_map = get_id_to_idx_map(txrx_dict)
    
    # Find any p2m file to extract project name
    # Format is: project_name.paths.t001_01.r001.p2m
    proj_name = list(p2m_folder.glob("*.p2m"))[0].name.split('.')[0]
    
    # Process each TX/RX pair
    for tx_id in tx_ids:
        for rx_id in rx_ids:
            # Generate filenames
            for tx_idx, tx_num in enumerate([1]):  # We assume each TX/RX SET only has one BS
                # Generate paths filename
                base_filename = f'{proj_name}.paths.t{tx_num:03}_{tx_id:02}.r{rx_id:03}.p2m'
                paths_p2m_file = p2m_folder / base_filename
                
                if not paths_p2m_file.exists():
                    print(f"Warning: Path file not found: {paths_p2m_file}")
                    continue
                
                # Parse path data
                data = paths_parser(str(paths_p2m_file))
                
                # Extract TX positions
                data[c.TX_POS_PARAM_NAME] = extract_tx_pos(str(paths_p2m_file))
                
                # Extract RX positions and path loss from .pl.p2m file
                pl_p2m_file = str(paths_p2m_file).replace('.paths.', '.pl.')
                data[c.RX_POS_PARAM_NAME], _, path_loss = read_pl_p2m_file(pl_p2m_file)
                
                # Get indices for saving
                tx_set_idx = id_to_idx_map[tx_id]
                rx_set_idx = id_to_idx_map[rx_id]
                
                # Update TX/RX point information
                update_txrx_points(txrx_dict, rx_set_idx, data[c.RX_POS_PARAM_NAME], path_loss)

                # Save each data key
                for key in data.keys():
                    cu.save_mat(data[key], key, output_folder, tx_set_idx, tx_idx, rx_set_idx)


if __name__ == "__main__":
    # Test directory with path files
    test_dir = r"./P2Ms/simple_street_canyon_test/"
    p2m_folder = r"./P2Ms/simple_street_canyon_test/p2m"
    output_folder = r"./P2Ms/simple_street_canyon_test/mat_files"

    print(f"\nTesting path data extraction from: {test_dir}")
    print("-" * 50)
    
    # First get TX/RX information
    from .insite_txrx import read_txrx
    txrx_dict = read_txrx(test_dir, p2m_folder, output_folder)
    
    # Create path data from test directory
    read_paths(p2m_folder, txrx_dict, output_folder) 