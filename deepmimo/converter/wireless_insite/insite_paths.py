"""
Path data handling for Wireless Insite conversion.

This module provides functionality for parsing and saving path data from Wireless Insite
path files (.paths.p2m).
"""

from pathlib import Path
from typing import Dict

from .paths_parser import paths_parser
from .insite_txrx import get_id_to_idx_map
from ..converter_utils import save_mat


def read_paths(p2m_folder: str, output_folder: str, txrx_dict: Dict) -> None:
    """Create path data from a folder containing Wireless Insite files.
    
    This function:
    1. Uses provided TX/RX set configurations
    2. Finds all path files for each TX/RX pair
    3. Parses and saves path data for each pair
    
    Args:
        p2m_folder: Path to folder containing .p2m files
        txrx_dict: Dictionary containing TX/RX set information from read_txrx
        output_folder: Path to folder where .mat files will be saved

    Raises:
        ValueError: If folder doesn't exist or required files not found
    """
    p2m_folder = Path(p2m_folder)
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
                
                # Get indices for saving
                tx_set_idx = id_to_idx_map[tx_id]
                rx_set_idx = id_to_idx_map[rx_id]
                
                # Save each data key
                for key in data.keys():
                    save_mat(data[key], key, output_folder, tx_set_idx, tx_idx, rx_set_idx)


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