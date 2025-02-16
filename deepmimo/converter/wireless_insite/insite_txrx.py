"""
TX/RX handling for Wireless Insite conversion.

This module provides functionality for parsing TX/RX configurations from Wireless Insite files
and converting them to the base TxRxSet format.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

from .setup_parser import parse_file
from ...txrx import TxRxSet


def read_txrx(sim_folder: str, p2m_folder: str) -> Dict:
    """Create TX/RX information from a folder containing Wireless Insite files.
    
    This function:
    1. Reads the .txrx file to get TX/RX set configurations
    2. Returns a dictionary with all TX/RX information
    
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
        txrx_obj = TxRxSet()
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
        if txrx_obj.is_rx:
            rx_ids += [insite_id]
        
        ant_dict = txrx.values['receiver'] if txrx_obj.is_rx else txrx.values['transmitter']
        txrx_obj.num_ant = ant_dict['pattern'].values['antenna']
        
        txrx_obj.dual_pol = False # for now, only single polarization is supported

        # The number of tx/rx points inside set is updated when reading the p2m
        txrx_objs.append(txrx_obj)
    
    # Create txrx_sets dictionary with idx-based keys (as integers)
    txrx_sets = {}
    for obj in txrx_objs:
        txrx_sets[f'txrx_set_{obj.idx}'] = obj.to_dict()

    return tx_ids, rx_ids, txrx_sets


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


if __name__ == "__main__":
    # Test directory with TX/RX files
    test_dir = r"./P2Ms/simple_street_canyon_test/"
    p2m_folder = r"./P2Ms/simple_street_canyon_test/p2m"
    output_folder = r"./P2Ms/simple_street_canyon_test/mat_files"
    
    print(f"\nTesting TX/RX set extraction from: {test_dir}")
    print("-" * 50)
    
    # Create TX/RX information
    txrx_dict = read_txrx(test_dir, p2m_folder, output_folder)
    
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