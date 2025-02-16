"""
Sionna Ray Tracing TX/RX Module.

This module handles loading and converting transmitter and receiver data from Sionna's format to DeepMIMO's format.
"""

from typing import Dict
from ...txrx import TxRxSet

def read_txrx(setup_dict: Dict) -> Dict:
    """Read and convert TX/RX data from Sionna format.
    
    Args:
        setup_dict: Dictionary containing Sionna setup parameters
        
    Returns:
        Dict containing TX/RX configuration in DeepMIMO format
    """
    txrx_dict = {}
    # Create TX and RX objects in a loop
    for i in range(2):
        is_tx = (i == 0)  # First iteration is TX, second is RX
        obj = TxRxSet()
        obj.is_tx = is_tx
        obj.is_rx = not is_tx
        
        obj.name = 'tx_array' if is_tx else 'rx_array'
        obj.id_orig = i + 1
        obj.idx = i + 1  # 1-indexed
        
        # Set antenna properties        
        obj.num_ant = 1 if setup_dict['array_synthetic'] else setup_dict[obj.name + '_num_ant']
        obj.ant_rel_positions = setup_dict[obj.name + '_ant_pos']        
        obj.dual_pol = setup_dict[obj.name + '_num_ant'] != setup_dict[obj.name + '_size']

        txrx_dict[f'txrx_set_{i+1}'] = obj.to_dict() # 1-indexed

    return txrx_dict 