"""
TX/RX set handling for Wireless Insite conversion.
"""
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

from .setup_parser import parse_file

@dataclass
class InsiteTxRxSet():
    """
    TX/RX set class
    """
    name: str = ''
    id: int = 0   # Wireless Insite ID 
    idx: int = 0  # TxRxSet index for saving after conversion and generation
    # id -> idx example: [3, 5, 7] -> [1, 2, 3]
    is_tx: bool = False
    is_rx: bool = False
    
    num_points: int = 0    # all points
    inactive_idxs: tuple = ()  # list of indices of points with at least one path
    num_inactive_points: int = 0
    
    # Antenna elements of tx / rx
    tx_num_ant: int = 1
    rx_num_ant: int = 1
    
    dual_pol: bool = False # if '_dual-pol' in name


def read_txrx(txrx_file, verbose: bool) -> Tuple[List[int], List[int], Dict]:
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
        txrx_obj.id = insite_id 
        # TX/RX set ID is used to abstract from the ray tracing configurations
        # (how the DeepMIMO dataset will be saved and generated)
        txrx_obj.idx = txrx_set_idx + 1 # 1-indexed
        
        # Locations
        txrx_obj.loc_lat = txrx.values['location'].values['reference'].values['latitude']
        txrx_obj.loc_lon = txrx.values['location'].values['reference'].values['longitude']
        txrx_obj.coord_ref = txrx.values['location'].values['reference'].labels[1]
        
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
    
    txrx_dict = {}
    for obj in txrx_objs:
        # Remove 'None' from dict (to be saved as .mat)
        obj_dict = {key: val for key, val in asdict(obj).items() if val is not None}
        
        # Index separate txrx-sets based on p_id
        txrx_dict = {**txrx_dict, **{f'txrx_set_{obj.idx}': obj_dict}}

    return tx_ids, rx_ids, txrx_dict


def get_id_to_idx_map(txrx_dict: Dict):
    ids = [txrx_dict[key]['id'] for key in txrx_dict.keys()]
    idxs = [i + 1 for i in range(len(ids))]
    return {key:val for key, val in zip(ids, idxs)} 