
import os

class PrintIfVerbose():
    def __init__(self, verbose):
        self.verbose = verbose
    
    def __call__(self, s):
        if self.verbose:
            print(s)

def get_txrx_str_id(tx_set_idx: int, tx_idx: int, rx_set_idx: int):
    return f't{tx_set_idx:03}_tx{tx_idx:03}_r{rx_set_idx:03}'

def get_mat_filename(key: str, tx_set_idx: int, tx_idx: int, rx_set_idx: int):
    str_id = get_txrx_str_id(tx_set_idx, tx_idx, rx_set_idx)
    return f'{key}_{str_id}.mat'