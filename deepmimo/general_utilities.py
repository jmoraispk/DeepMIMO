"""
General utility functions and classes for the DeepMIMO dataset generation.

This module provides utility functions and classes for handling printing,
file naming, and string ID generation used across the DeepMIMO toolkit.
"""

class PrintIfVerbose:
    """A callable class that conditionally prints messages based on verbosity setting.
    
    Args:
        verbose (bool): Flag to control whether messages should be printed.
    """
    
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
    
    def __call__(self, message: str) -> None:
        """Print the message if verbose mode is enabled.
        
        Args:
            message (str): The message to potentially print.
        """
        if self.verbose:
            print(message)

def get_txrx_str_id(tx_set_idx: int, tx_idx: int, rx_set_idx: int) -> str:
    """Generate a standardized string identifier for TX-RX combinations.
    
    Args:
        tx_set_idx (int): Index of the transmitter set.
        tx_idx (int): Index of the transmitter within its set.
        rx_set_idx (int): Index of the receiver set.
    
    Returns:
        str: Formatted string identifier in the form 't{tx_set_idx}_tx{tx_idx}_r{rx_set_idx}'.
    """
    return f't{tx_set_idx:03}_tx{tx_idx:03}_r{rx_set_idx:03}'

def get_mat_filename(key: str, tx_set_idx: int, tx_idx: int, rx_set_idx: int) -> str:
    """Generate a .mat filename for storing DeepMIMO data.
    
    Args:
        key (str): The key identifier for the data type.
        tx_set_idx (int): Index of the transmitter set.
        tx_idx (int): Index of the transmitter within its set.
        rx_set_idx (int): Index of the receiver set.
    
    Returns:
        str: Complete filename with .mat extension.
    """
    str_id = get_txrx_str_id(tx_set_idx, tx_idx, rx_set_idx)
    return f'{key}_{str_id}.mat'