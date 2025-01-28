"""
General utility functions and classes for the DeepMIMO dataset generation.

This module provides utility functions and classes for handling printing,
file naming, string ID generation, and dictionary utilities used across 
the DeepMIMO toolkit.
"""

from typing import Dict, Any

class DotDict:
    """A dictionary subclass that supports dot notation access to nested dictionaries.
    
    This class allows accessing dictionary items using both dictionary notation (d['key'])
    and dot notation (d.key). It automatically converts nested dictionaries to DotDict
    instances to maintain dot notation access at all levels.
    
    Example:
        >>> d = DotDict({'a': 1, 'b': {'c': 2}})
        >>> d.a
        1
        >>> d.b.c
        2
        >>> d['b']['c']
        2
    """
    def __init__(self, dictionary: Dict[str, Any]):
        """Initialize DotDict with a dictionary.
        
        Args:
            dictionary: Dictionary to convert to DotDict
        """
        self._data = {}
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self._data[key] = DotDict(value)
            else:
                self._data[key] = value
                
    def __getattr__(self, key):
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key)
    
    def __setattr__(self, key, value):
        if key == '_data':
            super().__setattr__(key, value)
        else:
            if isinstance(value, dict):
                self._data[key] = DotDict(value)
            else:
                self._data[key] = value
                
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        if isinstance(value, dict):
            self._data[key] = DotDict(value)
        else:
            self._data[key] = value
            
    def to_dict(self) -> Dict:
        """Convert DotDict back to a regular dictionary.
        
        Returns:
            dict: Regular dictionary representation
        """
        result = {}
        for key, value in self._data.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

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