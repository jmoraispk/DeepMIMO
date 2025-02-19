"""
General utility functions and classes for the DeepMIMO dataset generation.

This module provides utility functions and classes for handling printing,
file naming, string ID generation, and dictionary utilities used across 
the DeepMIMO toolkit.
"""

import numpy as np
import scipy.io
from pprint import pformat
from typing import Dict, Any, TypeVar, Mapping, Optional

K = TypeVar('K', bound=str)
V = TypeVar('V')

class DotDict(Mapping[K, V]):
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
        >>> list(d.keys())
        ['a', 'b']
    """
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize DotDict with a dictionary.
    
        Args:
            dictionary: Dictionary to convert to DotDict
        """
        # Store protected attributes in a set
        self._data = {}
        if data:
            for key, value in data.items():
                if isinstance(value, dict):
                    self._data[key] = DotDict(value)
                else:
                    self._data[key] = value

    def __getattr__(self, key: str) -> Any:
        """Enable dot notation access to dictionary items."""
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        """Enable dot notation assignment."""
        if key == '_data':
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dictionary-style assignment."""
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        self._data[key] = value

    def update(self, other: Dict[str, Any]) -> None:
        """Update the dictionary with elements from another dictionary."""
        # Convert any nested dicts to DotDicts first
        processed = {
            k: DotDict(v) if isinstance(v, dict) and not isinstance(v, DotDict) else v 
            for k, v in other.items()
        }
        self._data.update(processed)

    def __len__(self) -> int:
        """Return the length of the underlying data dictionary."""
        return len(self._data)
        
    def __iter__(self):
        """Return an iterator over the data dictionary keys."""
        return iter(self._data)
        
    def __dir__(self):
        """Return list of valid attributes."""
        return list(set(
            list(super().__dir__()) + 
            list(self._data.keys())
        ))
        
    @property
    def shape(self):
        """Return shape of the first array-like value in the dictionary."""
        for val in self._data.values():
            if hasattr(val, 'shape'):
                return val.shape
        return None
        
    @property
    def size(self):
        """Return size of the first array-like value in the dictionary."""
        for val in self._data.values():
            if hasattr(val, 'size'):
                return val.size
        return None

    def keys(self):
        """Return dictionary keys."""
        return self._data.keys()

    def values(self):
        """Return dictionary values."""
        return self._data.values()

    def items(self):
        """Return dictionary items as (key, value) pairs."""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value for key, returning default if key doesn't exist."""
        return self._data.get(key, default)
            
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
    
    def __repr__(self) -> str:
        """Return string representation of dictionary."""
        return pformat(self._data)

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

def load_mat_file_as_dict(file_path: str) -> Dict[str, Any]:
    """Load MATLAB .mat file as Python dictionary.
    
    Args:
        file_path (str): Path to .mat file to load
        
    Returns:
        dict: Dictionary containing loaded MATLAB data
        
    Raises:
        ValueError: If file cannot be loaded
    """
    mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    return {key: mat_struct_to_dict(value) for key, value in mat_data.items()
            if not key.startswith('__')}

def mat_struct_to_dict(mat_struct: Any) -> Dict[str, Any]:
    """Convert MATLAB structure to Python dictionary.
    
    This function recursively converts MATLAB structures and arrays to
    Python dictionaries and numpy arrays.

    Args:
        mat_struct (any): MATLAB structure to convert
        
    Returns:
        dict: Dictionary containing converted data
    """
    if isinstance(mat_struct, scipy.io.matlab.mat_struct):
        result = {}
        for field in mat_struct._fieldnames:
            result[field] = mat_struct_to_dict(getattr(mat_struct, field))
        return result
    elif isinstance(mat_struct, np.ndarray):
        # Process arrays recursively in case they contain mat_structs
        try:
            # First try to convert directly to numpy array
            return np.array([mat_struct_to_dict(item) for item in mat_struct])
        except ValueError:
            # If that fails due to inhomogeneous shapes, return as list instead
            return [mat_struct_to_dict(item) for item in mat_struct]
    return mat_struct  # Return the object as is for other types