"""
TX/RX set handling for DeepMIMO.

This module provides the base TxRxSet class used by all ray tracing converters
to represent transmitter and receiver configurations.
"""

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class TxRxSet:
    """
    Base TX/RX set class for DeepMIMO.
    
    This class represents a set of transmitters or receivers in a ray tracing simulation.
    It is used by all ray tracing converters (Wireless Insite, Sionna, etc.) to store
    TX/RX configuration information during conversion.
    
    Example:
        Wireless Insite IDXs = [3, 7, 8]
        DeepMIMO (after conversion) TX/RX Sets: [1, 2, 3]
        DeepMIMO (after generation): only individual tx and rx indices
    """
    name: str = ''
    id_orig: int = 0   # Original ray tracer ID 
    idx: int = 0       # TxRxSet index for saving after conversion and generation
    is_tx: bool = False
    is_rx: bool = False
    
    num_points: int = 0    # all points
    inactive_idxs: tuple = ()  # list of indices of points with at least one path
    num_active_points: int = 0  # number of points with at least one path
    
    # Antenna elements of tx / rx
    tx_num_ant: int = 1
    rx_num_ant: int = 1
    
    dual_pol: bool = False # if antenna supports dual polarization
    
    def to_dict(self) -> Dict:
        """Convert TxRxSet to a dictionary, filtering out None values.
        
        Returns:
            Dictionary containing all non-None attributes of the TxRxSet.
        """
        return {key: val for key, val in asdict(self).items() if val is not None}

