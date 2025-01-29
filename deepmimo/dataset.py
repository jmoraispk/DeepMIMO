"""
Dataset module for DeepMIMO.

This module provides the Dataset class for managing DeepMIMO datasets,
including channel matrices, path information, and metadata.
"""

from typing import Dict, Any, Optional
import numpy as np
from .general_utilities import DotDict

class Dataset(DotDict):
    """Class for managing DeepMIMO datasets.
    
    This class provides an interface for accessing dataset attributes including:
    - Channel matrices
    - Path information (angles, powers, delays)
    - Position information
    - Metadata
    
    Attributes can be accessed using both dot notation (dataset.channel) 
    and dictionary notation (dataset['channel']).
    
    Common attributes:
        channel: MIMO channel matrices
        power: Path powers in dBm
        power_linear: Path powers in linear scale
        phase: Path phases in degrees
        aoa_az/el: Angles of arrival (azimuth/elevation)
        aod_az/el: Angles of departure (azimuth/elevation)
        rx_pos: Receiver positions
        tx_pos: Transmitter position
        num_paths: Number of paths per user
        pathloss: Path loss in dB
        distances: Distances between TX and RXs
    """
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize dataset with optional data.
        
        Args:
            data: Initial dataset dictionary. If None, creates empty dataset.
        """
        super().__init__(data or {})
        
    @property
    def aoa(self) -> np.ndarray:
        """Get combined angles of arrival.
        
        Returns:
            Array of shape (n_users, n_paths, 2) containing azimuth and elevation angles
        """
        return np.stack([self.aoa_az, self.aoa_el], axis=-1)
    
    @property
    def aod(self) -> np.ndarray:
        """Get combined angles of departure.
        
        Returns:
            Array of shape (n_users, n_paths, 2) containing azimuth and elevation angles
        """
        return np.stack([self.aod_az, self.aod_el], axis=-1)
    

    def __getattr__2(self, key: str) -> Any:
        """Enable dot notation access with common aliases.
        
        Supports aliases like:
        - ch -> channel
        - pwr -> power
        - pwr_lin -> power_linear
        - rx_loc -> rx_pos
        - tx_loc -> tx_pos
        - pl -> pathloss
        - dist -> distances
        """
        aliases = {
            # Channel aliases
            'ch': 'channel',
            'chs': 'channel',
            'channels': 'channel',
            
            # Power aliases
            'pwr': 'power',
            'powers': 'power',
            'pwr_lin': 'power_linear',
            'power_lin': 'power_linear',
            'pwr_linear': 'power_linear',
            
            # Position aliases
            'rx_loc': 'rx_pos',
            'rx_position': 'rx_pos',
            'rx_locations': 'rx_pos',
            'tx_loc': 'tx_pos',
            'tx_position': 'tx_pos',
            'tx_locations': 'tx_pos',
            
            # Pathloss aliases
            'pl': 'pathloss',
            'path_loss': 'pathloss',
            
            # Distance aliases
            'dist': 'distance',
            'dists': 'distance',
            'distances': 'distance',
            
            # Angle aliases
            'aoa_azimuth': 'aoa_az',
            'aoa_elevation': 'aoa_el',
            'aod_azimuth': 'aod_az',
            'aod_elevation': 'aod_el',
            
            # Path count aliases
            'n_paths': 'num_paths',
            
            # Time of arrival alias
            'time_of_arrival': 'toa',
            
            # Interaction aliases
            'interactions': 'inter',
            'interaction_locations': 'inter_pos'
        }
        if key in aliases:
            return self[aliases[key]]
        return super().__getattr__(key)

