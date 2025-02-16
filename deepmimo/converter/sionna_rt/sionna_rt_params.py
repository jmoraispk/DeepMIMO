"""
Sionna Ray Tracing Parameters Module.

This module handles loading and converting ray tracing parameters from Sionna's format.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np

from .. import converter_utils as cu
# from ..raytracing_parameters import RayTracingParameters


@dataclass
class RayTracingParameters:
    """Class representing Sionna Ray Tracing parameters.
    
    This class encapsulates all parameters needed for Sionna ray tracing simulations,
    including frequency settings, array configurations, and interaction settings.
    """
    # Frequency and bandwidth settings
    frequency: float  # Center frequency in Hz
    bandwidth: float  # Bandwidth in Hz
    
    # Array configurations
    rx_array_size: int  # Size of RX array
    rx_array_num_ant: int  # Number of RX antennas
    rx_array_ant_pos: np.ndarray  # RX antenna positions relative to reference
    
    tx_array_size: int  # Size of TX array
    tx_array_num_ant: int  # Number of TX antennas
    tx_array_ant_pos: np.ndarray  # TX antenna positions relative to reference
    
    array_synthetic: bool  # Whether arrays are synthetic
    
    # Ray tracing settings
    max_depth: int = 3  # Maximum number of interactions
    method: str = 'fibonacci'  # Ray launching method
    num_samples: int = 1000000  # Number of rays to launch
    
    # Interaction flags
    los: bool = True  # Line of sight
    reflection: bool = True  # Reflections
    diffraction: bool = False  # Diffraction
    scattering: bool = False  # Scattering
    edge_diffraction: bool = False  # Edge diffraction
    
    # Scattering settings
    scat_keep_prob: float = 0.001  # Scattering keep probability
    scat_random_phases: bool = True  # Random phases for scattering
    
    # Version info
    raytracer_version: str = ''  # Sionna version used
    doppler_available: int = 0  # Whether Doppler information is available
    
    @classmethod
    def from_dict(cls, params_dict: Dict) -> 'RayTracingParameters':
        """Create RayTracingParameters from a dictionary.
        
        Args:
            params_dict: Dictionary containing parameter values
            
        Returns:
            RayTracingParameters object
        """
        return cls(**params_dict)
    
    def to_dict(self) -> Dict:
        """Convert parameters to dictionary format.
        
        Returns:
            Dictionary containing all parameters
        """
        return asdict(self)


def read_raytracing_parameters(load_folder: str) -> Dict:
    """Read Sionna RT parameters.
    
    Args:
        load_folder: Path to folder containing setup file
        
    Returns:
        RayTracingParameters object containing standardized parameters
    """
    params_dict = cu.load_pickle(load_folder + 'sionna_rt_params.pkl')
    rt_params = RayTracingParameters.from_dict(params_dict)
    return rt_params.to_dict()