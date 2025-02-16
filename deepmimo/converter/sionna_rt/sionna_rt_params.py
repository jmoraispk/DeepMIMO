"""
Sionna Ray Tracing Parameters Module.

This module handles loading and converting ray tracing parameters from Sionna's format.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np

from .. import converter_utils as cu
from ...rt_params import RayTracingParameters
from ...consts import RAYTRACER_NAME_SIONNA, RAYTRACER_VERSION_SIONNA


def read_rt_params(load_folder: str) -> Dict:
    """Read Sionna RT parameters from a folder."""
    return SionnaRayTracingParameters.read_rt_params(load_folder).to_dict()


@dataclass
class SionnaRayTracingParameters(RayTracingParameters):
    """Class representing Sionna Ray Tracing parameters.
    
    This class extends the base RayTracingParameters with Sionna-specific settings
    for array configurations and ray launching.
    
    Note: All required parameters must come before optional ones in dataclasses.
    First come the base class required parameters (inherited), then the class-specific
    required parameters, then all optional parameters.
    """
    # Required array parameters (no defaults)
    rx_array_size: int  # Size of RX array
    rx_array_num_ant: int  # Number of RX antennas
    rx_array_ant_pos: np.ndarray  # RX antenna positions relative to reference
    tx_array_size: int  # Size of TX array
    tx_array_num_ant: int  # Number of TX antennas
    tx_array_ant_pos: np.ndarray  # TX antenna positions relative to reference
    
    # Optional parameters (with defaults)
    array_synthetic: bool = False  # Whether arrays are synthetic
    method: str = 'fibonacci'  # Ray launching method
    num_samples: int = 1000000  # Number of rays to launch
    scat_keep_prob: float = 0.001  # Scattering keep probability
    scat_random_phases: bool = True  # Random phases for scattering
    doppler_available: int = 0  # Whether Doppler information is available
    
    def __post_init__(self):
        """Set Sionna-specific engine info and defaults."""
        # Call parent's post init first
        super().__post_init__()
        
        # Set Sionna-specific engine info
        self.raytracer_name = RAYTRACER_NAME_SIONNA
        self.raytracer_version = self.raw_params.get('raytracer_version', RAYTRACER_VERSION_SIONNA)
        
        # Map max_depth to max_reflections if needed
        if 'max_depth' not in self.raw_params and hasattr(self, 'max_depth'):
            self.max_reflections = self.max_depth
    
    @classmethod
    def read_rt_params(cls, load_folder: str) -> 'SionnaRayTracingParameters':
        """Read Sionna RT parameters and return a parameters object.
        
        Args:
            load_folder: Path to folder containing setup file
            
        Returns:
            SionnaRayTracingParameters object containing standardized parameters
        """
        # Load original parameters
        raw_params = cu.load_pickle(load_folder + 'sionna_rt_params.pkl')
        
        # Create standardized parameters
        params_dict = {
            # Base required parameters
            'frequency': raw_params['frequency'],
            'bandwidth': raw_params['bandwidth'],
            'max_depth': raw_params.get('max_depth', 3),
            'max_reflections': raw_params.get('max_reflections', raw_params.get('max_depth', 3)),
            'raytracer_name': RAYTRACER_NAME_SIONNA,
            'raytracer_version': raw_params.get('raytracer_version', RAYTRACER_VERSION_SIONNA),
            'los': raw_params.get('los', True),
            'reflection': raw_params.get('reflection', True),
            'diffraction': raw_params.get('diffraction', False),
            'scattering': raw_params.get('scattering', False),
            'raw_params': raw_params,
            
            # Required Sionna-specific parameters
            'rx_array_size': raw_params['rx_array_size'],
            'rx_array_num_ant': raw_params['rx_array_num_ant'],
            'rx_array_ant_pos': raw_params['rx_array_ant_pos'],
            'tx_array_size': raw_params['tx_array_size'],
            'tx_array_num_ant': raw_params['tx_array_num_ant'],
            'tx_array_ant_pos': raw_params['tx_array_ant_pos'],
            
            # Optional Sionna-specific parameters
            'array_synthetic': raw_params.get('array_synthetic', False),
            'method': raw_params.get('method', 'fibonacci'),
            'num_samples': raw_params.get('num_samples', 1000000),
            'scat_keep_prob': raw_params.get('scat_keep_prob', 0.001),
            'scat_random_phases': raw_params.get('scat_random_phases', True),
            'doppler_available': raw_params.get('doppler_available', 0)
        }
        
        # Create and return parameters object
        return cls.from_dict(params_dict)