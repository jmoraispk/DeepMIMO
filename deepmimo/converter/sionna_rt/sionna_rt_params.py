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
        
        # Raise error if los is not present
        if 'los' not in raw_params or not raw_params['los']:
            raise ValueError("los not found in Sionna RT parameters")
        
        # Raise error if arrays are not synthetic
        if not raw_params['synthetic_array']:
            raise ValueError("arrays are not synthetic in Sionna RT parameters. "
                             "Multi-antenna arrays are not supported yet.")
        
        # NOTE: Sionna distributes these samples across antennas AND TXs
        n_tx, n_tx_ant = raw_params['tx_array_size'], raw_params['tx_array_num_ant']
        n_emmitters = n_tx * n_tx_ant
        n_rays = raw_params['num_samples'] // n_emmitters
    
        # Create standardized parameters
        params_dict = {
            # Ray Tracing Engine info
            'raytracer_name': RAYTRACER_NAME_SIONNA,
            'raytracer_version': raw_params.get('raytracer_version', RAYTRACER_VERSION_SIONNA),

            # Base required parameters
            'frequency': raw_params['frequency'],
            
            # Ray tracing interaction settings
            'max_path_depth': raw_params['max_depth'],
            'max_reflections': raw_params['max_depth'] if raw_params['reflection'] else 0,
            'max_diffractions': int(raw_params['diffraction']),  # Sionna only supports 1 diffraction event
            'max_scatterings': int(raw_params['scattering']),   # Sionna only supports 1 scattering event
            'max_transmissions': 0, # Sionna does not support transmissions

            # Terrain interaction settings
            'terrain_reflection': bool(raw_params['reflection']), 
            'terrain_diffraction': raw_params['diffraction'],  # Sionna only supports 1 diffraction, may be on terrain
            'terrain_scattering': raw_params['scattering'],

            # Details on diffraction, scattering, and transmission
            'diffuse_reflections': raw_params['max_depth'] - 1, # Sionna only supports diffuse reflections
            'diffuse_diffractions': 0, # Sionna only supports 1 diffraction event, with no diffuse scattering
            'diffuse_transmissions': 0, # Sionna does not support transmissions
            'diffuse_final_interaction_only': True, # Sionna only supports diffuse scattering at final interaction
            'diffuse_random_phases': raw_params.get('scat_random_phases', True),

            'synthetic_array': raw_params.get('synthetic_array', True),
            'num_rays': -1 if raw_params['method'] == 'fibonacci' else n_rays, 
            'ray_casting_method': raw_params['method'].replace('fibonacci', 'uniform'),
            # The alternative to fibonacci is exhaustive, for which the number of rays is not predictable

            'raw_params': raw_params,
        }
        
        # Create and return parameters object
        return cls.from_dict(params_dict)