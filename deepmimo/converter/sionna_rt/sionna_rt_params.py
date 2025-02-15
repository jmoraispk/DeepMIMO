"""
Sionna Ray Tracing Parameters Module.

This module handles loading and converting ray tracing parameters from Sionna's format.
"""

from .. import converter_utils as cu
from typing import Dict
# from ..raytracing_parameters import RayTracingParameters

def read_raytracing_parameters(load_folder: str) -> Dict:
    """Read Sionna RT parameters.
    
    Args:
        load_folder: Path to folder containing setup file
        
    Returns:
        RayTracingParameters object containing standardized parameters
    """
    params_dict = cu.load_pickle(load_folder + 'sionna_rt_params.pkl')
    # rt_params = RayTracingParameters.from_sionna(params_dict)
    # return rt_params.to_dict()
    return params_dict