"""
Main converter module for processing raytracing data from different sources.

This module provides functionality to automatically detect and convert raytracing
data from various supported formats (AODT, Sionna RT, Wireless Insite) into
a standardized scenario format.
"""

# Standard library imports
import os
from typing import Dict, Any, Optional

# Local imports
from . import converter_utils as cu
from .aodt.aodt_converter import aodt_rt_converter
from .sionna_rt.sionna_converter import sionna_rt_converter
from .wireless_insite.insite_converter import insite_rt_converter


def create_scenario(path_to_rt_folder: str, **conversion_params: Dict[str, Any]) -> Optional[Any]:
    """Create a standardized scenario from raytracing data.
    
    This function automatically detects the raytracing data format based on file 
    extensions and uses the appropriate converter to generate a standardized scenario.
    It supports AODT, Sionna RT, and Wireless Insite formats.

    Args:
        path_to_rt_folder (str): Path to the folder containing raytracing data
        **conversion_params (Dict[str, Any]): Additional parameters for the conversion process

    Returns:
        Optional[Any]: Scenario object if conversion is successful, None otherwise
    """
    print('Determining converter...')
    
    files_in_dir = os.listdir(path_to_rt_folder)
    if cu.ext_in_list('.aodt', files_in_dir):
        print("Using AODT generator")
        rt_converter = aodt_rt_converter
    elif cu.ext_in_list('.path', files_in_dir):
        print("Using Sionna_RT generator")
        rt_converter = sionna_rt_converter
    elif cu.ext_in_list('.p2m', files_in_dir):
        print("Using Wireless Insite generator")
        rt_converter = insite_rt_converter
    else:
        print("Unknown raytracer type")
        return None
    
    scenario = rt_converter(path_to_rt_folder, **conversion_params)
    return scenario
