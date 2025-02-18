"""
Setup handling for Wireless Insite conversion.

This module provides functionality for parsing setup files (.setup) from Wireless Insite
into a standardized parameter format.
"""
import os
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
from pprint import pprint

from .setup_parser import parse_file
from ...rt_params import RayTracingParameters
from ...consts import RAYTRACER_NAME_WIRELESS_INSITE, RAYTRACER_VERSION_WIRELESS_INSITE


def read_rt_params(sim_folder: str | Path) -> Dict:
    """Read Wireless Insite RT parameters from a folder."""
    return InsiteRayTracingParameters.read_rt_params(sim_folder).to_dict()


@dataclass
class InsiteRayTracingParameters(RayTracingParameters):
    """Class representing Wireless Insite Ray Tracing parameters.
    
    This class extends the base RayTracingParameters with Wireless Insite-specific
    settings for antenna configuration, APG acceleration, and diffuse scattering.
    
    Note: All required parameters must come before optional ones in dataclasses.
    First come the base class required parameters (inherited), then the class-specific
    required parameters, then all optional parameters.
    """
    
    @classmethod
    def read_rt_params(cls, sim_folder: str | Path) -> 'InsiteRayTracingParameters':
        """Read a Wireless Insite setup file and return a parameters object.
        
        Args:
            sim_folder: Path to simulation folder containing .setup file
            
        Returns:
            InsiteRayTracingParameters object containing standardized parameters
            
        Raises:
            ValueError: If no .setup file found or multiple .setup files found
        """
        sim_folder = Path(sim_folder)
        if not sim_folder.exists():
            raise ValueError(f"Simulation folder does not exist: {sim_folder}")
        
        # Find .setup file
        setup_files = list(sim_folder.glob("*.setup"))
        if not setup_files:
            raise ValueError(f"No .setup file found in {sim_folder}")
        if len(setup_files) > 1:
            raise ValueError(f"Multiple .setup files found in {sim_folder}")
        
        # Parse setup file
        setup_file = str(setup_files[0])
        document = parse_file(setup_file)

        # Select study area 
        prim = list(document.keys())[0]
          
        prim_vals = document[prim].values
        antenna_vals = prim_vals['antenna'].values
        waveform_vals = prim_vals['Waveform'].values
        studyarea_vals = prim_vals['studyarea'].values
        model_vals = studyarea_vals['model'].values
        apg_accel_vals = studyarea_vals['apg_acceleration'].values
        diffuse_scat_vals = studyarea_vals['diffuse_scattering'].values
        
        # Store raw parameters
        raw_params = {
            'antenna': antenna_vals,
            'waveform': waveform_vals,
            'studyarea': studyarea_vals,
            'model': model_vals,
            'apg_acceleration': apg_accel_vals,
            'diffuse_scattering': diffuse_scat_vals
        }

        max_scat = sum([diffuse_scat_vals['diffuse_reflections'],
                        diffuse_scat_vals['diffuse_diffractions'],
                        diffuse_scat_vals['diffuse_transmissions']])
        
        num_rays = 360 // model_vals['ray_spacing'] * 180  

        # Build standardized parameter dictionary
        params_dict = {
            # Ray Tracing Engine info
            'raytracer_name': RAYTRACER_NAME_WIRELESS_INSITE,
            'raytracer_version': RAYTRACER_VERSION_WIRELESS_INSITE,

            # Frequency
            'frequency': waveform_vals['CarrierFrequency'],
            
            # Ray tracing interaction settings
            'max_path_depth': apg_accel_vals['path_depth'],
            'max_reflections': model_vals['max_reflections'],
            'max_diffractions': model_vals['terrain_diffractions'], 
            'max_scatterings': max_scat if bool(diffuse_scat_vals['enabled']) else 0,  # 1 if enabled, 0 if not
            'max_transmissions': 0,  # Insite does not support transmissions in our setup

            # Details on diffraction, scattering, and transmission
            'diffuse_reflections': diffuse_scat_vals['diffuse_reflections'],
            'diffuse_diffractions': diffuse_scat_vals['diffuse_diffractions'],
            'diffuse_transmissions': diffuse_scat_vals['diffuse_transmissions'],
            'diffuse_final_interaction_only': diffuse_scat_vals['final_interaction_only'],
            'diffuse_random_phases': False,  # Insite does not support random phases

            # Terrain interaction settings
            'terrain_reflection': bool(model_vals.get('terrain_reflections', 0)),
            'terrain_diffraction': 'Yes' == model_vals['terrain_diffractions'],
            'terrain_scattering': bool(model_vals.get('terrain_scattering', 0)),

            # Ray casting settings
            'num_rays': num_rays,  # Insite uses ray spacing instead of explicit ray count
            'ray_casting_method': 'uniform',  # Insite uses uniform ray casting
            'synthetic_array': True,  # Currently only synthetic arrays are supported

            # Store raw parameters
            'raw_params': raw_params,
        }
        
        # Create and return parameters object
        return cls.from_dict(params_dict)


if __name__ == "__main__":
    # Test directory with setup files
    test_dir = r"./P2Ms/simple_street_canyon_test/"
    
    # Find .setup file in test directory
    setup_file = None
    for root, _, filenames in os.walk(test_dir):
        for filename in filenames:
            if filename.endswith('.setup'):
                setup_file = os.path.join(root, filename)
                break
        if setup_file:
            break
            
    if not setup_file:
        print(f"No .setup file found in {test_dir}")
        exit(1)
        
    print(f"\nTesting setup extraction from: {setup_file}")
    print("-" * 50)
    
    # Extract setup information and print in a nicely formatted way
    setup_dict = InsiteRayTracingParameters.read_rt_params(setup_file)
    
    # Filter out raw_params to keep output cleaner
    pprint(setup_dict, sort_dicts=True, width=80)
    