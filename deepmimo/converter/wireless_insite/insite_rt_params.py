"""
Setup handling for Wireless Insite conversion.

This module provides functionality for parsing setup files (.setup) from Wireless Insite
into a standardized parameter format.
"""
import os
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

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
    # Required parameters (no defaults)
    antenna_type: str  # Type of antenna
    polarization: str  # Antenna polarization
    power_threshold: float  # Power threshold for rays
    path_depth: int  # Path depth for APG
    
    # Optional parameters (with defaults)
    # Study area settings
    initial_ray_mode: str = ''  # Initial ray mode
    foliage_model: str = ''  # Foliage model
    foliage_attenuation_vert: float = 0.0  # Vertical foliage attenuation
    foliage_attenuation_hor: float = 0.0  # Horizontal foliage attenuation
    terrain_diffractions: int = 0  # Number of terrain diffractions
    ray_spacing: float = 0.25  # Ray spacing
    
    # APG acceleration settings
    apg_acceleration: bool = False  # Whether APG acceleration is enabled
    workflow_mode: str = ''  # APG workflow mode
    adjacency_distance: float = 0.0  # Adjacency distance for APG
    
    # Diffuse scattering settings
    diffuse_scattering: bool = False  # Whether diffuse scattering is enabled
    diffuse_reflections: bool = False  # Whether diffuse reflections are enabled
    diffuse_diffractions: bool = False  # Whether diffuse diffractions are enabled
    diffuse_transmissions: bool = False  # Whether diffuse transmissions are enabled
    final_interaction_only: bool = False  # Whether to only consider final interactions
    
    
    def __post_init__(self):
        """Set Wireless Insite-specific engine info and defaults."""
        # Call parent's post init first
        super().__post_init__()
        
        # Set Wireless Insite-specific engine info
        self.raytracer_name = RAYTRACER_NAME_WIRELESS_INSITE
        self.raytracer_version = RAYTRACER_VERSION_WIRELESS_INSITE
        
        # Set max_depth based on path_depth if available
        if hasattr(self, 'path_depth') and not hasattr(self, 'max_depth'):
            self.max_depth = self.path_depth
    
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
        
        # Build standardized parameter dictionary
        params_dict = {
            # Base required parameters
            'frequency': waveform_vals['CarrierFrequency'],
            'bandwidth': waveform_vals['bandwidth'],
            'max_depth': model_vals.get('path_depth', 3),
            'max_reflections': model_vals.get('max_reflections', 3),
            'raytracer_name': RAYTRACER_NAME_WIRELESS_INSITE,
            'raytracer_version': RAYTRACER_VERSION_WIRELESS_INSITE,
            'los': True,  # Always enabled in Wireless Insite
            'reflection': True,  # Always enabled in Wireless Insite
            'diffraction': bool(model_vals.get('terrain_diffractions', 0)),
            'scattering': diffuse_scat_vals['enabled'],
            'raw_params': raw_params,
            
            # Required Insite-specific parameters
            'antenna_type': antenna_vals['type'],
            'polarization': antenna_vals['polarization'],
            'power_threshold': antenna_vals['power_threshold'],
            'path_depth': apg_accel_vals['path_depth'],
            
            # Optional Insite-specific parameters
            'initial_ray_mode': model_vals.get('initial_ray_mode', ''),
            'foliage_model': model_vals.get('foliage_model', ''),
            'foliage_attenuation_vert': model_vals.get('foliage_attenuation_vert', 0.0),
            'foliage_attenuation_hor': model_vals.get('foliage_attenuation_hor', 0.0),
            'terrain_diffractions': bool(model_vals.get('terrain_diffractions', 0)),
            'ray_spacing': model_vals.get('ray_spacing', 0.25),
            
            'apg_acceleration': apg_accel_vals['enabled'],
            'workflow_mode': apg_accel_vals['workflow_mode'],
            'adjacency_distance': apg_accel_vals['adjacency_distance'],
            
            'diffuse_scattering': diffuse_scat_vals['enabled'],
            'diffuse_reflections': diffuse_scat_vals['diffuse_reflections'],
            'diffuse_diffractions': diffuse_scat_vals['diffuse_diffractions'],
            'diffuse_transmissions': diffuse_scat_vals['diffuse_transmissions'],
            'final_interaction_only': diffuse_scat_vals['final_interaction_only'],
            
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
    
    # Extract setup information
    setup_dict = InsiteRayTracingParameters.read_rt_params(setup_file)
    
    # Print summary by categories
    print("\nAntenna Settings:")
    print(f"  Type: {setup_dict['antenna_type']}")
    print(f"  Polarization: {setup_dict['polarization']}")
    print(f"  Power Threshold: {setup_dict['power_threshold']}")
    
    print("\nWaveform Settings:")
    print(f"  Frequency: {setup_dict['frequency']} Hz")
    print(f"  Bandwidth: {setup_dict['bandwidth']} Hz")
    
    print("\nStudy Area Settings:")
    print(f"  Ray Spacing: {setup_dict['ray_spacing']}")
    print(f"  Max Reflections: {setup_dict['max_reflections']}")
    print(f"  Initial Ray Mode: {setup_dict['initial_ray_mode']}")
    
    print("\nAPG Settings:")
    print(f"  Enabled: {setup_dict['apg_acceleration']}")
    print(f"  Workflow Mode: {setup_dict['workflow_mode']}")
    print(f"  Path Depth: {setup_dict['path_depth']}")
    print(f"  Adjacency Distance: {setup_dict['adjacency_distance']}")
    
    print("\nDiffuse Scattering Settings:")
    print(f"  Enabled: {setup_dict['diffuse_scattering']}")
    print(f"  Reflections: {setup_dict['diffuse_reflections']}")
    print(f"  Diffractions: {setup_dict['diffuse_diffractions']}")
    print(f"  Transmissions: {setup_dict['diffuse_transmissions']}")
    print(f"  Final Interaction Only: {setup_dict['final_interaction_only']}")
    