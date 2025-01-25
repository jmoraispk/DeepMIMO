"""
Setup handling for Wireless Insite conversion.

This module provides functionality for parsing setup files (.setup) from Wireless Insite
into a dictionary format containing all simulation parameters and settings.
"""

import os
from pathlib import Path
from typing import Dict

from .setup_parser import parse_file


def read_setup(sim_folder: str | Path) -> Dict:
    """Read a Wireless Insite setup file and extract all configuration parameters.
    
    Simulation parameter include:
    - Antenna settings (type, polarization, power threshold)
    - Waveform settings (carrier frequency, bandwidth)
    - Study area settings (ray tracing parameters)
    - APG acceleration settings
    - Diffuse scattering settings
    - Boundary settings
    
    Args:
        sim_folder: Path to simulation folder containing .setup file
        
    Returns:
        Dictionary containing all extracted setup parameters
        
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
    
    setup_dict = {}
    
    # Antenna Settings
    setup_dict['antenna'] = antenna_vals['type']
    setup_dict['polarization'] = antenna_vals['polarization']
    setup_dict['power_threshold'] = antenna_vals['power_threshold']
    
    # Waveform Settings
    setup_dict['frequency'] = waveform_vals['CarrierFrequency']
    setup_dict['bandwidth'] = waveform_vals['bandwidth']
    
    # Study Area Settings
    model_vals = studyarea_vals['model'].values
    match_list = ['initial_ray_mode', 'foliage_model',
                  'foliage_attenuation_vert', 'foliage_attenuation_hor', 
                  'terrain_diffractions', 'ray_spacing', 'max_reflections', 
                  'initial_ray_mode']
    defaults = {'ray_spacing': 0.25, 'terrain_diffractions': 0}
    for key in match_list:
        try:
            setup_dict[key] = model_vals[key]
        except KeyError:
            print(f'key "{key}" not found in setup file')
            setup_dict[key] = defaults[key]
            
    # Verify that the required outputs were generated
    output_vals = model_vals['OutputRequests'].values
    necessary_output_files_exist = True
    necessary_outputs = ['Paths']
    for output in necessary_outputs:
        if not output_vals[output]:
            print(f'One of the NECESSARY outputs is missing. Output missing: {output}')
            necessary_output_files_exist = False
            
    if not necessary_output_files_exist:
        raise Exception('Missing output file. Please rerun the simulation '
                        'with the necessary outputs enabled.')
    
    # APG settings
    apg_accel_vals = studyarea_vals['apg_acceleration'].values
    setup_dict['apg_acceleration']   = apg_accel_vals['enabled']
    setup_dict['workflow_mode']      = apg_accel_vals['workflow_mode']
    setup_dict['path_depth']         = apg_accel_vals['path_depth']
    setup_dict['adjacency_distance'] = apg_accel_vals['adjacency_distance']
    
    # Diffuse scattering settings
    diffuse_scat_vals = studyarea_vals['diffuse_scattering'].values
    setup_dict['diffuse_scattering']     = diffuse_scat_vals['enabled']
    setup_dict['diffuse_reflections']    = diffuse_scat_vals['diffuse_reflections']
    setup_dict['diffuse_diffractions']   = diffuse_scat_vals['diffuse_diffractions']
    setup_dict['diffuse_transmissions']  = diffuse_scat_vals['diffuse_transmissions']
    setup_dict['final_interaction_only'] = diffuse_scat_vals['final_interaction_only']
    
    # Boundary settings
    setup_dict['boundary_zmin'] = studyarea_vals['boundary']['zmin']
    setup_dict['boundary_zmax'] = studyarea_vals['boundary']['zmax']
    setup_dict['boundary_xmin'] = studyarea_vals['boundary'].data[0][0]
    setup_dict['boundary_xmax'] = studyarea_vals['boundary'].data[2][0]
    setup_dict['boundary_ymin'] = studyarea_vals['boundary'].data[0][1]
    setup_dict['boundary_ymax'] = studyarea_vals['boundary'].data[2][1]
    
    return setup_dict


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
    setup_dict = read_setup(setup_file)
    
    # Print summary by categories
    print("\nAntenna Settings:")
    print(f"  Type: {setup_dict['antenna']}")
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
    
    print("\nBoundary Settings:")
    print(f"  X: [{setup_dict['boundary_xmin']}, {setup_dict['boundary_xmax']}]")
    print(f"  Y: [{setup_dict['boundary_ymin']}, {setup_dict['boundary_ymax']}]")
    print(f"  Z: [{setup_dict['boundary_zmin']}, {setup_dict['boundary_zmax']}]")
