"""
Setup file handling for Wireless Insite conversion.
"""
import os
import shutil
from typing import Dict

from .. import converter_utils as cu
from .setup_parser import parse_file

def read_setup(setup_file: str, verbose: bool) -> Dict:
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
