"""
Sionna Ray Tracing Converter Module.

This module provides functionality for converting Sionna Ray Tracing output files
into the DeepMIMO format. It handles reading and processing ray tracing data including:
- Path information (angles, delays, powers, interactions, ...)
- TX/RX locations and parameters 
- Scene geometry and materials
"""

import os
import shutil
from pprint import pprint

from ... import consts as c
from .. import converter_utils as cu

from .sionna_scene import read_scene
from .sionna_materials import read_materials
from .sionna_paths import read_paths
from .sionna_txrx import read_txrx
from .sionna_rt_params import read_raytracing_parameters

def sionna_rt_converter(rt_folder: str, copy_source: bool = False,
                        overwrite: bool = None, vis_scene: bool = False, 
                        scenario_name: str = '') -> str:
    """Convert Sionna ray-tracing data to DeepMIMO format.

    This function handles the conversion of Sionna ray-tracing simulation 
    data into the DeepMIMO dataset format. It processes path data, setup files,
    and transmitter/receiver configurations to generate channel matrices and metadata.

    Args:
        rt_folder (str): Path to folder containing Sionna ray-tracing data.
        copy_source (bool): Whether to copy ray-tracing source files to output.
        overwrite (bool): Whether to overwrite existing files. Prompts if None. Defaults to None.
        vis_scene (bool): Whether to visualize the scene layout. Defaults to False.
        scenario_name (str): Custom name for output folder. Uses rt folder name if empty.

    Returns:
        str: Path to output folder containing converted DeepMIMO dataset.
        
    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If transmitter or receiver IDs are invalid.
    """
    print('converting from sionna RT')

    # Setup scenario name
    scen_name = os.path.basename(rt_folder) 
    if scenario_name:
        scen_name = scenario_name

    # Setup output folder
    output_folder = os.path.join(rt_folder, scen_name + '_deepmimo')
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Read ray tracing parameters
    rt_params = read_raytracing_parameters(rt_folder)

    # Read TXRX
    txrx_dict = read_txrx(rt_params)

    # Read Paths (.paths)
    read_paths(rt_folder, output_folder, txrx_dict)

    # Read Materials (.materials)
    materials_dict, material_indices = read_materials(rt_folder, output_folder)

    # Read Scene data
    scene = read_scene(rt_folder, material_indices)
    scene_dict = scene.export_data(output_folder) if scene else {}
    
    # Visualize if requested
    if vis_scene and scene:
        scene.plot()
    
    # Save parameters to params.mat
    params = {
        c.LOAD_FILE_SP_VERSION: c.VERSION,
        c.PARAMSET_DYNAMIC_SCENES: 0, # only static currently
        c.LOAD_FILE_SP_RAYTRACER: c.RAYTRACER_NAME_SIONNA,
        c.LOAD_FILE_SP_RAYTRACER_VERSION: c.RAYTRACER_VERSION_SIONNA,
        c.RT_PARAMS_PARAM_NAME: rt_params,
        c.TXRX_PARAM_NAME: txrx_dict,
        c.MATERIALS_PARAM_NAME: materials_dict,
        c.SCENE_PARAM_NAME: scene_dict
    }
    cu.save_mat(params, 'params', output_folder)
    pprint(params)

    # Save scenario to deepmimo scenarios folder
    scen_name = cu.save_scenario(output_folder, scen_name=scenario_name, overwrite=overwrite)
    
    print(f'Zipping DeepMIMO scenario (ready to upload!): {output_folder}')
    cu.zip_folder(output_folder) # ready for upload
    
    # Copy and zip ray tracing source files as well
    if copy_source:
        cu.save_rt_source_files(rt_folder, ['.pkl'])
    
    return scen_name


if __name__ == '__main__':
    rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/' + \
                'all_runs/run_02-02-2025_15H45M26S/scen_0/DeepMIMO_folder'
    output_folder = os.path.join(rt_folder, 'test_deepmimo')

    rt_params = read_raytracing_parameters(rt_folder)
    txrx_dict = read_txrx(rt_params)
    read_paths(rt_folder, output_folder)
    read_materials(rt_folder, output_folder)

