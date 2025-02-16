"""Wireless Insite to DeepMIMO Scenario Converter.

This module serves as the main entry point for converting Wireless Insite raytracing simulation
outputs into DeepMIMO-compatible scenario files. It orchestrates the conversion process by:

1. Reading and processing setup files (.setup)
2. Reading TX/RX configurations (.txrx)
3. Converting path data from .p2m files
4. Processing material properties (.city, .ter, .veg)
5. Converting scene geometry
6. Saving all data in DeepMIMO format

Module Dependencies:
- insite_materials.py: Material property handling
- insite_setup.py: Setup file parsing
- insite_scene.py: Scene geometry conversion
- insite_txrx.py: TX/RX configuration handling
- insite_paths.py: Path data processing
  - p2m_parser.py: Low-level .p2m file parsing

The adapter assumes BSs are transmitters and users are receivers. Uplink channels
can be generated using (transpose) reciprocity.

Main Entry Point:
    insite_rt_converter(): Converts a complete Wireless Insite scenario to DeepMIMO format
"""

# Standard library imports
import os
import shutil
from typing import Optional

# Local imports
from .. import converter_utils as cu
from ... import consts as c

from .insite_rt_params import read_rt_params
from .insite_txrx import read_txrx
from .insite_paths import read_paths
from .insite_materials import read_materials
from .insite_scene import read_scene


# Constants
MATERIAL_FILES = ['.city', '.ter', '.veg']
SETUP_FILES = ['.setup', '.txrx'] + MATERIAL_FILES
SOURCE_EXTS = SETUP_FILES + ['.kmz']  # Files to copy to ray tracing source zip

def insite_rt_converter(p2m_folder: str, copy_source: bool = False,
                        overwrite: Optional[bool] = None, vis_scene: bool = False, 
                        scenario_name: str = '') -> str:
    """Convert Wireless InSite ray-tracing data to DeepMIMO format.

    This function handles the conversion of Wireless InSite ray-tracing simulation 
    data into the DeepMIMO dataset format. It processes path files (.p2m), setup files,
    and transmitter/receiver configurations to generate channel matrices and metadata.

    Args:
        p2m_folder (str): Path to folder containing .p2m path files.
        copy_source (bool): Whether to copy ray-tracing source files to output.
        overwrite (Optional[bool]): Whether to overwrite existing files. Prompts if None. Defaults to None.
        vis_scene (bool): Whether to visualize the scene layout. Defaults to False.
        scenario_name (str): Custom name for output folder. Uses p2m folder name if empty.

    Returns:
        str: Path to output folder containing converted DeepMIMO dataset.
        
    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If transmitter or receiver IDs are invalid.
    """

    # Setup output folder
    insite_sim_folder = os.path.dirname(p2m_folder)
    p2m_basename = os.path.basename(p2m_folder)
    out_fold_name = scenario_name if scenario_name else p2m_basename 
    
    output_folder = os.path.join(insite_sim_folder, out_fold_name + '_deepmimo')
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Read setup (.setup)
    rt_params = read_rt_params(insite_sim_folder)

    # Read TXRX (.txrx)
    txrx_dict = read_txrx(insite_sim_folder, p2m_folder)
    
    # Read and save path data
    read_paths(p2m_folder, output_folder, txrx_dict)
    
    # Read Materials of all objects (.city, .ter, .veg)
    materials_dict = read_materials(insite_sim_folder)
    
    # Read scene objects
    scene = read_scene(insite_sim_folder)
    scene_dict = scene.export_data(output_folder)
    
    # Visualize if requested
    if vis_scene: scene.plot()
    
    # Save parameters to params.mat
    params = {
        c.LOAD_FILE_SP_VERSION: c.VERSION,
        c.PARAMSET_DYNAMIC_SCENES: 0, # only static currently
        c.LOAD_FILE_SP_RAYTRACER: c.RAYTRACER_NAME_WIRELESS_INSITE,
        c.LOAD_FILE_SP_RAYTRACER_VERSION: c.RAYTRACER_VERSION_WIRELESS_INSITE,
        c.RT_PARAMS_PARAM_NAME: rt_params,
        c.TXRX_PARAM_NAME: txrx_dict,
        c.MATERIALS_PARAM_NAME: materials_dict,
        c.SCENE_PARAM_NAME: scene_dict
    }
    cu.save_mat(params, 'params', output_folder)
    
    from pprint import pprint
    pprint(params)

    # Save scenario to deepmimo scenarios folder
    scen_name = cu.save_scenario(output_folder, scen_name=scenario_name, overwrite=overwrite)
    
    print(f'Zipping DeepMIMO scenario (ready to upload!): {output_folder}')
    cu.zip_folder(output_folder) # ready for upload
    
    # Copy and zip ray tracing source files as well
    if copy_source:
        cu.save_rt_source_files(insite_sim_folder, SOURCE_EXTS)
    
    return scen_name