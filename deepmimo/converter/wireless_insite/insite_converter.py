"""Wireless Insite to DeepMIMO Scenario Converter.

This module provides functionality to convert Wireless Insite raytracing simulation
outputs into DeepMIMO-compatible scenario files. It handles:
- Channel data formatting and conversion
- Single and multi-user scenario support
- Single and multi-basestation configurations
- Path delay and coefficient extraction

The adapter assumes BSs are transmitters and users are receivers. Uplink channels
can be generated using (transpose) reciprocity.
"""

# Standard library imports
import os
import shutil
from typing import List, Dict, Optional

# Third-party imports
import numpy as np
import scipy.io

# Local imports
from .. import converter_utils as cu
from ... import consts as c
from .insite_materials import read_materials
from .insite_setup import read_setup
from .insite_scene import read_scene
from .insite_txrx import read_txrx
from .insite_paths import read_paths

# v3 (old)
from .ChannelDataLoader import WIChannelConverter
from .ChannelDataFormatter import DeepMIMODataFormatter

# Constants
MATERIAL_FILES = ['.city', '.ter', '.veg']
SETUP_FILES = ['.setup', '.txrx'] + MATERIAL_FILES
SOURCE_EXTS = SETUP_FILES + ['.kmz']  # Files to copy to ray tracing source zip

def insite_rt_converter_v3(p2m_folder: str, tx_ids: List[int], rx_ids: List[int], 
                          params_dict: Dict, scenario_name: str = '') -> str:
    """Convert Wireless Insite files to DeepMIMO format using legacy v3 converter.

    Args:
        p2m_folder (str): Path to folder containing .p2m files.
        tx_ids (List[int]): List of transmitter IDs to process.
        rx_ids (List[int]): List of receiver IDs to process.
        params_dict (Dict): Dictionary containing simulation parameters.
        scenario_name (str): Custom name for output folder. Uses p2m parent folder name if empty.

    Returns:
        str: Path to output folder containing converted files.
    """
    # Loads P2Ms (.cir, .doa, .dod, .paths[.t001_{tx_id}.r{rx_id}.p2m] eg: .t001_01.r001.p2m)
    
    insite_sim_folder = os.path.dirname(p2m_folder)

    intermediate_folder = os.path.join(insite_sim_folder, 'intermediate_files')
    output_folder = os.path.join(insite_sim_folder, 'mat_files') # SCEN_NAME!
    
    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Convert P2M files to mat format
    WIChannelConverter(p2m_folder, intermediate_folder)

    DeepMIMODataFormatter(intermediate_folder, output_folder, TX_order=tx_ids, RX_order=rx_ids)
    
    data_dict = {
                c.LOAD_FILE_SP_VERSION: c.VERSION,
                c.LOAD_FILE_SP_CF: params_dict['freq'], 
                c.LOAD_FILE_SP_USER_GRIDS: np.array([params_dict['user_grid']], dtype=float),
                c.LOAD_FILE_SP_NUM_BS: params_dict['num_bs'],
                c.LOAD_FILE_SP_TX_POW: 0,
                c.LOAD_FILE_SP_NUM_RX_ANT: 1,
                c.LOAD_FILE_SP_NUM_TX_ANT: 1,
                c.LOAD_FILE_SP_POLAR: 0,
                c.LOAD_FILE_SP_DOPPLER: 0
                }
    
    scipy.io.savemat(os.path.join(output_folder, 'params.mat'), data_dict)
    
    # export
    scen_name = scenario_name if scenario_name else os.path.basename(os.path.dirname(output_folder))
    scen_path = c.SCENARIOS_FOLDER + f'/{scen_name}'
    if os.path.exists(scen_path):
        shutil.rmtree(scen_path)
    shutil.copytree(output_folder, './' + scen_path)
    
    return output_folder


def insite_rt_converter(p2m_folder: str, copy_source: bool = False, tx_set_ids: Optional[List[int]] = None,
                       rx_set_ids: Optional[List[int]] = None, verbose: bool = True, 
                       overwrite: Optional[bool] = None, vis_buildings: bool = False, 
                       convert_buildings: bool = True, old: bool = False, 
                       old_params: Dict = {}, scenario_name: str = '') -> str:
    """Convert Wireless InSite ray-tracing data to DeepMIMO format.

    This function handles the conversion of Wireless InSite ray-tracing simulation 
    data into the DeepMIMO dataset format. It processes path files (.p2m), setup files,
    and transmitter/receiver configurations to generate channel matrices and metadata.

    Args:
        p2m_folder (str): Path to folder containing .p2m path files.
        copy_source (bool): Whether to copy ray-tracing source files to output.
        tx_set_ids (Optional[List[int]]): List of transmitter set IDs. Uses all if None. Defaults to None.
        rx_set_ids (Optional[List[int]]): List of receiver set IDs. Uses all if None. Defaults to None.
        verbose (bool): Whether to print progress messages. Defaults to True.
        overwrite (Optional[bool]): Whether to overwrite existing files. Prompts if None. Defaults to None.
        vis_buildings (bool): Whether to visualize building layouts. Defaults to False.
        convert_buildings (bool): Whether to convert extract buldings from the .city file. Defaults to True.
        old (bool): Whether to use legacy v3 converter. Defaults to False.
        old_params (Dict): Parameters for legacy v3 converter. Defaults to {}.
        scenario_name (str): Custom name for output folder. Uses p2m folder name if empty.

    Returns:
        str: Path to output folder containing converted DeepMIMO dataset.
        
    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If transmitter or receiver IDs are invalid.
    """
    if old: # v3
        return insite_rt_converter_v3(p2m_folder, tx_set_ids, rx_set_ids, old_params, scenario_name)
    
    # Setup output folder
    insite_sim_folder = os.path.dirname(p2m_folder)
    p2m_basename = os.path.basename(p2m_folder)
    out_fold_name = scenario_name if scenario_name else p2m_basename 
    
    output_folder = os.path.join(insite_sim_folder, out_fold_name + '_deepmimo')
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Read setup (.setup)
    setup_dict = read_setup(insite_sim_folder)

    # Read TXRX (.txrx)
    txrx_dict = read_txrx(insite_sim_folder, p2m_folder, output_folder)
    
    # Read and save path data
    read_paths(insite_sim_folder, p2m_folder, txrx_dict, output_folder)
    
    # Read Materials of Buildings, Terrain and Vegetation (.city, .ter, .veg)
    materials_dict = read_materials(insite_sim_folder, verbose=False)
    
    if convert_buildings:
        
        # Create scene from simulation folder
        scene = read_scene(insite_sim_folder)
        
        # Export scene data (save {building/terrain/vegetaion}_{faces/materials}.mat files)
        scene_dict = scene.export_data(output_folder)
        
        # Visualize if requested
        if vis_buildings:
            scene.plot_3d(show=True)#, save=True, filename=os.path.join(output_folder, 'scene_3d.png'))
    
    # Save params.mat
    cu.save_params_dict(output_folder, c.RAYTRACER_NAME_WIRELESS_INSITE, 
                       c.RAYTRACER_VERSION_WIRELESS_INSITE,
                       setup_dict, txrx_dict, materials_dict, scene_dict)
    
    # Save scenario to deepmimo scenarios folder
    scen_name = cu.save_scenario(output_folder, scen_name=scenario_name, overwrite=overwrite)
    
    print(f'Zipping DeepMIMO scenario (ready to upload!): {output_folder}')
    cu.zip_folder(output_folder) # ready for upload
    
    # Copy and zip ray tracing source files as well
    if copy_source:
        cu.save_rt_source_files(insite_sim_folder, SOURCE_EXTS)
    
    return scen_name