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
import re
from pprint import pprint
import shutil
from typing import List, Dict, Tuple, Optional

# Third-party imports
import numpy as np
import scipy.io

# Local imports
from .. import converter_utils as cu
from ...general_utilities import PrintIfVerbose
from ... import consts as c
from .insite_materials import read_materials
from .insite_setup import read_setup
from .insite_scene import create_scene_from_folder

# v3 (old)
from .ChannelDataLoader import WIChannelConverter
from .ChannelDataFormatter import DeepMIMODataFormatter
from .insite_txrx import create_txrx_from_folder
from .insite_paths import create_paths_from_folder

# Constants
MATERIAL_FILES = ['.city', '.ter', '.veg']
SETUP_FILES = ['.setup', '.txrx'] + MATERIAL_FILES

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

    # Check if necessary files exist
    verify_sim_folder(insite_sim_folder, verbose)
    
    # Copy ray tracing source files
    if copy_source:
        copy_rt_source_files(insite_sim_folder, verbose)
    
    # Read setup (.setup)
    setup_dict = read_setup(insite_sim_folder)

    # Read TXRX (.txrx)
    txrx_dict = create_txrx_from_folder(insite_sim_folder, p2m_folder, output_folder)
    
    # Read and save path data
    create_paths_from_folder(insite_sim_folder, p2m_folder, txrx_dict, output_folder)
    
    # Read Materials of Buildings, Terrain and Vegetation (.city, .ter, .veg)
    materials_dict = read_materials(insite_sim_folder, verbose=False)
    
    if convert_buildings:
        
        # Create scene from simulation folder
        scene = create_scene_from_folder(insite_sim_folder)
        
        # Export scene data (save {building/terrain/vegetaion}_{faces/materials}.mat files)
        scene_dict = scene.export_data(output_folder)
        
        # Visualize if requested
        if vis_buildings:
            scene.plot_3d(show=True)#, save=True, filename=os.path.join(output_folder, 'scene_3d.png'))
    
    # Export params.mat
    export_params_dict(output_folder, setup_dict, txrx_dict, materials_dict, scene_dict)
    
    # Move scenario to deepmimo scenarios folder
    scen_name = export_scenario(output_folder, scen_name=scenario_name, overwrite=overwrite)
    
    print(f'Zipping DeepMIMO scenario (ready to upload!): {output_folder}')
    cu.zip_folder(output_folder) # ready for upload
    
    return scen_name


def verify_sim_folder(sim_folder: str, verbose: bool) -> None:
    """Verify that required simulation files exist.
    
    Args:
        sim_folder (str): Path to simulation folder.
        verbose (bool): Whether to print progress messages.
        
    Raises:
        Exception: If required files are missing or duplicated.
    """
    files_in_sim_folder = os.listdir(sim_folder)
    for ext in ['.setup', '.txrx']:
        files_found_with_ext = cu.ext_in_list(ext, files_in_sim_folder)
        if verbose:
            print(f'Found {files_found_with_ext}')
        if len(files_found_with_ext) == 0:
            raise Exception(f'{ext} not found in {sim_folder}')
        elif len(files_found_with_ext) > 1:
            raise Exception(f'Several {ext} found in {sim_folder}')


def copy_rt_source_files(sim_folder: str, verbose: bool = True) -> None:
    """Copy raytracing source files to a new directory.
    
    Args:
        sim_folder (str): Path to simulation folder.
        verbose (bool): Whether to print progress messages. Defaults to True.
    """
    vprint = PrintIfVerbose(verbose) # prints if verbose 
    rt_source_folder = os.path.basename(sim_folder) + '_raytracing_source'
    files_in_sim_folder = os.listdir(sim_folder)
    print(f'Copying raytracing source files to {rt_source_folder}')
    zip_temp_folder = os.path.join(sim_folder, rt_source_folder)
    os.makedirs(zip_temp_folder)
    for ext in ['.setup', '.txrx', '.ter', '.city', '.kmz']:
        # copy all files with extensions to temp folder
        for file in cu.ext_in_list(ext, files_in_sim_folder):
            curr_file_path = os.path.join(sim_folder, file)
            new_file_path  = os.path.join(zip_temp_folder, file)
            
            # vprint(f'Adding {file}')
            shutil.copy(curr_file_path, new_file_path)
    
    vprint('Zipping')
    cu.zip_folder(zip_temp_folder)
    
    vprint(f'Deleting temp folder {os.path.basename(zip_temp_folder)}')
    shutil.rmtree(zip_temp_folder)
    
    vprint('Done')


def export_params_dict(output_folder: str, *dicts: Dict) -> None:
    """Export parameter dictionaries to a .mat file.
    
    Args:
        output_folder (str): Output directory path.
        *dicts: Variable number of dictionaries to merge and export.
    """
    base_dict = {
        c.LOAD_FILE_SP_VERSION: c.VERSION,
        c.LOAD_FILE_SP_RAYTRACER: c.RAYTRACER_NAME_WIRELESS_INSITE,
        c.LOAD_FILE_SP_RAYTRACER_VERSION: c.RAYTRACER_VERSION_WIRELESS_INSITE,
        c.PARAMSET_DYNAMIC_SCENES: 0, # only static currently
    }
    
    merged_dict = base_dict.copy()
    for d in dicts:
        merged_dict.update(d)
        
    pprint(merged_dict)
    scipy.io.savemat(os.path.join(output_folder, 'params.mat'), merged_dict)


def export_scenario(sim_folder: str, scen_name: str = '', 
                    overwrite: Optional[bool] = None) -> Optional[str]:
    """Export scenario to the DeepMIMO scenarios folder.
    
    Args:
        sim_folder (str): Path to simulation folder.
        overwrite (Optional[bool]): Whether to overwrite existing scenario. Defaults to None.
        
    Returns:
        Optional[str]: Name of the exported scenario.
    """
    default_scen_name = os.path.basename(os.path.dirname(sim_folder.replace('_deepmimo', '')))
    scen_name = scen_name if scen_name else default_scen_name
    scen_path = c.SCENARIOS_FOLDER + f'/{scen_name}'
    if os.path.exists(scen_path):
        if overwrite is None:
            print(f'Scenario with name "{scen_name}" already exists in '
                  f'{c.SCENARIOS_FOLDER}. Delete? (Y/n)')
            ans = input()
            overwrite = False if 'n' in ans.lower() else True
        if overwrite:
            shutil.rmtree(scen_path)
        else:
            return None
    
    shutil.copytree(sim_folder, scen_path)
    return scen_name