"""
Converts Wireless Insite raytracing files into DeepMIMO scenarios ready to upload.

TODOS:
    - support multi-antennas (includes polarization)
    - support dynamic scenarios
    - (optional) dictionary mapping between Wireless Insite and DeepMIMO names
    - (optional) expand support multiple tx_ids per tx_set
      (requires reading number of ids from .txrx and use them to index files right)
    - (optional) if we decide to drop the inactive positions, the changes are simple:
      1- remove "active_points<..>.mat"
      2- remove inactive positions from position array

Requirements:

    - keep the transmit power at its default (0 dBm)
    - for dual polarization, end the name of the antenna as '_pol', so the 
      converter knows alternated antennas should be considered for different 
      polarizations. Otherwise, a single polarization is used
    - Request the folowing outputs: path loss (.pl file) and paths (.path file)
      (the .pl.p2m is only needed to get the positions of points without paths)
    - across a tx/rx set, the same antenna is used
"""


import os
import re
from pprint import pprint # for debugging
import shutil
import numpy as np
import scipy.io

from .. import converter_utils as cu
from ...general_utilities import PrintIfVerbose, get_mat_filename
from ... import consts as c

from .ChannelDataLoader import WIChannelConverter
from .ChannelDataFormatter import DeepMIMODataFormatter

from typing import List, Dict

from .paths_parser import paths_parser, extract_tx_pos  # for: .paths

from .city_vis import city_vis

from .insite_materials import read_materials
from .insite_setup import read_setup
from .insite_txrx import read_txrx, get_id_to_idx_map

MATERIAL_FILES = ['.city', '.ter', '.veg']
SETUP_FILES = ['.setup', '.txrx'] + MATERIAL_FILES 


def insite_rt_converter_v3(p2m_folder, tx_ids, rx_ids, params_dict):
    # P2Ms (.cir, .doa, .dod, .paths[.t001_{tx_id}.r{rx_id}.p2m] eg: .t001_01.r001.p2m)
    
    insite_sim_folder = os.path.dirname(p2m_folder)

    intermediate_folder = os.path.join(insite_sim_folder, 'intermediate_files')
    output_folder = os.path.join(insite_sim_folder, 'mat_files') # SCEN_NAME!
    
    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Convert P2M files to mat format
    WIChannelConverter(p2m_folder, intermediate_folder)

    DeepMIMODataFormatter(intermediate_folder, output_folder, 
                          TX_order=tx_ids, RX_order=rx_ids)
    
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
    scen_name = os.path.basename(os.path.dirname(output_folder))
    scen_path = c.SCENARIOS_FOLDER + f'/{scen_name}'
    if os.path.exists(scen_path):
        shutil.rmtree(scen_path)
    shutil.copytree(output_folder, './' + scen_path)
    
    return output_folder


def insite_rt_converter(p2m_folder: str, copy_source: bool = False,
                        tx_set_ids: List[int] = None, rx_set_ids: List[int] = None,
                        verbose: bool = True, overwrite: bool | None = None, 
                        vis_buildings: bool = False, 
                        old: bool = False, old_params: Dict = {},
                        scenario_name=''):
    if old: # v3
        scen_name = insite_rt_converter_v3(p2m_folder, tx_set_ids, rx_set_ids, old_params)
        return scen_name
    
    # Setup output folder
    insite_sim_folder = os.path.dirname(p2m_folder)
    p2m_basename = os.path.basename(p2m_folder)
    scenario_name = scenario_name if scenario_name else p2m_basename 
    
    output_folder = os.path.join(insite_sim_folder, scenario_name + '_deepmimo')
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Check if necessary files exist
    verify_sim_folder(insite_sim_folder, verbose)
    
    # Copy ray tracing source files
    if copy_source:
        copy_rt_source_files(insite_sim_folder, verbose)
    
    files_in_sim_folder = [os.path.join(insite_sim_folder, file) 
                           for file in os.listdir(insite_sim_folder)]
    
    # Read setup (.setup)
    setup_file = cu.ext_in_list('.setup', files_in_sim_folder)[0]
    setup_dict = read_setup(setup_file, verbose=False)

    # Read TXRX (.txrx)
    txrx_file = cu.ext_in_list('.txrx', files_in_sim_folder)[0]
    avail_tx_set_ids, avail_rx_set_ids, txrx_dict = read_txrx(txrx_file, verbose)
    
    tx_set_ids = tx_set_ids if tx_set_ids else avail_tx_set_ids
    rx_set_ids = rx_set_ids if rx_set_ids else avail_rx_set_ids
    
    # Instead of Wireless Insite TX/RX SET IDs, we save and use only indices
    id_to_idx_map = get_id_to_idx_map(txrx_dict)
    
    # Read Materials of Buildings, Terrain and Vegetation (.city, .ter, .veg)
    materials_dict = read_materials(files_in_sim_folder, verbose=False)
    
    # Save Position Matrices and Populate Number of Points in Each TxRxSet 
    # NOTE: only necessary for the inactive positions. Active pos exists in .paths
    for tx_set_id in tx_set_ids: 
        for rx_set_id in rx_set_ids:
            # <Project name>.pl.t<tx number> <tx set number>.r<rx set number>.p2m
            proj_name = os.path.basename(insite_sim_folder)
            for tx_idx, tx_id in enumerate([1]): # We assume each TX/RX SET only has one BS    
                # 1- generate file names based on active txrx sets    
                base_filename = f'{proj_name}.pl.t{tx_id:03}_{tx_set_id:02}.r{rx_set_id:03}.p2m'
                pl_p2m_file = os.path.join(p2m_folder, base_filename)
                
                # 2- extract all rx positions from pathloss.p2m
                rx_pos, _, path_loss = read_pl_p2m_file(pl_p2m_file)
                
                rx_set_idx = id_to_idx_map[rx_set_id]
                tx_set_idx = id_to_idx_map[tx_set_id]
                
                save_mat(rx_pos, c.RX_POS_PARAM_NAME, output_folder, 
                         tx_set_idx, tx_idx, rx_set_idx)
                
                # 3- update number of (active/inactive) points in txrx sets
                txrx_dict[f'txrx_set_{rx_set_idx}']['num_points'] = rx_pos.shape[0]
                
                inactive_idxs = np.where(path_loss == 250.)[0]
                txrx_dict[f'txrx_set_{rx_set_idx}']['inactive_idxs'] = inactive_idxs
                txrx_dict[f'txrx_set_{rx_set_idx}']['num_inactive_points'] = len(inactive_idxs)
                
                # 4- save all path information
                # Paths P2M (.paths[.t{tx_id}_{??}.r{rx_id}.p2m] e.g. .t001_01.r001.p2m)
                paths_p2m_file = pl_p2m_file.replace('.pl.', '.paths.')
                data = paths_parser(paths_p2m_file)
                
                for key in data.keys():
                    save_mat(data[key], key, output_folder, tx_set_idx, tx_idx, rx_set_idx)
                
                # 5- also read tx position from path files
                # (can be done in many ways, but this is easiest on code & user requirements)
                tx_pos = extract_tx_pos(paths_p2m_file)
                save_mat(tx_pos, c.TX_POS_PARAM_NAME, output_folder, 
                         tx_set_idx, tx_idx, rx_set_idx)
                
    # Export params.mat
    export_params_dict(output_folder, setup_dict, txrx_dict, materials_dict)
    
    # Move scenario to deepmimo scenarios folder
    scen_name = export_scenario(output_folder, overwrite=overwrite)
    
    print(f'Zipping DeepMIMO scenario (ready to upload!): {output_folder}')
    cu.zip_folder(output_folder) # ready for upload
    
    if vis_buildings:
        city_files = cu.ext_in_list('.city', files_in_sim_folder)
        if city_files:
            city_vis(city_files[0])
        
    return scen_name


def save_mat(data, data_key, output_folder, tx_set_idx, tx_idx, rx_set_idx):
    mat_file_name = get_mat_filename(data_key, tx_set_idx, tx_idx, rx_set_idx)
    file_path = output_folder + '/' + mat_file_name
    scipy.io.savemat(file_path, {c.MAT_VAR_NAME: data})
    

def read_pl_p2m_file(filename: str):
    """
    Returns xyz, distance, pl from p2m file.
    """
    assert filename.endswith('.p2m') # should be a .p2m file
    assert '.pl.' in filename        # should be the pathloss p2m

    # Initialize empty lists for matrices
    xyz_list = []
    dist_list = []
    path_loss_list = []

    # Define (regex) patterns to match numbers (optionally signed floats)
    re_data = r"-?\d+\.?\d*"
    
    # To preallocate matrices, count lines: sum(1 for _ in open(filename, 'rb'))
    
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    
    for line in lines:
        if line[0] != '#':
            data = re.findall(re_data, line)
            xyz_list.append([float(data[1]), float(data[2]), float(data[3])]) # XYZ (m)
            dist_list.append([float(data[4])])       # distance (m)
            path_loss_list.append([float(data[5])])  # path loss (dB)

    # Convert lists to numpy arrays
    xyz_matrix = np.array(xyz_list, dtype=np.float32)
    dist_matrix = np.array(dist_list, dtype=np.float32)
    path_loss_matrix = np.array(path_loss_list, dtype=np.float32)

    return xyz_matrix, dist_matrix, path_loss_matrix


def verify_sim_folder(sim_folder: str, verbose: bool):
    
    files_in_sim_folder = os.listdir(sim_folder)
    for ext in ['.setup', '.txrx']:
        files_found_with_ext = cu.ext_in_list(ext, files_in_sim_folder)
        if verbose:
            print(f'Found {files_found_with_ext}')
        if len(files_found_with_ext) == 0:
            raise Exception(f'{ext} not found in {sim_folder}')
        elif len(files_found_with_ext) > 1:
            raise Exception(f'Several {ext} found in {sim_folder}')


def copy_rt_source_files(sim_folder: str, verbose: bool = True):
    
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


def export_params_dict(output_folder: str, setup_dict: Dict = {},
                       txrx_dict: Dict = {}, mat_dict: Dict = {}):
    
    data_dict = {
        c.LOAD_FILE_SP_VERSION: c.VERSION,
        c.LOAD_FILE_SP_RAYTRACER: c.RAYTRACER_NAME_WIRELESS_INSITE,
        c.LOAD_FILE_SP_RAYTRACER_VERSION: c.RAYTRACER_VERSION_WIRELESS_INSITE,
        c.PARAMSET_DYNAMIC_SCENES: 0, # only static currently
    }
    
    merged_dict = {**data_dict, **setup_dict, **txrx_dict, **mat_dict}
    pprint(merged_dict)
    scipy.io.savemat(os.path.join(output_folder, 'params.mat'), merged_dict)


def export_scenario(sim_folder, overwrite: bool | None = None):
    scen_name = os.path.basename(os.path.dirname(sim_folder.replace('_deepmimo', '')))
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