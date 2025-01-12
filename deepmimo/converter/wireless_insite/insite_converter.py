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

from dataclasses import dataclass, asdict

from .setup_parser import tokenize_file, parse_document # for: .setup, .txrx, .city, .ter, .veg
from .paths_parser import paths_parser, extract_tx_pos  # for: .paths
#from .pl_parser import ..    # for: .pl

from .city_vis import city_vis

MATERIAL_FILES = ['.city', '.ter', '.veg']
SETUP_FILES = ['.setup', '.txrx'] + MATERIAL_FILES 

@dataclass
class InsiteMaterial():
    """
    Materials in Wireless InSite.
    
    Notes:
    - Diffuse model implemented from [1] + extended with cross-polarization scattering terms
    - Diffuse scattering models explained in [2], slides 29-31. 
    
    - At present, all MATERIALS in Wireless InSite are nonmagnetic, 
      and the permeability for all materials is that of free space 
      (µ0 = 4π x 10e-7 H/m) [3]. 

    Sources:
        [1] A Diffuse Scattering Model for Urban Propagation Prediction - Vittorio Degli-Esposti 2001
            https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=933491
        [2] https://x.webdo.cc/userfiles/Qiwell/files/Remcom_Wireless%20InSite_5G_final.pdf
        [3] Wireless InSite 3.3.0 Reference Manual, section 10.5 - Dielectric Parameters
    """
    # These field names match the names in Material section of the feature file
    name: str = ''
    diffuse_scattering_model: str = ''    # 'labertian', 'directive', 'directive_w_backscatter'
    fields_diffusively_scattered: int = 0 # 0-1, fraction of incident fields that are scattered
    cross_polarized_power: str = 0        # 0-1, fraction of the scattered field that is cross pol
    directive_alpha: int = 0     # 1-10, defines how broad forward beam is
    directive_beta: int = 0      # 1-10, defines how broad backscatter beam is
    directive_lambda: int = 0    # 0-1, fraction of the scattered power in forward direction (vs back)
    conductivity: int = 0        # >=0, conductivity
    permittivity: int = 0        # >=0, permittivity
    roughness: int = 0           # >=0, roughness
    thickness: int = 0           # >=0, thickness [m]


@dataclass
class InsiteTxRxSet():
    """
    TX/RX set class
    """
    name: str = ''
    id: int = 0   # Wireless Insite ID 
    idx: int = 0  # TxRxSet index for saving after conversion and generation
    # id -> idx example: [3, 5, 7] -> [1, 2, 3]
    is_tx: bool = False
    is_rx: bool = False
    
    num_points: int = 0    # all points
    inactive_idxs: tuple = ()  # list of indices of points with at least one path
    num_inactive_points: int = 0
    
    # Antenna elements of tx / rx
    tx_num_ant: int = 1
    rx_num_ant: int = 1
    
    dual_pol: bool = False # if '_dual-pol' in name
    
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
                        old: bool = False, old_params: Dict = {}):
    if old: # v3
        scen_name = insite_rt_converter_v3(p2m_folder, tx_set_ids, rx_set_ids, old_params)
        return scen_name
    
    # Setup output folder
    insite_sim_folder = os.path.dirname(p2m_folder)
    p2m_basename = os.path.basename(p2m_folder)
    
    output_folder = os.path.join(insite_sim_folder, p2m_basename + '_deepmimo')
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
    

def get_id_to_idx_map(txrx_dict: Dict):
    ids = [txrx_dict[key]['id'] for key in txrx_dict.keys()]
    idxs = [i + 1 for i in range(len(ids))]
    return {key:val for key, val in zip(ids, idxs)}


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
    
    # If we want to preallocate matrices, count lines
    # num_lines = sum(1 for _ in open(filename, 'rb'))
    
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


def read_config_file(file):
    return parse_document(tokenize_file(file))
    

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


def read_setup(setup_file: str, verbose: bool):
    document = read_config_file(setup_file)
    
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
    # setup_dict['binary_output_mode'] = apg_accel_vals['binary_output_mode']
    # setup_dict['binary_rate']        = apg_accel_vals['binary_rate']
    # setup_dict['database_mode']      = apg_accel_vals['database_mode']
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

    
def read_txrx(txrx_file, verbose: bool):
    print(f'Reading txrx file: {os.path.basename(txrx_file)}')
    document = read_config_file(txrx_file)
    n_tx, n_rx = 0, 0
    tx_ids, rx_ids = [], []
    txrx_objs = []
    for txrx_set_idx, key in enumerate(document.keys()):
        txrx = document[key]
        txrx_obj = InsiteTxRxSet()
        txrx_obj.name = key
        
        # Insite ID is used during ray tracing
        insite_id = (int(txrx.name[-1]) if txrx.name.startswith('project_id')
                     else txrx.values['project_id'])
        txrx_obj.id = insite_id 
        # TX/RX set ID is used to abstract from the ray tracing configurations
        # (how the DeepMIMO dataset will be saved and generated)
        txrx_obj.idx = txrx_set_idx + 1 # 1-indexed
        
        # Locations
        txrx_obj.loc_lat = txrx.values['location'].values['reference'].values['latitude']
        txrx_obj.loc_lon = txrx.values['location'].values['reference'].values['longitude']
        txrx_obj.coord_ref = txrx.values['location'].values['reference'].labels[1]
        
        # Is TX or RX?
        txrx_obj.is_tx = txrx.values['is_transmitter']
        txrx_obj.is_rx = txrx.values['is_receiver']
        
        # Antennas and Power
        if txrx_obj.is_tx:
            tx_ids += [insite_id]
            tx_vals = txrx.values['transmitter']
            assert tx_vals.values['power'] == 0.0, 'Tx power should be 0 dBm!'
            txrx_obj.tx_num_ant = tx_vals['pattern'].values['antenna']
        if txrx_obj.is_rx:
            rx_ids += [insite_id]
            rx_vals = txrx.values['receiver']
            txrx_obj.rx_num_ant = rx_vals['pattern'].values['antenna']
        
        # The number of tx/rx points inside set is updated when reading the p2m
        txrx_objs.append(txrx_obj)
    
    txrx_dict = {}
    for obj in txrx_objs:
        # Remove 'None' from dict (to be saved as .mat)
        obj_dict = {key: val for key, val in asdict(obj).items() if val is not None}
        
        # Index separate txrx-sets based on p_id
        txrx_dict = {**txrx_dict, **{f'txrx_set_{obj.idx}': obj_dict}}

    return tx_ids, rx_ids, txrx_dict


def read_material_files(files: List[str], verbose: bool):
    if verbose:
        print(f'Reading materials in {[os.path.basename(f) for f in files]}')
    
    # Extract materials for each file
    material_list = []
    for file in files:
        material_list += read_single_material_file(file, verbose)

    # Filter the list of materials so they are unique
    unique_mat_list = make_mat_list_unique(material_list)
    
    return unique_mat_list


def read_single_material_file(file: str, verbose: bool):
    document = read_config_file(file)
    direct_fields = ['diffuse_scattering_model', 'fields_diffusively_scattered', 
                     'cross_polarized_power', 'directive_alpha',
                     'directive_beta', 'directive_lambda']
    dielectric_fields = ['conductivity', 'permittivity', 'roughness', 'thickness']
    
    mat_objs = []
    for prim in document.keys():
        materials = document[prim].values['Material']
        materials = [materials] if type(materials) != list else materials
        for mat in materials:
            material_obj = InsiteMaterial()
            material_obj.name = mat.name
            for field in direct_fields:
                setattr(material_obj, field, mat.values[field])
            
            for field in dielectric_fields:
                setattr(material_obj, field, mat.values['DielectricLayer'].values[field])
            
            mat_objs += [material_obj]

    return mat_objs


def make_mat_list_unique(mat_list):
    
    n_mats = len(mat_list)
    idxs_to_discard = []
    for i1 in range(n_mats):
        for i2 in range(n_mats):
            if i1 == i2:
                continue
            if mat_list[i1].get_dict() == mat_list[i2].get_dict():
                idxs_to_discard.append(i2)

    for idx in sorted(np.unique(idxs_to_discard), reverse=True):
        del mat_list[idx]
    
    return mat_list


def read_materials(files_in_sim_folder, verbose):

    city_files = cu.ext_in_list('.city', files_in_sim_folder)
    ter_files  = cu.ext_in_list('.ter', files_in_sim_folder)
    veg_files  = cu.ext_in_list('.veg', files_in_sim_folder)
    fpl_files  = cu.ext_in_list('.flp', files_in_sim_folder)
    obj_files  = cu.ext_in_list('.obj', files_in_sim_folder)
    
    city_materials = read_material_files(city_files, verbose)
    ter_materials = read_material_files(ter_files, verbose)
    veg_materials = read_material_files(veg_files, verbose)
    floor_plan_materials = read_material_files(fpl_files, verbose)
    obj_materials = read_material_files(obj_files, verbose)

    materials_dict = {'city': city_materials, 
                      'terrain': ter_materials,
                      'vegetation': veg_materials,
                      'floorplans': floor_plan_materials,
                      'obj_materials': obj_materials}
    if verbose:
        pprint(materials_dict)
    return materials_dict


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