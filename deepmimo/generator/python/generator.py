import os
import copy
import numpy as np

from ... import consts as c
from .construct_deepmimo import generate_MIMO_channel
from .utils import safe_print
from .params import Parameters
from .downloader import download_scenario_handler, extract_scenario
import scipy.io
from typing import List, Dict

from ...general_utilities import get_mat_filename

def load_scenario(scen_name: str, **load_params):
    if '\\' in scen_name or '/' in scen_name:
        scen_folder = scen_name
        scen_name = os.path.basename(scen_folder)
    else:
        # Use default deepmimo scenario folder
        scen_folder = os.path.join(c.SCENARIOS_FOLDER, scen_name)
    
    if not os.path.exists(scen_folder):
        print('Scenario not found. Would you like to download it? [Y/n]')
        ans = input()
        if not ('n' in ans.lower()):
            zip_path = download_scenario_handler(scen_name)
            extract_scenario(zip_path)
    
    params_mat_file = os.path.join(scen_folder, 'params.mat')
    rt_params = load_mat_file_as_dict(params_mat_file)
    
    # If dynamic scenario, load each scene separately
    n_scenes = rt_params[c.PARAMSET_DYNAMIC_SCENES]
    if n_scenes > 1: # dynamic
        dataset = []
        for scene_i in range(n_scenes):
            scene_folder = os.path.join(scen_folder, rt_params[c.PARAMSET_SCENARIO],
                                        f'scene_{scene_i}')
            print(f'Scene {scene_i + 1}/{n_scenes}')
            dataset.append(load_raytracing_scene(scene_folder, rt_params, **load_params))
    else: # static 
        dataset = load_raytracing_scene(scen_folder, rt_params, **load_params)

    return dataset


def mat_struct_to_dict(mat_struct):
    """
    Recursively converts scipy.io.mat_struct objects to Python dictionaries.
    """
    if isinstance(mat_struct, scipy.io.matlab.mio5_params.mat_struct):
        result = {}
        for field in mat_struct._fieldnames:
            result[field] = mat_struct_to_dict(getattr(mat_struct, field))
        return result
    elif isinstance(mat_struct, np.ndarray):
        # Process arrays recursively in case they contain mat_structs
        return np.array([mat_struct_to_dict(item) for item in mat_struct])
    return mat_struct  # Return the object as is for other types

def load_mat_file_as_dict(file_path):
    """
    Loads a .mat file and converts any mat_struct objects to Python dictionaries.
    """
    mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    return {key: mat_struct_to_dict(value) for key, value in mat_data.items()
            if not key.startswith('__')}


def load_raytracing_scene(folder, rt_params, 
                          tx_sets: Dict | List | str = 'all',
                          rx_sets: Dict | List | str = 'all'):
    
    tx_sets = validate_txrx_sets(tx_sets, rt_params, 'tx')
    rx_sets = validate_txrx_sets(rx_sets, rt_params, 'rx')
    
    dataset_dict = {}
    
    bs_idxs = []
    for tx_set_idx, tx_idxs in tx_sets.items():
        for rx_set_idx, rx_idxs in rx_sets.items():
            for tx_idx in tx_idxs:
                # Compute new index on the tx level - a BS relative index
                # e.g. tx_set_1: [0, 1, 2] and  tx_set_2: [0], then new_idxs = [0,1,2,3]
                bs_idx = len(bs_idxs)
                bs_idxs.append(bs_idx)

                print(f'\nTX set: {tx_set_idx} (basestation)')
                
                rx_id_str = 'basestation' if rx_set_idx == tx_set_idx else 'users'
                print(f'RX set: {rx_set_idx} ({rx_id_str})')
                
                dataset_dict[bs_idx] = load_tx_rx_raydata(folder,
                                                          tx_set_idx, rx_set_idx,
                                                          tx_idx, rx_idxs)
    return dataset_dict

def validate_txrx_sets(sets: Dict | List | str, rt_params: Dict, tx_or_rx: str = 'tx'):
    """
    Ensures the input to the generator is compatible with the available tx/rx sets
    and their points.
    """
    valid_tx_set_idxs = []
    valid_rx_set_idxs = []
    for key, val in rt_params.items():
        if key.startswith('txrx_set_'):
            if val['is_tx']:
                valid_tx_set_idxs.append(val['idx'])
            if val['is_rx']:
                valid_rx_set_idxs.append(val['idx'])
    
    valid_set_idxs = valid_tx_set_idxs if tx_or_rx == 'tx' else valid_rx_set_idxs
    set_str = 'Tx' if tx_or_rx == 'tx' else 'Rx'
    
    info_str = "To see supported TX/RX sets and indices run dm.info(<scenario_name>)"
    if type(sets) is dict:
        for set_idx, idxs in sets.items():    
            # check the the tx/rx_set indices are valid
            if set_idx not in valid_set_idxs:
                raise Exception(f"{set_str} set {set_idx} not in allowed sets {valid_set_idxs}\n"
                                + info_str)
            
            all_idxs_available = np.arange(rt_params[f'txrx_set_{set_idx}']['num_points'])
            if type(idxs) is np.ndarray:
                pass # correct
            elif type(idxs) is list:
                sets[set_idx] = np.array(idxs)
            elif idxs == 'all':
                sets[set_idx] = all_idxs_available
            else:
                raise Exception('Only <list> of <np.ndarray> allowed as tx/rx sets indices')
                
            # check that the specific tx/rx indices inside the sets are valid
            if not set(idxs).issubset(all_idxs_available):
                raise Exception(f'Some indices of {idxs} are not in {all_idxs_available}. '
                                + info_str)
        sets_dict = sets
    elif type(sets) is list:
        # Generate all user indices
        sets_dict = {}
        for set_idx in sets:
            if set_idx not in valid_set_idxs:
                raise Exception(f"{set_str} set {set_idx} not in allowed sets {valid_set_idxs}\n"
                                + info_str)
                
            sets_dict[set_idx] = np.arange(rt_params[f'txrx_set_{set_idx}']['num_points'])
    elif type(sets) is str:
        if sets != 'all':
            raise Exception(f"String '{sets}' not understood. Only string allowed "
                            "is 'all' to generate all available sets and indices")
        
        # Generate dict with all sets and indices available
        sets_dict = {}
        for set_idx in valid_set_idxs:
            sets_dict[set_idx] = np.arange(rt_params[f'txrx_set_{set_idx}']['num_points'])
    
    return sets_dict


def load_tx_rx_raydata(rayfolder, tx_set_idx, rx_set_idx, tx_idx, rx_idxs):
    
    tx_dict = {c.AOA_AZ_PARAM_NAME: None,
               c.AOA_EL_PARAM_NAME: None,
               c.AOD_AZ_PARAM_NAME: None,
               c.AOD_EL_PARAM_NAME: None,
               c.TOA_PARAM_NAME: None,
               c.PWR_PARAM_NAME: None,
               c.PHASE_PARAM_NAME: None,
               c.RX_POS_PARAM_NAME: None,
               c.TX_POS_PARAM_NAME: None,
               c.INTERACTIONS_PARAM_NAME: None,
               c.INTERACTIONS_POS_PARAM_NAME: None}
    
    for key in tx_dict.keys():
        
        load_key = key
        
        # Small logic to prevent writing repeated files for rx and tx locations
        if key in [c.RX_POS_PARAM_NAME, c.TX_POS_PARAM_NAME]:
            load_key = c.POS_MAT_NAME
        rx_set_to_load = rx_set_idx if key != c.TX_POS_PARAM_NAME else tx_set_idx
        
        mat_filename = get_mat_filename(load_key, tx_set_idx, tx_idx, rx_set_to_load)
        
        mat_path = os.path.join(rayfolder, mat_filename)
        
        if os.path.exists(mat_path):
            print(f'Loading {mat_filename}..')
            tx_dict[key] = scipy.io.loadmat(mat_path)['data']#[rx_idxs]
        else:
            print(f'File {mat_path} could not be found')

    return tx_dict

def generate_channels(dataset, params):
    
    if params is None:
        params = Parameters()
    elif type(params) is str:
        params = Parameters(params)
        
    np.random.seed(1001)
    
    validate_ch_gen_params(params)
    
    num_active_bs = len(params[c.PARAMSET_ACTIVE_BS])
    for i in range(num_active_bs):
        (dataset[i][c.DICT_UE_IDX][c.OUT_CHANNEL], 
            dataset[i][c.DICT_UE_IDX][c.OUT_LOS]) = \
            generate_MIMO_channel(raydata=dataset[i][c.DICT_UE_IDX][c.OUT_PATH], 
                                  params=params, 
                                  tx_ant_params=params[c.PARAMSET_ANT_BS][i], 
                                  rx_ant_params=params[c.PARAMSET_ANT_UE])
                
    return dataset

# TODO: Move validation into another file
def validate_ch_gen_params(params):

    # Notify the user if some keyword is not used (likely set incorrectly)
    additional_keys = compare_two_dicts(params, Parameters().get_params_dict())
    if len(additional_keys):
        print('The following parameters seem unnecessary:')
        print(additional_keys)
    
    params['dynamic_scenario'] = is_dynamic_scenario(params)
    
    params['data_version'] = check_data_version(params)
    params[c.PARAMSET_SCENARIO_PARAMS_PATH] = get_scenario_params_path(params)

    # Active user IDs and related parameter
    assert_str = f"The subsampling parameter '{c.PARAMSET_USER_SUBSAMP}' needs to be in (0, 1]"
    assert params[c.PARAMSET_USER_SUBSAMP] > 0 and params[c.PARAMSET_USER_SUBSAMP] <= 1, assert_str
    
    # BS antenna format
    params[c.PARAMSET_ANT_BS_DIFF] = True
    if type(params[c.PARAMSET_ANT_BS]) is dict: # Replicate BS Antenna for each active BS in a list
        ant = params[c.PARAMSET_ANT_BS]
        params[c.PARAMSET_ANT_BS] = []
        for i in range(len(params[c.PARAMSET_ACTIVE_BS])):
            params[c.PARAMSET_ANT_BS].append(ant)
    else:
        if len(params[c.PARAMSET_ACTIVE_BS]) == 1:
            params[c.PARAMSET_ANT_BS_DIFF] = False 
            
    # BS Antenna Rotation
    for i in range(len(params[c.PARAMSET_ACTIVE_BS])):
        if (c.PARAMSET_ANT_ROTATION in params[c.PARAMSET_ANT_BS][i].keys() and \
            params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_ROTATION] is not None):
            rotation_shape = params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_ROTATION].shape
            assert  (len(rotation_shape) == 1 and rotation_shape[0] == 3) \
                    ,'The BS antenna rotation must be a 3D vector'
                    
        else:
            params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_ROTATION] = None                                            
      
    # UE Antenna Rotation
    if (c.PARAMSET_ANT_ROTATION in params[c.PARAMSET_ANT_UE].keys() and \
        params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] is not None):
        rotation_shape = params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION].shape
        cond_1 = len(rotation_shape) == 1 and rotation_shape[0] == 3
        cond_2 = len(rotation_shape) == 2 and rotation_shape[0] == 3 and rotation_shape[1] == 2
        cond_3 = rotation_shape[0] == len(params[c.PARAMSET_ACTIVE_UE])
        
        assert_str = ('The UE antenna rotation must either be a 3D vector for ' +
                      'constant values or 3 x 2 matrix for random values')
        assert cond_1 or cond_2 or cond_3, assert_str
                
        if len(rotation_shape) == 1 and rotation_shape[0] == 3:
            rotation = np.zeros((len(params[c.PARAMSET_ACTIVE_UE]), 3))
            rotation[:] =  params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION]
            params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = rotation
        elif (len(rotation_shape) == 2 and rotation_shape[0] == 3 and rotation_shape[1] == 2):
            params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = np.random.uniform(
                              params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION][:, 0], 
                              params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION][:, 1], 
                              (len(params[c.PARAMSET_ACTIVE_UE]), 3))
    else:
        params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = \
            np.array([None] * len(params[c.PARAMSET_ACTIVE_UE])) # List of None
     
    # BS Antenna Radiation Pattern
    for i in range(len(params[c.PARAMSET_ACTIVE_BS])):
        if c.PARAMSET_ANT_RAD_PAT in params[c.PARAMSET_ANT_BS][i].keys():
            assert_str = (f"The antenna radiation pattern for BS-{i} must have " + 
                          f"one of the following values: {str(c.PARAMSET_ANT_RAD_PAT_VALS)}")
            assert params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_RAD_PAT] in c.PARAMSET_ANT_RAD_PAT_VALS, assert_str
        else:
            params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]
                     
    # UE Antenna Radiation Pattern
    if c.PARAMSET_ANT_RAD_PAT in params[c.PARAMSET_ANT_UE].keys():
        assert_str = ("The antenna radiation pattern for UEs must have one of the " + 
                      f"following values: {str(c.PARAMSET_ANT_RAD_PAT_VALS)}")
        assert params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT] in c.PARAMSET_ANT_RAD_PAT_VALS, assert_str
    else:
        params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]
                                             
    return params


def is_dynamic_scenario(params):
    return 'dyn' in params[c.PARAMSET_SCENARIO]

def check_data_version(params):
    v3_params_path = get_scenario_params_path(params)
    return os.path.isfile(v3_params_path)
    
    
def get_scenario_params_path(params):
    folder_path = os.path.abspath(params[c.PARAMSET_DATASET_FOLDER])
    return os.path.join(folder_path, params[c.PARAMSET_SCENARIO], 'params.mat')

def compare_two_dicts(dict1, dict2):
    
    additional_keys = dict1.keys() - dict2.keys()
    for key, item in dict1.items():
        if isinstance(item, dict):
            if key in dict2:
                additional_keys = additional_keys | compare_two_dicts(dict1[key], dict2[key])

    return additional_keys

