import os
import copy
import numpy as np

from ... import consts as c
from .construct_deepmimo import generate_MIMO_channel
from .utils import safe_print
from .params import Parameters
from .downloader import download_scenario_handler, extract_scenario
import scipy.io

def load_scenario(scen_name: str):
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

    dataset = load_raytracing_scene(scen_folder, rt_params)

    # If dynamic scenario, load each scene separately
    scene_list = rt_params[c.PARAMSET_DYNAMIC_SCENES]
    if len(scene_list) > 1: # dynamic scenario
        dataset = []
        for scene_i, scene in enumerate(scene_list):
            scene_folder = os.path.join(scen_folder, rt_params[c.PARAMSET_SCENARIO],
                                        'scene_' + str(scene))
            print(f'Scene {scene_i + 1}/{len(scene_list)}')
            dataset.append(load_raytracing_scene(scene_folder, rt_params))
    else: # static scenario
        dataset = load_raytracing_scene(scen_folder, rt_params)

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



def process_mat_data2(data):
    """
    Recursively process a Python dictionary loaded from a .mat file.
    Replaces data in cell arrays if suitable.
    """
    for key, value in data.items():
        data[key] = process_mat_data(value)
    
    return data  # Return processed or original data


def process_mat_data(data):
    """
    Recursively process a Python dictionary loaded from a .mat file.
    Replaces data based on custom logic, ensuring operations are applied only to numeric types.
    """
    if isinstance(data, dict):  # If it's a dictionary (MATLAB struct)
        for key, value in data.items():
            data[key] = process_mat_data(value)
    elif isinstance(data, np.ndarray):
        if data.dtype == 'object':  # Likely a cell array
            # Iterate over each cell and process
            return np.array([process_mat_data(cell) for cell in data.flat]).reshape(data.shape)
        elif np.issubdtype(data.dtype, np.number):  # Numeric array
            # Replace data based on custom logic (e.g., replace negatives with 0)
            return np.where(data < 0, 0, data)  # Example logic
        else:
            # Non-numeric arrays (e.g., strings) are returned unchanged
            return data
    return data  # Return processed or original data


def load_raytracing_scene(params):
    num_active_bs = len(params[c.PARAMSET_ACTIVE_BS])
    dataset = [{c.DICT_UE_IDX: dict(), c.DICT_BS_IDX: dict(), c.OUT_LOC: None}
               for x in range(num_active_bs)]
    
    # JTODO: iterate on tx sets & rx sets (instead of BS)
    for i in range(num_active_bs):
        bs_indx = params[c.PARAMSET_ACTIVE_BS][i]
        
        safe_print('\nBasestation %i' % bs_indx)
        
        safe_print('\nUE-BS Channels')
        (dataset[i][c.DICT_UE_IDX], dataset[i][c.OUT_LOC]) = \
            params['raytracing_fn'](bs_indx, params, user=True)

def generate_channels(dataset, params):
    
    if params_obj is None:
        params_obj = Parameters()
    elif type(params_obj) is str:
        params_obj = Parameters(params_obj)
    
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

    # Notify the user if something was not set correctly
    # JTODO: better have a object...
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

