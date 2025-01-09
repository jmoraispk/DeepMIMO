import os
import copy
import numpy as np

from ... import consts as c
from .construct_deepmimo import generate_MIMO_channel
from .utils import safe_print
from .params import Parameters
from .downloader import download_scenario_handler, extract_scenario

def generate_data(params_obj=None):
    
    if params_obj is None:
        params_obj = Parameters()
    
    if type(params_obj) is str:
        params_obj = Parameters(params_obj)
    
    np.random.seed(1001)
    
    ext_params = params_obj.get_params_dict()

    if not os.path.exists(params_obj.get_path()):
        print('Scenario not found. Would you like to download it? (Y/n)')
        ans = input()
        if not ('n' in ans.lower()):
            zip_path = download_scenario_handler(params_obj.get_name())
            extract_scenario(zip_path)
        
    try:
        params = validate_params(copy.deepcopy(ext_params))
    except FileNotFoundError:
        print('Scenario not found. ')
        return
            
    # If dynamic scenario
    if is_dynamic_scenario(params):
        scene_list = params[c.PARAMSET_DYNAMIC_SCENES]
        num_of_scenes = len(scene_list)
        dataset = []
        for scene_i in range(num_of_scenes):
            scene = scene_list[scene_i]
            params[c.PARAMSET_SCENARIO_FIL] = \
                os.path.join(os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]), 
                             params[c.PARAMSET_SCENARIO],
                             'scene_' + str(scene))
                
            print('\nScene %i/%i' % (scene_i+1, num_of_scenes))
            dataset.append(generate_scene_data(params))
    else: # static scenario
        params[c.PARAMSET_SCENARIO_FIL] = \
            os.path.join(os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]), 
                         params[c.PARAMSET_SCENARIO])
            
        dataset = generate_scene_data(params)
    return dataset

def generate_scene_data(params):
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
    num_active_bs = len(params[c.PARAMSET_ACTIVE_BS])
    for i in range(num_active_bs):
        (dataset[i][c.DICT_UE_IDX][c.OUT_CHANNEL], 
            dataset[i][c.DICT_UE_IDX][c.OUT_LOS]) = \
            generate_MIMO_channel(dataset[i][c.DICT_UE_IDX][c.OUT_PATH], 
                                    params, 
                                    params[c.PARAMSET_ANT_BS][i], 
                                    params[c.PARAMSET_ANT_UE])
                
    return dataset

# TODO: Move validation into another script
def validate_params(params):

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
    assert params[c.PARAMSET_USER_SUBSAMP] > 0 and params[c.PARAMSET_USER_SUBSAMP] <= 1, 'The subsampling parameter \'%s\' needs to be in (0, 1]'%c.PARAMSET_USER_SUBSAMP
    
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
        if c.PARAMSET_ANT_ROTATION in params[c.PARAMSET_ANT_BS][i].keys() and params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_ROTATION] is not None:
            rotation_shape = params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_ROTATION].shape
            assert  (len(rotation_shape) == 1 and rotation_shape[0] == 3) \
                    ,'The BS antenna rotation must be a 3D vector'
                    
        else:
            params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_ROTATION] = None                                            
      
    # UE Antenna Rotation
    if c.PARAMSET_ANT_ROTATION in params[c.PARAMSET_ANT_UE].keys() and params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] is not None:
        rotation_shape = params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION].shape
        assert (len(rotation_shape) == 1 and rotation_shape[0] == 3) or \
                (len(rotation_shape) == 2 and rotation_shape[0] == 3 and rotation_shape[1] == 2) or \
                (rotation_shape[0] == len(params[c.PARAMSET_ACTIVE_UE])) \
                ,'The UE antenna rotation must either be a 3D vector for constant values or 3 x 2 matrix for random values'
                
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
        params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = np.array([None] * len(params[c.PARAMSET_ACTIVE_UE])) # List of None
     
    # BS Antenna Radiation Pattern
    for i in range(len(params[c.PARAMSET_ACTIVE_BS])):
        if c.PARAMSET_ANT_RAD_PAT in params[c.PARAMSET_ANT_BS][i].keys():
            assert_str = f"The antenna radiation pattern for BS-{i} must have one of the following values: [{', '.join(c.PARAMSET_ANT_RAD_PAT_VALS)}]"
            assert params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_RAD_PAT] in c.PARAMSET_ANT_RAD_PAT_VALS, assert_str
        else:
            params[c.PARAMSET_ANT_BS][i][c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]
                     
    # UE Antenna Radiation Pattern
    if c.PARAMSET_ANT_RAD_PAT in params[c.PARAMSET_ANT_UE].keys():
        assert_str = f"The antenna radiation pattern for UEs must have one of the following values: [{', '.join(c.PARAMSET_ANT_RAD_PAT_VALS)}]"
        assert params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT] in c.PARAMSET_ANT_RAD_PAT_VALS, assert_str
    else:
        params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]
                                             
    return params


def is_dynamic_scenario(params):
    return 'dyn' in params[c.PARAMSET_SCENARIO]

def check_data_version(params):
    v3_params_path = os.path.join(os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]), 
                                    params[c.PARAMSET_SCENARIO],
                                    'params.mat')
    if os.path.isfile(v3_params_path):
        return 'v3'
    else:
        return 'v2'
    
def get_scenario_params_path(params):
    if params['data_version'] == 'v2':
        if params['dynamic_scenario']:
            params_path = os.path.join(
                os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]),
                params[c.PARAMSET_SCENARIO],
                'scene_' + str(params[c.PARAMSET_DYNAMIC_SCENES][0]),
                params[c.PARAMSET_SCENARIO] + c.LOAD_FILE_SP_EXT)
        else:
            params_path = os.path.join(
                os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]), 
                params[c.PARAMSET_SCENARIO], 
                params[c.PARAMSET_SCENARIO] + c.LOAD_FILE_SP_EXT)
    elif params['data_version'] == 'v3':
        params_path = os.path.join(
            os.path.abspath(params[c.PARAMSET_DATASET_FOLDER]), 
            params[c.PARAMSET_SCENARIO], 'params.mat')
    else:
        raise NotImplementedError
        
    return params_path

def compare_two_dicts(dict1, dict2):
    
    additional_keys = dict1.keys() - dict2.keys()
    for key, item in dict1.items():
        if isinstance(item, dict):
            if key in dict2:
                additional_keys = additional_keys | compare_two_dicts(dict1[key], dict2[key])

    return additional_keys

