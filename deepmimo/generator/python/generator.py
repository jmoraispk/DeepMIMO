import os
import numpy as np

from ... import consts as c
from .construct_deepmimo import generate_MIMO_channel

from .utils import dbm2watt
from .channel_params import ChannelGenParameters
from .downloader import download_scenario_handler, extract_scenario
import scipy.io
from typing import List, Dict

from ...general_utilities import get_mat_filename

def generate(scen_name: str, load_params: Dict = {}, ch_gen_params: Dict = {}):
    
    if len(load_params) == 0:
        # Option 1 - dictionaries per tx/rx set and tx/rx index inside the set)
        tx_sets = {1: [0]}
        rx_sets = {2: 'active'}
        
        # Option 2 - lists with tx/rx set (assumes all points inside the set)
        # tx_sets = [1]
        # rx_sets = [2]
        
        # Option 3 - string 'all' (generates all points of all tx/rx sets) (default)
        # tx_sets = rx_sets = 'all'
        
        load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': 5}
    
    dataset = load_scenario(scen_name, **load_params)
    
    # Add load params to dataset
    dataset['load_params'] = load_params  # c.LOAD_PARAMS_PARAM_NAME
    
    # dataset.info() # print available tx-rx information
    
    # Compute num_paths and power_linear - necessary for channel generation
    dataset['num_paths'] = compute_num_paths(dataset)    # c.NUM_PATHS_PARAM_NAME
    dataset['power_linear'] = dbm2watt(dataset['power']) # c.PWR_LINEAR_PARAM_NAME
    
    channel_generation_params = ch_gen_params if ch_gen_params else ChannelGenParameters()
    dataset['channel'] = compute_channels(dataset, channel_generation_params)
    
    return dataset


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


def load_raytracing_scene(folder, rt_params, max_paths: int = 5,
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
                                                          tx_idx, rx_idxs,
                                                          max_paths)
                dataset_dict[bs_idx][c.RT_PARAMS_PARAM_NAME] = rt_params
    
    return dataset_dict if len(dataset_dict) != 1 else dataset_dict[0]

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
            elif type(idxs) is str:
                if idxs == 'all':
                    sets[set_idx] = all_idxs_available
                elif idxs == 'active':
                    inactive_idx = rt_params[f'txrx_set_{set_idx}']['inactive_idxs']
                    sets[set_idx] = np.array(list(set(all_idxs_available.tolist()) - 
                                                  set(inactive_idx.tolist())))
                else:
                    raise Exception(f"String '{idxs}' not recognized for tx/rx indices " )
            else:
                raise Exception('Only <list> of <np.ndarray> allowed as tx/rx indices')
                
            # check that the specific tx/rx indices inside the sets are valid
            if not set(sets[set_idx]).issubset(set(all_idxs_available.tolist())):
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


def load_tx_rx_raydata(rayfolder: str, tx_set_idx: int, rx_set_idx: int,
                       tx_idx: int, rx_idxs: np.ndarray | List, max_paths: int):
    
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
        
        mat_filename = get_mat_filename(key, tx_set_idx, tx_idx, rx_set_idx)
        
        mat_path = os.path.join(rayfolder, mat_filename)
        
        if os.path.exists(mat_path):
            print(f'Loading {mat_filename}..')
            tx_dict[key] = scipy.io.loadmat(mat_path)[c.MAT_VAR_NAME]
        else:
            print(f'File {mat_path} could not be found')
        
        # Filter by selected rx indices (all but tx positions)
        if key != c.TX_POS_PARAM_NAME: 
            tx_dict[key] = tx_dict[key][rx_idxs]
            
        # Trim by max paths
        if key not in [c.RX_POS_PARAM_NAME, c.TX_POS_PARAM_NAME]:
            tx_dict[key] = tx_dict[key][:, :max_paths, ...]
        
        print(f'shape = {tx_dict[key].shape}')
    return tx_dict


def compute_num_paths(dataset):
    # any matrix with paths in last dim
    max_paths = dataset[c.AOA_AZ_PARAM_NAME].shape[-1] # 
    
    nan_count_matrix = np.isnan(dataset[c.AOA_AZ_PARAM_NAME]).sum(axis=1)
    
    return max_paths - nan_count_matrix

def compute_num_interactions(dataset):
    
    # NOTE: dm.info('inter') (or dataset['inter'].info())
    # number of paths = number of digits in this array. 
    
    # Compute next power of 10 (fast way to know the number of digits
    
    # Zero = no bounce/interaction
    result = np.zeros_like(dataset['inter'], dtype=int)
    
    # For non-zero values, calculate order
    non_zero = dataset['inter'] > 0
    result[non_zero] = np.floor(np.log10(dataset['inter'][non_zero])).astype(int) + 1
    
    # dataset['num_interaction'] = result
    return result




def compute_distances(rx, tx):
    return np.linalg.norm(rx - tx, axis=1)

def compute_pathloss(received_powers_dbm, phases_degrees, transmitted_power_dbm=0, coherent=True):
    """
    Computes the total path loss of a link.
    
    Parameters:
        received_powers_dbm (list or np.array): Received powers in dBm for each path.
        phases_degrees (list or np.array): Phases in degrees for each path.
        transmitted_power_dbm (float): Transmitted power in dBm (default is 0 dBm).
        coherent (bool): If True, considers the phases of the paths (coherent sum).
                         If False, ignores the phases (non-coherent sum).

    Returns:
        float: The total path loss in dB.
    
    # Example Usage:
    received_powers = [-83.1005, -92.8987]
    phases = [-15.8284, -126.971]
    coherent_pathlos = compute_pathloss(received_powers, phases, coherent=True)     # = 83.25 dB
    non_coherent_result = compute_pathloss(received_powers, phases, coherent=False) # = 82.67 dB
    """
    # Convert received powers to linear scale (mW)
    received_powers_linear = 10 ** (np.array(received_powers_dbm) / 10)

    if coherent:
        # Coherent sum: Considering phases
        total_complex_power = np.sum(received_powers_linear * np.exp(1j * np.radians(phases_degrees)))
    else:
        # Non-coherent sum: Ignoring phases (set all phases to 0)
        total_complex_power = np.sum(received_powers_linear)

    # Compute the total received power magnitude (linear scale) and convert to dBm
    total_received_power_dbm = 10 * np.log10(np.abs(total_complex_power))

    # Compute total path loss
    path_loss = transmitted_power_dbm - total_received_power_dbm

    return path_loss

def compute_channels(dataset, params):
    
    if params is None:
        params_obj = ChannelGenParameters()
    elif type(params) is str:
        params_obj = ChannelGenParameters(params)
    else:
        params_obj = params
        
    np.random.seed(1001)
    
    params = params_obj.get_params_dict()
    
    dataset['num_ues'] = dataset[c.RX_POS_PARAM_NAME].shape[0]
    validate_ch_gen_params(params, n_active_ues=dataset['num_ues'])
    
    chs = generate_MIMO_channel(dataset=dataset,
                                ofdm_params=params[c.PARAMSET_OFDM],
                                tx_ant_params=params[c.PARAMSET_ANT_BS], 
                                rx_ant_params=params[c.PARAMSET_ANT_UE],
                                freq_domain=params[c.PARAMSET_FD_CH])
    
    return chs

def validate_ch_gen_params(params, n_active_ues):

    # Notify the user if some keyword is not used (likely set incorrectly)
    additional_keys = compare_two_dicts(params, ChannelGenParameters().get_params_dict())
    if len(additional_keys):
        print('The following parameters seem unnecessary:')
        print(additional_keys)
    
    # BS Antenna Rotation
    if c.PARAMSET_ANT_ROTATION in params[c.PARAMSET_ANT_BS].keys():
        rotation_shape = params[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_ROTATION].shape
        assert  (len(rotation_shape) == 1 and rotation_shape[0] == 3), \
                'The BS antenna rotation must be a 3D vector'
    else:
        params[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_ROTATION] = None                                            
    
    # UE Antenna Rotation
    if (c.PARAMSET_ANT_ROTATION in params[c.PARAMSET_ANT_UE].keys() and \
        params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] is not None):
        rotation_shape = params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION].shape
        cond_1 = len(rotation_shape) == 1 and rotation_shape[0] == 3
        cond_2 = len(rotation_shape) == 2 and rotation_shape[0] == 3 and rotation_shape[1] == 2
        cond_3 = rotation_shape[0] == n_active_ues
        
        assert_str = ('The UE antenna rotation must either be a 3D vector for ' +
                      'constant values or 3 x 2 matrix for random values')
        assert cond_1 or cond_2 or cond_3, assert_str
                
        if len(rotation_shape) == 1 and rotation_shape[0] == 3:
            rotation = np.zeros((n_active_ues, 3))
            rotation[:] = params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION]
            params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = rotation
        elif (len(rotation_shape) == 2 and rotation_shape[0] == 3 and rotation_shape[1] == 2):
            params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = np.random.uniform(
                              params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION][:, 0], 
                              params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION][:, 1], 
                              (n_active_ues, 3))
    else:
        params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_ROTATION] = \
            np.array([None] * n_active_ues) # List of None
    
    # BS Antenna Radiation Pattern
    if (c.PARAMSET_ANT_RAD_PAT in params[c.PARAMSET_ANT_BS].keys() and \
        params[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_ROTATION] is not None):
        assert_str = ("The BS antenna radiation pattern must have " + 
                      f"one of the following values: {str(c.PARAMSET_ANT_RAD_PAT_VALS)}")
        assert params[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_RAD_PAT] in c.PARAMSET_ANT_RAD_PAT_VALS, assert_str
    else:
        params[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]
                     
    # UE Antenna Radiation Pattern
    if c.PARAMSET_ANT_RAD_PAT in params[c.PARAMSET_ANT_UE].keys():
        assert_str = ("The UE antenna radiation pattern must have one of the " + 
                      f"following values: {str(c.PARAMSET_ANT_RAD_PAT_VALS)}")
        assert params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT] in c.PARAMSET_ANT_RAD_PAT_VALS, assert_str
    else:
        params[c.PARAMSET_ANT_UE][c.PARAMSET_ANT_RAD_PAT] = c.PARAMSET_ANT_RAD_PAT_VALS[0]
                                             
    return params


def compare_two_dicts(dict1, dict2):
    
    additional_keys = dict1.keys() - dict2.keys()
    for key, item in dict1.items():
        if isinstance(item, dict):
            if key in dict2:
                additional_keys = additional_keys | compare_two_dicts(dict1[key], dict2[key])

    return additional_keys

def compute_los(interactions):
    """
    Computes the Line of Sight (LoS) status for each receiver based on path interactions.
    
    Parameters:
        interactions (np.ndarray): Matrix containing interaction codes for each path
                                 Shape: (num_receivers, num_paths)
    
    Returns:
        np.ndarray: LoS status for each receiver
                   1: LoS path exists
                   0: Only NLoS paths exist
                   -1: No paths exist
    """
    # Initialize result array with -1 (no paths)
    result = np.full(interactions.shape[0], -1)
    
    # For receivers with at least one path (non-zero interaction)
    has_paths = np.any(interactions > 0, axis=1)
    result[has_paths] = 0  # Set to NLoS by default if has paths
    
    # Check first path (paths are sorted by power, so first path is strongest)
    first_path = interactions[:, 0]
    
    # If first path has no interactions (0) then it's a LoS path
    los_mask = first_path == 0
    result[los_mask & has_paths] = 1
    
    return result

