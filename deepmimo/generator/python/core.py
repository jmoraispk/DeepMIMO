"""
DeepMIMO Core Generation Module.

This module provides the core functionality for generating and managing DeepMIMO datasets.
It handles:
- Dataset generation and scenario management
- Ray-tracing data loading and processing
- Channel computation and parameter validation
- Multi-user MIMO channel generation

The module serves as the main entry point for creating DeepMIMO datasets from ray-tracing data.
"""

# Standard library imports
import os
from typing import Dict, List, Optional, Any

# Third-party imports
import numpy as np
import scipy.io

# Local imports
from .utils import dbm2watt
from .channel import generate_MIMO_channel, ChannelGenParameters
from ... import consts as c
from ...general_utilities import get_mat_filename
from ..python.downloader import download_scenario_handler, extract_scenario


def generate(scen_name: str, load_params: Dict[str, Any] = {},
            ch_gen_params: Dict[str, Any] = {}) -> Dict[str, Any] | List[Dict[str, Any]]:
    """Generate a DeepMIMO dataset for a given scenario.
    
    This function wraps loading scenario data, computing channels, and organizing results.

    Args:
        scen_name (str): Name of the scenario to generate data for
        load_params (dict): Parameters for loading the scenario. Defaults to {}.
        ch_gen_params (dict): Parameters for channel generation. Defaults to {}.

    Returns:
        dict or list: Generated DeepMIMO dataset containing channel matrices and metadata
        
    Raises:
        ValueError: If scenario name is invalid or required files are missing
    """
    if len(load_params) == 0:
        tx_sets = {1: [0]}
        rx_sets = {2: 'active'}
        load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': 5}
    
    dataset = load_scenario(scen_name, **load_params)
    dataset['load_params'] = load_params
    
    dataset['num_paths'] = compute_num_paths(dataset)    
    dataset['power_linear'] = dbm2watt(dataset['power'])
    
    channel_generation_params = ch_gen_params if ch_gen_params else ChannelGenParameters()
    dataset['channel'] = compute_channels(dataset, channel_generation_params)
    
    return dataset

def load_scenario(scen_name: str, **load_params) -> Dict[str, Any] | List[Dict[str, Any]]:
    """Load a DeepMIMO scenario from disk or download if not available.
    
    This function handles scenario data loading, including automatic downloading
    if the scenario is not found locally.

    Args:
        scen_name (str): Name of the scenario to load
        **load_params (dict): Additional loading parameters including tx_sets, rx_sets, max_paths

    Returns:
        dict or list: Loaded scenario data including ray-tracing results and parameters
        
    Raises:
        ValueError: If scenario files cannot be loaded
    """
    if '\\' in scen_name or '/' in scen_name:
        scen_folder = scen_name
        scen_name = os.path.basename(scen_folder)
    else:
        scen_folder = os.path.join(c.SCENARIOS_FOLDER, scen_name)
    
    # Download scenario if needed
    if not os.path.exists(scen_folder):
        print('Scenario not found. Would you like to download it? [Y/n]')
        ans = input()
        if not ('n' in ans.lower()):
            zip_path = download_scenario_handler(scen_name)
            extract_scenario(zip_path)
    
    params_mat_file = os.path.join(scen_folder, 'params.mat')
    rt_params = load_mat_file_as_dict(params_mat_file)
    
    # Load scenario data
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

def load_raytracing_scene(scene_folder: str, rt_params: dict, max_paths: int = 5,
                         tx_sets: Dict[int, list | str] | list | str = 'all',
                         rx_sets: Dict[int, list | str] | list | str = 'all',
                         matrices: List[str] = None) -> Dict[str, Any]:
    """Load raytracing data for a scene.

    Args:
        scene_folder (str): Path to scene folder containing raytracing data files
        rt_params (dict): Dictionary containing raytracing parameters 
        max_paths (int): Maximum number of paths to load. Defaults to 5
        tx_sets (dict or list or str): Transmitter sets to load. Defaults to 'all'
        rx_sets (dict or list or str): Receiver sets to load. Defaults to 'all'
        matrices (list of str): List of matrix names to load. Defaults to None

    Returns:
        dict: Dataset containing the requested matrices for each tx-rx pair
    """
    tx_sets = validate_txrx_sets(tx_sets, rt_params, 'tx')
    rx_sets = validate_txrx_sets(rx_sets, rt_params, 'rx')
    
    dataset_dict = {}
    bs_idxs = []
    
    for tx_set_idx, tx_idxs in tx_sets.items():
        for rx_set_idx, rx_idxs in rx_sets.items():
            for tx_idx in tx_idxs:
                bs_idx = len(bs_idxs)
                bs_idxs.append(bs_idx)

                print(f'\nTX set: {tx_set_idx} (basestation)')
                rx_id_str = 'basestation' if rx_set_idx == tx_set_idx else 'users'
                print(f'RX set: {rx_set_idx} ({rx_id_str})')
                
                dataset_dict[bs_idx] = load_tx_rx_raydata(scene_folder,
                                                        tx_set_idx, rx_set_idx,
                                                        tx_idx, rx_idxs,
                                                        max_paths, matrices)
                dataset_dict[bs_idx][c.RT_PARAMS_PARAM_NAME] = rt_params
    
    return dataset_dict if len(dataset_dict) != 1 else dataset_dict[0]

def compute_num_paths(dataset: Dict[str, Any]) -> np.ndarray:
    """Compute number of valid paths for each user.
    
    Args:
        dataset (dict): DeepMIMO dataset containing path information
        
    Returns:
        numpy.ndarray: Array containing number of valid paths for each user
    """
    max_paths = dataset[c.AOA_AZ_PARAM_NAME].shape[-1]
    nan_count_matrix = np.isnan(dataset[c.AOA_AZ_PARAM_NAME]).sum(axis=1)
    return max_paths - nan_count_matrix

def compute_num_interactions(dataset: Dict[str, Any]) -> np.ndarray:
    """Compute number of interactions for each path.
    
    Args:
        dataset (dict): DeepMIMO dataset containing interaction information
        
    Returns:
        numpy.ndarray: Array containing number of interactions for each path
    """
    result = np.zeros_like(dataset['inter'], dtype=int)
    non_zero = dataset['inter'] > 0
    result[non_zero] = np.floor(np.log10(dataset['inter'][non_zero])).astype(int) + 1
    return result

def compute_distances(rx: np.ndarray, tx: np.ndarray) -> np.ndarray:
    """Compute Euclidean distances between receivers and transmitter.
    
    Args:
        rx (numpy.ndarray): Receiver positions array of shape (n_receivers, 3)
        tx (numpy.ndarray): Transmitter position array of shape (3,)
        
    Returns:
        numpy.ndarray: Array of distances between each receiver and the transmitter
    """
    return np.linalg.norm(rx - tx, axis=1)

def compute_pathloss(received_powers_dbm: np.ndarray, phases_degrees: np.ndarray, 
                    transmitted_power_dbm: float = 0, coherent: bool = True) -> float:
    """Compute path loss from received powers and phases.

    Args:
        received_powers_dbm (numpy.ndarray): Received powers in dBm for each path
        phases_degrees (numpy.ndarray): Phases in degrees for each path 
        transmitted_power_dbm (float): Transmitted power in dBm. Defaults to 0
        coherent (bool): Whether to use coherent sum. Defaults to True

    Returns:
        float: Path loss in dB
    """
    received_powers_linear = 10 ** (np.array(received_powers_dbm) / 10)

    if coherent:
        total_complex_power = np.sum(received_powers_linear * 
                                   np.exp(1j * np.radians(phases_degrees)))
    else:
        total_complex_power = np.sum(received_powers_linear)

    total_received_power_dbm = 10 * np.log10(np.abs(total_complex_power))
    return transmitted_power_dbm - total_received_power_dbm

def compute_channels(dataset: Dict[str, Any], 
                    params: Optional[str | ChannelGenParameters] = None) -> np.ndarray:
    """Compute MIMO channel matrices for all users.
    
    Args:
        dataset (dict): DeepMIMO dataset containing path information
        params (str or ChannelGenParameters, optional): Channel generation parameters. Defaults to None
        
    Returns:
        numpy.ndarray: MIMO channel matrix
    """
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
    
    return generate_MIMO_channel(dataset=dataset,
                               ofdm_params=params[c.PARAMSET_OFDM],
                               tx_ant_params=params[c.PARAMSET_ANT_BS], 
                               rx_ant_params=params[c.PARAMSET_ANT_UE],
                               freq_domain=params[c.PARAMSET_FD_CH])

def compute_los(interactions: np.ndarray) -> np.ndarray:
    """Calculate Line of Sight status for each receiver.

    Args:
        interactions (numpy.ndarray): Matrix containing interaction codes for each path.
            The codes are read from left to right, starting from the transmitter end.
            0: Line-of-sight (direct path)
            1: Reflection 
            2: Diffraction
            3: Transmission
            4: Scattering
    Returns:
        numpy.ndarray: LoS status for each receiver (1: LoS, 0: NLoS, -1: No paths)
    """
    result = np.full(interactions.shape[0], -1)
    has_paths = np.any(interactions > 0, axis=1)
    result[has_paths] = 0
    
    first_path = interactions[:, 0]
    los_mask = first_path == 0
    result[los_mask & has_paths] = 1
    
    return result

# Helper functions
def validate_txrx_sets(sets: Dict[int, list | str] | list | str,
                      rt_params: Dict[str, Any], tx_or_rx: str = 'tx') -> Dict[int, list]:
    """Validate and process TX/RX set specifications.

    This function validates and processes transmitter/receiver set specifications,
    ensuring they match the available sets in the raytracing parameters.

    Args:
        sets (dict or list or str): TX/RX set specifications as dict, list, or string
        rt_params (dict): Raytracing parameters containing valid set information
        tx_or_rx (str): Whether validating TX or RX sets. Defaults to 'tx'

    Returns:
        dict: Dictionary mapping set indices to lists of valid TX/RX indices
        
    Raises:
        ValueError: If invalid TX/RX sets are specified
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

def validate_ch_gen_params(params: Dict[str, Any], n_active_ues: int) -> None:
    """Validate channel generation parameters.
    
    This function checks that channel generation parameters are valid and
    consistent with the dataset configuration.

    Args:
        params (dict): Channel generation parameters to validate
        n_active_ues (int): Number of active users in the dataset

    Returns:
        dict: Validated parameters

    Raises:
        ValueError: If parameters are invalid or inconsistent
    """
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

def compare_two_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> bool:
    """Compare two dictionaries for equality.
    
    This function performs a deep comparison of two dictionaries, handling
    nested dictionaries and numpy arrays.

    Args:
        dict1 (dict): First dictionary to compare
        dict2 (dict): Second dictionary to compare
        
    Returns:
        set: Set of keys in dict1 that are not in dict2
    """
    additional_keys = dict1.keys() - dict2.keys()
    for key, item in dict1.items():
        if isinstance(item, dict):
            if key in dict2:
                additional_keys = additional_keys | compare_two_dicts(dict1[key], dict2[key])
    return additional_keys

def load_mat_file_as_dict(file_path: str) -> Dict[str, Any]:
    """Load MATLAB .mat file as Python dictionary.
    
    Args:
        file_path (str): Path to .mat file to load
        
    Returns:
        dict: Dictionary containing loaded MATLAB data
        
    Raises:
        ValueError: If file cannot be loaded
    """
    mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    return {key: mat_struct_to_dict(value) for key, value in mat_data.items()
            if not key.startswith('__')}

def mat_struct_to_dict(mat_struct: Any) -> Dict[str, Any]:
    """Convert MATLAB structure to Python dictionary.
    
    This function recursively converts MATLAB structures and arrays to
    Python dictionaries and numpy arrays.

    Args:
        mat_struct (any): MATLAB structure to convert
        
    Returns:
        dict: Dictionary containing converted data
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

def load_tx_rx_raydata(rayfolder: str, tx_set_idx: int, rx_set_idx: int, tx_idx: int, 
                        rx_idxs: np.ndarray | List, max_paths: int, 
                        matrices_to_load: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load raytracing data for a transmitter-receiver pair.

    This function loads raytracing data files containing path information
    between a transmitter and set of receivers.

    Args:
        rayfolder (str): Path to folder containing raytracing data
        tx_set_idx (int): Index of transmitter set
        rx_set_idx (int): Index of receiver set
        tx_idx (int): Index of transmitter within its set
        rx_idxs (numpy.ndarray or list): Indices of receivers to load data for
        max_paths (int): Maximum number of paths to load per receiver
        matrices_to_load (list of str, optional): Optional list of specific matrices to load
        
    Returns:
        dict: Dictionary containing loaded raytracing data
        
    Raises:
        ValueError: If required data files are missing or invalid
    """
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
    
    if matrices_to_load is None:
        matrices_to_load = tx_dict.keys()
    else:
        valid_matrices = set(tx_dict.keys())
        invalid = set(matrices_to_load) - valid_matrices
        if invalid:
            raise ValueError(f"Invalid matrix names: {invalid}. "
                           f"Valid names are: {valid_matrices}")
        
    for key in tx_dict.keys():
        if key not in matrices_to_load:
            continue
        
        mat_filename = get_mat_filename(key, tx_set_idx, tx_idx, rx_set_idx)
        mat_path = os.path.join(rayfolder, mat_filename)
        
        if os.path.exists(mat_path):
            print(f'Loading {mat_filename}..')
            tx_dict[key] = scipy.io.loadmat(mat_path)[key]
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