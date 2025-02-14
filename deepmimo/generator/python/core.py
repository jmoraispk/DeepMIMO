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
from typing import Dict, List, Any

# Third-party imports
import numpy as np
import scipy.io

# Local imports
from ... import consts as c
from ...general_utilities import get_mat_filename
from ...scene import Scene
from .dataset import Dataset, MacroDataset
from ...materials import MaterialList

# Channel generation
from .channel import ChannelGenParameters

# Scenario management
from .downloader import download_scenario_handler, extract_scenario

def generate(scen_name: str, load_params: Dict[str, Any] = {},
            ch_gen_params: Dict[str, Any] = {}) -> Dataset:
    """Generate a DeepMIMO dataset for a given scenario.
    
    This function wraps loading scenario data, computing channels, and organizing results.

    Args:
        scen_name (str): Name of the scenario to generate data for
        load_params (dict): Parameters for loading the scenario. Defaults to {}.
        ch_gen_params (dict): Parameters for channel generation. Defaults to {}.

    Returns:
        Dataset: Generated DeepMIMO dataset containing channel matrices and metadata
        
    Raises:
        ValueError: If scenario name is invalid or required files are missing
    """
    dataset = load_scenario(scen_name, **load_params)
    
    # Create channel generation parameters
    ch_params = ch_gen_params if ch_gen_params else ChannelGenParameters()
    
    # Compute channels - will be propagated to all child datasets if MacroDataset
    _ = dataset._compute_channels(ch_params)

    return dataset

def load_scenario(scen_name: str, **load_params) -> Dataset | MacroDataset:
    """Load a DeepMIMO scenario.
    
    This function loads raytracing data and creates a Dataset or MacroDataset instance.
    
    Args:
        scen_name (str): Name of the scenario to load
        **load_params: Additional parameters for loading the scenario
        
    Returns:
        Dataset or MacroDataset: Loaded dataset(s)
        
    Raises:
        ValueError: If scenario files cannot be loaded
    """
    # Handle absolute paths
    if os.path.isabs(scen_name):
        scen_folder = scen_name
        scen_name = os.path.basename(scen_folder)
    else:
        scen_folder = os.path.join(c.SCENARIOS_FOLDER, scen_name)
    
    # Download scenario if needed
    if not os.path.exists(scen_folder):
        print('Scenario not found. Would you like to download it? [Y/n]')
        response = input().lower()
        if response in ['', 'y', 'yes']:
            download_scenario_handler(scen_name)
            extract_scenario(scen_name)
        else:
            raise ValueError(f'Scenario {scen_name} not found')
    
    # Load parameters file
    params_mat_file = os.path.join(scen_folder, 'params.mat')
    if not os.path.exists(params_mat_file):
        raise ValueError(f'Parameters file not found in {scen_folder}')
    params = load_mat_file_as_dict(params_mat_file)['params']
    
    # Load scenario data
    n_snapshots = params[c.PARAMSET_DYNAMIC_SCENES]
    if n_snapshots > 1: # dynamic
        raise NotImplementedError('Dynamic scenarios not implemented yet')
        dataset = {}
        for snapshot_i in range(n_snapshots):
            snapshot_folder = os.path.join(scen_folder, rt_params[c.PARAMSET_SCENARIO],
                                           f'scene_{snapshot_i}')
            print(f'Scene {snapshot_i + 1}/{n_snapshots}')
            dataset[snapshot_i] = load_raytracing_scene(snapshot_folder, rt_params, **load_params)
    else: # static
        dataset = load_raytracing_scene(scen_folder, params[c.TXRX_PARAM_NAME], **load_params)
    
    # Set shared parameters
    dataset[c.LOAD_PARAMS_PARAM_NAME] = load_params
    dataset[c.RT_PARAMS_PARAM_NAME] = params[c.RT_PARAMS_PARAM_NAME]
    dataset[c.SCENE_PARAM_NAME] = Scene.from_data(params[c.SCENE_PARAM_NAME], scen_folder)
    dataset[c.MATERIALS_PARAM_NAME] = MaterialList.from_dict(params[c.MATERIALS_PARAM_NAME])

    return dataset

def load_raytracing_scene(scene_folder: str, txrx_dict: dict, max_paths: int = 5,
                         tx_sets: Dict[int, list | str] | list | str = 'all',
                         rx_sets: Dict[int, list | str] | list | str = 'all',
                         matrices: List[str] | str = 'all') -> Dataset:
    """Load raytracing data for a scene.

    Args:
        scene_folder (str): Path to scene folder containing raytracing data files
        rt_params (dict): Dictionary containing raytracing parameters 
        max_paths (int): Maximum number of paths to load. Defaults to 5
        tx_sets (dict or list or str): Transmitter sets to load. Defaults to 'all'
        rx_sets (dict or list or str): Receiver sets to load. Defaults to 'all'
        matrices (list of str): List of matrix names to load. Defaults to None

    Returns:
        Dataset: Dataset containing the requested matrices for each tx-rx pair
    """
    tx_sets = validate_txrx_sets(tx_sets, txrx_dict, 'tx')
    rx_sets = validate_txrx_sets(rx_sets, txrx_dict, 'rx')
    
    dataset_list = []
    bs_idxs = []
    
    for tx_set_idx, tx_idxs in tx_sets.items():
        for rx_set_idx, rx_idxs in rx_sets.items():
            dataset_list.append({})
            for tx_idx in tx_idxs:
                bs_idx = len(bs_idxs)
                bs_idxs.append(bs_idx)

                print(f'\nTX set: {tx_set_idx} (basestation)')
                rx_id_str = 'basestation' if rx_set_idx == tx_set_idx else 'users'
                print(f'RX set: {rx_set_idx} ({rx_id_str})')
                dataset_list[bs_idx] = load_tx_rx_raydata(scene_folder,
                                                          tx_set_idx, rx_set_idx,
                                                          tx_idx, rx_idxs,
                                                          max_paths, matrices)

                dataset_list[bs_idx]['info'] = {
                    'tx_set_idx': tx_set_idx,
                    'rx_set_idx': rx_set_idx,
                    'tx_idx': tx_idx,
                    'rx_idxs': rx_idxs
                }

    # Convert dictionary to Dataset at the end
    if len(dataset_list):
        final_dataset = MacroDataset([Dataset(d_dict) for d_dict in dataset_list])
    else:
        final_dataset = Dataset(dataset_list[0])
    return final_dataset


def load_tx_rx_raydata(rayfolder: str, tx_set_idx: int, rx_set_idx: int, tx_idx: int, 
                        rx_idxs: np.ndarray | List, max_paths: int, 
                        matrices_to_load: List[str] | str = 'all') -> Dict[str, Any]:
    """Load raytracing data for a transmitter-receiver pair.
    
    This function loads raytracing data files containing path information
    between a transmitter and set of receivers.

    Args:
        rayfolder (str): Path to folder containing raytracing data
        tx_set_idx (int): Index of transmitter set
        rx_set_idx (int): Index of receiver set
        tx_idx (int): Index of transmitter within set
        rx_idxs (numpy.ndarray or list): Indices of receivers to load
        max_paths (int): Maximum number of paths to load
        matrices_to_load (list of str, optional): List of matrix names to load. 

    Returns:
        dict: Dictionary containing loaded raytracing data

    Raises:
        ValueError: If required data files are missing or invalid
    """
    tx_dict = {c.AOA_AZ_PARAM_NAME: None,
               c.AOA_EL_PARAM_NAME: None,
               c.AOD_AZ_PARAM_NAME: None,
               c.AOD_EL_PARAM_NAME: None,
               c.DELAY_PARAM_NAME: None,
               c.POWER_PARAM_NAME: None,
               c.PHASE_PARAM_NAME: None,
               c.RX_POS_PARAM_NAME: None,
               c.TX_POS_PARAM_NAME: None,
               c.INTERACTIONS_PARAM_NAME: None,
               c.INTERACTIONS_POS_PARAM_NAME: None}
    
    if matrices_to_load == 'all':
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

# Helper functions
def validate_txrx_sets(sets: Dict[int, list | str] | list | str,
                      txrx_dict: Dict[str, Any], tx_or_rx: str = 'tx') -> Dict[int, list]:
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
    
    for key, val in txrx_dict.items():
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
            
            # Get the txrx_set info for this index
            txrx_set_key = f'txrx_set_{set_idx}'
            txrx_set = txrx_dict[txrx_set_key]
            all_idxs_available = np.arange(txrx_set['num_points'])
            
            if type(idxs) is np.ndarray:
                pass # correct
            elif type(idxs) is list:
                sets[set_idx] = np.array(idxs)
            elif type(idxs) is str:
                if idxs == 'all':
                    sets[set_idx] = all_idxs_available
                elif idxs == 'active':
                    inactive_idx = txrx_set['inactive_idxs']
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
        
            sets_dict[set_idx] = np.arange(txrx_dict[f'txrx_set_{set_idx}']['num_points'])
    elif type(sets) is str:
        if sets != 'all':
            raise Exception(f"String '{sets}' not understood. Only string allowed "
                          "is 'all' to generate all available sets and indices")
        
        # Generate dict with all sets and indices available
        sets_dict = {}
        for set_idx in valid_set_idxs:
            sets_dict[set_idx] = np.arange(txrx_dict[f'txrx_set_{set_idx}']['num_points'])
    return sets_dict
        
def validate_ch_gen_params(params: ChannelGenParameters, n_active_ues: int) -> ChannelGenParameters:
    """Validate channel generation parameters.
        
    This function checks that channel generation parameters are valid and
    consistent with the dataset configuration.s
    
    Args:
        params (dict): Channel generation parameters to validate
        n_active_ues (int): Number of active users in the dataset
        
    Returns:
        dict: Validated parameters

    Raises:
        ValueError: If parameters are invalid or inconsistent
    """
    # Notify the user if some keyword is not used (likely set incorrectly)
    additional_keys = compare_two_dicts(params, ChannelGenParameters())
    if len(additional_keys):
        print('The following parameters seem unnecessary:')
        print(additional_keys)
    
    # BS Antenna Rotation
    if c.PARAMSET_ANT_ROTATION in params[c.PARAMSET_ANT_BS].keys():
        rotation_shape = params[c.PARAMSET_ANT_BS][c.PARAMSET_ANT_ROTATION].shape
        assert (len(rotation_shape) == 1 and rotation_shape[0] == 3), \
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
    
    # TODO: Remove the None option from here
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
    if isinstance(mat_struct, scipy.io.matlab.mat_struct):
        result = {}
        for field in mat_struct._fieldnames:
            result[field] = mat_struct_to_dict(getattr(mat_struct, field))
        return result
    elif isinstance(mat_struct, np.ndarray):
        # Process arrays recursively in case they contain mat_structs
        return np.array([mat_struct_to_dict(item) for item in mat_struct])
    return mat_struct  # Return the object as is for other types

