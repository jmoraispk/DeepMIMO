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
from .. import consts as c
from ..general_utilities import (get_mat_filename, load_dict_from_json, unzip, 
                                 get_scenario_folder, get_params_path, compare_two_dicts)
from ..scene import Scene
from .dataset import Dataset, MacroDataset
from ..materials import MaterialList

# Channel generation
from .channel import ChannelGenParameters

# Scenario management
from ..api import download

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
    dataset = load(scen_name, **load_params)
    
    # Create channel generation parameters
    ch_params = ch_gen_params if ch_gen_params else ChannelGenParameters()
    
    # Compute channels - will be propagated to all child datasets if MacroDataset
    _ = dataset._compute_channels(ch_params)

    return dataset

def load(scen_name: str, **load_params) -> Dataset | MacroDataset:
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
        scen_folder = get_scenario_folder(scen_name)
    
    # Download scenario if needed
    if not os.path.exists(scen_folder):
        print('Scenario not found. Would you like to download it? [Y/n]')
        response = input().lower()
        if response in ['', 'y', 'yes']:
            download(scen_name)
        else:
            raise ValueError(f'Scenario {scen_name} not found')
    
    # Load parameters file
    params_file = get_params_path(scen_name)
    if not os.path.exists(params_file):
        raise ValueError(f'Parameters file not found in {scen_folder}')
    params = load_dict_from_json(params_file)
    
    # Load scenario data
    n_snapshots = params[c.SCENE_PARAM_NAME][c.SCENE_PARAM_NUMBER_SCENES]
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
    dataset[c.NAME_PARAM_NAME] = scen_name
    dataset[c.LOAD_PARAMS_PARAM_NAME] = load_params
    dataset[c.RT_PARAMS_PARAM_NAME] = params[c.RT_PARAMS_PARAM_NAME]
    dataset[c.SCENE_PARAM_NAME] = Scene.from_data(params[c.SCENE_PARAM_NAME], scen_folder)
    dataset[c.MATERIALS_PARAM_NAME] = MaterialList.from_dict(params[c.MATERIALS_PARAM_NAME])

    return dataset

def load_raytracing_scene(scene_folder: str, txrx_dict: dict, max_paths: int = c.MAX_PATHS,
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
        matrices (list of str): List of matrix names to load. Defaults to 'all'

    Returns:
        Dataset: Dataset containing the requested matrices for each tx-rx pair
    """
    tx_sets = validate_txrx_sets(tx_sets, txrx_dict, 'tx')
    rx_sets = validate_txrx_sets(rx_sets, txrx_dict, 'rx')
    dataset_list = []
    bs_idx = 0
    
    for tx_set_id, tx_idxs in tx_sets.items():
        for rx_set_id, rx_idxs in rx_sets.items():
            dataset_list.append({})
            for tx_idx in tx_idxs:
                print(f'\nTX set: {tx_set_id} (basestation)')
                rx_id_str = 'basestation' if rx_set_id == tx_set_id else 'users'
                print(f'RX set: {rx_set_id} ({rx_id_str})')
                dataset_list[bs_idx] = load_tx_rx_raydata(scene_folder,
                                                          tx_set_id, rx_set_id,
                                                          tx_idx, rx_idxs,
                                                          max_paths, matrices)

                dataset_list[bs_idx]['txrx'] = {
                    'tx_set_id': tx_set_id,
                    'rx_set_id': rx_set_id,
                    'tx_idx': int(tx_idx),
                }
                bs_idx += 1

    # Convert dictionary to Dataset at the end
    if len(dataset_list):
        final_dataset = MacroDataset([Dataset(d_dict) for d_dict in dataset_list])
    else:
        final_dataset = Dataset(dataset_list[0])
    return final_dataset


def load_tx_rx_raydata(rayfolder: str, tx_set_id: int, rx_set_id: int, tx_idx: int, 
                        rx_idxs: np.ndarray | List, max_paths: int, 
                        matrices_to_load: List[str] | str = 'all') -> Dict[str, Any]:
    """Load raytracing data for a transmitter-receiver pair.
    
    This function loads raytracing data files containing path information
    between a transmitter and set of receivers.

    Args:
        rayfolder (str): Path to folder containing raytracing data
        tx_set_id (int): Index of transmitter set
        rx_set_id (int): Index of receiver set
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
               c.POWER_PARAM_NAME: None,
               c.PHASE_PARAM_NAME: None,
               c.DELAY_PARAM_NAME: None,
               c.RX_POS_PARAM_NAME: None,
               c.TX_POS_PARAM_NAME: None,
               c.INTERACTIONS_PARAM_NAME: None,
               c.INTERACTIONS_POS_PARAM_NAME: None}
    
    if matrices_to_load == 'all':
        matrices_to_load = tx_dict.keys()
    else:
        matrices_to_load = [] if matrices_to_load is None else matrices_to_load
        valid_matrices = set(tx_dict.keys())
        invalid = set(matrices_to_load) - valid_matrices
        if invalid:
            raise ValueError(f"Invalid matrix names: {invalid}. "
                             f"Valid names are: {valid_matrices}")
        
    for key in tx_dict.keys():
        if key not in matrices_to_load:
            continue
        
        mat_filename = get_mat_filename(key, tx_set_id, tx_idx, rx_set_id)
        mat_path = os.path.join(rayfolder, mat_filename)
    
        if os.path.exists(mat_path):
            print(f'Loading {mat_filename}...', end='')
            tx_dict[key] = scipy.io.loadmat(mat_path)[key]
        else:
            print(f'File {mat_path} could not be found')
        
        if tx_dict[key] is None:
            continue

        # Filter by selected rx indices (all but tx positions)
        if key != c.TX_POS_PARAM_NAME: 
            tx_dict[key] = tx_dict[key][rx_idxs]
            
        # Trim by max paths
        if key not in [c.RX_POS_PARAM_NAME, c.TX_POS_PARAM_NAME]:
            tx_dict[key] = tx_dict[key][:, :max_paths, ...]
        
        print(f'Done. Shape: {tx_dict[key].shape}')
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
    # Get valid TX/RX sets in a deterministic order
    tx_sets = [txrx_dict[key] for key in sorted(txrx_dict.keys()) if txrx_dict[key]['is_tx']]
    rx_sets = [txrx_dict[key] for key in sorted(txrx_dict.keys()) if txrx_dict[key]['is_rx']]
    
    valid_tx_set_ids = [tx_set['id'] for tx_set in tx_sets]
    valid_rx_set_ids = [rx_set['id'] for rx_set in rx_sets]
    
    valid_set_ids = valid_tx_set_ids if tx_or_rx == 'tx' else valid_rx_set_ids
    set_str = 'Tx' if tx_or_rx == 'tx' else 'Rx'
    
    info_str = "To see supported TX/RX sets and indices run dm.info(<scenario_name>)"
    if type(sets) is dict:
        for set_id, idxs in sets.items():
            # check the the tx/rx_set indices are valid
            if set_id not in valid_set_ids:
                raise Exception(f"{set_str} set {set_id} not in allowed sets {valid_set_ids}\n"
                                + info_str)
            
            # Get the txrx_set info for this index
            txrx_set_key = f'txrx_set_{set_id}'  # Use id for internal operations
            txrx_set = txrx_dict[txrx_set_key]
            all_idxs_available = np.arange(txrx_set['num_points'])
            
            if type(idxs) is np.ndarray:
                pass # correct
            elif type(idxs) is list:
                sets[set_id] = np.array(idxs)
            elif type(idxs) is str:
                if idxs == 'all':
                    sets[set_id] = all_idxs_available
                else:
                    raise Exception(f"String '{idxs}' not recognized for tx/rx indices " )
            else:
                raise Exception('Only <list> of <np.ndarray> allowed as tx/rx indices')
            
            # check that the specific tx/rx indices inside the sets are valid
            if not set(sets[set_id]).issubset(set(all_idxs_available.tolist())):
                raise Exception(f'Some indices of {idxs} are not in {all_idxs_available}. '
                                 + info_str)
            
        sets_dict = sets
    elif type(sets) is list:
        # Generate all user indices
        sets_dict = {}
        for set_id in sets:
            if set_id not in valid_set_ids:
                raise Exception(f"{set_str} set {set_id} not in allowed sets {valid_set_ids}\n"
                              + info_str)
        
            sets_dict[set_id] = np.arange(txrx_dict[f'txrx_set_{set_id}']['num_points'])
    elif type(sets) is str:
        if sets != 'all':
            raise Exception(f"String '{sets}' not understood. Only string allowed "
                          "is 'all' to generate all available sets and indices")
        
        # Generate dict with all sets and indices available
        sets_dict = {}
        for set_id in valid_set_ids:
            sets_dict[set_id] = np.arange(txrx_dict[f'txrx_set_{set_id}']['num_points'])
    return sets_dict
    