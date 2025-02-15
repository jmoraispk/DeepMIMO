"""
Utility functions for file operations and compression.

This module provides helper functions for working with file extensions,
creating zip archives of folders, and managing scenario files.
"""

import os
from typing import List, Dict, Optional, Any
import zipfile
import numpy as np
import scipy.io
import shutil
import pickle

from ..general_utilities import get_mat_filename
from .. import consts as c

def save_pickle(obj: Any, filename: str) -> None:
    """Saves an object to a pickle file.
    
    Args:
        obj: Object to save
        filename: Path to save the pickle file
    """
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(filename: str) -> Any:
    """Loads an object from a pickle file.
    
    Args:
        filename: Path to the pickle file
        
    Returns:
        The unpickled object
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)

def save_mat(data: np.ndarray, data_key: str, output_folder: str,
             tx_set_idx: Optional[int] = None, tx_idx: Optional[int] = None, 
             rx_set_idx: Optional[int] = None) -> None:
    """Save data to a .mat file with standardized naming.
    
    This function saves data to a .mat file using standardized naming conventions.
    If transmitter/receiver indices are provided, the filename will include those indices.
    Otherwise, it will use just the data_key as the filename.

    For example:
    - With indices: {data_key}_t{tx_set_idx}_{tx_idx}_r{rx_set_idx}.mat
    - Without indices: {data_key}.mat
    
    Args:
        data: Data array to save
        data_key: Key identifier for the data type
        output_folder: Output directory path
        tx_set_idx: Transmitter set index. Use None for no index.
        tx_idx: Transmitter index within set. Use None for no index.
        rx_set_idx: Receiver set index. Use None for no index.
    """
    if tx_set_idx is None:
        mat_file_name = data_key + '.mat'
    else:
        mat_file_name = get_mat_filename(data_key, tx_set_idx, tx_idx, rx_set_idx)
    file_path = os.path.join(output_folder, mat_file_name)
    scipy.io.savemat(file_path, {data_key: data}) 


def ext_in_list(extension: str, file_list: List[str]) -> List[str]:
    """Filter files by extension.
    
    This function filters a list of filenames to only include those that end with
    the specified extension.
    
    Args:
        extension (str): File extension to filter by (e.g. '.txt')
        file_list (List[str]): List of filenames to filter
        
    Returns:
        List[str]: Filtered list containing only filenames ending with extension
    """
    return [el for el in file_list if el.endswith(extension)]


def zip_folder(folder_path: str) -> None:
    """Create zip archive of folder contents.
    
    This function creates a zip archive containing all files in the specified
    folder. The archive is created in the same directory as the folder with
    '.zip' appended to the folder name.
    
    Args:
        folder_path (str): Path to folder to be zipped
    """
    files_in_folder = os.listdir(folder_path)
    file_full_paths = [os.path.join(folder_path, file) 
                       for file in files_in_folder]
    
    # Create a zip file
    with zipfile.ZipFile(folder_path + '.zip', 'w') as zipf:
        for file_path in file_full_paths:
            zipf.write(file_path, os.path.basename(file_path))


def save_rt_source_files(sim_folder: str, source_exts: List[str]) -> None:
    """Save raytracing source files to a new directory and create a zip archive.
    
    Args:
        sim_folder (str): Path to simulation folder.
        source_exts (List[str]): List of file extensions to copy.
        verbose (bool): Whether to print progress messages. Defaults to True.
    """
    rt_source_folder = os.path.basename(sim_folder) + '_raytracing_source'
    files_in_sim_folder = os.listdir(sim_folder)
    print(f'Copying raytracing source files to {rt_source_folder}')
    zip_temp_folder = os.path.join(sim_folder, rt_source_folder)
    os.makedirs(zip_temp_folder)
    
    for ext in source_exts:
        # copy all files with extensions to temp folder
        for file in ext_in_list(ext, files_in_sim_folder):
            curr_file_path = os.path.join(sim_folder, file)
            new_file_path  = os.path.join(zip_temp_folder, file)
            
            # vprint(f'Adding {file}')
            shutil.copy(curr_file_path, new_file_path)
    
    # Zip the temp folder
    zip_folder(zip_temp_folder)
    
    # Delete the temp folder (not the zip)
    shutil.rmtree(zip_temp_folder)

    return


def save_scenario(sim_folder: str, scen_name: str = '', 
                 overwrite: Optional[bool] = None) -> Optional[str]:
    """Save scenario to the DeepMIMO scenarios folder.
    
    Args:
        sim_folder (str): Path to simulation folder.
        scen_name (str): Custom name for scenario. Uses folder name if empty.
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

################################################################################
### Utils for compressing path data (likely to be moved outward to paths.py) ###
################################################################################

def compress_path_data(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Remove unused paths and interactions to optimize memory usage.
    
    This function compresses the path data by:
    1. Finding the maximum number of actual paths used
    2. Computing maximum number of interactions (bounces)
    3. Trimming arrays to remove unused entries
    
    Args:
        data (Dict[str, np.ndarray]): Dictionary containing path information arrays
        num_paths_key (str): Key in data dict containing number of paths. Defaults to 'n_paths'
        
    Returns:
        Dict[str, np.ndarray]: Compressed data dictionary with unused entries removed
    """
    # Compute max paths
    max_paths = get_max_paths(data)

    # Compute max bounces if interaction data exists
    max_bounces = 0
    if c.INTERACTIONS_PARAM_NAME in data:
        max_bounces = np.max(comp_next_pwr_10(data[c.INTERACTIONS_PARAM_NAME]))
    
    # Compress arrays to not take more than that space
    for key in data.keys():
        if len(data[key].shape) >= 2:
            data[key] = data[key][:, :max_paths, ...]
        if len(data[key].shape) >= 3:
            data[key] = data[key][:, :max_paths, :max_bounces]
    
    return data

def comp_next_pwr_10(arr: np.ndarray) -> np.ndarray:
    """Calculate number of interactions from interaction codes.
    
    This function computes the number of interactions (bounces) from the
    interaction code array by calculating the number of digits.
    
    Args:
        arr (np.ndarray): Array of interaction codes
        
    Returns:
        np.ndarray: Array containing number of interactions for each path
    """
    # Handle zero separately
    result = np.zeros_like(arr, dtype=int)
    
    # For non-zero values, calculate order
    non_zero = arr > 0
    result[non_zero] = np.floor(np.log10(arr[non_zero])).astype(int) + 1
    
    return result

def get_max_paths(arr: Dict[str, np.ndarray], angle_key: str = c.AOA_AZ_PARAM_NAME,
                  max_paths: int = 25) -> int:
    """Find maximum number of valid paths in the dataset.
    
    This function determines the maximum number of valid paths by finding
    the first path index where all entries (across all receivers) are NaN.
    
    Args:
        arr (Dict[str, np.ndarray]): Dictionary containing path information arrays
        angle_key (str): Key to use for checking valid paths. Defaults to AOA_AZ
        max_paths (int): Maximum number of paths to consider. Defaults to 25
        
    Returns:
        int: Maximum number of valid paths, or max_paths if all paths contain data
    """
    # The first path index with all entries at NaN
    all_nans_per_path_idx = np.all(np.isnan(arr[angle_key]), axis=0)
    n_max_paths = np.where(all_nans_per_path_idx)[0]
    return n_max_paths[0] if len(n_max_paths) else max_paths
            