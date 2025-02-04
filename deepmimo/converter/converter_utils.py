"""
Utility functions for file operations and compression.

This module provides helper functions for working with file extensions,
creating zip archives of folders, and managing scenario files.
"""

import os
from typing import List, Dict, Optional
import zipfile
import numpy as np
import scipy.io
import shutil
from pprint import pprint

from ..general_utilities import get_mat_filename, PrintIfVerbose
from .. import consts as c


def save_mat(data: np.ndarray, data_key: str, output_folder: str, tx_set_idx: int,
             tx_idx: int, rx_set_idx: int) -> None:
    """Save data to a .mat file with standardized naming.
    
    Args:
        data: Data array to save
        data_key: Key identifier for the data type
        output_folder: Output directory path
        tx_set_idx: Transmitter set index
        tx_idx: Transmitter index within set
        rx_set_idx: Receiver set index
    """
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
            

def copy_rt_source_files(sim_folder: str, verbose: bool = True) -> None:
    """Copy raytracing source files to a new directory and create a zip archive.
    
    Args:
        sim_folder (str): Path to simulation folder.
        verbose (bool): Whether to print progress messages. Defaults to True.
    """
    vprint = PrintIfVerbose(verbose) # prints if verbose 
    rt_source_folder = os.path.basename(sim_folder) + '_raytracing_source'
    files_in_sim_folder = os.listdir(sim_folder)
    print(f'Copying raytracing source files to {rt_source_folder}')
    zip_temp_folder = os.path.join(sim_folder, rt_source_folder)
    os.makedirs(zip_temp_folder)
    for ext in ['.setup', '.txrx', '.ter', '.city', '.kmz']:
        # copy all files with extensions to temp folder
        for file in ext_in_list(ext, files_in_sim_folder):
            curr_file_path = os.path.join(sim_folder, file)
            new_file_path  = os.path.join(zip_temp_folder, file)
            
            # vprint(f'Adding {file}')
            shutil.copy(curr_file_path, new_file_path)
    
    vprint('Zipping')
    zip_folder(zip_temp_folder)
    
    vprint(f'Deleting temp folder {os.path.basename(zip_temp_folder)}')
    shutil.rmtree(zip_temp_folder)
    
    vprint('Done')


def export_params_dict(output_folder: str, *dicts: Dict) -> None:
    """Export parameter dictionaries to a .mat file.
    
    Args:
        output_folder (str): Output directory path.
        *dicts: Variable number of dictionaries to merge and export.
    """
    base_dict = {
        c.LOAD_FILE_SP_VERSION: c.VERSION,
        c.LOAD_FILE_SP_RAYTRACER: c.RAYTRACER_NAME_WIRELESS_INSITE,
        c.LOAD_FILE_SP_RAYTRACER_VERSION: c.RAYTRACER_VERSION_WIRELESS_INSITE,
        c.PARAMSET_DYNAMIC_SCENES: 0, # only static currently
    }
    
    merged_dict = base_dict.copy()
    for d in dicts:
        merged_dict.update(d)
        
    pprint(merged_dict)
    scipy.io.savemat(os.path.join(output_folder, 'params.mat'), merged_dict)


def export_scenario(sim_folder: str, scen_name: str = '', 
                   overwrite: Optional[bool] = None) -> Optional[str]:
    """Export scenario to the DeepMIMO scenarios folder.
    
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
            