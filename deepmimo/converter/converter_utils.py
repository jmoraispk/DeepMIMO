"""
Utility functions for file operations and compression.

This module provides helper functions for working with file extensions
and creating zip archives of folders.
"""

import os
from typing import List
import zipfile
import numpy as np
import scipy.io

from ..general_utilities import get_mat_filename


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
            