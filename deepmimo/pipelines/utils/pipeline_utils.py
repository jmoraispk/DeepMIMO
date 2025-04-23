""""""

import os
import subprocess
from typing import List, Tuple
import numpy as np


def run_command(command: List[str], description: str) -> None:
    """Run a shell command and stream output in real-time.
    
    Args:
        command (List[str]): Command to run
        description (str): Description of the command for logging
    """
    print(f"\n🚀 Starting: {description}...\n")
    print('\t Running: ', ' '.join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               text=True, encoding="utf-8", errors="replace")

    # Stream the output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end="")  # Print each line as it arrives

    process.stdout.close()
    process.wait()

    print(f"\n✅ {description} completed!\n")


def get_origin_coords(osm_folder: str) -> Tuple[float, float]:
    """Read the origin coordinates from the OSM folder.
    
    Args:
        osm_folder (str): Path to the OSM folder
    
    Returns:
        Tuple[float, float]: Origin coordinates (latitude, longitude)
    """
    origin_file = os.path.join(osm_folder, 'osm_gps_origin.txt')
    # Check if the file exists
    if not os.path.exists(origin_file):
        raise FileNotFoundError(f"❌ Origin coordinates file not found at {origin_file}\n"
                                "Ensure that Blender has been run successfully.")
    
    with open(origin_file, "r") as f:
        origin_coords = f.read().split('\n')
    return float(origin_coords[0]), float(origin_coords[1])


def _split_coords(x: str) -> np.ndarray:
    """Split comma-separated coordinates into float array."""
    return np.array(x.split(',')).astype(np.float32)


def load_params_from_row(row, params_dict):
    """Load parameters from a DataFrame row into a parameters dictionary.
    
    Args:
        row (pandas.Series): Row from a DataFrame containing parameters
        params_dict (Dict): Dictionary of parameters to update
    """
    # Update parameters that exist in both the row and params dict
    for key in params_dict.keys():
        if key in row.index:
            params_dict[key] = row[key]
    
    # Handle base station coordinates separately
    params_dict['bs_lats'] = _split_coords(row['bs_lat'])
    params_dict['bs_lons'] = _split_coords(row['bs_lon'])
    params_dict['bs_heights'] = _split_coords(row['bs_height'])

