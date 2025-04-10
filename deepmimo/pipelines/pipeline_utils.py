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


def call_blender(min_lat, min_lon, max_lat, max_lon, osm_folder: str,
                 blender_path: str, blender_script_path: str, outputs: List[str]) -> None:
    """Process OSM extraction directly without calling a separate script.
    
    Args:
        min_lat (float): Minimum latitude
        min_lon (float): Minimum longitude
        max_lat (float): Maximum latitude
        max_lon (float): Maximum longitude
        osm_folder (str): Path to the OSM folder
        blender_path (str): Path to the Blender executable
        blender_script_path (str): Path to the Blender script
        outputs (List[str]): List of outputs to generate
    """
    
    # Check if the folder already exists
    if os.path.exists(osm_folder):
        print(f"⏩ Folder '{osm_folder}' already exists. Skipping OSM extraction.")
        return
    
    # Validate paths
    if not os.path.exists(blender_path):
        raise FileNotFoundError(f"❌ Blender executable not found at {blender_path}")
        
    if not os.path.exists(blender_script_path):
        raise FileNotFoundError(f"❌ Blender script not found at {blender_script_path}")
    
    # Build command to run Blender
    command = [
        blender_path, 
        "--background", 
        "--python", 
        blender_script_path, 
        "--", 
        "--minlat", str(min_lat), 
        "--minlon", str(min_lon), 
        "--maxlat", str(max_lat), 
        "--maxlon", str(max_lon),
        "--output", osm_folder   # Output folder to the Blender script
        
    ]
    
    # Run the command
    run_command(command, "OSM Extraction")
    
    return


def get_origin_coords(osm_folder: str) -> Tuple[float, float]:
    """Read the origin coordinates from the OSM folder.
    
    Args:
        osm_folder (str): Path to the OSM folder
    
    Returns:
        Tuple[float, float]: Origin coordinates (latitude, longitude)
    """
    with open(os.path.join(osm_folder, 'osm_gps_origin.txt'), "r") as f:
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

