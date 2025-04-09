""""""

import os
import subprocess
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def run_command(command: List[str], description: str) -> None:
    """Run a shell command and stream output in real-time.
    
    Args:
        command (List[str]): Command to run
        description (str): Description of the command for logging
    """
    print(f"\n🚀 Starting: {description}...\n")
    print('running command: ', ' '.join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

    # Stream the output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end="")  # Print each line as it arrives

    process.stdout.close()
    process.wait()

    print(f"\n✅ {description} completed!\n")


def call_blender1(rt_params: Dict[str, Any], osm_folder: str, blender_path: str, blender_script_path: str) -> None:
    """Process OSM extraction directly without calling a separate script.
    
    Args:
        rt_params (Dict[str, Any]): Ray tracing parameters
        osm_folder (str): Path to the OSM folder
    """
    
    # Extract coordinates from rt_params
    minlat = rt_params['min_lat']
    minlon = rt_params['min_lon']
    maxlat = rt_params['max_lat']
    maxlon = rt_params['max_lon']
    
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
        "--minlat", str(minlat), 
        "--minlon", str(minlon), 
        "--maxlat", str(maxlat), 
        "--maxlon", str(maxlon),
        "--output", str(osm_folder)  # Pass the output folder to the Blender script
    ]
    
    # Run the command
    run_command(command, "OSM Extraction")


def read_rt_configs(row: pd.Series) -> Dict[str, Any]:
    """Read ray tracing configurations from a pandas Series.
    
    Args:
        row (pd.Series): Row from the parameters DataFrame
        
    Returns:
        Dict[str, Any]: Ray tracing parameters
    """
    bs_lats = np.array(row['bs_lat'].split(',')).astype(np.float32)
    bs_lons = np.array(row['bs_lon'].split(',')).astype(np.float32)
    carrier_freq = row['freq (ghz)'] * 1e9
    diffraction = scattering = bool(row['ds_enable'])

    rt_params = {
        'name': row['name'],
        'city': row['city'],
        'min_lat': row['min_lat'],
        'min_lon': row['min_lon'],
        'max_lat': row['max_lat'],
        'max_lon': row['max_lon'],
        'bs_lats': bs_lats,
        'bs_lons': bs_lons,
        'carrier_freq': carrier_freq,

        # Ray-tracing parameters -> Efficient if they match the dataclass in SetupEditor.py
        'max_reflections': row['n_reflections'],
        'diffraction': diffraction,
        'scattering': scattering,
        'max_paths': row['max_paths'],
        'ray_spacing': row['ray_spacing'],
        'max_transmissions': row['max_transmissions'],
        'max_diffractions': row['max_diffractions'],
        'ds_enable': row['ds_enable'],
        'ds_max_reflections': row['ds_max_reflections'],
        'ds_max_transmissions': row['ds_max_transmissions'],
        'ds_max_diffractions': row['ds_max_diffractions'],
        'ds_final_interaction_only': row['ds_final_interaction_only'],
        'conform_to_terrain': row['conform_to_terrain'] == 1
    }
    return rt_params
