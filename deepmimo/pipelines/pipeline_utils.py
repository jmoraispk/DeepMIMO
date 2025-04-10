""""""

import os
import subprocess
from typing import List, Dict, Any, Tuple


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
        "--output", osm_folder   # Output folder to the Blender script
    ]
    
    # Run the command
    run_command(command, "OSM Extraction")


def get_origin_coords(osm_folder: str) -> Tuple[float, float]:
    """Read the origin coordinates from the OSM folder.
    
    Args:
        osm_folder (str): Path to the OSM folder
    
    Returns:
        Tuple[float, float]: Origin coordinates (latitude, longitude)
    """
    with open(os.path.join(osm_folder, 'osm_gps_origin.txt'), "r") as f:
        origin_coords = f.read().split(',')
    return float(origin_coords[0]), float(origin_coords[1])

