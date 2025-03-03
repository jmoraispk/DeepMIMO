# main_blender.py
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import os
import csv
from datetime import datetime as dt
from utils.addon_utils import install_blender_addon
from core.scene_builder import create_scene
from constants import PROJ_ROOT, ADDONS

if __name__ == '__main__':
    """Main entry point for automated scene creation."""
    # Install required add-ons
    for addon_name in ADDONS:
        install_blender_addon(addon_name)

    # Setup run directory
    time_str = dt.now().strftime("%m-%d-%Y_%HH%MM%SS")
    osm_folder = os.path.join(PROJ_ROOT, 'all_runs', f'run_{time_str}')
    csv_path = os.path.join(PROJ_ROOT, 'params.csv')

    # Read positions from CSV
    with open(csv_path, 'r') as file:
        reader = csv.DictReader(file)
        positions = list(reader)

    # Save scenes folder path
    with open(os.path.join(PROJ_ROOT, 'scenes_folder.txt'), 'w') as fp:
        fp.write(osm_folder + '\n')

    # Create scenes
    create_scene(positions, osm_folder, time_str)