# main_blender.py
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.simulation_params import PROJ_ROOT, ADDONS
from utils.addon_utils import install_blender_addon
from core.scene_builder import SceneBuilder

if __name__ == '__main__':
    for addon_name, zip_name in ADDONS.items():
        install_blender_addon(addon_name, zip_name, PROJ_ROOT)
    builder = SceneBuilder()
    builder.build_scenes()