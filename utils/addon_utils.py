import os
import sys
import bpy
import subprocess
import time
from constants import PROJ_ROOT, ADDONS

def install_python_package(pckg_name):
    """
    Install a Python package using Blender's Python executable.
    
    Args:
        pckg_name (str): Name of the package to install (e.g., 'mitsuba==3.5.0')
    """
    python_exe = sys.executable
    subprocess.call([python_exe, "-m", "ensurepip"])
    subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.call([python_exe, "-m", "pip", "install", pckg_name])

def install_blender_addon(addon_name):
    """
    Install and enable a Blender add-on from a zip file if not already installed.
    
    Args:
        addon_name (str): Name of the add-on to install (e.g., 'blosm')
    """
    zip_name = ADDONS.get(addon_name)
    if not zip_name:
        print(f"Error: No zip file defined for add-on '{addon_name}'")
        return
    
    print("Installed add-ons:", list(bpy.context.preferences.addons.keys()))
    
    # Check if add-on is already installed
    if addon_name in bpy.context.preferences.addons.keys():
        print(f"The add-on '{addon_name}' is already installed.")
        if bpy.context.preferences.addons[addon_name].module:
            print(f"The add-on '{addon_name}' is enabled.")
        else:
            bpy.ops.preferences.addon_enable(module=addon_name)
            bpy.ops.wm.save_userpref()
            print(f"The add-on '{addon_name}' has been enabled.")
    else:
        print(f"The add-on '{addon_name}' is not installed or enabled.")
        addon_zip_path = os.path.join(PROJ_ROOT, "blender_addons", zip_name)
        bpy.ops.preferences.addon_install(filepath=addon_zip_path)
        bpy.ops.preferences.addon_enable(module=addon_name)
        bpy.ops.wm.save_userpref()
        print(f"Add-on '{addon_name}' installed and enabled.")
    
    # Special handling for Mitsuba
    if addon_name == 'mitsuba-blender':
        try:
            import mitsuba
        except ImportError:
            install_python_package('mitsuba==3.5.0')
            print('Packages installed! Restarting Blender to update imports.')
            time.sleep(5)
            sys.exit()