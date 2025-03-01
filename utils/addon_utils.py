# utils/addon_utils.py
import bpy
import os

def install_blender_addon(addon_name, zip_name, proj_root):
    """Installs a Blender add-on from a zip file if not yet installed."""
    addon_zip_path = os.path.join(proj_root, "blender_addons", zip_name)
    
    if addon_name in bpy.context.preferences.addons.keys():
        if not bpy.context.preferences.addons[addon_name].module:
            bpy.ops.preferences.addon_enable(module=addon_name)
            bpy.ops.wm.save_userpref()
        print(f"Add-on '{addon_name}' is already installed and enabled.")
    else:
        bpy.ops.preferences.addon_install(filepath=addon_zip_path)
        bpy.ops.preferences.addon_enable(module=addon_name)
        bpy.ops.wm.save_userpref()
        print(f"Add-on '{addon_name}' installed and enabled.")