#%%
import os
import shutil

# Main execution
EXECUTION_MODE = 'collect_errors' # 'collect_errors' | 'retry_errors'
ERROR_LOG_FILE = 'conversion_errors.json'

base_path = r"M:\AutoRayTracingSionna\all_runs_sionna\run_04-07-2025_18H13M23S"
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

new_base_path = r"M:\AutoRayTracingSionna\all_runs_sionna\run_04-12-2025_18H13M23S_cleaned"

#%%
# 1- Copy all folders in subfolders to new folder
# Set to True to only print actions without executing them
SAFE_MODE = False

for subfolder in subfolders:
    new_subfolder = os.path.join(new_base_path, os.path.basename(subfolder))
    
    print(f"Would copy {subfolder} to {new_subfolder}")
    if not SAFE_MODE:
        shutil.copytree(subfolder, new_subfolder)

    # 2- Delete everything inside each subfolder except for the sionna_export_full folder
    for item in os.listdir(new_subfolder if not SAFE_MODE else subfolder):
        item_path = os.path.join(new_subfolder if not SAFE_MODE else subfolder, item)
        if item != 'sionna_export_full' and item != 'osm_gps_origin.txt':
            print(f"Would delete {item_path}")
            if not SAFE_MODE:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)

    # 3- Move the contents of sionna_export_full folder up to the subfolder level
    export_full_path = os.path.join(new_subfolder if not SAFE_MODE else subfolder, 'sionna_export_full')
    for item in os.listdir(export_full_path):
        print(f"Would move {os.path.join(export_full_path, item)} to {new_subfolder}")
        if not SAFE_MODE:
            shutil.move(os.path.join(export_full_path, item), new_subfolder)
    
    print(f"Would delete {export_full_path}")
    if not SAFE_MODE:
        shutil.rmtree(os.path.join(new_subfolder, 'sionna_export_full'))

