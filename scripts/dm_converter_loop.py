#%% Imports

import time
import deepmimo as dm
import os
import json
import shutil
import matplotlib.pyplot as plt
from api_keys import DEEPMIMO_API_KEY

# Main execution
EXECUTION_MODE = 'collect_errors' # 'collect_errors' | 'retry_errors'
ERROR_LOG_FILE = 'conversion_errors.json'

# base_path = "F:/deepmimo_loop_ready"
base_path = r"M:\AutoRayTracingSionna\all_runs_sionna\run_04-07-2025_18H13M23S"
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

# Load previous errors if in retry mode
error_scenarios = []
if EXECUTION_MODE == 'retry_errors' and os.path.exists(ERROR_LOG_FILE):
    with open(ERROR_LOG_FILE, 'r') as f:
        error_scenarios_to_retry = {item[0] for item in json.load(f)}
    subfolders = [f for f in subfolders if os.path.basename(f) in error_scenarios_to_retry]

#%%

timing_results = {}

# For conversion
# for subfolder in subfolders:
#     scen_name = os.path.basename(subfolder)
#     print(f'running: {subfolder}')

# For zipping
for scen_name in dm.get_available_scenarios():

    # if 'asu' in scen_name:
    #     continue

    if not scen_name.startswith('city_') or not scen_name.endswith('_s'):
        continue

    start = time.time()
    try:
        # scen_name = dm.convert(subfolder, overwrite=True, scenario_name=scen_name, 
        #                        vis_scene=True, print_params=False)
        # print(f"Conversion successful: {scen_name}")
        stop = time.time()

        # dm.upload(scen_name, key=DEEPMIMO_API_KEY)
        # print(f"Upload successful: {scen_name}")
        
        full_dataset = dm.load(scen_name)
        dataset = dm.load(scen_name)[4]
        
        grid_size = dataset.grid_size - 1
        import numpy as np
        cols = np.arange(grid_size[0], step=6)
        rows = np.arange(grid_size[1], step=6)
        idxs = np.array([j + i*grid_size[0] for i in rows for j in cols])
        # idxs = dataset.get_uniform_idxs([4,4])
        
        fig, ax = plt.subplots(figsize=(15, 15), subplot_kw={'projection': '3d'})
        ax.scatter(xs=dataset.rx_pos[idxs,0], 
                   ys=dataset.rx_pos[idxs,1], 
                   zs=dataset.rx_pos[idxs,2]-3, 
                   s=20, color='red', label='Users', marker='o', alpha=0.2)
        
        ax = full_dataset.scene.plot(title=False, ax=ax)
        ax.view_init(elev=90, azim=-90)
        ax.scatter(xs=full_dataset[0].bs_pos[0,0], 
                          ys=full_dataset[0].bs_pos[0,1], 
                          zs=full_dataset[0].bs_pos[0,2], 
                          s=250, color='red', label='BS 1', marker='*')
        ax.scatter(xs=full_dataset[1].bs_pos[0,0], 
                          ys=full_dataset[1].bs_pos[0,1], 
                          zs=full_dataset[1].bs_pos[0,2], 
                          s=250, color='blue', label='BS 2', marker='*')
        ax.scatter(xs=full_dataset[2].bs_pos[0,0], 
                          ys=full_dataset[2].bs_pos[0,1], 
                          zs=full_dataset[2].bs_pos[0,2], 
                          s=250, color='green', label='BS 3', marker='*')

        ax.legend(ncol=4, loc='center', bbox_to_anchor=(0.5, 0.85), fontsize=20).set_zorder(1e9)
        plt.show()
        
        # ax.set_position([-0.1, -0.1, 1.2, 1.2])
        # img_path = f'{scen_name}_scene.png'
        # plt.savefig(img_path, dpi=150, bbox_inches='tight')#, pad_inches=0)
        # plt.close()
        
        # call PIL to crop image
        # from PIL import Image
        # img = Image.open(img_path)
        # img = img.crop((150, 150, 1550, 1550))
        # img.save(img_path)
        break
        # dm.upload_images(scen_name, key=DEEPMIMO_API_KEY, img_paths=[img_path])
        # print(f"Upload successful: {scen_name}")
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Deleted image: {img_path}")
        
    except Exception as e:
        stop = start
        if EXECUTION_MODE == 'retry_errors':
            raise  # Re-raise the exception in retry mode
        error_scenarios.append((scen_name, str(e)))
        if EXECUTION_MODE == 'collect_errors' and error_scenarios:
            with open(ERROR_LOG_FILE, 'w') as f:
                json.dump(error_scenarios, f, indent=2)
    
    timing_results[scen_name] = stop - start

#%%
if timing_results:
    print("\nPer-scenario timing:")
    print("-" * 50)
    print(f"{'Scenario':<30} | {'Time (s)':<10}")
    print("-" * 50)
    for scen, duration in sorted(timing_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{scen:<30} | {duration:>10.2f}")
    
if error_scenarios:
    print(f"\nErrors ({len(error_scenarios)}):")
    for name, err in error_scenarios:
        print(f"{name:<30} | {err[:47]}")
    print(f"\nError scenarios saved to {ERROR_LOG_FILE}")

# Cleanup in retry mode if all successful
if EXECUTION_MODE == 'retry_errors' and not error_scenarios and os.path.exists(ERROR_LOG_FILE):
    os.remove(ERROR_LOG_FILE)
    print("All retries successful - error log removed")
