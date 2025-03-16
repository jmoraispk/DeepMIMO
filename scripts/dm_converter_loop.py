#%% Imports

import time
import deepmimo as dm
import os
import json

# from my_api_key import API_KEY as MY_API_KEY

# Main execution
EXECUTION_MODE = 'collect_errors' # 'collect_errors' | 'retry_errors'
ERROR_LOG_FILE = 'conversion_errors.json'

base_path = "F:/deepmimo_loop_ready"
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

# For zipping
for scen_name in dm.get_available_scenarios():

    # if 'asu' in scen_name:
    #     continue
    print(f"\nProcessing: {scen_name}")
    # break
    # continue
    start = time.time()
    try:
        # scen_name = dm.convert(subfolder, overwrite=True, scenario_name=scen_name, vis_scene=False)
        # print(f"Conversion successful: {scen_name}")
        
        dm.upload(scen_name, key='')
        print(f"Zip successful: {scen_name}")
    except Exception as e:
        if EXECUTION_MODE == 'retry_errors':
            raise  # Re-raise the exception in retry mode
        error_scenarios.append((scen_name, str(e)))
        if EXECUTION_MODE == 'collect_errors' and error_scenarios:
            with open(ERROR_LOG_FILE, 'w') as f:
                json.dump(error_scenarios, f, indent=2)
    
    timing_results[scen_name + '+' + 'upload'] = time.time() - start


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
