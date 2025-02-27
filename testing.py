#%% Imports

import time
import numpy as np
import deepmimo as dm
import matplotlib.pyplot as plt
import os
from pprint import pprint

from my_api_key import API_KEY as MY_API_KEY

#%% V3 & V4 Conversion

# Example usage
# rt_folder = './P2Ms/asu_campus'
rt_folder = './P2Ms/simple_street_canyon_test'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/DeepMIMO_folder'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/sionna_test'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/sionna_export_test2'

scen_name = os.path.basename(rt_folder)
dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=True)

#%% V4 Generation

# Start timing
start_time = time.time()

# scen_name = 'DeepMIMO_folder'
# scen_name = 'simple_street_canyon_test'
scen_name = 'asu_campus'

# Option 1 - dictionaries per tx/rx set and tx/rx index inside the set)
tx_sets = {1: [0]}
rx_sets = {2: 'all'}

load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': 25}
dataset = dm.load(scen_name, **load_params)

# Create channel generation parameters
ch_params = dm.ChannelGenParameters()

# Using direct dot notation for parameters
ch_params.num_paths = 5
ch_params.ue_antenna.shape = np.array([1,1])

# Basic computations
p = dataset.power_linear  # Will be computed from dataset.power

dataset.power_linear *= 1000  # JUST TO BE COMPATIBLE WITH V3

# Other computations
dataset.compute_channels(ch_params)

# End timing
end_time = time.time()
print(f"Time elapsed: {end_time - start_time:.2f} seconds")

#%%
import deepmimo as dm
# dataset = dm.load('simple_street_canyon_test', tx_sets={1: [0]}, rx_sets={2: 'all'})
# dataset = dm.load('asu_campus', tx_sets={1: [0]}, rx_sets={2: 'all'})
dataset = dm.load('sionna_export_test', tx_sets={1: [0]}, rx_sets={2: 'all'})
idxs = dataset.get_uniform_idxs([1,1])
# idxs = np.arange(dataset.rx_pos.shape[0])
dm.plot_coverage(dataset.rx_pos[idxs], dataset.aoa_az[idxs, 0], bs_pos=dataset.tx_pos.T)


# import matplotlib.pyplot as plt
# plt.scatter(dataset['rx_pos'][10,0], dataset['rx_pos'][10,1], c='k', s=20)
#%%

idxs = dataset.get_uniform_idxs([1,1])

#%%

active_mask = dataset.num_paths > 0
print(f"\nNumber of active positions: {np.sum(active_mask)}")
print(f"Number of inactive positions: {np.sum(~active_mask)}")

# Create scatter plot showing active vs inactive positions
plt.figure(figsize=(12, 8))
plt.scatter(dataset.rx_pos[~active_mask, 0], dataset.rx_pos[~active_mask, 1], 
           alpha=0.5, s=10, c='red', label='Inactive')
plt.scatter(dataset.rx_pos[active_mask, 0], dataset.rx_pos[active_mask, 1], 
           alpha=0.5, s=10, c='green', label='Active')
plt.legend()

#%%

dm.plot_rays(dataset['rx_pos'][10], dataset['tx_pos'][0],
             dataset['inter_pos'][10], dataset['inter'][10],
             proj_3D=True, color_by_type=True)

# Next: determine which buildings interact with each ray. 
#       make a set of those buildings for all the rays in the user.
#       plot the buildings that matter to that user along with the rays.
#       (based on the building bounding boxes)
#       Use the PhysicalObjects class to plot a group of buildings.
#### NEXT: Make a plot of just SOME of the buildings
# buildings_scene = dm.Scene()
# for obj in buildings[:3]:
#     buildings_scene.add_object(obj)
    
# buildings_scene.plot()
#####

#%%
import deepmimo as dm
dm.summary('i1_2p4')
d = dm.load('i1_2p4')

#%%

d = dm.load('city_0_newyork_3p5', tx_sets={1: [0]}, rx_sets={4: 'all'})

dm.plot_coverage(d.rx_pos, d.aoa_az[:,0], bs_pos=d.tx_pos.T, proj_3D=True)

#%%

import os
base_path = "F:/deepmimo_loop_ready"
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]
subfolders = subfolders[31:]

start_time = time.time()
timing_results = {}  # Dictionary to store scenario timing results
error_scenarios = []
for subfolder in subfolders:
    iter_start_time = time.time()  # Start timing for this iteration
    scen_name = os.path.basename(subfolder)
    if scen_name not in error_scenarios:
        continue
    print(scen_name)
    print(subfolder)
    
    sub_subfolders = [f.path for f in os.scandir(subfolder) if f.is_dir()]
    
    if not sub_subfolders:
        print(f"No subfolders found in {subfolder}")
        continue
    
    possible_rt_folders = [sub_subfolder for sub_subfolder in sub_subfolders
                            if not sub_subfolder.endswith('_deepmimo')]
    if len(possible_rt_folders) > 1:
        print(f"Warning: Multiple subfolders found in {subfolder}, using last one")    
    
    rt_folder = possible_rt_folders[-1]
    
    try:
        scen_name = dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=False)
    except Exception as e:
        print(f"Error converting {scen_name}: {e}")
        # error_scenarios.append(scen_name)
        raise e
        # break
        continue
    
    timing_results[scen_name] = time.time() - iter_start_time
    print(f"Time for {scen_name}: {timing_results[scen_name]:.2f} seconds")
    #dm.upload(scen_name, MY_API_KEY)

print(f"\nTotal time elapsed: {time.time() - start_time:.2f} seconds")

# Print timing results table
print("\nTiming Results:")
print("-" * 50)
print(f"{'Scenario Name':<30} | {'Time (seconds)':<15}")
print("-" * 50)
for scen_name, time_taken in timing_results.items():
    if scen_name:
        print(f"{scen_name:<30} | {time_taken:>15.2f}")

#%%
base_path = "./deepmimo_scenarios3"
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

city_coordinates = {
    "city_0_newyork": [40.68503298, -73.84682129, 40.68597435, -73.84336302],
    "city_1_losangeles": [34.06430723, -118.2630866, 34.06560881, -118.2609365],
    "city_2_chicago": [41.88311487, -87.64432816, 41.88437915, -87.64194209],
    "city_3_houston": [29.7776392, -95.32486774, 29.77906384, -95.32281642],
    "city_4_phoenix": [33.45334109, -112.0848325, 33.455141, -112.0825603],
    "city_5_philadelphia": [39.94798907, -75.16421661, 39.95013822, -75.16230388],
    "city_6_miami": [25.77456239, -80.2254557, 25.77634401, -80.22329908],
    "city_7_sandiego": [32.71630718, -117.142513, 32.71789333, -117.1403115],
    "city_8_dallas": [32.83121188, -96.67452405, 32.83303719, -96.672456],
    "city_9_sanfrancisco": [37.7639299, -122.4718829, 37.76568531, -122.4695364],
    "city_10_austin": [30.27461507, -97.74694154, 30.2768963, -97.74549154],
    "city_11_santaclara": [37.35120556, -121.9472933, 37.35223013, -121.944066],
    "city_12_fortworth": [32.86085834, -97.2834463, 32.86275613, -97.28150025],
    "city_13_columbus": [39.96728387, -83.00179295, 39.9689303, -82.99903808],
    "city_14_charlotte": [35.26000315, -80.84962936, 35.26194658, -80.84768392],
    "city_15_indianapolis": [39.77786199, -86.15576074, 39.77964283, -86.15345984],
    "city_16_sanfrancisco": [37.7556508, -122.4885839, 37.75744553, -122.4862133],
    "city_17_seattle": [47.62319627, -122.3372192, 47.62484021, -122.3344778],
    "city_18_denver": [39.7383816, -105.0463622, 39.74029077, -105.0439858],
    "city_19_oklahoma": [35.47943919, -97.51608389, 35.4812478, -97.51398485]
}

# scenarios with conversion errors: o1_28, o1_60
# scenarios with missing files: o1b_3p5, o1_28, i3_2p4, 2x boston

static = ['F:/deepmimo_loop_ready\\o1_140',
        'F:/deepmimo_loop_ready\\o1_3p4',
        'F:/deepmimo_loop_ready\\o1_3p5',
        'F:/deepmimo_loop_ready\\o1_drone_200',
        'F:/deepmimo_loop_ready\\o1b_28',
        'F:/deepmimo_loop_ready\\o1b_3p5',]

for subfolder in subfolders[25:26]:
    scen_name = os.path.basename(subfolder)
    print(scen_name)
    if 'boston' in scen_name:
        print('Skipping Boston')
        continue
    for key in city_coordinates.keys():
        if key in scen_name:
            desc = 'GPS bounding box: ' + str(city_coordinates[key])
            break
        else:
            desc = ''
        
    dm.upload(scen_name, MY_API_KEY,
              details=[desc] if desc else None)
    
#%%
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

# Print summary of all scenarios
for subfolder in subfolders[:-5]:
    scen_name = os.path.basename(subfolder)
    dm.summary(scen_name)
    # break

#%%
scen_name = 'asu_campus_3p5'
dm.summary(scen_name)

dataset = dm.load(scen_name, tx_sets={1: [0]}, rx_sets={2: 'all'})

#%% auto load and plot
import deepmimo as dm
import matplotlib.pyplot as plt
base_path = 'F:/deepmimo_loop_ready'
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

# Load all scenarios
for subfolder in subfolders[:-5]:
    scen_name = os.path.basename(subfolder)
    # dm.load(scen_name)
    try:
        d = dm.load(scen_name, tx_sets={1: []}, rx_sets={2: []})
    except Exception as e:
        d = dm.load(scen_name, tx_sets={1: []}, rx_sets={4: []})
    _, ax = d.scene.plot(show=False)
    ax.set_title(scen_name + ': ' + ax.get_title())
    plt.show()

