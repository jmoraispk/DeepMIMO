#%% Imports

import time
import numpy as np
import deepmimo as dm

from pprint import pprint

#%% V3 & V4 Conversion

def convert_scenario(rt_folder: str, use_v3: bool = False) -> str:
    """Convert a Wireless Insite scenario to DeepMIMO format.
    
    Args:
        rt_folder (str): Path to the ray tracing folder
        use_v3 (bool): Whether to use v3 converter. Defaults to False.
        
    Returns:
        str: Name of the converted scenario
    """
    # Set parameters based on scenario
    if 'asu_campus' in rt_folder:
        old_params_dict = {'num_bs': 1, 'user_grid': [1, 411, 321], 'freq': 3.5e9} # asu
    else:
        old_params_dict = {'num_bs': 1, 'user_grid': [1, 91, 61], 'freq': 3.5e9} # simple canyon

    # Convert to unix path
    rt_folder = rt_folder.replace('\\', '/')
    # Make sure it ends with a /
    if 'P2Ms' in rt_folder:
        i = -2
    else:
        rt_folder += '/' if not rt_folder.endswith('/') else ''
        i = -1
    # -2 for Wireless Insite, -1 for other raytracers
    # because for Wireless Insite, we give the P2M INSIDE the rt_folder.. 

    # Get scenario name
    scen_name = rt_folder.split('/')[i] + ('_old' if use_v3 else '')

    # Convert using appropriate converter
    if use_v3:
        return dm.insite_rt_converter_v3(rt_folder, None, None, old_params_dict, scen_name)
    else:
        return dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=True)

# Example usage
# rt_folder = './P2Ms/asu_campus/study_area_asu5'
# rt_folder = './P2Ms/simple_street_canyon_test/study_rays=0.25_res=2m_3ghz'
rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/DeepMIMO_folder'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/sionna_test'

# Convert using v4 converter
scen_name = convert_scenario(rt_folder, use_v3=False)

#%% V4 Generation

# Start timing
start_time = time.time()

# scen_name = 'DeepMIMO_folder'
# scen_name = 'simple_street_canyon_test'
# scen_name = 'asu_campus'

# Option 1 - dictionaries per tx/rx set and tx/rx index inside the set)
tx_sets = {1: [0]}
# rx_sets = {1: [0]}
rx_sets = {2: 'all'}#[0,1,2,3,4,5,6,7,8,9,10]}

# Option 2 - lists with tx/rx set (assumes all points inside the set)
# tx_sets = [1]
# rx_sets = [2]

# Option 3 - string 'all' (generates all points of all tx/rx sets) (default)
# tx_sets = rx_sets = 'all'

load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets}
dataset = dm.load_scenario(scen_name, **load_params)
# pprint(dataset)

# dataset.info() # print available tx-rx information

# V4 from Dataset

# Create channel generation parameters
ch_params = dm.ChannelGenParameters()

# Using direct dot notation for parameters
ch_params.bs_antenna.rotation = np.array([30,40,30])
ch_params.bs_antenna.fov = np.array([360, 180])
ch_params.ue_antenna.fov = np.array([120, 180])
ch_params.freq_domain = True

# Basic computations
p = dataset.power_linear  # Will be computed from dataset.power

dataset.power_linear *= 1000  # JUST TO BE COMPATIBLE WITH V3

# Other computations
dataset.compute_channels(ch_params)

var_names = ['channel', 'num_paths', 'distances', 'pathloss', 'grid_size', 
             'grid_spacing', 'los']

for var_name in list(dataset.keys())+var_names:
    print(f'dataset.{var_name}: {dataset[var_name]}')

dataset.info()

# End timing
end_time = time.time()
print(f"Time elapsed: {end_time - start_time:.2f} seconds")

#%% V3 Generation

# Start timing
start_time = time.time()
params = dm.Parameters_old(scen_name + '_old')
params['bs_antenna']['rotation'] = np.array([30,40,30])
params['bs_antenna']['fov'] = np.array([360, 180])
params['ue_antenna']['fov'] = np.array([120, 180])
params['freq_domain'] = True

params['user_rows'] = np.arange(1)
dataset2 = dm.generate_old(params)

# End timing
end_time = time.time()
print(f"Time elapsed: {end_time - start_time:.2f} seconds")

# Verification
i = 10
a = dataset['ch'][i]
b = dataset2[0]['user']['channel'][i]
pprint(a.flatten()[-10:])
pprint(b.flatten()[-10:])
pprint(np.max(np.abs(a-b)))

#%% Demo




#%% Visualization check
import deepmimo as dm
dataset = dm.load_scenario('asu_campus', tx_sets={1: [0]}, rx_sets={2: 'all'})
idxs = dm.uniform_sampling([8,4], 321, 411)
dm.plot_coverage(dataset.rx_pos[idxs], dataset.aoa_az[idxs, 0], bs_pos=dataset.tx_pos.T)

#%%

import deepmimo as dm
dataset = dm.load_scenario('simple_street_canyon_test', tx_sets={1: [0]}, rx_sets={2: 'all'})
idxs = dataset.get_uniform_idxs([1,1])
idxs = np.arange(dataset.rx_pos.shape[0])
dm.plot_coverage(dataset.rx_pos[idxs], dataset.aoa_az[idxs, 0], bs_pos=dataset.tx_pos.T)


# import matplotlib.pyplot as plt
# plt.scatter(dataset['rx_pos'][10,0], dataset['rx_pos'][10,1], c='k', s=20)

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
dm.summary('asu_campus')
# %%
import deepmimo as dm

dm.upload('./asu_campus.zip', '<your-upload-key>')

# download_url = dm.download('asu_campus')
# print(download_url)


#%%
import shutil
import deepmimo.consts as c
import deepmimo.general_utilities as gu

scen_name = 'asu_campus'
scen_folder = c.SCENARIOS_FOLDER + '/' + scen_name

# Get params.mat path
params_path = scen_folder + f'/{c.PARAMS_FILENAME}.mat'

# Zip scenario and get path
zip_path = gu.zip(scen_folder)

# Upload to DeepMIMO
dm.upload(zip_path, '<your-upload-key>') # get key from DeepMIMO website 

# Download from DeepMIMO
#downloaded_zip_path = dm.download(scen_name)

# (simulate a downloaded file while the download is not working)
downloaded_zip_path = zip_path.replace('.zip', '_downloaded.zip') # a simulation folder
shutil.copy(zip_path, downloaded_zip_path)

# Unzip downloaded scenario
unzipped_folder = gu.unzip(downloaded_zip_path)

# Move unzipped folder to scenarios folder
# shutil.move(unzipped_folder, scen_folder)
import os
basename = os.path.basename(unzipped_folder)

# Try loading the scenario and check if: a) find the path b) 
dataset = dm.load_scenario(basename)

# Check if the original and downloaded scenarios are the same













# %%
