#%% Imports

import os
import numpy as np
import deepmimo as dm
import matplotlib.pyplot as plt


from my_api_key import API_KEY as MY_API_KEY

#%% V4 Conversion

# Example usage
rt_folder = './P2Ms/asu_campus'
# rt_folder = './P2Ms/simple_street_canyon_test'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/DeepMIMO_folder'
# rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/sionna_test'
rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/all_runs/run_02-02-2025_15H45M26S/scen_0/sionna_export_test2'

scen_name = os.path.basename(rt_folder)
dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=True)

#%%

dataset = dm.load(scen_name)

dm.plot_coverage(dataset.rx_pos, dataset.los, bs_pos=dataset.tx_pos[0].T)


#%% V4 Generation

# scen_name = 'DeepMIMO_folder'
# scen_name = 'simple_street_canyon_test'
scen_name = 'asu_campus'

# Option 1 - dictionaries per tx/rx set and tx/rx index inside the set)
tx_sets = {1: [0]}
rx_sets = {0: 'all'}

load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': 25}
dataset = dm.load(scen_name, **load_params)

# Create channel generation parameters
ch_params = dm.ChannelGenParameters()

# Using direct dot notation for parameters
ch_params.num_paths = 5
ch_params.ue_antenna.shape = np.array([1,1])

dataset.compute_channels(ch_params)

print(f'channel parameters = {ch_params}')
print(f'channel.shape = {dataset.channel.shape}')


#%% Test available_txrx_pairs

# Get all available TX-RX pairs
txrx_sets = dm.get_txrx_sets(scen_name)
pairs = dm.get_txrx_pairs(txrx_sets)

#%% PLOT RAYS

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

#%% LOOP for zip (no upload)

# Get all available scenarios using function
scenarios = dm.get_available_scenarios()

# Zip the filtered scenarios
for scenario in scenarios:
    scen_path = dm.get_scenario_folder(scenario)
    # dm.zip(scen_path)
    if not ('boston' in scenario):
        continue
    print(f"\nProcessing: {scenario}")
    # continue
    dm.upload(scenario, MY_API_KEY, skip_zip=False)

#%% LOOP all scenarios (summary, load, plot)

import deepmimo as dm
import matplotlib.pyplot as plt
base_path = 'F:/deepmimo_loop_ready'
subfolders = [f.path for f in os.scandir(base_path) if f.is_dir()]

# Load all scenarios
for subfolder in subfolders[:-5]:
    scen_name = os.path.basename(subfolder)
    
    # dm.summary(scen_name)

    # dm.load(scen_name, 'matrices': None)

    # TODO: Make this work for loading only the first tx and rx set
    try:
        d = dm.load(scen_name, tx_sets={1: []}, rx_sets={2: []})
    except Exception as e:
        d = dm.load(scen_name, tx_sets={1: []}, rx_sets={4: []})
    _, ax = d.scene.plot(show=False)
    ax.set_title(scen_name + ': ' + ax.get_title())
    plt.show()


# TODO: Give argument to also not load any matrices (only scene)
