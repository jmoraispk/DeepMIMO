#%% V3 & V4 Conversion

import deepmimo as dm
path_to_p2m_outputs = r'.\P2Ms\asu_campus\study_area_asu5'
# path_to_p2m_outputs = r'.\P2Ms\simple_street_canyon_test\study_rays=0.25_res=2m_3ghz'

if 'asu_campus' in path_to_p2m_outputs:
    old_params_dict = {'num_bs': 1, 'user_grid': [1, 411, 321], 'freq': 3.5e9} # asu
else:
    old_params_dict = {'num_bs': 1, 'user_grid': [1, 91, 61],   'freq': 3.5e9}   # simple canyon

old = True
scen_name = path_to_p2m_outputs.split('\\')[2] + ('_old' if old else '')
scen_name = dm.create_scenario(path_to_p2m_outputs,
                               overwrite=True, 
                               old=old,
                               old_params=old_params_dict,
                               scenario_name=scen_name)

#%% V4 
import deepmimo as dm
# scen_name = 'simple_street_canyon_test'
scen_name = 'asu_campus'

# Option 1 - dictionaries per tx/rx set and tx/rx index inside the set)
tx_sets = {1: [0]}
rx_sets = {2: 'all'}

# Option 2 - lists with tx/rx set (assumes all points inside the set)
# tx_sets = [1]
# rx_sets = [2]

# Option 3 - string 'all' (generates all points of all tx/rx sets) (default)
# tx_sets = rx_sets = 'all'

load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': 5,
               'matrices': None}#['aoa_az']}
dataset = dm.load_scenario(scen_name, **load_params)
dataset['load_params'] = load_params  # c.LOAD_PARAMS_PARAM_NAME

# dataset.info() # print available tx-rx information
# from pprint import pprint
# pprint(dataset)

#%%
params = dm.ChannelGenParameters()

# num_paths and power_linear are necessary for channel
dataset['num_paths'] = dm.compute_num_paths(dataset)          # c.NUM_PATHS_PARAM_NAME
dataset['power_linear'] = dm.utils.dbm2watt(dataset['power']) # c.PWR_LINEAR_PARAM_NAME
dataset['channel'] = dm.compute_channels(dataset, params)     # c.CHANNEL_PARAM_NAME
dataset['pathloss'] = dm.compute_pathloss(dataset['power'][10], 
                                          dataset['phase'][10]) # c.PATHLOSS_PARAM_NAME
dataset['distances'] = dm.compute_distances(dataset['rx_pos'], 
                                            dataset['tx_pos'])  # c.DIST_PARAM_NAME


# Aliases for convenience
dataset['pwr'] = dataset['power']
dataset['pwr_lin'] = dataset['power_linear']
dataset['ch'] = dataset['channel']
dataset['pl'] = dataset['pathloss']
dataset['rx_loc'] = dataset['rx_pos']
dataset['tx_loc'] = dataset['tx_pos']
dataset['dist'] = dataset['distances']

# TODO: eliminate c.MAT_VAR_NAME -> use the name of the variable!
#%%
import deepmimo as dm
dataset = dm.generate(scen_name)
# ADD dataset['power_linear']?
# ADD 'fov trim' and 'ant pat pwr' requirements to the compute channels

#%% V3 Generation

import deepmimo as dm
# scen_name = 'simple_street_canyon_test'
scen_name = 'asu_campus'
params = dm.Parameters_old(scen_name)
# params.get_params_dict()['user_rows'] = np.arange(91)
dataset2 = dm.generate_old(params)

chs2 = dataset2[0]['user']['channel']

# TODOOOO: test PARAMSET_OFDM_LPF = 1! (crashing on the other.)

#%%

import deepmimo as dm
scen_name = dm.create_scenario(r'.\P2Ms\asu_campus\study_area_asu5')
dataset = dm.generate(scen_name)
#%%
# load_params = {'tx_sets': [1], 'rx_sets': [2], 'max_paths': 1}
load_params = {'tx_sets': [1], 'rx_sets': {2: 'active'}}
# load_params = {'tx_sets': [1], 'rx_sets': {2: [1,2,3]}}
dataset = dm.load_scenario(scen_name, **load_params)
# dataset = dm.load_scenario('city_10_austin')

#%%

import deepmimo as dm
import matplotlib.pyplot as plt

dm.visualization.plot_coverage(dataset['rx_pos'], dataset['aoa_az'][:, 0],
                               bs_pos=dataset['tx_pos'].T)

plt.scatter(dataset['rx_pos'][100,0], dataset['rx_pos'][100,1], c='k', s=100)

#%% Dream

# 12- REFACTORING CONVERSION: move MATERIAL and TXRX to separate files
# 13- REFACTORING GENERATION: move validation into channel gen script
# 10- Save building matrix & plot building

###############

# -- PRINT INFO: (after loading)
# Wireless Insite IDXs = [3, 7, 8]
# DeepMIMO (after conversion) TX/RX Sets: [1, 2, 3]
# DeepMIMO (after generation) : only individual tx and rx indices

###########
# Essential for release

# 8- Add dataset.info() to dataset (and documentation throughout)
# dm.info('inter') (or dataset['inter'].info())

# 9- Add new smart object: 
# - compute_angles_with_fov()            -> unlocks 'aoa_el_fov', ...
# - compute_power_with_antenna_pattern() -> unlocks 'power_with_ant_pattern'
#   - dataset.compute_los_status()       -> unlocks 'los_status'
#   - dataset.compute_channels()         -> unlocks 'channels'
#   - dataset.compute_pl()               -> unlocks 'pathloss'
#   - dataset.compute_dists()            -> unlocks 'distance'
#   - dataset.compute_num_paths()        -> unlocks 'num_paths'
#   - dataset.compute_num_interactions() -> unlocks 'num_interactions'

# 9.2) dataset.gen_chs() = dataset[i].gen_chs()

"""
Plan: possibly 3 objects

1- Macro dataset (containing the information from several transmitters)
2- Dataset (containing 1tx worth of information)
3- Matrix (like a numpy matrix, but supports a few more operations

Other notes:

* make dataset.aoa_az and dataset[‘aoa_az’] both possible
* @property can be used to define functions to be called when doing a.smth
* @descriptors can be used to define meta properties
* __help__, __getitem__, __setitem__, __getattr__, can be used to make our dataset object
"""




# Other features:

# 11- RUN RUFF to format all code (consistently - https://github.com/astral-sh/ruff)

# 13- Option to include (or not) interaction locations in dataset (3x more space and time needed)

# 15- Enable generating the channel only for R/D/... paths

# 16- Enable generating only up to a given number of interactions

# 17- Add more LinearPath functions
# - To append paths use, for example: linpath3.append(linpath1, linpath2)
# - To repeat back and forth: linpath3.append(linpath3.flip())
# - Supposing the last position of linpath3 coincides with the first, it can be looped like:
# linpath3.append(linpath3, linpath3, linpath3) or linpath3.repeat(3)

# 18 - Add list of variables (keys) to load.
# This will make the generation faster, leaner and customized (for space-constraint applications)
# +++ Since only information is loaded, we can avoid downloading the scenario with
# interactions locations! (and save 3x the speed and space requirements!)
# If access is attempted via dataset['inter_loc'], we can explain how to add this
# variable to the load chain, etc...

# 19 - If anyone asks for aoa or aod, ask if they would like to concatenate them. 
#      np.concatenate(dataset['aoa_az'], dataset['aoa_el'] )
# dataset['aoa'] # N x PATHS x 2 (azimuth / elevation)
# dataset['aoa'] = # concat...aoa_el / aoa_az

# 20 - Add space requirements to auto downloader information

# 21 - Add dual-polarization AND multi-antenna support

#%% READ Setup

# Generation:
# 6) Generate a <info> field with all sorts of information
# dataset[tx]['info']
# maybe also dm.info('chs') | dm.info('aoa_az') | ..

dataset.info() # Print general scenario info, including about available TX/RX Sets
dm.info()
dm.info('params num_paths')
dm.info('params')['num_paths']

def info(s):
    if ' ' in s:
        s1, s2 = s.split()
        info(s2)
    
    a = 'num_paths'
    b = 'Influences the sizes of the matrices aoa, aod, etc... and the generated channels'

