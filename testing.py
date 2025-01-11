#%%
# The directives below are for ipykernel to auto reload updated modules (e.g. in vscode)
# %reload_ext autoreload
# %autoreload 2

#%% V3 & V4 Conversion

import deepmimo as dm
# path_to_p2m_outputs = r'.\P2Ms\asu_campus\study_area_asu5'
path_to_p2m_outputs = r'.\P2Ms\simple_street_canyon_test\study_rays=0.25_res=2m_3ghz'

old_params_dict = {'num_bs': 1, 'user_grid': [1, 91, 61],   'freq': 3.5e9}   # simple canyon
# old_params_dict = {'num_bs': 1, 'user_grid': [1, 411, 321], 'freq': 3.5e9} # asu


scen_name = dm.create_scenario(path_to_p2m_outputs,
                               overwrite=True, 
                               old=False,
                               old_params=old_params_dict) # V3 or V4 conversion flag
#                              old has STATIC PARAMS! (asu scenario needs diff grids)

#%% V4 
import deepmimo as dm
scen_name = 'simple_street_canyon_test'

# Option 1 - dictionaries per tx/rx set and tx/rx index inside the set)
tx_sets = {1: [0]}
rx_sets = {2: [1,2,3]}

# Option 2 - lists with tx/rx set (assumes all points inside the set)
# tx_sets = [1]
# rx_sets = [2]

# Option 3 - string 'all' (generates all points of all tx/rx sets) (default)
# tx_sets = rx_sets = 'all'

load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': 5}
dataset = dm.load_scenario(scen_name, **load_params)

# make sure the rt_params are in each dataset
# dataset[0].info() # -> bs to bs? bs to ue?

# from pprint import pprint
# pprint(dataset)

#%%
params = dm.ChannelGenParameters()
dataset[0]['chs'] = dm.compute_channels(dataset[0], params)

#%% V3 Generation

import deepmimo as dm
scen_name = 'simple_street_canyon_test'
params = dm.Parameters_old(scen_name)#asu_campus')
# params.get_params_dict()['user_rows'] = np.arange(91)
dataset = dm.generate_old(params)

#%% Dream

# ------
# Wireless Insite IDXs = [3, 7, 8]
# DeepMIMO (after conversion) TX/RX Sets: [1, 2, 3]
# DeepMIMO (after generation) : only individual tx and rx indices

# 6- Make new DeepMIMO work with channel generation

# IMPORTANT: There are functions inside compute_channels() that MODIFY the dataset
# (I'll rename them to what I'd call them when implementing them outside)
# - compute_angles_with_fov()            -> unlocks 'aoa_el_fov', ...
# - compute_power_with_antenna_pattern() -> unlocks 'power_with_ant_pattern'
    # (assumes pattern doesn't mess with polarizations)

# DECISION:
# The channel generation needs these quantities computed. 
# In a first iteration, they will be left inside for now.
# Afterwards, the channel_generation will try to access them and trigger a computation
# in case they don't exist yet.

# JTODO: make them explicitely available outside, so
# people can use them WITHOUT computing the channels (and compare them with the originals!)

# NOTE 2: better to have an antenna "orientation" than an antenna "rotation"

# NOTE 3: having doppler here will only bring trouble because it's RT/conversion dependent, 
#         NOT channel dependent. Since there is only ONE fixed amount
#         of Doppler that can be added, it will be confusing. Better remove it 
#         for now and add it to the right place: generation (not conversion!)

# 7- Add dataset.info() to dataset (and documentation throughout)

# QUESTION: if there's only one BS, why the list? -> flatten dataset?

# 8 - Add option to load only active users

# ---- (later) ----

# 9- Add new smart object: 
#   - dataset.compute_los_status()       -> unlocks 'los_status'
#   - dataset.compute_channels()         -> unlocks 'channels'
#   - dataset.compute_pl()               -> unlocks 'pathloss'
#   - dataset.compute_dists()            -> unlocks 'distance'
#   - dataset.compute_num_paths()        -> unlocks 'num_paths'
#   - dataset.compute_num_interactions() -> unlocks 'num_interactions'


# 9.1) embed compute_channels in dataset (using the outside function?) -> dataset.gen_channels()
# 9.2) make gen_channels() that computes all BSs (like dataset.gen_channels() vs dataset[0].gen_channels())
#       where dataset.gen_channels() just does a for over the len of the dataset and calls gen_channels()
# 9.3) make generate function that does the loading and channel gen.
#       dataset = dm.generate(params) -> with both load and ch_gen


# 10- Save building matrix & plot building

# 11- RUN BLACK to format all code (consistent)

# 12- REFACTORING CONVERSION: move MATERIAL and TXRX to separate files

# 13- REFACTORING GENERATION: move validation into channel gen script

# 13- Option to include (or not) interaction locations in dataset (3x more space and time needed)

# 14- Enable loading x paths

# 15- Enable generating the channel only for R/D/... paths

# 16- Enable generating only up to a given number of interactions


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

