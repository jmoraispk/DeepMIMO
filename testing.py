#%%
# The directives below are for ipykernel to auto reload updated modules (e.g. in vscode)
# %reload_ext autoreload
# %autoreload 2

#%% V3 & V4 Conversion

import deepmimo as dm
# path_to_p2m_outputs = r'.\P2Ms\asu_campus\study_area_asu5'
path_to_p2m_outputs = r'.\P2Ms\simple_street_canyon_test\study_rays=0.25_res=2m_3ghz'

scen_name = dm.create_scenario(path_to_p2m_outputs,
                               overwrite=True, 
                               old=True) # V3 or V4 conversion flag
#                              old has STATIC PARAMS! (asu scenario needs diff grids)

#%% V4 
import deepmimo as dm
scen_name = 'simple_street_canyon_test'

load_params = {}
dataset = dm.load(scen_name, **load_params)
# dataset.gen_channels()

#dataset = dm.generate(params)

#%% V3 Generation

import deepmimo as dm
scen_name = 'simple_street_canyon_test'
params = dm.Parameters_old(scen_name)#asu_campus')
# params.get_params_dict()['user_rows'] = np.arange(91)
dataset = dm.generate_old(params)

#%% Dream

# Wireless Insite IDXs = [3, 7, 8]
# DeepMIMO (after conversion) TX/RX Sets: [1, 2, 3]
# DeepMIMO (after generation) : only individual tx and rx indices

# 0- Conversion is general. Make bindings for a general generation
#    (by having an old and new version of the code and calling diff funcs)
# [DONE]

# 1- Make parameters optional in dm.generate (so we don't need to create params)
# [DONE]

# 2- Remove BS-BS specific function (use normal generation)
# [DONE]

# 3- Decouple channel generation from the dataset generation

# 4- Add new (multi-txrx) way of generating data

# tx_set = ...
# Option 1: {1: [0,2,4], 2: [3,4,5,], 3: 'all'}
# Option 2: [0, 1]
# Option 3: 'all' (default)

# 5- IMPLEMENT new structure of dataset and generate from new matrices

# 6- Add new smart object: 
#   - dataset.compute_channels()
#   - dataset.compute_pl() -> unlocks 'pathloss'
#   - dataset.compute_dists()
#   - dataset.compute_num_paths()
#   - dataset.compute_num_interactions() -> unlocks 'num_interactions'

# ---- (later) ----

# 7- Make it work with ['chs'], ['channels'], etc..

# 8- Simplified building save matrix & plots

# 9- RUN BLACK to format all code (consistent)

# 10- REFACTORING CONVERSION: move MATERIAL and TXRX to separate files

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

# 7) Generate scenario automatically for ASU and street canyon
# 8) Save channels for validation
# 9) Time conversion and Generation speeds to compare with new formats

# 10) Redo Insite Converter to use the new format (and don't store empty users?)

