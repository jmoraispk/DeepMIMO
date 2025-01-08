#%%
# The directives below are for ipykernel to auto reload updated modules (e.g. in vscode)
# %reload_ext autoreload
# %autoreload 2

#%%

import deepmimo as dm
# path_to_p2m_folder = r'.\P2Ms\ASU_campus_just_p2m\study_area_asu5'
path_to_p2m_outputs = r'.\P2Ms\simple_street_canyon_test\study_rays=0.25_res=2m_3ghz'

scen_name = dm.create_scenario(path_to_p2m_outputs)

#%%
scen_name = dm.create_scenario(path_to_p2m_outputs,
                               copy_source=True, 
                               tx_set_ids=[1], rx_set_ids=[2],
                               overwrite=True, 
                               old=True)

# NOTE: the old version requires copying the params file & has static params

#%%
import deepmimo as dm
scen_name = 'simple_street_canyon_test'
params = dm.Parameters(scen_name)#asu_campus')
# params.get_params_dict()['user_rows'] = np.arange(91)
dataset = dm.generate(params)

#%% Dream

import deepmimo as dm
scen_name = dm.create_scenario(r'.\P2Ms\simple_street_canyon')
dataset = dm.generate(scen_name)

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

