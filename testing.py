#%%

# The directives below are for ipykernel to auto reload updated modules (avoid importlib)
%reload_ext autoreload
%autoreload 2

import deepmimo as dm

#%%

dm.create_scenario('asu_campus2')

#%%
params = dm.Parameters('city_2_chicago')#asu_campus')

dataset = dm.generate(params)

#%%
from deepmimo.generator.python.downloader import download_scenario_handler, NAME_TO_LINK

for scen in NAME_TO_LINK.keys():
    download_scenario_handler(scen)
    