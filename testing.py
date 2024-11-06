#%%

# The directives below are for ipykernel to auto reload updated modules (avoid importlib)
%reload_ext autoreload
%autoreload 2

import deepmimo as dm

# params = dm.default_params('asu_campus')
# dataset = dm.generate(params)

dm.create_scenario('asu_campus2')