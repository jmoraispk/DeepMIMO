# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:16:57 2025

@author: joao

Comparing DeepMIMOv3 with DeepMIMO(v4)
"""

import DeepMIMOv3 as dm_v3
import deepmimo as dm

# 1- add converter(s)
# 2- add new features to enhance user experience
# 3- improve the format (simpler, more efficient, more user friendly)
#   - dictionary bottleneck ("Dicts of Lists >> Lists of Dicts")
# 4- organize, document, and make all code readable -> prepare for wide-scale open source

#%% Creating a scenario (v3)

# Generators not available in dm_v3
# Need to go into personal repository and run the generator function
# We have loaded those function into dm

# V3
dm.old_conv.v3('')

# V4
dm.create('')

# Diff
# - space
# - no intermediate files
# - speed
# - ready to upload to DeepMIMO database
# - other:
#   - outputs to deepmimo_scenarios
#   - saves interactions and interactions types (R, D, S, T, F/X)
#   - no dynamic scenes

#%% Generating a scenario (v4)

# V3

# V4
# dm.generate(scen)

# or
# dm.load()
# dataset.compute_chs()

# Diff
# - speed: ... vs ...
# - less necessary files: 
#   - .cir, .doa, .dod, .pl, .paths[.t001_{tx_id}.r{rx_id}.p2m], 
#   vs
#    - .pl and .paths (and only .paths if we didn't want positions without paths)
# - fair speed (with channel gen): ... vs ...
# - simpler and more consistent API
#   - dataset['aoa']
#   vs
#   - dataset[0]['user']['paths'][0]['DoA_phi']
# - gen features
#   - generates all TX-RX sets by default
#   - generates only selected users (not rows)
#   - generates users with xyz subsampling
#   - downloads scenarios automatically by name (no need for manual download)
#   - can generate only active users (that have paths)
#   - decouple pl, dist, and channel generation
#     (via smart object that says when something needs computation)
#       - compute_pl()
#       - compute_pg()
#       - compute_dist()
#       - compute_channels()
#       - get_min_usr_spacing()
#       - ...
#   - multiple antenna (extended API from dual polarization - cue dual pol from name)
# - info features
#   - help about any function or parameter (including website ref)
#   - scenario information
#   - buildings
# - plot features
#   - buildings
#
