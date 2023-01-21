# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:55:17 2023

@author: demir
"""

from ChannelDataLoader import WIChannelLoader
from scenario_utils import ScenarioParameters

#%% Parameters to be given for the scenario

p2m_folder = r'C:\Users\Umt\Desktop\p2m_data\I3_60_p2m'

sp = ScenarioParameters()
sp.add_BS(TX_ID=13, TX_subID=1, FOV_az=0, FOV_el=0)
sp.add_BS(TX_ID=14, TX_subID=1, FOV_az=0, FOV_el=0)

sp.set_transmission(carrier_freq=60e9, tx_power=0)
sp.set_user_grids(user_grids=[[1, 551, 121], 
                              [552, 1159, 86]]) # Start row - end row - num users


#%% Auto Read-Write

# Save Params File
mat_folder = p2m_folder[:-4]
sp.save(mat_folder)

# Save Channel Files
channel_data = WIChannelLoader(p2m_folder)
channel_data.save_channels(mat_folder, scene_idx=1, interacts=False)

