# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:55:17 2023

@author: demir
"""

from ChannelDataLoader import WIChannelConverter
from ChannelDataFormatter import DeepMIMODataFormatter
from scenario_utils import ScenarioParameters
import os

#%% Parameters to be given for the scenario
p2m_folder = r'C:\Users\Umt\Desktop\Boston5G_3p5_small\Boston5G_3p5'
intermediate_folder = os.path.join(os.path.dirname(p2m_folder), 'intermediate_files')
output_folder = os.path.join(os.path.dirname(p2m_folder), 'mat_files')

#%% Convert P2M files to mat format
channel_data = WIChannelConverter(p2m_folder, intermediate_folder)

#%%
DeepMIMODataFormatter(intermediate_folder, output_folder, max_channels=10000)

#%%
import numpy as np
import scipy.io

output_folder = r'C:\Users\Umt\Desktop\Boston5G_3p5_small\Boston5G_3p5_v1'
data_dict = {
              'version': 2,
              'carrier_freq': 3.4e9,
              'transmit_power': 0.0, #dB from the scenario
              'user_grids': np.array([[1, 151, 111], # Start row - end row - num users
                                      [152, 302, 111]], 
                                     dtype=float),
              'num_BS': 2,
              #'BS_grids': np.array([[i+1, i+1, 1] for i in range(self.num_BS)]).astype(float)
             }
    
scipy.io.savemat(os.path.join(output_folder, 'params.mat'), data_dict)