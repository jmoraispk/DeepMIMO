# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:55:17 2023

@author: demir
"""

from .ChannelDataLoader import WIChannelConverter
from .ChannelDataFormatter import DeepMIMODataFormatter
from .scenario_utils import ScenarioParameters
import os


def insite_rt_converter(p2m_folder):

    #%% Parameters to be given for the scenario
    # p2m_folder = r'C:\Users\Umt\Desktop\RIS_Indoor_v2\p2m'
    intermediate_folder = os.path.join(os.path.dirname(p2m_folder), 'intermediate_filesv2')
    output_folder = os.path.join(os.path.dirname(p2m_folder), 'mat_filesv2')
    
    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    #%% Convert P2M files to mat format
    # WIChannelConverter(p2m_folder, intermediate_folder)

    #%%
    dm = DeepMIMODataFormatter(intermediate_folder, output_folder, max_channels=100000,
                               TX_order=[1], RX_order=[1])

    #%%
    import numpy as np
    import scipy.io

    # output_folder = r'C:\Users\Umt\Desktop\dynamic_scenario\dyn3'
    data_dict = {
                'version': 2,
                'carrier_freq': 28e9,
                'transmit_power': 0.0, #dB from the scenario
                'user_grids': np.array([[1, 171, 151] # Start row - end row - num users - Num users must be larger than the maximum number of dynamic receivers
                                        ], 
                                        dtype=float),
                'num_BS': len(dm.TX_order),
                'dual_polar_available': 0,
                'doppler_available': 0
                #'BS_grids': np.array([[i+1, i+1, 1] for i in range(self.num_BS)]).astype(float)
                }
        
    scipy.io.savemat(os.path.join(output_folder, 'params.mat'), data_dict)