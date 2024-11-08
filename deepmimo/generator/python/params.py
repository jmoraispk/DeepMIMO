# -*- coding: utf-8 -*-
"""
DeepMIMOv2 Python Implementation

Description: Parameters file

Authors: Umut Demirhan, Ahmed Alkhateeb
Date: 12/10/2021
"""

import os 
import numpy as np
from . import consts as c

# JTODO: make this into class

class Parameters():
    def __init__(self, scen_name=None):
        self.params = {
            c.PARAMSET_DATASET_FOLDER: c.SCENARIOS_FOLDER,
            c.PARAMSET_SCENARIO: scen_name if scen_name else'O1_60' ,
            
            # Dynamic scenario scene selection
            c.PARAMSET_DYNAMIC_SCENES: np.array([1]),
            
            # Maximum # of paths to be loaded
            c.PARAMSET_NUM_PATHS: 5,
            
            # Active Basestation IDs
            c.PARAMSET_ACTIVE_BS: np.array([1]), # To be implemented
            
            # Selected user rows
            c.PARAMSET_USER_ROWS: np.array([1]),
            
            # Subsampling
            # c.PARAMSET_USER_ROW_SUBSAMP: 1,
            c.PARAMSET_USER_SUBSAMP: 1,
            
            # BS Antenna Parameters
            c.PARAMSET_ANT_BS: {
                c.PARAMSET_ANT_SHAPE: np.array([8, 4]), # Antenna dimensions in X - Y - Z
                c.PARAMSET_ANT_SPACING: 0.5,
                c.PARAMSET_ANT_ROTATION: np.array([0, 0, 0]), # Rotation around X - Y - Z axes
                c.PARAMSET_ANT_FOV: np.array([360, 180]), # Horizontal-Vertical FoV
                c.PARAMSET_ANT_RAD_PAT: c.PARAMSET_ANT_RAD_PAT_VALS[0] # 'omni-directional'
                },
            
            # UE Antenna Parameters
            c.PARAMSET_ANT_UE: {
                c.PARAMSET_ANT_SHAPE: np.array([4, 2]), # Antenna dimensions in X - Y - Z
                c.PARAMSET_ANT_SPACING: 0.5,
                c.PARAMSET_ANT_ROTATION: np.array([0, 0, 0]), # Rotation around X - Y - Z axes
                c.PARAMSET_ANT_FOV: np.array([360, 180]), # Horizontal-Vertical FoV
                c.PARAMSET_ANT_RAD_PAT: c.PARAMSET_ANT_RAD_PAT_VALS[0] # 'omni-directional'
                },
            
            c.PARAMSET_DOPPLER_EN: 0,
            c.PARAMSET_POLAR_EN: 0,
            c.PARAMSET_BS2BS: 0,
            
            c.PARAMSET_FDTD: 1 , 
            # OFDM if 1 
            # Time domain if 0.
            # In time domain, the channel of 
            # RX antennas x TX antennas x Number of available paths is generated.
            # Each matrix of RX ant x TX ant represent the response matrix of that path.
            
            # OFDM Parameters
            c.PARAMSET_OFDM: {
                c.PARAMSET_OFDM_SC_NUM: 512, # Number of total subcarriers
                c.PARAMSET_OFDM_SC_SAMP: np.arange(1), # Select subcarriers to generate
                                    
                c.PARAMSET_OFDM_BW: 0.05, # GHz
                
                # Receive Low Pass / ADC Filter
                # 0: No Filter - Delta Function 
                # 1: Ideal (Rectangular) Low Pass Filter - Sinc Function
                c.PARAMSET_OFDM_LPF: 0
                }
            }
        
    def get_params_dict(self):
        return self.params
    
    def get_name(self):
        return self.params[c.PARAMSET_SCENARIO]
    
    def get_folder(self):
        return os.path.abspath(self.params[c.PARAMSET_DATASET_FOLDER])
    
    def get_path(self):
        return os.path.join(self.get_folder(), self.params[c.PARAMSET_SCENARIO])
    