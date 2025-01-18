import os 
import numpy as np
from ... import consts as c
from pprint import pformat # returns pprint string

class ChannelGenParameters():
    def __init__(self, scen_name=None):
        self.params = {

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
            
            c.PARAMSET_FD_CH: 1, 
            # OFDM channel if 1 (True)
            # Time domain if 0 (False.
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
    
    def __repr__(self):
        return pformat(self.get_params_dict())
    