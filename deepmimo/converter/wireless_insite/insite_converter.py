# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:55:17 2023

@author: demir
"""
import os
import numpy as np
import scipy.io

from .. import converter_utils as cu

from .ChannelDataLoader import WIChannelConverter
from .ChannelDataFormatter import DeepMIMODataFormatter
from .scenario_utils import ScenarioParameters

from typing import List

class Material(): # TODO: move to parameters
    def __init__(self):
        """
        Diffuse model implemented from [1] + extended with cross-polarization scattering terms
        
        Diffuse scattering models explained in [2], slides 29-31. 
        
        Sources:
            [1] A Diffuse Scattering Model for Urban Propagation Prediction - Vittorio Degli-Esposti 2001
                https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=933491
            [2] https://x.webdo.cc/userfiles/Qiwell/files/Remcom_Wireless%20InSite_5G_final.pdf

        """
        self.name = ''
        self.type = 0
        self.diffuse_scat_model = '' # 'labertian', 'directive', 'directive_w_backscatter'
        self.scat_fact = 0           # 0-1, fraction of incident fields that are scattered
        self.cross_pol_frac = 0      # 0-1, fraction of the scattered field that is cross pol
        self.directive_alpha = 0     # 1-10, defines how broad forward beam is
        self.directive_beta = 0      # 1-10, defines how broad backscatter beam is
        self.directive_lambda = 0    # 0-1, fraction of the scattered power in forward direction (vs back)
        
        self.conductivity = 0        # ..
        self.permittivity = 0        # ..
        self.roughness = 0           # .. affects ...
        self.thickness = 0           # [m]

    


def insite_rt_converter(p2m_folder: str, copy_source: bool = False,
                        tx_ids: List[int] = None, rx_ids: List[int] = None):

    # Parameters to be given for the scenario
    intermediate_folder = os.path.join(os.path.dirname(p2m_folder), 'intermediate_files')
    output_folder = os.path.join(os.path.dirname(p2m_folder), 'mat_files')
    
    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    # Check that the necessary files exist:
    files_in_folder = os.listdir(p2m_folder)
    for ext in ['.setup', '.txrx']:
        if not cu.ext_in_list(ext, files_in_folder):
            raise Exception(f'{ext} not found in {p2m_folder}')
        
    if copy_source:
        ['.setup', '.txrx', '.ter', '.city']
        pass

    # Setup (.setup)
    # tx power
    # dual pol (antennas)

    # TXRX (.txrx)
    # txrx <grid>
    # side1 410.00000
    # side2 320.00000
    # spacing 1.00000
    
    # power 0.00000

    # Terrain (.ter, .city):
    # read .ter and .city to extract the name of the objects, the materials, and their properties.

    # materials used in buildings: ...
    # materials used in terrains: ...
    # For more info, inspect the raytracing source available in {website}. (or offer option to dload RT source)

    # P2Ms (.cir, .doa, .dod, .paths[.t{tx_id}_{??}.r{rx_id}.p2m] e.g. .t001_01.r001.p2m)

    # Convert P2M files to mat format
    # WIChannelConverter(p2m_folder, intermediate_folder)

    if tx_ids is None:
        # read all tx idxs...
        pass
    if rx_ids is None:
        # read ...
        pass

    dm = DeepMIMODataFormatter(intermediate_folder, output_folder, TX_order=[1], RX_order=[4])
                                                                 # TODO: read this automatically from P2M

    data_dict = {
                'version': 2,
                'carrier_freq': 28e9, ############# REAAAAD
                'transmit_power': 0.0, #dB from the scenario ############# REAAAAD
                # Start row - end row - num users - Num users must be larger than the maximum number of dynamic receivers
                'user_grids': np.array([[1, 411, 321]], dtype=float), ############# REAAAAD
                'num_BS': len(dm.TX_order), ############# REAAAAD
                'dual_polar_available': 0, ############# REAAAAD
                'doppler_available': 0
                #'BS_grids': np.array([[i+1, i+1, 1] for i in range(self.num_BS)]).astype(float)

                # n_paths, n_reflections, type of difusion, ...

                }
        
    scipy.io.savemat(os.path.join(output_folder, 'params.mat'), data_dict)

    return output_folder

    # JTODO 1: copy raytracing source to zip
    # JTODO 2: read parameters to DeepMIMO metadata (accessible via print) -> these will also be used in the website
    # JTODO 3: some parameters are used for actual generation, like the size of user grids, etc.. 
    # JTODO 4: convert all users, but add option to generate only users with channels
    # JTODO 5: DoA DoD -> AoA AoD

    # TODO: eliminate the intermediate files -> ****** no need ******

    keys = [
        'diffuse_reflections', 
        'diffuse_diffractions',
        'diffuse_transmissions',
        'max_reflections', 

    ]