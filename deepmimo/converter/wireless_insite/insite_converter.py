# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:55:17 2023

@author: demir
"""
import os
import shutil
import numpy as np
import scipy.io

from .. import converter_utils as cu
from ...general_utilities import PrintIfVerbose

from .ChannelDataLoader import WIChannelConverter
from .ChannelDataFormatter import DeepMIMODataFormatter

from typing import List, Dict

class InsiteMaterial():
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
                        tx_ids: List[int] = None, rx_ids: List[int] = None,
                        verbose: bool = True):

    # Setup output folder
    insite_sim_folder = os.path.dirname(p2m_folder)
    output_folder = os.path.join(insite_sim_folder, 'mat_files') # SCEN_NAME!
    os.makedirs(output_folder, exist_ok=True)

    # DELETE??
    intermediate_folder = os.path.join(insite_sim_folder, 'intermediate_files')
    os.makedirs(intermediate_folder, exist_ok=True)

    # Check if necessary files exist
    verify_sim_folder(insite_sim_folder, verbose)
    
    # Copy ray tracing source files
    if copy_source: copy_rt_source_files(insite_sim_folder, verbose)
    
    files_in_sim_folder = [os.path.join(insite_sim_folder, file) 
                           for file in os.listdir(insite_sim_folder)]

    # Read setup (.setup)
    setup_file = cu.ext_in_list('.setup', files_in_sim_folder)[0]
    setup_dict = read_setup(setup_file, verbose)

    # Read TXRX (.txrx)
    txrx_file = cu.ext_in_list('.txrx', files_in_sim_folder)[0]
    avail_tx_idxs, avail_rx_idxs, txrx_dict = read_txrx(txrx_file, verbose)
    
    tx_ids = tx_ids if tx_ids else avail_tx_idxs
    rx_ids = rx_ids if rx_ids else avail_rx_idxs

    # Read Terrain and Buildings Materials (.ter, .city):
    city_files = cu.ext_in_list('.city', files_in_sim_folder)
    ter_files = cu.ext_in_list('.ter', files_in_sim_folder)
    city_materials = read_city(city_files, verbose)
    ter_materials = read_ter(ter_files, verbose)
    # For more info, inspect the raytracing source available in {website}. 
    # (or offer option to dload RT source)

    export_params_dict(output_folder, setup_dict, txrx_dict, 
                       city_materials, ter_materials)

    # P2Ms (.cir, .doa, .dod, .paths[.t{tx_id}_{??}.r{rx_id}.p2m] e.g. .t001_01.r001.p2m)

    # Convert P2M files to mat format
    # WIChannelConverter(p2m_folder, intermediate_folder)

    dm = DeepMIMODataFormatter(intermediate_folder, output_folder, 
                               TX_order=tx_ids, RX_order=rx_ids)
    #                            # TODO: read this automatically from P2M
    name = os.path.basename(insite_sim_folder)
    shutil.move(output_folder, 'deepmimo_scenarios/{name}')
    return name

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

def verify_sim_folder(sim_folder, verbose):
    
    files_in_sim_folder = os.listdir(sim_folder)
    for ext in ['.setup', '.txrx']:
        files_found_with_ext = cu.ext_in_list(ext, files_in_sim_folder)
        if verbose:
            print(f'Found {files_found_with_ext}')
        if len(files_found_with_ext) == 0:
            raise Exception(f'{ext} not found in {sim_folder}')
        elif len(files_found_with_ext) > 1:
            raise Exception(f'Several {ext} found in {sim_folder}')

def copy_rt_source_files(sim_folder: str, verbose: bool = True):
    
    vprint = PrintIfVerbose(verbose) # prints if verbose 
    rt_source_folder = 'raytracing_source'
    files_in_sim_folder = os.listdir(sim_folder)
    vprint('Copying raytracing source files to "rt_source_folder"')
    zip_temp_folder = os.path.join(sim_folder, rt_source_folder)
    os.makedirs(zip_temp_folder)
    for ext in ['.setup', '.txrx', '.ter', '.city']:
        # copy all files with extensions to temp folder
        for file in cu.ext_in_list(ext, files_in_sim_folder):
            curr_file_path = os.path.join(sim_folder, file)
            new_file_path  = os.path.join(zip_temp_folder, file)
            
            vprint(f'Copying {file}')
            shutil.copy(curr_file_path, new_file_path)
    
    vprint('Zipping')
    cu.zip_folder(zip_temp_folder)
    
    vprint(f'Deleting temp folder {os.path.basename(zip_temp_folder)}')
    shutil.rmtree(zip_temp_folder)
    
    vprint('Done')

def read_setup(file: str, verbose: bool):
    print(f'Reading setup file: {os.path.basename(file)}')
    
    setup_dict = {
        'tx_power': 0,
        'dual_pol': 0
    }
    return setup_dict

def read_txrx(file: str, verbose: bool):
    print(f'Reading txrx file: {os.path.basename(file)}')
    
    txrx_dict = {}

    # txrx <grid>
    # side1 410.00000
    # side2 320.00000
    # spacing 1.00000
    
    # power 0.00000

    return [], [], txrx_dict

def read_city(files: List[str], verbose: bool):
    return read_components_file(files, verbose)

def read_ter(files: List[str], verbose: bool):
    return read_components_file(files, verbose)

def read_components_file(files: List[str], verbose: bool):
    print(f'Reading materials files: {[os.path.basename(f) for f in files]}')
    material_list = [InsiteMaterial() for i in range(1)]
    return material_list

def export_params_dict(output_folder: str, setup_dict: Dict, txrx_dict: Dict, 
                       city_mat_dict: Dict, ter_mat_dict: Dict):
    data_dict = {
                'version': 2,
                'carrier_freq': 28e9, ############# REAAAAD
                'transmit_power': 0.0, #dB from the scenario ############# REAAAAD
                # Start row - end row - num users - Num users must be larger than the maximum number of dynamic receivers
                'user_grids': np.array([[1, 411, 321]], dtype=float), ############# REAAAAD
                'num_BS': 1, #len(dm.TX_order), ############# REAAAAD
                'dual_polar_available': 0, ############# REAAAAD
                'doppler_available': 0
                #'BS_grids': np.array([[i+1, i+1, 1] for i in range(self.num_BS)]).astype(float)

                # n_paths, n_reflections, type of difusion, ...

                }
        
    scipy.io.savemat(os.path.join(output_folder, 'params.mat'), data_dict)