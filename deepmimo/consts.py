"""
Constants and configuration parameters for the DeepMIMO dataset generation.

This module contains all constant definitions used throughout the DeepMIMO toolkit,
including parameter names, file paths, and configuration options for both v4 and
legacy versions of DeepMIMO.
"""

import numpy as np


# Core Configuration - defines DeepMIMO version and numerical precision
VERSION = 4
FP_TYPE = np.float32  # floating point precision for saving values

# Supported ray tracers and their versions
RAYTRACER_NAME_WIRELESS_INSITE = 'Remcom Wireless Insite'
RAYTRACER_VERSION_WIRELESS_INSITE = 3.3
RAYTRACER_NAME_SIONNA = 'Sionna Ray Tracing'           # not supported yet
RAYTRACER_NAME_AODT = 'Aerial Omniverse Digital Twin'  # not supported yet


# Interaction Codes
# The codes are read from left to right, starting from the transmitter end
INTERACTION_LOS = 0         # Line-of-sight (direct path)
INTERACTION_REFLECTION = 1  # Reflection
INTERACTION_DIFFRACTION = 2 # Diffraction
INTERACTION_TRANSMISSION = 3 # Transmission
INTERACTION_SCATTERING = 4  # Scattering



# DEEPMIMOv4 Fundamental Parameters
AOA_AZ_PARAM_NAME = 'aoa_az'
AOA_EL_PARAM_NAME = 'aoa_el'
AOD_AZ_PARAM_NAME = 'aod_az'
AOD_EL_PARAM_NAME = 'aod_el'

TOA_PARAM_NAME = 'toa'
PWR_PARAM_NAME = 'power'
PHASE_PARAM_NAME = 'phase'
RX_POS_PARAM_NAME = 'rx_pos'
TX_POS_PARAM_NAME = 'tx_pos'
INTERACTIONS_PARAM_NAME = 'inter'
INTERACTIONS_POS_PARAM_NAME = 'inter_pos'
RT_PARAMS_PARAM_NAME = 'rt_params'
LOAD_PARAMS_PARAM_NAME = 'load_params'
SCENE_PARAM_NAME = 'scene'  # Scene parameters and configuration

# DEEPMIMOv4 Computed Parameters
CHANNEL_PARAM_NAME = 'channel'
NUM_PATHS_PARAM_NAME = 'num_paths'
NUM_PATHS_FOV_PARAM_NAME = 'num_paths_fov'  # Number of paths within FoV for each user
PWR_LINEAR_PARAM_NAME = 'power_linear'
PATHLOSS_PARAM_NAME = 'pathloss'
DIST_PARAM_NAME = 'distance'

# Rotated angles parameters (after antenna rotation)
AOA_AZ_ROT_PARAM_NAME = 'aoa_az_rot'
AOA_EL_ROT_PARAM_NAME = 'aoa_el_rot'
AOD_AZ_ROT_PARAM_NAME = 'aod_az_rot'
AOD_EL_ROT_PARAM_NAME = 'aod_el_rot'

# Field of view filtered angles
AOD_EL_FOV_PARAM_NAME = 'aod_el_rot_fov'  # Elevation angles after rotation and FoV filtering
AOD_AZ_FOV_PARAM_NAME = 'aod_az_rot_fov'  # Azimuth angles after rotation and FoV filtering
AOA_EL_FOV_PARAM_NAME = 'aoa_el_rot_fov'  # Elevation angles after rotation and FoV filtering
AOA_AZ_FOV_PARAM_NAME = 'aoa_az_rot_fov'  # Azimuth angles after rotation and FoV filtering
FOV_MASK_PARAM_NAME = 'fov_mask'      # Boolean mask for FoV filtering

# Power parameters
PWR_LINEAR_ANT_GAIN_PARAM_NAME = 'power_linear_ant_gain'

# Aliases
# Channel aliases
CHANNEL_PARAM_NAME_2 = 'channels'
CHANNEL_PARAM_NAME_3 = 'ch'
CHANNEL_PARAM_NAME_4 = 'chs'
# Pathloss aliases
PATHLOSS_PARAM_NAME_2 = 'path_loss'
PATHLOSS_PARAM_NAME_3 = 'pl'
# Distance aliases
DIST_PARAM_NAME_2 = 'distances'
DIST_PARAM_NAME_3 = 'dist'
DIST_PARAM_NAME_4 = 'dists'
# Number of paths aliases
NUM_PATHS_PARAM_NAME_2 = 'n_paths'
# Power aliases
PWR_PARAM_NAME_2 = 'pwr'
PWR_PARAM_NAME_3 = 'powers'
PWR_LINEAR_PARAM_NAME_2 = 'pwr_lin'
PWR_LINEAR_PARAM_NAME_3 = 'power_lin'
PWR_LINEAR_PARAM_NAME_4 = 'pwr_linear'
# Position aliases
RX_POS_PARAM_NAME_2 = 'rx_loc'
RX_POS_PARAM_NAME_3 = 'rx_position'
RX_POS_PARAM_NAME_4 = 'rx_locations'
TX_POS_PARAM_NAME_2 = 'tx_loc'
TX_POS_PARAM_NAME_3 = 'tx_position'
TX_POS_PARAM_NAME_4 = 'tx_locations'
# Angle aliases
AOA_AZ_PARAM_NAME_2 = 'aoa_azimuth'
AOA_EL_PARAM_NAME_2 = 'aoa_elevation'
AOD_AZ_PARAM_NAME_2 = 'aod_azimuth'
AOD_EL_PARAM_NAME_2 = 'aod_elevation'
# Time aliases
TOA_PARAM_NAME_2 = 'time_of_arrival'
# Interaction aliases
INTERACTIONS_PARAM_NAME_2 = 'interactions'
INTERACTIONS_POS_PARAM_NAME_2 = 'interaction_locations'

# Load Params
LOAD_PARAM_MAX_PATH = 'max_paths'

# ######### BOTH DM v4 and older versions ###########

SCENARIOS_FOLDER = 'deepmimo_scenarios2'

# Channel parameters
PARAMSET_POLAR_EN = 'enable_dual_polar'
PARAMSET_DOPPLER_EN = 'enable_doppler' # Doppler from Ray Tracer
PARAMSET_FD_CH = 'freq_domain' # Time Domain / Frequency Domain (OFDM)

PARAMSET_OFDM = 'ofdm'
PARAMSET_OFDM_SC_NUM = 'subcarriers'
PARAMSET_OFDM_SC_SAMP = 'selected_subcarriers'
PARAMSET_OFDM_BW = 'bandwidth'
PARAMSET_OFDM_BW_MULT = 1e9 # Bandwidth input is GHz, multiply by this
PARAMSET_OFDM_LPF = 'rx_filter'

PARAMSET_ANT_BS = 'bs_antenna'
PARAMSET_ANT_UE = 'ue_antenna'
PARAMSET_ANT_SHAPE = 'shape'
PARAMSET_ANT_SPACING = 'spacing'
PARAMSET_ANT_ROTATION = 'rotation'
PARAMSET_ANT_RAD_PAT = 'radiation_pattern'
PARAMSET_ANT_RAD_PAT_VALS = ['isotropic', 'halfwave-dipole']
PARAMSET_ANT_FOV = 'fov'

# ######### OLDER VERSIONS ###########

# Dict names
DICT_UE_IDX = 'user'
DICT_BS_IDX = 'basestation'

# NAME OF PARAMETER VARIABLES
PARAMSET_DATASET_FOLDER = 'dataset_folder'
PARAMSET_SCENARIO = 'scenario'
PARAMSET_DYNAMIC_SCENES = 'dynamic_scenario_scenes'

PARAMSET_NUM_PATHS = 'num_paths'
PARAMSET_ACTIVE_BS = 'active_BS'
PARAMSET_USER_ROWS = 'user_rows'
PARAMSET_USER_SUBSAMP = 'user_subsampling'

PARAMSET_BS2BS = 'enable_BS2BS'

# INNER VARIABLES
PARAMSET_ACTIVE_UE = 'active_UE'
PARAMSET_SCENARIO_FIL = 'scenario_files'
PARAMSET_ANT_BS_DIFF = 'BS2BS_isnumpy'
# Based on this paramater, the BS-BS channels won't be converted from a 
# list of matrices to a single matrix


# SCENARIO PARAMS
PARAMSET_SCENARIO_PARAMS = 'scenario_params'
PARAMSET_SCENARIO_PARAMS_CF = 'carrier_freq'
PARAMSET_SCENARIO_PARAMS_TX_POW = 'tx_power'
PARAMSET_SCENARIO_PARAMS_NUM_BS = 'num_BS'
PARAMSET_SCENARIO_PARAMS_USER_GRIDS = 'user_grids'
PARAMSET_SCENARIO_PARAMS_POLAR_EN = 'dual_polar_available'
PARAMSET_SCENARIO_PARAMS_DOPPLER_EN = 'doppler_available'

PARAMSET_SCENARIO_PARAMS_PATH = 'scenario_params_path'

# OUTPUT VARIABLES
OUT_CHANNEL = 'channel'
OUT_PATH = 'paths'
OUT_LOS = 'LoS'
OUT_LOC = 'location'
OUT_DIST = 'distance'
OUT_PL = 'pathloss'

OUT_PATH_NUM = 'num_paths'
OUT_PATH_DOD_PHI = 'DoD_phi'
OUT_PATH_DOD_THETA = 'DoD_theta'
OUT_PATH_DOA_PHI = 'DoA_phi'
OUT_PATH_DOA_THETA = 'DoA_theta'
OUT_PATH_PHASE = 'phase'
OUT_PATH_TOA = 'ToA'
OUT_PATH_RX_POW = 'power'
OUT_PATH_LOS = 'LoS'
OUT_PATH_DOP_VEL = 'Doppler_vel'
OUT_PATH_DOP_ACC = 'Doppler_acc'
OUT_PATH_ACTIVE = 'active_paths'

# FILE LISTS - raytracing.load_ray_data()
LOAD_FILE_EXT = ['DoD', 'DoA', 'CIR', 'LoS', 'PL', 'Loc']
LOAD_FILE_EXT_FLATTEN =[1, 1, 1, 1, 0, 0]
LOAD_FILE_EXT_UE = ['DoD.mat', 'DoA.mat', 'CIR.mat', 'LoS.mat', 'PL.mat', 'Loc.mat']
LOAD_FILE_EXT_BS = ['DoD.BSBS.mat', 'DoA.BSBS.mat', 'CIR.BSBS.mat', 'LoS.BSBS.mat', 'PL.BSBS.mat', 'BSBS.RX_Loc.mat']

# TX LOCATION FILE VARIABLE NAME - load_scenario_params()
LOAD_FILE_TX_LOC = 'TX_Loc_array_full'

# SCENARIO PARAMS FILE VARIABLE NAMES - load_scenario_params()
LOAD_FILE_SP_VERSION = 'version'
LOAD_FILE_SP_RAYTRACER = 'raytracer'
LOAD_FILE_SP_RAYTRACER_VERSION = 'raytracer_version'
LOAD_FILE_SP_EXT = '.params.mat'
LOAD_FILE_SP_CF = 'frequency'
LOAD_FILE_SP_TX_POW = 'transmit_power' # delete!
LOAD_FILE_SP_NUM_BS = 'num_BS'
LOAD_FILE_SP_USER_GRIDS = 'user_grids'
LOAD_FILE_SP_DOPPLER = 'doppler_available'
LOAD_FILE_SP_POLAR = 'dual_polar_available'
LOAD_FILE_SP_NUM_TX_ANT = 'num_tx_ant' 
LOAD_FILE_SP_NUM_RX_ANT = 'num_rx_ant' 

# Physical Constants
LIGHTSPEED = 299792458  # Speed of light in m/s
