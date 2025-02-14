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
RAYTRACER_VERSION_WIRELESS_INSITE = '3.3'
RAYTRACER_NAME_SIONNA = 'Sionna Ray Tracing'           # not supported yet
RAYTRACER_VERSION_SIONNA = '0.19.1'
RAYTRACER_NAME_AODT = 'Aerial Omniverse Digital Twin'  # not supported yet
RAYTRACER_VERSION_AODT = '1.x'



# Interaction Codes
# The codes are read from left to right, starting from the transmitter end
INTERACTION_LOS = 0         # Line-of-sight (direct path)
INTERACTION_REFLECTION = 1  # Reflection
INTERACTION_DIFFRACTION = 2 # Diffraction
INTERACTION_SCATTERING = 3 # Scattering
INTERACTION_TRANSMISSION = 4 # Transmission


# Path Processing Constants
MAX_PATHS = 25  # Maximum number of paths per receiver
MAX_INTER_PER_PATH = 10  # Maximum number of interactions per path


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
TXRX_PARAM_NAME = 'txrx'
LOAD_PARAMS_PARAM_NAME = 'load_params'
SCENE_PARAM_NAME = 'scene'  # Scene parameters and configuration
MATERIALS_PARAM_NAME = 'materials'  # Materials list and properties


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
# PARAMSET_OFDM_CH = 'ofdm_channels' # Time Domain / Frequency Domain (OFDM)
# PARAMSET_OTFS_CH = 'otfs_channels'
# If OTFS, OFDM, or other channel models are OFF, generate time-domain channels

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

# Physical Constants
LIGHTSPEED = 299792458  # Speed of light in m/s

# Scenarios folder
SCENARIOS_FOLDER = 'deepmimo_scenarios2'

# Scenario params file variable names - for v3 compatibility
LOAD_FILE_SP_VERSION = 'version'
LOAD_FILE_SP_RAYTRACER = 'raytracer'
LOAD_FILE_SP_RAYTRACER_VERSION = 'raytracer_version'
LOAD_FILE_SP_EXT = '.params.mat'

# Dynamic scenario params - for v3 compatibility
PARAMSET_DYNAMIC_SCENES = 'dynamic_scenario_scenes'
