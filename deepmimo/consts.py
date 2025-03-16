"""
Constants and configuration parameters for the DeepMIMO dataset generation.

This module contains all constant definitions used throughout the DeepMIMO toolkit,
organized into the following categories:

1. Core Configuration
   - Version information
   - File paths and data types
   - Supported ray tracers

2. Ray-Tracing Parameters
   - Basic parameters (frequency, raytracer info)
   - Interaction limits
   - Diffuse scattering settings
   - Terrain interaction settings
   - Ray casting configuration

3. Scene Parameters
   - Scene structure (objects, faces, vertices)

4. Materials Parameters
   - Material properties

5. TXRX Parameters
   - Transmitter/Receiver configuration

6. Paths Parameters
   - Interaction codes
   - Path processing limits

7. DeepMIMO Matrices
   - Fundamental quantities (power, phase, delay, angles, tx/rx positions, interactions)
   - Derived quantities (channel, number of paths, pathloss, distance)
   - Rotated and filtered quantities (after antenna rotation and FoV filtering)

8. Channel Generation Parameters
   - Basic channel parameters
   - OFDM configuration
   - Antenna configuration

IMPORTANT: The string values of these constants MUST match exactly the field names 
used in the respective files (params.mat, scene files, dataclasses, etc.). 
"""

import numpy as np
from . import __version__

#==============================================================================
# 1. Core Configuration
#==============================================================================

# Version information
VERSION_PARAM_NAME = 'version'
VERSION = __version__

# File and folder paths
SCENARIOS_FOLDER = 'deepmimo_scenarios'  # Folder with extracted scenarios
SCENARIOS_DOWNLOAD_FOLDER = 'deepmimo_scenarios_downloaded'  # Folder with downloaded ZIP files
PARAMS_FILENAME = 'params'

# Data types
FP_TYPE = np.float32  # floating point precision for saving values

# Utility/Configuration parameters
NAME_PARAM_NAME = 'name'
LOAD_PARAMS_PARAM_NAME = 'load_params'

# Supported ray tracers and their versions
RAYTRACER_NAME_WIRELESS_INSITE = 'Remcom Wireless Insite'
RAYTRACER_VERSION_WIRELESS_INSITE = '3.3'
RAYTRACER_NAME_SIONNA = 'Sionna Ray Tracing'
RAYTRACER_VERSION_SIONNA = '0.19.1'
RAYTRACER_NAME_AODT = 'Aerial Omniverse Digital Twin'
RAYTRACER_VERSION_AODT = '1.x'

#==============================================================================
# 2. Ray-Tracing Parameters
#==============================================================================

# Main RT params dictionary key
RT_PARAMS_PARAM_NAME = 'rt_params'

# Basic RT parameters
RT_PARAM_FREQUENCY = 'frequency'
RT_PARAM_RAYTRACER = 'raytracer_name'
RT_PARAM_RAYTRACER_VERSION = 'raytracer_version'

# Interaction limits
RT_PARAM_PATH_DEPTH = 'max_path_depth'
RT_PARAM_MAX_REFLECTIONS = 'max_reflections'
RT_PARAM_MAX_DIFFRACTIONS = 'max_diffractions'
RT_PARAM_MAX_SCATTERING = 'max_scattering'
RT_PARAM_MAX_TRANSMISSIONS = 'max_transmissions'

# Diffuse scattering parameters
RT_PARAM_DIFFUSE_REFLECTIONS = 'diffuse_reflections'
RT_PARAM_DIFFUSE_DIFFRACTIONS = 'diffuse_diffractions'
RT_PARAM_DIFFUSE_TRANSMISSIONS = 'diffuse_transmissions'
RT_PARAM_DIFFUSE_FINAL_ONLY = 'diffuse_final_interaction_only'
RT_PARAM_DIFFUSE_RANDOM_PHASES = 'diffuse_random_phases'

# Terrain interaction parameters
RT_PARAM_TERRAIN_REFLECTION = 'terrain_reflection'
RT_PARAM_TERRAIN_DIFFRACTION = 'terrain_diffraction'
RT_PARAM_TERRAIN_SCATTERING = 'terrain_scattering'

# Ray casting parameters
RT_PARAM_NUM_RAYS = 'num_rays'
RT_PARAM_RAY_CASTING_METHOD = 'ray_casting_method'
RT_PARAM_SYNTHETIC_ARRAY = 'synthetic_array'
RT_PARAM_RAY_CASTING_RANGE_AZ = 'ray_casting_range_az'
RT_PARAM_RAY_CASTING_RANGE_EL = 'ray_casting_range_el'

#==============================================================================
# 3. Scene Parameters
#==============================================================================

# Scene parameters
SCENE_PARAM_NAME = 'scene'  # Scene parameters and configuration
SCENE_PARAM_NUMBER_SCENES = 'num_scenes'
SCENE_PARAM_OBJECTS = 'objects'
SCENE_PARAM_FACES = 'faces'
SCENE_PARAM_N_OBJECTS = 'n_objects'
SCENE_PARAM_N_VERTICES = 'n_vertices'
SCENE_PARAM_N_FACES = 'n_faces'
SCENE_PARAM_N_TRIANGULAR_FACES = 'n_triangular_faces'

#==============================================================================
# 4. Materials Parameters
#==============================================================================

# Materials parameters
MATERIALS_PARAM_NAME = 'materials'  # Materials list and properties
MATERIALS_PARAM_NAME_FIELD = 'name'
MATERIALS_PARAM_PERMITTIVITY = 'permittivity'
MATERIALS_PARAM_CONDUCTIVITY = 'conductivity'
MATERIALS_PARAM_SCATTERING_MODEL = 'scattering_model'
MATERIALS_PARAM_SCATTERING_COEF = 'scattering_coefficient'
MATERIALS_PARAM_CROSS_POL_COEF = 'cross_polarization_coefficient'

#==============================================================================
# 5. TXRX Parameters
#==============================================================================

# TXRX configuration
TXRX_PARAM_NAME = 'txrx_sets'
TXRX_PARAM_NAME_FIELD = 'name'
TXRX_PARAM_IS_TX = 'is_tx'
TXRX_PARAM_IS_RX = 'is_rx'
TXRX_PARAM_NUM_POINTS = 'num_points'
TXRX_PARAM_NUM_ACTIVE_POINTS = 'num_active_points'
TXRX_PARAM_NUM_ANT = 'num_ant'
TXRX_PARAM_DUAL_POL = 'dual_pol'
TXRX_PARAM_ANT_REL_POS = 'ant_rel_pos'
TXRX_PARAM_ANT_ARRAY_ORIENTATION = 'ant_array_orientation'

#==============================================================================
# 6. Paths Parameters
#==============================================================================

# Interaction Codes (read from left to right, starting from transmitter end)
INTERACTION_LOS = 0           # Line-of-sight (direct path)
INTERACTION_REFLECTION = 1    # Reflection
INTERACTION_DIFFRACTION = 2   # Diffraction
INTERACTION_SCATTERING = 3    # Scattering
INTERACTION_TRANSMISSION = 4  # Transmission

# Path Processing Constants
MAX_PATHS = 25  # Maximum number of paths per receiver
MAX_INTER_PER_PATH = 10  # Maximum number of interactions per path

#==============================================================================
# 7. DeepMIMO Matrices
#==============================================================================

# Fundamental quantities (11 matrices)
POWER_PARAM_NAME = 'power'
PHASE_PARAM_NAME = 'phase'
DELAY_PARAM_NAME = 'delay'
AOA_AZ_PARAM_NAME = 'aoa_az'
AOA_EL_PARAM_NAME = 'aoa_el'
AOD_AZ_PARAM_NAME = 'aod_az'
AOD_EL_PARAM_NAME = 'aod_el'
RX_POS_PARAM_NAME = 'rx_pos'
TX_POS_PARAM_NAME = 'tx_pos'
INTERACTIONS_PARAM_NAME = 'inter'
INTERACTIONS_POS_PARAM_NAME = 'inter_pos'

# Computed Parameters
CHANNEL_PARAM_NAME = 'channel'
CH_PARAMS_PARAM_NAME = 'ch_params'  # Channel generation parameters
LOS_PARAM_NAME = 'los'
NUM_PATHS_PARAM_NAME = 'num_paths'
PWR_LINEAR_PARAM_NAME = 'power_linear'
PATHLOSS_PARAM_NAME = 'pathloss'
DIST_PARAM_NAME = 'distance'
INTER_STR_PARAM_NAME = 'inter_str'
N_UE_PARAM_NAME = 'n_ue'
NUM_INTERACTIONS_PARAM_NAME = 'num_interactions'
NUM_PATHS_FOV_PARAM_NAME = '_num_paths_fov'  # Number of paths within FoV for each user

# Rotated angles (after antenna rotation)
AOA_AZ_ROT_PARAM_NAME = '_aoa_az_rot'
AOA_EL_ROT_PARAM_NAME = '_aoa_el_rot'
AOD_AZ_ROT_PARAM_NAME = '_aod_az_rot'
AOD_EL_ROT_PARAM_NAME = '_aod_el_rot'

# Field of view filtered angles
AOD_EL_FOV_PARAM_NAME = '_aod_el_rot_fov'  # Elevation angles after rotation and FoV filtering
AOD_AZ_FOV_PARAM_NAME = '_aod_az_rot_fov'  # Azimuth   angles after rotation and FoV filtering
AOA_EL_FOV_PARAM_NAME = '_aoa_el_rot_fov'  # Elevation angles after rotation and FoV filtering
AOA_AZ_FOV_PARAM_NAME = '_aoa_az_rot_fov'  # Azimuth   angles after rotation and FoV filtering
FOV_MASK_PARAM_NAME = '_fov_mask'          # Boolean mask for FoV filtering

# Power with antenna gain
PWR_LINEAR_ANT_GAIN_PARAM_NAME = '_power_linear_ant_gain'

#==============================================================================
# 8. Channel Generation Parameters
#==============================================================================

# Base channel parameters
PARAMSET_POLAR_EN = 'enable_dual_polar'
PARAMSET_DOPPLER_EN = 'enable_doppler'  # Doppler from Ray Tracer
PARAMSET_FD_CH = 'freq_domain'  # Time Domain / Frequency Domain (OFDM)
PARAMSET_NUM_PATHS = 'num_paths'  # Number of paths to consider for channel generation

# OFDM parameters
PARAMSET_OFDM = 'ofdm'
PARAMSET_OFDM_SC_NUM = 'subcarriers'
PARAMSET_OFDM_SC_SAMP = 'selected_subcarriers'
PARAMSET_OFDM_BANDWIDTH = 'bandwidth'
PARAMSET_OFDM_LPF = 'rx_filter'

# Antenna parameters
PARAMSET_ANT_BS = 'bs_antenna'
PARAMSET_ANT_UE = 'ue_antenna'
PARAMSET_ANT_SHAPE = 'shape'
PARAMSET_ANT_SPACING = 'spacing'
PARAMSET_ANT_ROTATION = 'rotation'
PARAMSET_ANT_RAD_PAT = 'radiation_pattern'
PARAMSET_ANT_RAD_PAT_VALS = ['isotropic', 'halfwave-dipole']

#==============================================================================
# 9. Other Constants
#==============================================================================

# Invalid characters in scenario names
SCENARIO_NAME_INVALID_CHARS = ['/', '\\', ':', '*', '?', '"', "'", '<', '>', '|', 
                               '\n']
