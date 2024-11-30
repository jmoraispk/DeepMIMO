#%%

# The directives below are for ipykernel to auto reload updated modules
# %reload_ext autoreload
# %autoreload 2

import deepmimo as dm

#%%

# path_to_p2m_folder = r'.\P2Ms\ASU_campus_just_p2m\study_area_asu5'
# path_to_p2m_folder = r'.\P2Ms\simple_street_canyon\study_rays=0.25_res=2m_3ghz'
path_to_p2m_folder = r'.\P2Ms\simple_street_canyon'

scen_name = dm.create_scenario(path_to_p2m_folder,
                               copy_source=False, tx_ids=[1], rx_ids=[2])

# TODO: map p2m dict to universal dict to write to params (almost the same)

#%%
scen_name = 'simple_street_canyon'
params = dm.Parameters(scen_name)#asu_campus')

dataset = dm.generate(params)

#%% READ Setup
import os
from deepmimo.converter.wireless_insite.parser import tokenize_file, parse_document
file = "P2Ms/simple_street_canyon/simple_street_canyon_test.setup"
tks = tokenize_file(file)

document = parse_document(tks)
setup_dict = {}

# Select study area (the one that matches the file)
p2m_folder = os.path.basename(os.path.dirname(file))
prim = os.path.basename(file)[:-6] # remove .setup

if prim not in document.keys():
    raise Exception("Couldn't find '{basename}' in {os.path.basename(p2m_folder)}")
  
prim_vals = document[prim].values
antenna_vals = prim_vals['antenna'].values
waveform_vals = prim_vals['Waveform'].values
studyarea_vals = prim_vals['studyarea'].values

# Antenna Settings
setup_dict['antenna'] = antenna_vals['type']
setup_dict['polarization'] = antenna_vals['polarization']
setup_dict['power_threshold'] = antenna_vals['power_threshold']

# Waveform Settings
setup_dict['frequency'] = waveform_vals['CarrierFrequency']
setup_dict['bandwidth'] = waveform_vals['bandwidth']

# Study Area Settings
# Model Settings
model_vals = studyarea_vals['model'].values
setup_dict['initial_ray_mode']         = model_vals['initial_ray_mode']
setup_dict['foliage_model']            = model_vals['foliage_model']
setup_dict['foliage_attenuation_vert'] = model_vals['foliage_attenuation_vert']
setup_dict['foliage_attenuation_hor']  = model_vals['foliage_attenuation_hor']
setup_dict['terrain_diffractions']     = model_vals['terrain_diffractions']
setup_dict['ray_spacing']              = model_vals['ray_spacing']
setup_dict['max_reflections']          = model_vals['max_reflections']
setup_dict['initial_ray_mode']         = model_vals['initial_ray_mode']

# Verify that the required outputs were generated
output_vals = model_vals['OutputRequests'].values
necessary_output_files_exist = True
necessary_outputs = ['ComplexImpulseResponse', 'DirectionOfArrival', 
                     'DirectionOfDeparture', 'Paths']
for output in necessary_outputs:
    if not output_vals[output]:
        print(f'One of the NECESSARY outputs is missing. Output missing: {output}')
        necessary_output_files_exist = False
        
if not necessary_output_files_exist:
    raise Exception('At list one of the necessary output files was not generated.'
                    'Please rerun the simulation to enable the output of CIR, DoA, DoD and Paths.')

# APG settings
apg_accel_vals = studyarea_vals['apg_acceleration'].values
setup_dict['apg_acceleration']   = apg_accel_vals['enabled']
setup_dict['workflow_mode']      = apg_accel_vals['workflow_mode']
# setup_dict['binary_output_mode'] = apg_accel_vals['binary_output_mode']
# setup_dict['binary_rate']        = apg_accel_vals['binary_rate']
# setup_dict['database_mode']      = apg_accel_vals['database_mode']
setup_dict['path_depth']         = apg_accel_vals['path_depth']
setup_dict['adjacency_distance'] = apg_accel_vals['adjacency_distance']

# Diffuse scattering settings
diffuse_scat_vals = studyarea_vals['diffuse_scattering'].values
setup_dict['diffuse_scattering']     = diffuse_scat_vals['enabled']
setup_dict['diffuse_reflections']    = diffuse_scat_vals['diffuse_reflections']
setup_dict['diffuse_diffractions']   = diffuse_scat_vals['diffuse_diffractions']
setup_dict['diffuse_transmissions']  = diffuse_scat_vals['diffuse_transmissions']
setup_dict['final_interaction_only'] = diffuse_scat_vals['final_interaction_only']

# Boundary settings
boundary_vals = studyarea_vals['boundary'].values
setup_dict['boundary_zmin'] = studyarea_vals['boundary']['zmin']
setup_dict['boundary_zmax'] = studyarea_vals['boundary']['zmax']
setup_dict['boundary_xmin'] = studyarea_vals['boundary'].data[0][0]
setup_dict['boundary_xmax'] = studyarea_vals['boundary'].data[2][0]
setup_dict['boundary_ymin'] = studyarea_vals['boundary'].data[0][1]
setup_dict['boundary_ymax'] = studyarea_vals['boundary'].data[2][1]

# feature_vals = prim_vals['feature'].values # LOADED in another function
# txrx_vals =  prim_vals['txrx_sets'].values # LOADED in another function

#%% READ TXRX

import numpy as np
from deepmimo.converter.wireless_insite.parser import tokenize_file, parse_document
from dataclasses import dataclass, asdict

@dataclass
class InsiteTxRxSet():
    """
    TX or RX
    """
    # These field names match the names in Material section of the feature file
    name: str = ''
    kind: str = '' # type = 'points' or 'grid'
    p_id: int = 0
    is_tx: bool = False
    is_rx: bool = False
    loc_lat: float = 0.0
    loc_lon: float = 0.0
    loc_xy: np.ndarray | None = None # N_points x 3
    side_x: float  = 0.0 # [m]
    side_y: float  = 0.0 # [m]
    spacing: float = 0.0 # [m]
    
    # Indices of individual TXs and RXs
    tx_id_start: int | None = None
    rx_id_start: int | None = None
    tx_id_end: int | None = None
    rx_id_end: int | None = None
    
file = "P2Ms/simple_street_canyon/simple_street_canyon_test.txrx"
tks = tokenize_file(file)
document = parse_document(tks)

n_tx = 0
n_rx = 0
txrx_objs = []
for key in document.keys():
    txrx = document[key]
    txrx_obj = InsiteTxRxSet()
    txrx_obj.name = key
    txrx_obj.kind = txrx.kind
    txrx_obj.p_id = (int(txrx.name[-1]) if txrx.name.startswith('project_id')
                     else txrx.values['project_id'])
    txrx_obj.loc_lat = txrx.values['location'].values['reference'].values['latitude']
    txrx_obj.loc_lon = txrx.values['location'].values['reference'].values['longitude']
    txrx_obj.coord_ref = txrx.values['location'].values['reference'].labels[1] # 'terrain'?
    # If ref = terrain, then z
    
    if txrx_obj.kind == 'points':
        txrx_obj.loc_xy = np.array(txrx.values['location'].data[0]).reshape((1,3))
    if txrx_obj.kind == 'grid':
        corner = txrx.values['location'].data[0] # lower left
        txrx_obj.side_x = txrx.values['location'].values['side1']
        txrx_obj.side_y = txrx.values['location'].values['side2']
        txrx_obj.spacing = txrx.values['location'].values['spacing']
        
        # Generate grid points : [-90, -60, 1.5], [-88, -60, 1.5], ...
        xs = corner[0] + np.arange(0, txrx_obj.side_x + 1e-9, txrx_obj.spacing)
        ys = corner[1] + np.arange(0, txrx_obj.side_y + 1e-9, txrx_obj.spacing)
        points = np.array([[x,y, corner[2]] for y in ys for x in xs], dtype=np.float32)
        txrx_obj.loc_xy = points
    
    txrx_obj.is_tx = txrx.values['is_transmitter']
    txrx_obj.is_rx = txrx.values['is_receiver']
    
    # Update indices of individual tx/rx points
    num_txrx = len(txrx_obj.loc_xy)#.shape[0]
    if txrx_obj.is_tx:
        txrx_obj.tx_id_start = n_tx
        txrx_obj.tx_id_end = n_tx + num_txrx - 1
        n_tx += num_txrx
    if txrx_obj.is_rx:
        txrx_obj.rx_id_start = n_rx
        txrx_obj.rx_id_end = n_rx + num_txrx - 1
        n_rx += num_txrx
        
    txrx_objs.append(txrx_obj)
    
    
#%%
# Generate full grid of individual RXs and TXs
tx_pos = np.zeros((n_tx, 3), dtype=np.float32) * np.nan
rx_pos = np.zeros((n_rx, 3), dtype=np.float32) * np.nan

for txrx_obj in txrx_objs:
    if txrx_obj.is_tx:
        idxs = np.arange(txrx_obj.tx_id_start, txrx_obj.tx_id_end+1)
        tx_pos[idxs] = txrx_obj.loc_xy
    if txrx_obj.is_rx:
        idxs = np.arange(txrx_obj.rx_id_start, txrx_obj.rx_id_end+1)
        rx_pos[idxs] = txrx_obj.loc_xy

# Note: TxRx SETS != individual Tx/Rx points.
#       Sets are <points> or <grid>. 
#           - <points> contains a single point (currently - LIMITATION)
#           - <grid> typically contains more than a single point
#       The indices of the sets are only useful for reading data into the right arrays
#       The indices of the individual points will be used for data storage and generation

#%% READ Materials

# file = "P2Ms/simple_street_canyon/simple_street_canyon_floor.ter"
file = './P2Ms/ASU_campus_just_p2m/asu_plane.ter'
# file = "P2Ms/simple_street_canyon/simple_street_canyon_buildings.city"
tks = tokenize_file(file)
document = parse_document(tks)

import os
from dataclasses import dataclass, asdict

@dataclass
class InsiteMaterial():
    """
    Materials in Wireless InSite.
    
    Notes:
    - Diffuse model implemented from [1] + extended with cross-polarization scattering terms
    - Diffuse scattering models explained in [2], slides 29-31. 
    
    - At present, all MATERIALS in Wireless InSite are nonmagnetic, 
      and the permeability for all materials is that of free space 
      (µ0 = 4π x 10e-7 H/m) [3]. 

    Sources:
        [1] A Diffuse Scattering Model for Urban Propagation Prediction - Vittorio Degli-Esposti 2001
            https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=933491
        [2] https://x.webdo.cc/userfiles/Qiwell/files/Remcom_Wireless%20InSite_5G_final.pdf
        [3] Wireless InSite 3.3.0 Reference Manual, section 10.5 - Dielectric Parameters
    """
    # These field names match the names in Material section of the feature file
    name: str = ''
    diffuse_scattering_model: str = ''    # 'labertian', 'directive', 'directive_w_backscatter'
    fields_diffusively_scattered: int = 0 # 0-1, fraction of incident fields that are scattered
    cross_polarized_power: str = 0        # 0-1, fraction of the scattered field that is cross pol
    directive_alpha: int = 0     # 1-10, defines how broad forward beam is
    directive_beta: int = 0      # 1-10, defines how broad backscatter beam is
    directive_lambda: int = 0    # 0-1, fraction of the scattered power in forward direction (vs back)
    conductivity: int = 0        # >=0, conductivity
    permittivity: int = 0        # >=0, permittivity
    roughness: int = 0           # >=0, roughness
    thickness: int = 0           # >=0, thickness [m]
    
direct_fields = ['diffuse_scattering_model', 'fields_diffusively_scattered', 
                 'cross_polarized_power', 'directive_alpha', 'directive_beta',
                 'directive_lambda']
dielectric_fields = ['conductivity', 'permittivity', 'roughness', 'thickness']

mat_objs = []
for prim in document.keys():
    materials = document[prim].values['Material']
    materials = [materials] if type(materials) != list else materials
    for mat in materials:
        material_obj = InsiteMaterial()
        material_obj.name = mat.name
        for field in direct_fields:
            setattr(material_obj, field, mat.values[field])
        
        for field in dielectric_fields:
            setattr(material_obj, field, mat.values['DielectricLayer'].values[field])
        
        mat_objs += [material_obj]


# If .ter exists, measure the terrain height
terrain_height = document['asu_plane'].values['structure_group'].values['structure']\
    .values['sub_structure'].values['face'].data[0][2]

# Current approach:
# - Read and generate positions from the txrx files
# - Adjust them based on the floor height (read from the materials file)
# - Override them based on the ones read from the paths file (all users with paths)
# Note: this won't work for uneven terrains without diffuse scattering, but 
#       that is an unrealistic case since it would cause major problems anyway






