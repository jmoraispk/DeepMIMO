"""

Current approach for extracting data from wireless insite files:
- Read and generate positions from the txrx files
- Adjust them based on the floor height (read from the materials file)
- Override them based on the ones read from the paths file (all users with paths)

Note: this won't work for uneven terrains without diffuse scattering, but 
      that is an unrealistic case since it would cause major problems anyway

"""


import os
# from pprint import pprint # for debugging
import shutil
import numpy as np
import scipy.io

from .. import converter_utils as cu
from ...general_utilities import PrintIfVerbose
from ... import consts as c

from .ChannelDataLoader import WIChannelConverter
from .ChannelDataFormatter import DeepMIMODataFormatter

from typing import List, Dict

from dataclasses import dataclass, asdict

from .setup_parser import tokenize_file, parse_document # for .setup, .txrx, .city, .ter, .veg
from .paths_parser import paths_parser # for paths.p2m


MATERIAL_FILES = ['.city', '.ter', '.veg']
SETUP_FILES = ['.setup', '.txrx'] + MATERIAL_FILES 

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


@dataclass
class InsiteTxRxSet():
    """
    TX/RX set class
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
    

def insite_rt_converter(rt_folder: str, copy_source: bool = False,
                        tx_ids: List[int] = None, rx_ids: List[int] = None,
                        verbose: bool = True, p2m_folder: str = None):

    # Setup output folder
    if p2m_folder:
        insite_sim_folder = os.path.dirname(p2m_folder)
    else:
        insite_sim_folder = rt_folder
        # if p2m folder is not provided, choose the last folder available
        p2m_folder = [name for name in os.listdir(insite_sim_folder)
                      if os.path.isdir(os.path.join(insite_sim_folder, name))][-1]
        p2m_folder = os.path.join(insite_sim_folder, p2m_folder)

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
    
    # OLD WAY:
    old_way = True
    if old_way:
        # Read setup (.setup)
        setup_file = cu.ext_in_list('.setup', files_in_sim_folder)[0]
        setup_dict = read_setup(setup_file, p2m_folder, verbose=False)
    
        # Read TXRX (.txrx)
        txrx_file = cu.ext_in_list('.txrx', files_in_sim_folder)[0]
        avail_tx_idxs, avail_rx_idxs, txrx_dict = read_txrx(txrx_file, verbose)
        
        # tx_ids = tx_ids if tx_ids else avail_tx_idxs
        # rx_ids = rx_ids if rx_ids else avail_rx_idxs
    
        # Read Materials of Buildings, Terrain and Vegetation (.city, .ter, .veg):
        materials_dict = read_materials(files_in_sim_folder, verbose=False)
    
    export_params_dict(output_folder, setup_dict, txrx_dict, materials_dict)
    
    # P2Ms (.cir, .doa, .dod, .paths[.t{tx_id}_{??}.r{rx_id}.p2m] e.g. .t001_01.r001.p2m)

    # # Convert P2M files to mat format
    WIChannelConverter(p2m_folder, intermediate_folder)

    dm = DeepMIMODataFormatter(intermediate_folder, output_folder, 
                               TX_order=tx_ids, RX_order=rx_ids)
    
    scen_name = export_scenario(insite_sim_folder, output_folder, overwrite=False)
    return scen_name

def read_config_file(file):
    return parse_document(tokenize_file(file))


def measure_terrain_height(terrain_file):
    # If .ter exists, measure the terrain height
    ter_document = read_config_file(terrain_file)
    ter_name = ter_document.keys()[0] # Assumes the first terrain spans the whole area
    terrain_height = document[ter_name].values['structure_group'].values['structure']\
        .values['sub_structure'].values['face'].data[0][2]
    return terrain_height 




def verify_sim_folder(sim_folder: str, verbose: bool):
    
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
    print('Copying raytracing source files to "rt_source_folder"')
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

def read_setup(file: str, p2m_folder: str, verbose: bool):
    if verbose:
        print(f'Reading setup file: {os.path.basename(file)}')
    
    # Read the study area that matches the p2m folder name
    study_area_name = os.path.basename(p2m_folder)
    
    setup_dict = { # make sure the keys exist in .setup study areas
        'diffuse_scattering': 0, # ony one not in .setup -> has specific mapping
        'diffuse_reflections': 0,
        'diffuse_diffractions': 0,
        'diffuse_transmissions': 0,
        'final_interaction_only': 0,

        'max_reflections': 0,
        'max_transmissions': 0,
        'max_wedge_diffractions': 1,
        'foliage_attenuation_vert': 1, # 0/1 = OFF/ON attenuation
        'foliage_attenuation_hor': 1,  # 0/1 = OFF/ON attenuation

        'CarrierFrequency': 0, # [hz]
        'bandwidth': 0,        # [Hz]
    } # Tip: make them follow the same order as .setup for max performance

    inside_study_area = False
    line = ''
    with open(file, 'r') as fp:
        while True:
            last_line = line # needed because "enabled" refers to the previous line
            line = fp.readline()
            if line == '': # end of file
                raise Exception(f'Reached end of file - ensure {p2m_folder} matches'
                                f' study area name in {file}')
            
            if line != f'begin_<studyarea> {study_area_name}\n' and not inside_study_area:
                continue
            
            inside_study_area = True
            if line == 'end_<studyarea>\n': break

            line_split = line.split(' ')
            key = line_split[0]
            val = line_split[-1][:-1]
            if key == 'enabled':
                if last_line == 'begin_<diffuse_scattering> \n':
                    if val == 'yes':
                        setup_dict['diffuse_scattering'] = 1

            if key in setup_dict.keys():
                if not setup_dict['diffuse_scattering']:
                    if key in ['diffuse_reflections', 'diffuse_reflections',
                               'diffuse_diffractions', 'diffuse_transmissions',
                               'final_interaction_only']:
                        continue # ignore certain keys if DS is off
                if len(val) > 1:
                    val = 1 if val == 'yes' else 0
                else:
                    val = int(val)
                
                setup_dict[key] = val
        
        # Continue and fill also the subcarrier and bandwidth
        while True:
            line = fp.readline()
            if line == 'end_<Waveform>\n': # picks the first waveform
                break
            line_split = line.split(' ')
            key, val = line_split[0], line_split[-1][:-1]
            if key in ['CarrierFrequency', 'bandwidth']:
                setup_dict[key] = float(val)

    if verbose:
        print(f'RT info extracted from setup: {os.path.basename(file)}')
        pprint(setup_dict)

    return setup_dict


def read_setup_new(setup_file):
    document = read_config_file(tks)
    
    # Select study area (the one that matches the file)
    p2m_folder = os.path.basename(os.path.dirname(file))
    prim = os.path.basename(file)[:-6] # remove .setup
    
    if prim not in document.keys():
        raise Exception("Couldn't find '{basename}' in {os.path.basename(p2m_folder)}")
      
    prim_vals = document[prim].values
    antenna_vals = prim_vals['antenna'].values
    waveform_vals = prim_vals['Waveform'].values
    studyarea_vals = prim_vals['studyarea'].values
    
    setup_dict = {}
    
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
    return setup_dict


def read_txrx(file: str, verbose: bool):
    print(f'Reading txrx file: {os.path.basename(file)}')
    
    txrx_dict = {}

    # How many transmitters (and who they are - uniform index)
    # How many receivers (and who they are - uniform index)

    # For both: 
    # latitude
    # longitude
    # location
    # is_transmitter
    # is_receiver

    # begin_<points> BS

    # begin_<grid> ue_grid
    # side1 410.00000
    # side2 320.00000
    # spacing 1.00000
    
    # power 0.00000

    return [], [], txrx_dict

def read_txrx_new(txrx_file):
    document = read_config_file(txrx_file)
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
    
def gen_rx_tx_grid(n_tx, n_rx, txrx_objs):
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
    return tx_pos, rx_pos

# Note: TxRx SETS != individual Tx/Rx points.
#       Sets are <points> or <grid>. 
#           - <points> contains a single point (currently - LIMITATION)
#           - <grid> typically contains more than a single point
#       The indices of the sets are only useful for reading data into the right arrays
#       The indices of the individual points will be used for data storage and generation



def read_feature_files(files: List[str], verbose: bool):
    if verbose:
        print(f'Reading materials in {[os.path.basename(f) for f in files]}')
    
    # Extract materials for each file
    material_list = []
    for file in files:
        material_list += read_feat_file(file, verbose)

    # Filter the list of materials so they are unique
    unique_mat_list = make_mat_list_unique(material_list)
    
    return unique_mat_list


def read_feat_file(file: str, verbose: bool):
    
    mat_list = []
    inside_material = False
    with open(file, 'r') as fp:
        while True:
            line = fp.readline()
            if line == 'begin_<structure_group> \n': # end of materials
                break
            
            if not line.startswith('begin_<Material>'):
                if not inside_material:
                    continue
            else:
                inside_material = True
                curr_mat_dict = {}
                curr_mat_dict['name'] = ' '.join(line.split(' ')[1:])[:-1]
                continue

            if line == 'end_<Material>\n': 
                inside_material = False
                mat_list += [InsiteMaterial(curr_mat_dict)]
                continue

            line_split = line.split(' ')
            if len(line_split) != 2: continue

            key, val = line_split 
            val = val[:-1] # remove '\n'
            
            if val: curr_mat_dict[key] = convert_val_to_materials_dict(val)

    return mat_list

def read_feat_file_new(file: str, verbose: bool):
    
    direct_fields = ['diffuse_scattering_model', 'fields_diffusively_scattered', 
                     'cross_polarized_power', 'directive_alpha',
                     'directive_beta', 'directive_lambda']
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

    return mat_objs


def make_mat_list_unique(mat_list):
    
    n_mats = len(mat_list)
    idxs_to_discard = []
    for i1 in range(n_mats):
        for i2 in range(n_mats):
            if i1 == i2:
                continue
            if mat_list[i1].get_dict() == mat_list[i2].get_dict():
                idxs_to_discard.append(i2)

    for idx in sorted(np.unique(idxs_to_discard), reverse=True):
        del mat_list[idx]
    
    return mat_list

def convert_val_to_materials_dict(value):
    try:
        value = float(value) if '.' in value else int(value)
    except ValueError:
        # Keep as string if it can't be converted to int/float
        pass
    return value

def read_materials(files_in_sim_folder, verbose):

    city_files = cu.ext_in_list('.city', files_in_sim_folder)
    ter_files  = cu.ext_in_list('.ter', files_in_sim_folder)
    veg_files  = cu.ext_in_list('.veg', files_in_sim_folder)
    fpl_files  = cu.ext_in_list('.flp', files_in_sim_folder)
    obj_files  = cu.ext_in_list('.obj', files_in_sim_folder)
    
    city_materials = read_feature_files(city_files, verbose)
    ter_materials = read_feature_files(ter_files, verbose)
    veg_materials = read_feature_files(veg_files, verbose)
    floor_plan_materials = read_feature_files(fpl_files, verbose)
    obj_materials = read_feature_files(obj_files, verbose)

    materials_dict = {'city': city_materials, 
                      'terrain': ter_materials,
                      'vegetation': veg_materials,
                      'other': floor_plan_materials + obj_materials}
    if verbose:
        pprint(materials_dict)
    return materials_dict

def export_params_dict(output_folder: str, setup_dict: Dict, txrx_dict: Dict, 
                       mat_dict: Dict):
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

def export_scenario(sim_folder, output_folder, overwrite=False):
    name = os.path.basename(sim_folder)
    scen_path = c.SCENARIOS_FOLDER + f'/{name}'
    if os.path.exists(scen_path) and not overwrite:
        
        print(f'Scenario with name "{name}" already exists in {c.SCENARIOS_FOLDER}. Delete? (Y/n)')
        ans = input()
        if ('n' in ans.lower()):
            return
        else:
            shutil.rmtree(scen_path)
    
    shutil.move(output_folder, scen_path)

    return name