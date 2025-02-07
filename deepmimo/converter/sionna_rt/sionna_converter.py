"""
Sionna Ray Tracing Converter Module.

This module provides functionality for converting Sionna Ray Tracing output files
into the DeepMIMO format. It handles reading and processing ray tracing data including:
- Path information (angles, delays, powers, interactions, ...)
- TX/RX locations and parameters 
- Scene geometry and materials
"""

import os
import shutil
import numpy as np
from typing import Dict
import pickle

# Internal DeepMIMO imports
from ...txrx import TxRxSet  # For TX/RX set handling
from ... import consts as c  # Constants and configuration parameters
from .. import converter_utils as cu  # Shared converter utilities

# Interaction Type Map for Wireless Insite
INTERACTIONS_MAP = {
    0:  c.INTERACTION_LOS,           # LoS
    1:  c.INTERACTION_REFLECTION,    # Reflection
    2:  c.INTERACTION_DIFFRACTION,   # Diffraction
    3:  c.INTERACTION_SCATTERING,    # Diffuse Scattering
    4:  None,  # Sionna RIS is not supported yet
}

SOURCE_EXTS = ['.pkl']

def save_to_pickle(obj, filename):
    """Saves an object to a pickle file."""
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_from_pickle(filename):
    """Loads an object from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

def sionna_rt_converter(rt_folder: str, scenario_name: str = ''):
    print('converting from sionna RT')

    # Setup scenario name
    scen_name = os.path.basename(rt_folder) 
    if scenario_name:
        scen_name = scenario_name

    # Setup output folder
    output_folder = os.path.join(rt_folder, scen_name + '_deepmimo')
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # Read Setup (ray tracing parameters)
    setup_dict = read_setup(rt_folder)

    # Read TXRX
    txrx_dict = read_txrx(setup_dict)

    # Read Paths (.paths)
    read_paths(rt_folder, output_folder)

    # Read Materials (.materials)
    materials_dict = read_materials(rt_folder, output_folder)

    # Read Scene data
    scene = load_scene(rt_folder)
    scene_dict = scene.export_data(output_folder) if scene else {}
    
    # Save parameters to params.mat
    params = {
        'raytracer_name': c.RAYTRACER_NAME_SIONNA,
        'raytracer_version': c.RAYTRACER_VERSION_SIONNA,
        'setup': setup_dict,
        'txrx': txrx_dict,
        'materials': materials_dict,
        'scene': scene_dict
    }
    cu.save_mat(params, 'params', output_folder)
    
    # Save scenario to deepmimo scenarios folder
    scen_name = cu.save_scenario(output_folder, scen_name=scenario_name)
    
    print(f'Zipping DeepMIMO scenario (ready to upload!): {output_folder}')
    cu.zip_folder(output_folder) # ready for upload
    
    # Copy and zip ray tracing source files as well
    cu.save_rt_source_files(rt_folder, SOURCE_EXTS)
    
    return scen_name


def read_setup(load_folder: str) -> Dict:
    return load_from_pickle(load_folder + 'sionna_rt_params.pkl')

def read_txrx(setup_dict: Dict) -> Dict:
    
    # There will be two txrx sets, one for the sources and one for the targets

    txrx_dict = {}
    # Create TX and RX objects in a loop
    for i in range(2):
        is_tx = (i == 0)  # First iteration is TX, second is RX
        obj = TxRxSet()
        obj.is_tx = is_tx
        obj.is_rx = not is_tx
        
        obj.name = 'tx_array' if is_tx else 'rx_array'
        obj.id_orig = i + 1
        obj.idx = i + 1  # 1-indexed
        
        # Set antenna properties        
        obj.num_ant = 1 if setup_dict['array_synthetic'] else setup_dict[obj.name + '_num_ant']
        obj.ant_rel_positions = setup_dict[obj.name + '_ant_pos']        
        obj.dual_pol = setup_dict[obj.name + '_num_ant'] != setup_dict[obj.name + '_size']

        txrx_dict[f'tx_rx_set_{i+1}'] = obj.to_dict() # 1-indexed

    return txrx_dict

def read_paths(load_folder: str, save_folder: str) -> None:
    path_dict_list = load_from_pickle(load_folder + 'sionna_paths.pkl')

    # Get TX positions (assuming constant for all paths)
    tx_pos = path_dict_list[0]['sources']

    # Get RX number and positions
    rx_pos_all = []
    for paths_dict in path_dict_list:
        rx_pos_all += paths_dict['targets'].tolist()
    rx_pos = np.unique(rx_pos_all, axis=0)
    n_rx = rx_pos.shape[0]

    # Pre-allocate matrices
    max_iteract = min(c.MAX_INTER_PER_PATH, path_dict_list[0]['vertices'].shape[0])

    data = {
        'n_paths': np.zeros((n_rx), dtype=np.float32),
        'rx_pos': rx_pos,
        'tx_pos': tx_pos,
        'aoa_az': np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'aoa_el': np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'aod_az': np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'aod_el': np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'toa':    np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'power':  np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'phase':  np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'inter':  np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'inter_pos': np.zeros((n_rx, c.MAX_PATHS, max_iteract, 3), dtype=np.float32) * np.nan,
    }

    # make squeeze and slice function to be applied to each Sionna array
    ss = lambda array: array.squeeze()[..., :c.MAX_PATHS]

    # Read paths 
    last_idx = 0
    for paths_dict in path_dict_list:
        batch_size = paths_dict['a'].shape[1]
        idxs = last_idx + np.arange(batch_size)
        last_idx = idxs[-1]
        
        a = ss(paths_dict['a'])
        not_nan_mask = a != 0
        data['power'][idxs][not_nan_mask] = 20 * np.log10(np.absolute(a[not_nan_mask]))
        data['phase'][idxs][not_nan_mask] = np.angle(a[not_nan_mask], deg=True)
        data['toa'][idxs][not_nan_mask]   = ss(paths_dict['tau'])

        data['aoa_az'][idxs][not_nan_mask] = ss(paths_dict['phi_r'])   * 180 / np.pi
        data['aoa_el'][idxs][not_nan_mask] = ss(paths_dict['theta_r']) * 180 / np.pi
        data['aod_az'][idxs][not_nan_mask] = ss(paths_dict['phi_t'])   * 180 / np.pi
        data['aod_el'][idxs][not_nan_mask] = ss(paths_dict['theta_t']) * 180 / np.pi

        inter_pos = paths_dict['vertices'].squeeze()[:max_iteract, :, :c.MAX_PATHS, :]
        data['inter_pos'][idxs] = np.transpose(inter_pos, (1,2,0,3))

        data['inter'][idxs][not_nan_mask] = get_sionna_interaction_types(ss(paths_dict['types']),
                                                                         data['inter_pos'][idxs])
                                                                          
        
    
    # Compress data before saving
    data = cu.compress_path_data(data)
    
    # Save each data key
    for key in data.keys():
        cu.save_mat(data[key], key, save_folder, 1, 1, 2)

    return

def get_sionna_interaction_types(types: np.ndarray, inter_pos: np.ndarray) -> np.ndarray:
    """
    Convert Sionna interaction types to DeepMIMO interaction codes.
    
    Similarities between Sionna and DeepMIMO:
        - Sionna uses 0 for LoS. DeepMIMO too.

    Important differences between Sionna and DeepMIMO:
        - Sionna uses 1 to refer to paths with only reflections. The number of reflections in the 
        path, however, can be 1, 2, 3, etc... The way we are able to tell which DeepMIMO code 
        to use (respectively, 1, 11, 111, etc...) is by the size of the corresponding path vertices. 
        - Sionna uses 2 to refer to paths with A SINGLE diffraction. It doesn't allow paths with 
        two diffractions or paths with a diffraction and any other phenomena. So this is simple 
        to equate in DeepMIMO, since we use the single code for diffraction, 2.
        - Sionna uses 3 to refer to paths with a diffuse reflection / scattering event at the end. 
        These paths end with a diffusion, but may have any number of reflections (up to max_depth-1)
        before the diffusion. Therefore, like in the first case, we have to use the size of the vertices
        to determine whether to use DeepMIMO code 3, 13, 113, 1113, 11113, etc. 
        - Sionna uses 4 to refer to RIS. DeepMIMO does not support RIS yet, so we ignore this.
        (technically, DeepMIMO supports RIS, but not yet as a path interaction type)

    Args:
        types: Array of interaction types from Sionna (N_USERS x MAX_PATHS)
        inter_pos: Array of interaction positions (N_USERS x MAX_PATHS x MAX_INTERACTIONS x 3)

    Returns:
        np.ndarray: Array of DeepMIMO interaction codes (N_USERS x MAX_PATHS)
    """
    n_users, max_paths = inter_pos.shape[:2]
    result = np.zeros((n_users, max_paths), dtype=np.float32)
    
    # For each path
    for rx_idx in range(n_users):
        for path_idx in range(max_paths):
            
            # Skip if no type (nan or 0)
            if np.isnan(types[rx_idx, path_idx]) or types[rx_idx, path_idx] == 0:
                continue
                
            sionna_type = int(types[rx_idx, path_idx])
            
            # Handle LoS case (type 0)
            if sionna_type == 0:
                result[rx_idx, path_idx] = c.INTERACTION_LOS
                continue
                
                
            # Count number of actual interactions by checking non-nan positions
            n_interactions = np.nansum(~np.isnan(inter_pos[rx_idx, path_idx, :, 0]))
            if n_interactions == 0:  # Skip if no interactions
                continue
                
            # Handle different Sionna interaction types
            if sionna_type == 1:  # Pure reflection path
                # Create string of '1's with length = number of reflections
                code = '1' * n_interactions
                result[rx_idx, path_idx] = np.float32(code)
                
            elif sionna_type == 2:  # Single diffraction path
                # Always just '2' since Sionna only allows single diffraction
                result[rx_idx, path_idx] = c.INTERACTION_DIFFRACTION
                
            elif sionna_type == 3:  # Scattering path with possible reflections
                # Create string of '1's for reflections + '3' at the end for scattering
                if n_interactions > 1:
                    code = '1' * (n_interactions - 1) + '3'
                else:
                    code = '3'
                result[rx_idx, path_idx] = np.float32(code)
                
            else:
                if sionna_type == 4:
                    raise NotImplementedError('RIS code not supported yet')
                else:
                    raise ValueError(f'Unknown Sionna interaction type: {sionna_type}')
    
    return result

# Unwrap materials
def import_sionna_for_deepmimo(save_folder: str):

    
    saved_vars_names = [
        'sionna_paths.pkl',            # PHASE 2: DONE
        'sionna_materials.pkl',        # PHASE 3: NOW...
        'sionna_material_indices.pkl', # PHASE 3: NOW...
        'sionna_rt_params.pkl',        # PHASE 1: DONE
        'sionna_vertices.pkl',         # PHASE 4: not started
        'sionna_faces.pkl',            # PHASE 4: not started
        'sionna_objects.pkl',          # PHASE 4: not started

    ]
    
    for filename in saved_vars_names:
        variable = load_from_pickle(save_folder + filename)

    return



from ...materials import (
    Material,
    MaterialList
)

def read_materials(load_folder: str, save_folder: str) -> Dict:
    """Read materials from a Sionna RT simulation folder.
    
    Args:
        load_folder: Path to simulation folder containing material files
        save_folder: Path to save converted materials
        
    Returns:
        Dict containing materials and their categorization
    """
    # Load Sionna materials
    material_properties = load_from_pickle(load_folder + 'sionna_materials.pkl')
    material_indices = load_from_pickle(load_folder + 'sionna_material_indices.pkl')

    # Initialize material list
    material_list = MaterialList()
    
    # Attribute matching for scattering models
    scat_model = {
        'none?': Material.SCATTERING_NONE,  # if scattering coeff = 0
        'LambertianPattern': Material.SCATTERING_LAMBERTIAN,
        'DirectivePattern': Material.SCATTERING_DIRECTIVE,
        'BackscatteringPattern': Material.SCATTERING_DIRECTIVE  # directive = backscattering
    }

    # Convert each Sionna material to DeepMIMO Material
    materials = []
    for i, mat_property in enumerate(material_properties):
        # Get scattering model type and handle case where scattering is disabled
        scattering_model = scat_model[type(mat_property.scattering_pattern).__name__]
        scat_coeff = mat_property.scattering_coefficient.numpy()
        scattering_model = Material.SCATTERING_NONE if not scat_coeff else scattering_model
        
        # Create Material object
        material = Material(
            id=i,
            name=f'material_{i}',  # Default name if not provided
            permittivity=mat_property.relative_permittivity.numpy(),
            conductivity=mat_property.conductivity.numpy(),
            scattering_model=scattering_model,
            scattering_coefficient=scat_coeff,
            cross_polarization_coefficient=mat_property.xpd_coefficient.numpy(),
            alpha=mat_property.scattering_pattern.alpha_r,
            beta=mat_property.scattering_pattern.alpha_i,
            lambda_param=mat_property.scattering_pattern.lambda_.numpy()
        )
        materials.append(material)
    
    # Add all materials to buildings category by default
    # This can be modified if Sionna provides material categorization
    material_list.add_materials(materials, CATEGORY_BUILDINGS)
    
    # Get dictionary representation
    materials_dict = material_list.get_materials_dict()
    
    # Save materials to matlab file
    cu.save_mat(material_indices, 'materials', save_folder)
    
    return materials_dict


def load_scene(load_folder):

    pass


if __name__ == '__main__':
    rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/' + \
                'all_runs/run_02-02-2025_15H45M26S/scen_0/DeepMIMO_folder'
    output_folder = os.path.join(rt_folder, 'test_deepmimo')

    setup_dict = read_setup(rt_folder)
    txrx_dict = read_txrx(setup_dict)
    read_paths(rt_folder, output_folder)
    read_materials(rt_folder, output_folder)

