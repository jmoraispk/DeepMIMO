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
from typing import Dict, Optional, Tuple, List
import pickle
from pprint import pprint
from tqdm import tqdm

# Internal DeepMIMO imports
from ...txrx import TxRxSet  # For TX/RX set handling
from ... import consts as c  # Constants and configuration parameters
from .. import converter_utils as cu  # Shared converter utilities
from ...materials import (
    Material,
    MaterialList
)
from ...scene import (
    PhysicalElement, 
    Face, 
    Scene,
    CAT_BUILDINGS,
    CAT_TERRAIN,
    CAT_VEGETATION,
    CAT_FLOORPLANS,
    CAT_OBJECTS,
    get_object_faces
)

# saved_vars_names = [
#     'sionna_paths.pkl',            
#     'sionna_materials.pkl',        
#     'sionna_material_indices.pkl', 
#     'sionna_rt_params.pkl',       
#     'sionna_vertices.pkl',        
#     'sionna_faces.pkl',           
#     'sionna_objects.pkl',         
# ]


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

def sionna_rt_converter(rt_folder: str, copy_source: bool = False,
                        overwrite: Optional[bool] = None, vis_scene: bool = False, 
                        scenario_name: str = '') -> str:
    """Convert Sionna ray-tracing data to DeepMIMO format.

    This function handles the conversion of Sionna ray-tracing simulation 
    data into the DeepMIMO dataset format. It processes path data, setup files,
    and transmitter/receiver configurations to generate channel matrices and metadata.

    Args:
        rt_folder (str): Path to folder containing Sionna ray-tracing data.
        copy_source (bool): Whether to copy ray-tracing source files to output.
        overwrite (Optional[bool]): Whether to overwrite existing files. Prompts if None. Defaults to None.
        vis_scene (bool): Whether to visualize the scene layout. Defaults to False.
        scenario_name (str): Custom name for output folder. Uses rt folder name if empty.

    Returns:
        str: Path to output folder containing converted DeepMIMO dataset.
        
    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If transmitter or receiver IDs are invalid.
    """
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
    materials_dict, material_indices = read_materials(rt_folder, output_folder)

    # Read Scene data
    scene = load_scene(rt_folder, material_indices)
    scene_dict = scene.export_data(output_folder) if scene else {}
    
    # Visualize if requested
    if vis_scene and scene:
        scene.plot()
    
    # Save parameters to params.mat
    params = {
        c.LOAD_FILE_SP_VERSION: c.VERSION,
        c.PARAMSET_DYNAMIC_SCENES: 0, # only static currently
        c.LOAD_FILE_SP_RAYTRACER: c.RAYTRACER_NAME_SIONNA,
        c.LOAD_FILE_SP_RAYTRACER_VERSION: c.RAYTRACER_VERSION_SIONNA,
        c.RT_PARAMS_PARAM_NAME: setup_dict,
        c.TXRX_PARAM_NAME: txrx_dict,
        c.MATERIALS_PARAM_NAME: materials_dict,
        c.SCENE_PARAM_NAME: scene_dict
    }
    cu.save_mat(params, 'params', output_folder)
    pprint(params)

    # Save scenario to deepmimo scenarios folder
    scen_name = cu.save_scenario(output_folder, scen_name=scenario_name, overwrite=overwrite)
    
    print(f'Zipping DeepMIMO scenario (ready to upload!): {output_folder}')
    cu.zip_folder(output_folder) # ready for upload
    
    # Copy and zip ray tracing source files as well
    if copy_source:
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
        'delay':  np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'power':  np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'phase':  np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'inter':  np.zeros((n_rx, c.MAX_PATHS), dtype=np.float32) * np.nan,
        'inter_pos': np.zeros((n_rx, c.MAX_PATHS, max_iteract, 3), dtype=np.float32) * np.nan,
    }

    # make squeeze and slice function to be applied to each Sionna array
    ss = lambda array: array.squeeze()[..., :c.MAX_PATHS]

    # Calculate total number of users for progress bar
    total_users = sum(paths_dict['a'].shape[1] for paths_dict in path_dict_list)
    
    # Create progress bar
    pbar = tqdm(total=total_users, desc="Processing users")
    
    last_idx = 0
    for paths_dict in path_dict_list:
        batch_size = paths_dict['a'].shape[1]
        idxs = last_idx + np.arange(batch_size)
        last_idx = idxs[-1]
        
        a = ss(paths_dict['a'])
        not_nan_mask = a != 0
        
        # Process each field with proper masking
        for rx_idx, path_mask in enumerate(not_nan_mask):
            if rx_idx >= len(idxs):
                break
                
            abs_idx = idxs[rx_idx]

            n_paths = np.sum(path_mask)

            data['power'][abs_idx,:n_paths] = 20 * np.log10(np.absolute(a[rx_idx][path_mask]))
            data['phase'][abs_idx,:n_paths] = np.angle(a[rx_idx][path_mask], deg=True)
            data['delay'][abs_idx,:n_paths] = ss(paths_dict['tau'])[rx_idx][path_mask]
            data['aoa_az'][abs_idx,:n_paths] = ss(paths_dict['phi_r'])[rx_idx][path_mask] * 180 / np.pi
            data['aoa_el'][abs_idx,:n_paths] = ss(paths_dict['theta_r'])[rx_idx][path_mask] * 180 / np.pi
            data['aod_az'][abs_idx,:n_paths] = ss(paths_dict['phi_t'])[rx_idx][path_mask] * 180 / np.pi
            data['aod_el'][abs_idx,:n_paths] = ss(paths_dict['theta_t'])[rx_idx][path_mask] * 180 / np.pi

            # Handle interactions for this receiver
            types = ss(paths_dict['types'])[path_mask]

            # Get number of paths from the mask
            if n_paths > 0:
                # Ensure types has the right shape
                if len(types) < n_paths:
                    print('types has less paths than n_paths')
                    types = np.repeat(types, n_paths)
                types = types[:n_paths]
                
                inter_pos_rx = data['inter_pos'][abs_idx, :n_paths]
                interactions = get_sionna_interaction_types(types, inter_pos_rx)
                data['inter'][abs_idx, :n_paths] = interactions

                
            # Update progress bar for each user processed
            pbar.update(1)

        # Handle interaction positions
        inter_pos = paths_dict['vertices'].squeeze()[:max_iteract, :, :c.MAX_PATHS, :]
        data['inter_pos'][idxs, :len(path_mask), :inter_pos.shape[0]] = np.transpose(inter_pos, (1,2,0,3))

    
    pbar.close()

    # Compress data before saving
    data = cu.compress_path_data(data)
    
    # Save each data key
    for key in data.keys():
        cu.save_mat(data[key], key, save_folder, 1, 1, 2)

    return

def get_sionna_interaction_types(types: np.ndarray, inter_pos: np.ndarray) -> np.ndarray:
    """
    Convert Sionna interaction types to DeepMIMO interaction codes.
    
    Args:
        types: Array of interaction types from Sionna (N_PATHS,)
        inter_pos: Array of interaction positions (N_PATHS x MAX_INTERACTIONS x 3)

    Returns:
        np.ndarray: Array of DeepMIMO interaction codes (N_PATHS,)
    """
    # Ensure types is a numpy array
    types = np.asarray(types)
    if types.ndim == 0:
        types = np.array([types])
    
    # Get number of paths
    n_paths = len(types)
    result = np.zeros(n_paths, dtype=np.float32)
    
    # For each path
    for path_idx in range(n_paths):
        # Skip if no type (nan or 0)
        if np.isnan(types[path_idx]) or types[path_idx] == 0:
            continue
            
        sionna_type = int(types[path_idx])
        
        # Handle LoS case (type 0)
        if sionna_type == 0:
            result[path_idx] = c.INTERACTION_LOS
            continue
            
        # Count number of actual interactions by checking non-nan positions
        if inter_pos.ndim == 2:  # Single path case
            n_interactions = np.nansum(~np.isnan(inter_pos[:, 0]))
        else:  # Multiple paths case
            n_interactions = np.nansum(~np.isnan(inter_pos[path_idx, :, 0]))
            
        if n_interactions == 0:  # Skip if no interactions
            continue
            
        # Handle different Sionna interaction types
        if sionna_type == 1:  # Pure reflection path
            # Create string of '1's with length = number of reflections
            code = '1' * n_interactions
            result[path_idx] = np.float32(code)
            
        elif sionna_type == 2:  # Single diffraction path
            # Always just '2' since Sionna only allows single diffraction
            result[path_idx] = c.INTERACTION_DIFFRACTION
            
        elif sionna_type == 3:  # Scattering path with possible reflections
            # Create string of '1's for reflections + '3' at the end for scattering
            if n_interactions > 1:
                code = '1' * (n_interactions - 1) + '3'
            else:
                code = '3'
            result[path_idx] = np.float32(code)
            
        else:
            if sionna_type == 4:
                raise NotImplementedError('RIS code not supported yet')
            else:
                raise ValueError(f'Unknown Sionna interaction type: {sionna_type}')
    
    return result

def read_materials(load_folder: str, save_folder: str) -> Tuple[Dict, Dict[str, int]]:
    """Read materials from a Sionna RT simulation folder.
    
    Args:
        load_folder: Path to simulation folder containing material files
        save_folder: Path to save converted materials
        
    Returns:
        Tuple of (Dict containing materials and their categorization,
                 Dict mapping object names to material indices)
    """
    # Load Sionna materials
    material_properties = load_from_pickle(load_folder + 'sionna_materials.pkl')
    material_indices = load_from_pickle(load_folder + 'sionna_material_indices.pkl')

    # Initialize material list
    material_list = MaterialList()
    
    # Attribute matching for scattering models
    scat_model = {
        'LambertianPattern': Material.SCATTERING_LAMBERTIAN,
        'DirectivePattern': Material.SCATTERING_DIRECTIVE,
        'BackscatteringPattern': Material.SCATTERING_DIRECTIVE  # directive = backscattering
    }

    # Convert each Sionna material to DeepMIMO Material
    materials = []
    for i, mat_property in enumerate(material_properties):
        # Get scattering model type and handle case where scattering is disabled
        scattering_model = scat_model[mat_property['scattering_pattern']]
        scat_coeff = mat_property['scattering_coefficient']
        scattering_model = Material.SCATTERING_NONE if not scat_coeff else scattering_model
        
        # Create Material object
        material = Material(
            id=i,
            name=f'material_{i}',  # Default name if not provided
            permittivity=mat_property['relative_permittivity'],
            conductivity=mat_property['conductivity'],
            scattering_model=scattering_model,
            scattering_coefficient=scat_coeff,
            cross_polarization_coefficient=mat_property['xpd_coefficient'],
            alpha_r=mat_property['alpha_r'],
            alpha_i=mat_property['alpha_i'],
            lambda_param=mat_property['lambda_']
        )
        materials.append(material)
    
    # Add all materials to buildings category by default
    # This can be modified if Sionna provides material categorization
    material_list.add_materials(materials)
    
    # Save materials indices to matrix file
    cu.save_mat(material_indices, 'materials', save_folder)
    
    return material_list.to_dict(), material_indices

def load_scene(load_folder: str, material_indices: List[int]) -> Scene:
    """Load scene data from Sionna format.
    
    This function converts Sionna's triangular mesh representation into DeepMIMO's
    scene format. While we receive the scene as triangular faces, we store it using
    convex hull faces for efficiency. The Face class in DeepMIMO can handle both
    representations:
    1. Convex hull faces (more efficient for storage and most operations)
    2. Triangular faces (available when needed for detailed visualization)
    
    Args:
        load_folder: Path to folder containing Sionna scene files
        material_indices: List of material indices, one per object
        
    Returns:
        Scene: Loaded scene with all objects
    """
    # Load raw data - already in correct format
    vertices = load_from_pickle(load_folder + 'sionna_vertices.pkl') # (N_VERTICES, 3)
    tri_faces = load_from_pickle(load_folder + 'sionna_faces.pkl').astype(np.int32) # (N_FACES, 3)  
    objects = load_from_pickle(load_folder + 'sionna_objects.pkl') # Dict with vertex ranges
    
    print("\nInitial data shapes and types:")
    print("Vertices shape:", vertices.shape)
    print("Tri faces shape:", tri_faces.shape)
    print("Objects structure:", objects)
    
    # Create scene
    scene = Scene()
    
    # Process each object
    for id_counter, (name, vertex_range) in enumerate(objects.items()):
        print(f"\nProcessing object {id_counter}: {name}")
        try:
            # Get vertex range for this object
            start_idx, end_idx = vertex_range
            print(f"Vertex range: {start_idx} to {end_idx}")
            
            obj_name = name[5:] if name.startswith('mesh-') else name
            
            # Attribute the correct label to the object
            is_floor = obj_name.lower() in ['plane', 'floor']
            obj_label = CAT_TERRAIN if is_floor else CAT_BUILDINGS
            print(f"Processing object: {obj_name}, label: {obj_label}")
            
            # Get material index for this object
            material_idx = material_indices[id_counter]
            
            # Get vertices for this object
            object_vertices = []
            for i in range(start_idx, end_idx):
                vertex = vertices[i]
                vertex_tuple = (float(vertex[0]), float(vertex[1]), float(vertex[2]))
                object_vertices.append(vertex_tuple)
            
            print(f"Object has {len(object_vertices)} vertices")
            
            # Generate faces using convex hull approach
            # Note: While we store faces as convex hulls for efficiency,
            # the Face class maintains the ability to generate triangular faces
            # when needed (e.g., for detailed visualization)
            generated_faces = get_object_faces(object_vertices)
            print(f"Generated {len(generated_faces)} faces using convex hull")
            
            # Create Face objects with material indices
            object_faces = []
            for face_vertices in generated_faces:
                face = Face(
                    vertices=face_vertices,
                    material_idx=material_idx
                )
                object_faces.append(face)
            
            # Create object
            obj = PhysicalElement(
                faces=object_faces,
                object_id=id_counter,
                label=obj_label
            )
            scene.add_object(obj)
            
        except Exception as e:
            print(f"Error processing object {name}: {str(e)}")
            raise

    return scene


if __name__ == '__main__':
    rt_folder = 'C:/Users/jmora/Documents/GitHub/AutoRayTracing/' + \
                'all_runs/run_02-02-2025_15H45M26S/scen_0/DeepMIMO_folder'
    output_folder = os.path.join(rt_folder, 'test_deepmimo')

    setup_dict = read_setup(rt_folder)
    txrx_dict = read_txrx(setup_dict)
    read_paths(rt_folder, output_folder)
    read_materials(rt_folder, output_folder)

