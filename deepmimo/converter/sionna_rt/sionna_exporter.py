"""Sionna Ray Tracing Exporter.

This module provides functionality to export Sionna ray tracing data. 
This is necessary because Sionna (as of v0.19.1) does not provide sufficient built-in
tools for saving ray tracing results to disk.

The module handles exporting Paths and Scene objects from Sionna's ray tracer
into dictionary formats that can be serialized. This allows ray tracing
results to be saved and reused without re-running computationally expensive
ray tracing simulations.

This has been tested with Sionna v0.19.1 and may work with earlier versions.

DeepMIMO does not require sionna to be installed.
To keep it this way AND use this module, you need to import it explicitly:

# Import the module:
from deepmimo.converter.sionna_rt import sionna_exporter

sionna_exporter.export_to_deepmimo(scene, path_list, my_compute_path_params, save_folder)

"""

import os
import numpy as np
from typing import Tuple, List, Dict, Any

from .. import converter_utils as cu

from ...config import config

# Define types at module level
try:
    import sionna.rt
    Paths = sionna.rt.Paths
    Scene = sionna.rt.Scene
except ImportError:
    print("Sionna is not installed. To use sionna_exporter, please install it.")

def to_dict(paths: Paths) -> List[dict]:
    """Exports paths to a filtered dictionary with only selected keys """
    members_names = dir(paths)
    members_objects = [getattr(paths, attr) for attr in members_names]
    data = {attr_name[1:] : attr_obj for (attr_obj, attr_name)
            in zip(members_objects,members_names)
            if not callable(attr_obj) and
                not isinstance(attr_obj, Scene) and
                not attr_name.startswith("__") and
                attr_name.startswith("_")}
    return data

def export_paths(path_list: List[Paths] | Paths) -> List[dict]:
    """Exports paths to a filtered dictionary with only selected keys """
    relevant_keys = ['sources', 'targets', 'a', 'tau', 'phi_r', 'phi_t', 
                     'theta_r', 'theta_t', 'types', 'vertices']
    
    path_list = [path_list] if type(path_list) != list else path_list
    
    paths_dict_list = []
    for path_obj in path_list:
        path_dict = to_dict(path_obj)
        
        # filter unnecessary keys
        dict_filtered = {key: path_dict[key].numpy() for key in relevant_keys}
        
        # add dict to final list
        paths_dict_list += [dict_filtered]
    return paths_dict_list

def scene_to_dict(scene: Scene) -> Dict[str, Any]: 
    """ Export a Sionna Scene to a dictionary, like to Paths.to_dict() """
    members_names = dir(scene)
    members_objects = [getattr(scene, attr) for attr in members_names]
    data = {attr_name[1:] : attr_obj for (attr_obj, attr_name)
            in zip(members_objects, members_names)
            if not callable(attr_obj) and
               not isinstance(attr_obj, sionna.rt.Scene) and
               not attr_name.startswith("__") and
               attr_name.startswith("_")}
    return data

def scene_to_dict2(scene: Scene) -> Dict[str, Any]: 
    """ Export a Sionna Scene to a dictionary, like to Paths.to_dict() """
    members_names = dir(scene)
    bug_attrs =  ['paths_solver']
    members_objects = [getattr(scene, attr) for attr in members_names
                       if attr not in bug_attrs]
    data = {attr_name[1:] : attr_obj for (attr_obj, attr_name)
            in zip(members_objects, members_names)
            if not callable(attr_obj) and
               not isinstance(attr_obj, sionna.rt.Scene) and
               not attr_name.startswith("__")}
    return data

def export_scene_materials(scene: Scene) -> Tuple[List[Dict[str, Any]], List[int]]:
    """ Export the materials in a Sionna Scene to a list of dictionaries """
    
    obj_materials = []
    for _, obj in scene._scene_objects.items():
        obj_materials += [obj.radio_material]
    
    unique_materials = set(obj_materials)
    unique_mat_names = [mat.name for mat in unique_materials]
    
    n_objs = len(scene._scene_objects)
    obj_mat_indices = np.zeros(n_objs, dtype=int)
    for obj_idx, obj_mat in enumerate(obj_materials):
        obj_mat_indices[obj_idx] = unique_mat_names.index(obj_mat.name)
    
    # Do some light processing to add dictionaries to a list in a pickable format
    materials_dict_list = []
    for material in unique_materials:
        materials_dict = {
            'name': material.name,
            'conductivity': material.conductivity.numpy(),
            'relative_permeability': material.relative_permeability.numpy(),
            'relative_permittivity': material.relative_permittivity.numpy(),
            'scattering_coefficient': material.scattering_coefficient.numpy(),
            'scattering_pattern': type(material.scattering_pattern).__name__,
            'alpha_r': material.scattering_pattern.alpha_r,
            'alpha_i': material.scattering_pattern.alpha_i,
            'lambda_': material.scattering_pattern.lambda_.numpy(),
            'xpd_coefficient': material.xpd_coefficient.numpy(),   
        }
        materials_dict_list += [materials_dict]

    return materials_dict_list, obj_mat_indices

def export_scene_rt_params(scene: Scene, **compute_paths_kwargs) -> Dict[str, Any]:
    """ Extract parameters from Scene (and from compute_paths arguments)"""
    
    scene_dict = scene_to_dict(scene)
    rt_params_dict = dict(
        bandwidth=scene_dict['bandwidth'].numpy(),
        frequency=scene_dict['frequency'].numpy(),
        
        rx_array_size=scene_dict['rx_array'].array_size,  # dual-pol if diff than num_ant
        rx_array_num_ant=scene_dict['rx_array'].num_ant,
        rx_array_ant_pos=scene_dict['rx_array'].positions.numpy(),  # relative to ref.
        
        tx_array_size=scene_dict['tx_array'].array_size, 
        tx_array_num_ant=scene_dict['tx_array'].num_ant,
        tx_array_ant_pos=scene_dict['tx_array'].positions.numpy(),
    
        synthetic_array=scene_dict['synthetic_array'],
    
        # custom
        raytracer_version=sionna.__version__,
        doppler_available=0,
    )

    default_compute_paths_params = dict( # with Sionna default values
        max_depth=3, 
        method='fibonacci',
        num_samples=1000000,
        los=True,
        reflection=True,
        diffraction=False,
        scattering=False,
        scat_keep_prob=0.001,
        edge_diffraction=False,
        scat_random_phases=True
    )
    
    # Note 1: Sionna considers only last-bounce diffusion (except in compute_coverage(.), 
    #         but that one doesn't return paths)
    # Note 2: Sionna considers only one diffraction (first-order diffraction), 
    #         though it may occur anywhere in the path
    # Note 3: Sionna does not save compute_path(.) argument values. 
    #         Many of them cannot be derived from the paths and scenes.
    #         For this reason, we ask the user to define a dictionary with the 
    #         parameters we care about and raytrace using that dict.
    #         Alternatively, the user may fill the dictionary after ray tracing with 
    #         the parameters that changed from their default values in Sionna.

    # Update default parameters of compute_path(.) with parameters that changed (in kwargs)
    default_compute_paths_params.update(compute_paths_kwargs)

    return {**rt_params_dict, **default_compute_paths_params}

def export_scene_rt_params2(scene: Scene, **compute_paths_kwargs) -> Dict[str, Any]:
    """ Extract parameters from Scene (and from compute_paths arguments)"""
    
    scene_dict = scene_to_dict2(scene)
    rt_params_dict = dict(
        bandwidth=scene_dict['bandwidth'].numpy(),
        frequency=scene_dict['frequency'].numpy(),
        
        rx_array_size=scene_dict['rx_array'].array_size,  # dual-pol if diff than num_ant
        rx_array_num_ant=scene_dict['rx_array'].num_ant,
        rx_array_ant_pos=scene_dict['rx_array'].positions.numpy(),  # relative to ref.
        
        tx_array_size=scene_dict['tx_array'].array_size, 
        tx_array_num_ant=scene_dict['tx_array'].num_ant,
        tx_array_ant_pos=scene_dict['tx_array'].positions.numpy(),
    
        synthetic_array=scene_dict['synthetic_array'],
    
        # custom
        raytracer_version=sionna.__version__,
        doppler_available=0,
    )

    default_compute_paths_params = dict( # with Sionna default values
        max_depth=3, 
        method='fibonacci',
        num_samples=1000000,
        los=True,
        reflection=True,
        diffraction=False,
        scattering=False,
        scat_keep_prob=0.001,
        edge_diffraction=False,
        scat_random_phases=True
    )
    
    # Note 1: Sionna considers only last-bounce diffusion (except in compute_coverage(.), 
    #         but that one doesn't return paths)
    # Note 2: Sionna considers only one diffraction (first-order diffraction), 
    #         though it may occur anywhere in the path
    # Note 3: Sionna does not save compute_path(.) argument values. 
    #         Many of them cannot be derived from the paths and scenes.
    #         For this reason, we ask the user to define a dictionary with the 
    #         parameters we care about and raytrace using that dict.
    #         Alternatively, the user may fill the dictionary after ray tracing with 
    #         the parameters that changed from their default values in Sionna.

    # Update default parameters of compute_path(.) with parameters that changed (in kwargs)
    default_compute_paths_params.update(compute_paths_kwargs)

    return {**rt_params_dict, **default_compute_paths_params}

def export_scene_buildings(scene: Scene) -> Tuple[np.ndarray, Dict]:
    """ Export the vertices and faces of buildings in a Sionna Scene.
    Output:
        vertice_matrix: n_vertices_in_scene x 3 (xyz coordinates)
        obj_index_map: Dict with object name as key and (start_idx, end_idx) as value
    """
    all_vertices = []
    obj_index_map = {}  # Stores the name and starting index of each object
    
    vertex_offset = 0
    
    for obj_name, obj in scene._scene_objects.items():
    
        # Get vertices
        n_v = obj._mi_shape.vertex_count()
        obj_vertices = np.array(obj._mi_shape.vertex_position(np.arange(n_v)))
        
        # Append vertices to global list
        all_vertices.append(obj_vertices)
    
        # Store object index range
        obj_index_map[obj_name] = (vertex_offset, vertex_offset + n_v)
        
        # Update vertex offset
        vertex_offset += n_v
    
    # Convert lists to numpy arrays
    all_vertices = np.vstack(all_vertices)

    return all_vertices, obj_index_map

def export_to_deepmimo(scene: Scene, path_list: List[Paths] | Paths, 
                       my_compute_path_params: Dict, save_folder: str):
    """ Export a complete Sionna simulation to a format that can be converted by DeepMIMO """
    
    paths_dict_list = export_paths(path_list)
    materials_dict_list, material_indices = export_scene_materials(scene)
    rt_params = export_scene_rt_params(scene, **my_compute_path_params)
    vertice_matrix, obj_index_map = export_scene_buildings(scene)
    
    os.makedirs(save_folder, exist_ok=True)
    
    save_vars_dict = {
        # filename: variable_to_save
        'sionna_paths.pkl': paths_dict_list,
        'sionna_materials.pkl': materials_dict_list,
        'sionna_material_indices.pkl': material_indices,
        'sionna_rt_params.pkl': rt_params,
        'sionna_vertices.pkl': vertice_matrix,
        'sionna_objects.pkl': obj_index_map,
    }
    
    for filename, variable in save_vars_dict.items():
        cu.save_pickle(variable, os.path.join(save_folder, filename))

    return

def export_to_deepmimo_v2(scene: Scene, path_list: List[Paths] | Paths, 
                       my_compute_path_params: Dict, save_folder: str):
    """ Export a complete Sionna simulation to a format that can be converted by DeepMIMO """
    
    paths_dict_list = path_list # export moved to pipeline (needs to be called during RT)
    materials_dict_list, material_indices = {}, [] #export_scene_materials(scene)
    rt_params = export_scene_rt_params2(scene, **my_compute_path_params) # some params are broken
    vertice_matrix, obj_index_map = export_scene_buildings(scene)
    
    os.makedirs(save_folder, exist_ok=True)
    
    save_vars_dict = {
        # filename: variable_to_save
        'sionna_paths.pkl': paths_dict_list,
        'sionna_materials.pkl': materials_dict_list,
        'sionna_material_indices.pkl': material_indices,
        'sionna_rt_params.pkl': rt_params,
        'sionna_vertices.pkl': vertice_matrix,
        'sionna_objects.pkl': obj_index_map,
    }
    
    for filename, variable in save_vars_dict.items():
        cu.save_pickle(variable, os.path.join(save_folder, filename))

    return