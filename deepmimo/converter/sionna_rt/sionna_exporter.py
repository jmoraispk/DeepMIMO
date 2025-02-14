"""Sionna Ray Tracing Exporter.

This module provides functionality to export Sionna ray tracing data. 
This is necessary because Sionna (as of v0.19.1) does not provide sufficient built-in
tools for saving ray tracing results to disk.

The module handles exporting Paths and Scene objects from Sionna's ray tracer
into dictionary formats that can be serialized. This allows ray tracing
results to be saved and reused without re-running computationally expensive
ray tracing simulations.

This has been tested with Sionna v0.19.1 and may work with earlier versions.

Note:
- DeepMIMO does not require sionna to be installed. To keep it this way AND use
  this module, you need to import it explicitly:

  # Import the module:
  from deepmimo.converter.sionna_rt import sionna_exporter

  # Usage:
  sionna_exporter.export_to_deepmimo(scene, path_list, 
                                     my_compute_path_params, save_folder)

"""

import os
import numpy as np
from typing import Tuple, List, Dict, Any
import pickle

# Only export the pickle functions by default (robust to "from deepmimo import *")
__all__ = ['save_to_pickle', 'load_from_pickle']

# Define types at module level
try:
    import sionna
    Paths = sionna.rt.Paths
    Scene = sionna.rt.Scene
except ImportError:
    print("Sionna is not installed. To use sionna_exporter, please install it.")

def save_to_pickle(obj, filename):
    """Saves an object to a pickle file."""
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_from_pickle(filename):
    """Loads an object from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

def export_paths(path_list: List[Paths] | Paths) -> List[dict]:
    """Exports paths to a filtered dictionary with only selected keys """
    relevant_keys = ['sources', 'targets', 'a', 'tau', 'phi_r', 'phi_t', 
                     'theta_r', 'theta_t', 'types', 'vertices']
    
    path_list = [path_list] if not isinstance(path_list, list) else path_list
    
    paths_dict_list = []
    for path_obj in path_list:
        path_dict = path_obj.to_dict()
        
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
               not isinstance(attr_obj, Scene) and
               not attr_name.startswith("__") and
               attr_name.startswith("_")}
    return data

def export_scene_materials(scene: Scene) -> Tuple[List[Dict[str, Any]], List[int]]:
    """ Extract materials from Scene. 
    Outputs list of unique material dictionaries and a list of the material of each shape.
    """
    # Get scene in dictionary format
    scene_dict = scene_to_dict(scene)
    materials_predefined = scene_dict['radio_materials']
    
    # Materials in each object:
    materials = []
    for shape in scene.mi_shapes:
        shape_name = shape.id()
        shape_material = shape_name.split('-itu_')[-1]
        materials += [shape_material]
    
    unique_materials = np.unique(materials).tolist()
    
    material_indices = [unique_materials.index(material) for material in materials]
    
    # Terrain added manually in Blender (made of Concrete)
    if 'mesh-Plane' in unique_materials:
        plane_idx = unique_materials.index('mesh-Plane')
        unique_materials[plane_idx] = 'concrete'
    
    # Add 'itu_' preffixes back
    unique_materials_w_preffix = ['itu_' + mat for mat in unique_materials]
    
    # Get material indices and material properties from the predefined material list
    material_properties = {key: value for key, value in materials_predefined.items()
                           if key in unique_materials_w_preffix}

    # Do some light processing to add dictionaries to a list in a pickable format
    materials_dict_list = []
    for material_name, mat_property in material_properties.items():
        materials_dict = {
            'name': material_name,
            'conductivity': mat_property.conductivity.numpy(),
            'relative_permeability': mat_property.relative_permeability.numpy(),
            'relative_permittivity': mat_property.relative_permittivity.numpy(),
            'scattering_coefficient': mat_property.scattering_coefficient.numpy(),
            'scattering_pattern': type(mat_property.scattering_pattern).__name__,
            'alpha_r': mat_property.scattering_pattern.alpha_r,
            'alpha_i': mat_property.scattering_pattern.alpha_i,
            'lambda_': mat_property.scattering_pattern.lambda_.numpy(),
            'xpd_coefficient': mat_property.xpd_coefficient.numpy(),   
        }
        materials_dict_list += [materials_dict]
        
    return materials_dict_list, material_indices

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
    
        array_synthetic=scene_dict['synthetic_array'],
    
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

def export_scene_buildings(scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    """ Export the vertices and faces of buildings in a Sionna Scene.
    Output:
        vertice_matrix: n_vertices_in_scene x 3 (xyz coordinates)
        face_matrix: n_faces_in_scene x 3 (indices of each vertex in triangular face)
    """
    # Count all faces of all shapes
    n_tot_vertices = 0
    n_tot_faces = 0
    for i, shape in enumerate(scene.mi_shapes):
        n_tot_vertices += shape.vertex_count()
        n_tot_faces += shape.face_count()
    
    # Pre-allocate matrices
    vertice_matrix = np.array(np.zeros((n_tot_vertices, 3)))  # each vertice has 3 coordinates (xyz)
    face_matrix = np.array(np.zeros((n_tot_faces, 3)))  # each face has the indices of 3 vertices
    
    # Load matrices of vertices and faces
    last_vertice_idx = 0
    last_face_idx = 0
    objects_dict = {}  # store object-to-face mapping
    for i, shape in enumerate(scene.mi_shapes):
        n_vertices = shape.vertex_count()
        vertice_idxs = last_vertice_idx + np.arange(n_vertices)
        vertice_matrix[vertice_idxs] = np.array(shape.vertex_position(np.arange(n_vertices)))
        last_vertice_idx = vertice_idxs[-1]
    
        n_faces = shape.face_count()
        face_idxs = last_face_idx + np.arange(n_faces)
        face_matrix[face_idxs] = np.array(shape.vertex_position(np.arange(n_faces)))
        last_face_idx = face_idxs[-1]

        objects_dict[shape.id()] = face_matrix[face_idxs]
    
    return vertice_matrix, face_matrix, objects_dict

def export_to_deepmimo(scene: Scene, path_list: List[Paths] | Paths, 
                       my_compute_path_params: Dict, save_folder: str):
    """ Export a complete Sionna simulation to a format that can be converted by DeepMIMO """
    
    paths_dict_list = export_paths(path_list)
    materials_dict_list, material_indices = export_scene_materials(scene)
    rt_params = export_scene_rt_params(scene, **my_compute_path_params)
    vertice_matrix, face_matrix, objects_dict = export_scene_buildings(scene)
    
    os.makedirs(save_folder, exist_ok=True)
    
    save_vars_dict = {
        # filename: variable_to_save
        'sionna_paths.pkl': paths_dict_list,
        'sionna_materials.pkl': materials_dict_list,
        'sionna_material_indices.pkl': material_indices,
        'sionna_rt_params.pkl': rt_params,
        'sionna_vertices.pkl': vertice_matrix,
        'sionna_faces.pkl': face_matrix,
        'sionna_objects.pkl': objects_dict,
    }
    
    for filename, variable in save_vars_dict.items():
        save_to_pickle(variable, save_folder + filename)

    return