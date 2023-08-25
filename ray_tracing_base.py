#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:35:31 2023

@author: joao
"""
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')
tf.random.set_seed(1) # Set global random seed for reproducibility

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, DirectivePattern
from sionna.channel import ApplyTimeChannel, cir_to_ofdm_channel

from tqdm import tqdm
import mitsuba as mi
import drjit as dr
import time
from scipy.io import loadmat, savemat
import pandas as pd

with open('scenes_folder.txt', 'r') as fp:
    root_folder = fp.read()[:-1] # no newline

print(root_folder)


# In[2]:


tf.config.list_physical_devices()


# In[37]:


# Classes
class Shape:
    
    # bbox Mitsuba API: https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.ScalarBoundingBox3f
    def __init__(self, shape):
        self.id = shape.id()
        self.shape = shape
        params = mi.traverse(shape)
        self.faces = dr.unravel(mi.Point3f, params['faces']).numpy()
        self.vertices = dr.unravel(mi.Point3f, params['vertex_positions']).numpy()
        self.bbox = shape.bbox()
        
    def get_vertices(self):
        return self.vertices
    
    def get_faces(self):
        return self.faces

    def contains(self, pos):
        return self.bbox.contains(pos)

    def get_size(self):
        return self.bbox.extents()

    def get_3d_area(self):
        return self.bbox.surface_area()

    def get_volume(self):
        return self.bbox.volume()

    def get_bbox_corners(self):
        return self.bbox.min, self.bbox.max

    def get_height(self):
        return self.bbox.max[2] - self.bbox.min[2]
        
    def get_center(self):
        return self.bbox.center

    def get_distance(self, target):
        return self.bbox.distance(target)

    def get_bbox(self):
        return self.bbox

    def get_bbox_bottom_vertices(self):
        return [self.bbox.corner(i) for i in range(4)]
    
    def get_bbox_top_vertices(self):
        return [self.bbox.corner(4+i) for i in range(4)]

    def get_bbox_vertices(self):
        return [self.bbox.corner(i) for i in range(8)]
    

# Functions
def print_all_shapes_in_scene(scene):
    for shape in scene.mi_scene.shapes():
        print(shape.id())

def get_floor_from_scene(scene):
    floor_shape = None
    for shape in scene.mi_scene.shapes():
        if shape.id() == 'mesh-floor':
            floor_shape = shape
            break

    return shape

def get_buildings_from_scene(scene):
    buildings = []
    for shape in scene.mi_scene.shapes():
        if shape.id() != 'mesh-floor':
            buildings.append(shape)

    return buildings

def compute_array_combinations(arrays):
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))

def gen_user_grid(box_corners, steps, no_zones=None, box_offsets=None):
    """
    box_corners is = [bbox_min_corner, bbox_max_corner]
    steps = [x_step, y_step, z_step]
    no_zones = [list of Shapes with a contains() method]
    """

    # Sample the ranges of coordinates
    ndim = len(box_corners[0])
    dim_ranges = []
    for dim in range(ndim):
        if steps[dim]:
            dim_range = np.arange(box_corners[0][dim], box_corners[1][dim], steps[dim])
        else:
            dim_range = np.array([box_corners[0][dim]]) # select just the first limit
        
        dim_ranges.append(dim_range + box_offsets[dim] if box_offsets else 0)
    
    dims = [len(r) if len(r) else 1 for r in dim_ranges]
    print(f'Grid dimensions: {dims}')
    
    n_total = np.prod(dims)
    print(f'Total positions covering the ground plane: {n_total}')
    
    # Compute combination of sampled ranges
    positions = compute_array_combinations(dim_ranges)
    
    # Determine which positions are inside no_zones
    idxs_in_nozone = np.zeros(positions.shape[0], dtype=bool)
    for pos_idx in tqdm(range(n_total), desc='Intersecting positions with no-zones (e.g. buildings)'):
        for no_zone in no_zones:
            if no_zone.contains(positions[pos_idx]):
                idxs_in_nozone[pos_idx] = True
                break
    
    # Include only the positions that are outside no zones
    idxs_to_include = np.invert(idxs_in_nozone)
    
    n_total_after_filtering = sum(idxs_to_include)
    print(f'Total positions outside of buildings: {n_total_after_filtering}')
    
    return positions[idxs_to_include]

def is_notebook() -> bool:
    is_notebook = False
    try:
        shell = get_ipython().__class__.__name__
        module = get_ipython().__class__.__module__
    except NameError:
        return False # Probably standard Python interpreter
        
    if shell == 'ZMQInteractiveShell':
        is_notebook = True   # Jupyter notebook or qtconsole
    elif module == 'google.colab._shell':
        is_notebook = True   # Colab notebook
    elif shell == 'TerminalInteractiveShell':
        is_notebook = False  # Terminal running IPython
    else:
        is_notebook = False  # Other type (?)

    return is_notebook

def create_base_scene(scene_path, center_frequency):
    scene = load_scene(scene_path)
    scene.frequency = center_frequency
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")
    
    scene.rx_array = scene.tx_array
    scene.synthetic_array = True
    
    return scene


# In[38]:


# Read parameters from CSV
df = pd.read_csv('params.csv')

# Compute simulations for each row
n_rows = df.index.stop

for row_idx in range(n_rows):
    carrier_freq = df['freq (ghz)'][row_idx] * 1e9
    n_reflections = df['n_reflections'][row_idx]
    
    if not np.isnan(df['bs_lat'][row_idx]):
        tx_pos = [df['bs_lat'][row_idx], df['bs_lon'][row_idx], df['bs_alt'][row_idx]]
    else:
        tx_pos = None # placed automatically on building closest to the center
        
    scattering = bool(df['scattering'][row_idx])
    diffraction = bool(df['diffraction'][row_idx])
    
    x_step = df['x_step'][row_idx]
    y_step = df['y_step'][row_idx]
    z_step = df['z_step'][row_idx]

    print(f'Running RT simulation for row {row_idx+1} (starts at 1):\n'
          f'n_reflections = {n_reflections}\n'
          f'tx_pos = {tx_pos}\n'
          f'scattering = {scattering}\n'
          f'diffraction = {diffraction}\n'
          f'[x_step, y_step, z_step] = [{x_step}, {y_step}, {z_step}]\n')

    # 0- Create/Fetch scene and get buldings in the scene
    #scene_name = sionna.rt.scene.simple_street_canyon
    scene_folder = root_folder + f'scen_{row_idx}/'
    scene_name = scene_folder + 'scene.xml'
    scene = create_base_scene(scene_name, center_frequency=carrier_freq)
    buildings = [Shape(building) for building in get_buildings_from_scene(scene)][:-1] 
    # (unkown last building in bottom left corner...)

    # 1- Compute TX position
    print('Computing BS position')
    
    # 1.1- Find the building closest to the center of the scene ([0,0,0])
    distances = [building.get_distance([0,0,0]) for building in buildings]
    heights = [building.get_height() for building in buildings]
    building_score = [heights[b]**2/distances[b] for b in range(len(buildings))]
    best_building_idx = np.argmax(building_score)
    closest_building = buildings[best_building_idx]

    # 1.2- Find closest vertice to the origin (at height)
    best_building_vertices = closest_building.get_vertices()
    # use a high point at the origin to force the selection of a roof vertice
    vertice_distances = [np.linalg.norm(vert - [0,0,1e5]) for vert in best_building_vertices]
    closest_vertice_idx = np.argmin(vertice_distances)
    closest_vertice = best_building_vertices[closest_vertice_idx]
    
    # 1.3- Put transmitter 2 metters above that vertice
    tx_pos = closest_vertice + [0,0,2] if not tx_pos else tx_pos

    # 1.4- Add transmitter to the scene
    scene.add(Transmitter(name="tx",
                          position=tx_pos,
                          orientation=[0,0,0]))

    # 2- Compute RXs positions
    print('Computing UEs positions')
    
    # 2.1- Get limits of the floor
    floor_shape = Shape(get_floor_from_scene(scene))
    min_corner, max_corner = floor_shape.get_bbox_corners()
    c = 1.2 # constant of floor overscaling to account for edge effects

    # 2.2- Distribute users uniformely 1.5m above the floor
    rxs = gen_user_grid(box_corners=[min_corner/c, max_corner/c],
                        steps=[x_step,y_step,z_step],
                        no_zones=buildings,
                        box_offsets=[0,0,1.5])

    # 2.3- Add (ONLY SOME OF THE) receivers to the scene
    n_rx = len(rxs)
    n_rx_in_scene = 20 #n_rx if not scattering else 1
    print(f'Adding users to the scene ({n_rx_in_scene} at a time)')
    for rx_idx in range(n_rx_in_scene):
        scene.add(Receiver(name=f"rx_{rx_idx}",
                           position=rxs[rx_idx],
                           orientation=[0,0,0]))

    # 3- Compute paths
    # 3.1- Enable scattering in the radio materials
    if scattering:
        for rm in scene.radio_materials.values():
            rm.scattering_coefficient = 1/np.sqrt(3) # [0,1]
            rm.scattering_pattern = DirectivePattern(alpha_r=10)
            

    # 3.2- Compute the paths for each set of receiver positions
    path_list = []
    n_rx_remaining = n_rx
    for x in tqdm(range(int(n_rx / n_rx_in_scene)+1), desc='Path computation'):
        if n_rx_remaining > 0:
            n_rx_remaining -= n_rx_in_scene
        else:
            break
        if x != 0:
            # modify current RXs in scene
            for rx_idx in range(n_rx_in_scene):
                if rx_idx + n_rx_in_scene*x < n_rx:
                    scene.receivers[f'rx_{rx_idx}'].position = rxs[rx_idx + n_rx_in_scene*x]
                else:
                    # remove the last receivers in the scene
                    scene.remove(f'rx_{rx_idx}')
            
        paths = scene.compute_paths(max_depth=n_reflections,
                                    num_samples=1e6,
                                    scattering=scattering,
                                    diffraction=diffraction)
        path_list.append(paths)
    
    # 4- Save paths
    print('Building path matrices')
    n_rx = len(rxs)
    path_matrices = []
    for idx, paths in enumerate(path_list):
        if False:
            print(f'paths idx = {idx}')
            # [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, max_num_paths, num_time_steps]
            print(f'a = {paths.a.shape}')
            # print(f'tau = {paths.tau.shape}')         # [batch_size, num_rx, num_tx, max_num_paths]
            # print(f'phi_r = {paths.phi_r.shape}')     # [batch_size, num_rx, num_tx, max_num_paths],
            # print(f'phi_t = {paths.phi_t.shape}')     # [batch_size, num_rx, num_tx, max_num_paths],
            # print(f'theta_r = {paths.theta_r.shape}') # [batch_size, num_rx, num_tx, max_num_paths],
            # print(f'theta_t = {paths.theta_t.shape}') # [batch_size, num_rx, num_tx, max_num_paths],
            # print(f'type = {paths.types.shape}')      # [batch_size, max_num_paths]
        
        phase = np.angle(paths.a.numpy(), deg=True)
        ToA   = paths.tau.numpy()
        power = 20 * np.log10(np.absolute(paths.a.numpy()))
        DoA_phi   = paths.phi_r.numpy()   * 180 / np.pi
        DoA_theta = paths.theta_r.numpy() * 180 / np.pi
        DoD_phi   = paths.phi_t.numpy()   * 180 / np.pi
        DoD_theta = paths.theta_t.numpy() * 180 / np.pi
    
        # Generate 8 by X matrices, X = number of paths
        empty_paths_warning = False
        for i in range(paths.a.shape[1]):
            # determine which paths exist (non-existing paths have negative delays)
            non_zero_paths = np.where(paths.tau.numpy()[0,i,0,:] > 0)[0]

            if np.size(non_zero_paths) == 0:
                if not empty_paths_warning:
                    print('Found empty paths: number of reflections may not be enough')
                    empty_paths_warning = True
                path_matrix = np.zeros((8, len(non_zero_paths)))
                path_matrix[1,:] = -1 # negative delays!
            else:
                path_matrix = np.zeros((8, len(non_zero_paths)))
    
                # determine which paths are  LoS
                los = np.zeros_like(non_zero_paths)
                los[0] = 1 if (0 in non_zero_paths and paths.types.numpy()[0,0] == 0) else 0
                
                path_matrix = np.vstack((
                    phase[0, i, 0, 0, 0, non_zero_paths, 0],
                    ToA[0, i, 0, non_zero_paths],
                    power[0, i, 0, 0, 0, non_zero_paths, 0],
                    DoA_phi[0, i, 0, non_zero_paths],
                    DoA_theta[0, i, 0, non_zero_paths],
                    DoD_phi[0, i, 0, non_zero_paths],
                    DoD_theta[0, i, 0, non_zero_paths],
                    los))
                
                # sort paths based on received power
                path_matrix = path_matrix[:, np.flip(path_matrix[2,:].argsort())] 
                
            path_matrices.append(path_matrix)
        
    # 5- Save data for DeepMIMO 
    # 5.1- Create BS file
    dict_bs = {
        'channels': [{'p': []}],
        'rx_locs': [tx_pos[0], tx_pos[1], tx_pos[2], 0, 0],
    }
    # 5.2- Create users file
    rx_locs = np.zeros((n_rx, 5))
    rx_locs[:, :3] = rxs
    ues_channels = [{'p': path_matrices[i]} for i in range(n_rx)]
    dict_ues = {
        'channels': ues_channels,
        'rx_locs': rx_locs,
    }

    # CHECK why the -INF and why the (8,0) matrices!!!!!
                
    # 5.3- Create parameters file
    params_file = {
        'carrier_freq': carrier_freq,
        'doppler_available': 0,
        'dual_polar_available': 0,
        'num_BS': 1,
        'transmit_power': 0.0,
        'user_grids': [1.0, 1.0, n_rx],
        'version': 2,
    }
    
    # 5.4- Save data
    mat_folder = scene_folder + 'DeepMIMO_folder/'
    os.makedirs(mat_folder, exist_ok=True)
    savemat(mat_folder +  'BS1_BS.mat', dict_bs)
    savemat(mat_folder + f'BS1_UE_0-{n_rx}.mat', dict_ues)
    savemat(mat_folder + f'params.mat', params_file)

    # NOTE: warning is normal
    # RuntimeWarning: divide by zero encountered in log10power = 20 * np.log10(np.absolute(paths.a.numpy()))
    # This happens because some positions don't have any paths to them


# In[39]:


scene.preview()


# In[ ]:


m_test = loadmat(mat_folder + f'BS1_UE0-{n_rx}.mat')

# Check for -inf powers
n_users = len(m_test['channels'][0])
count_infs = 0
for i in range(n_users):
    count_infs = np.sum(len(np.where(np.isinf(m_test['channels'][0][i][0][0][0]))[0]))
    
print(f'count_infs = {count_infs}')

# Check for empty matrices (users with no paths)
emtpy_matrices = 0
for i in range(n_users):
    if m_test['channels'][0][i][0][0][0].shape[1] == 0:
        emtpy_matrices += 1
print(f'empty_matrices = {emtpy_matrices}')

