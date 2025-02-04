import numpy as np
from typing import Tuple, List, Dict, Any
import pickle


def save_to_pickle(obj, filename):
    """Saves an object to a pickle file."""
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_from_pickle(filename):
    """Loads an object from a pickle file."""
    with open(filename, 'rb') as file:
        return pickle.load(file)

def sionna_rt_converter(a):
    print('converting from sionna RT')
    
# Unwrap materials
def import_sionna_for_deepmimo(save_folder: str):
    
    saved_vars_names = [
        'sionna_paths.pkl',
        'sionna_materials.pkl',
        'sionna_material_indices.pkl',
        'sionna_rt_params.pkl',
        'sionna_vertices.pkl',
        'sionna_faces.pkl',
        'sionna_objects.pkl',
    ]
    
    for filename in saved_vars_names:
        variable = load_from_pickle(save_folder + filename)

    return

def load_materials(load_folder):

    material_properties = 0

    # Attribute matching
    scat_model = {'none?': 'none', # if scattering coeff = 0
                'LambertianPattern': 'lambertian',
                'DirectivePattern': 'directive',
                'BackscatteringPattern': 'directive'} # directive = backscattering

    for mat_property in material_properties:
        a = mat_property.conductivity.numpy()
        b = mat_property.relative_permeability.numpy()
        c = mat_property.relative_permittivity.numpy()
        scat_coeff = mat_property.scattering_coefficient.numpy()
        e = mat_property.scattering_pattern.alpha_r
        f = mat_property.scattering_pattern.alpha_i
        g = mat_property.scattering_pattern.lambda_.numpy()
        h = mat_property.xpd_coefficient.numpy()
        scattering_model = scat_model[type(mat_property.scattering_pattern).__name__]
        scattering_model = 'none' if not scat_coeff else scattering_model


def load_paths(load_folder):
    path_dict_list = load_from_pickle(load_folder + 'sionna_paths.pkl')


    tx_pos = path_dict_list[0]['sources']

    rx_pos_all = []
    for obj_idx, paths_dict in enumerate(path_dict_list):
        rx_pos_all += paths_dict['targets'].tolist()
    rx_pos = np.unique(rx_pos_all, axis=0)
    n_rx = rx_pos.shape[0]

    # Pre-allocate matrices
    MAX_PATHS = 25
    MAX_INTER_PER_PATH = 10

    max_iteract = min(MAX_INTER_PER_PATH, path_dict_list[0]['vertices'].shape[0])

    data = {
        'n_paths': np.zeros((n_rx), dtype=np.float32),
        'aoa_az': np.zeros((n_rx, MAX_PATHS), dtype=np.float32) * np.nan,
        'aoa_el': np.zeros((n_rx, MAX_PATHS), dtype=np.float32) * np.nan,
        'aod_az': np.zeros((n_rx, MAX_PATHS), dtype=np.float32) * np.nan,
        'aod_el': np.zeros((n_rx, MAX_PATHS), dtype=np.float32) * np.nan,
        'toa': np.zeros((n_rx, MAX_PATHS), dtype=np.float32) * np.nan,
        'power': np.zeros((n_rx, MAX_PATHS), dtype=np.float32) * np.nan,
        'phase': np.zeros((n_rx, MAX_PATHS), dtype=np.float32) * np.nan,
        'inter': np.zeros((n_rx, MAX_PATHS), dtype=np.float32) * np.nan,
        'inter_pos': np.zeros((n_rx, MAX_PATHS, max_iteract, 3), dtype=np.float32) * np.nan,
    }

    # squeeze and slice function to be applied to each array
    ss = lambda array: array.squeeze()[..., :MAX_PATHS]

    last_idx = 0
    for obj_idx, paths_dict in enumerate(path_dict_list):

        batch_size = paths_dict['a'].shape[1]
        idxs = last_idx + np.arange(batch_size)
        last_idx = idxs[-1]
        
        a = ss(paths_dict['a'])
        not_nan_mask = a != 0
        data['power'][idxs][not_nan_mask] = 20 * np.log10(np.absolute(a[not_nan_mask]))
        data['phase'][idxs] = np.angle(a, deg=True)
        data['toa'][idxs]   = ss(paths_dict['tau'])
        data['aoa_az'][idxs] = ss(paths_dict['phi_r'])   * 180 / np.pi
        data['aoa_el'][idxs] = ss(paths_dict['theta_r']) * 180 / np.pi
        data['aod_az'][idxs] = ss(paths_dict['phi_t'])   * 180 / np.pi
        data['aod_el'][idxs] = ss(paths_dict['theta_t']) * 180 / np.pi
        data['inter'][idxs]     = ss(paths_dict['types'])
        
        inter_pos = paths_dict['vertices'].squeeze()[:max_iteract, :, :MAX_PATHS, :]
        data['inter_pos'][idxs] = np.transpose(inter_pos, (1,2,0,3))


def load_vertices(load_folder):
    pass

