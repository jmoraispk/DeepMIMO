"""
Utilities module for DeepMIMO.
Contains commonly used functions across the package, including:
- Unit conversions
- Sampling utilities
- Path calculations
- Display helpers
"""
import time
import numpy as np
from ... import consts as c

################################# Internal ####################################

def safe_print(text, stop_dur=0.3):
    """Print with delay for better display (usually after tqdm)"""
    print(text)
    time.sleep(stop_dur)
        
################################## For User ###################################

def dbm2watt(val):
    """Convert dBm to Watts"""
    return 10**(val/10 - 3)

def steering_vec(array, phi=0, theta=0, spacing=0.5):
    """
    Creates the array steering vector for uniform (linear and rectangular) arrays.

    Parameters
    ----------
    array : tuple, list or numpy.ndarray
        Number of [elements along the horizontal, elements along the vertical]
    phi : float, optional    
        Azimuth angle in degrees. 0 azimuth is normal to the array. 
        Positive azimuth points beams to the right. The default is 0.
    theta : float, optional
        Elevation angle in degrees. 0 elevation is horizon (normal to the array). 
        Positive elevation tilts the beam downards. The default is 0.
    spacing : flaot, optional
        Antenna spacing in wavelengths. The default is 0.5.

    Returns
    -------
    numpy.ndarray
        The normalized array response vector.

    """
    idxs = ant_indices(array)
    resp = array_response(idxs, phi*np.pi/180, theta*np.pi/180 + np.pi/2, 2*np.pi*spacing)
    return resp / np.linalg.norm(resp)


def uniform_sampling(steps, n_rows, users_per_row):
    """
    Returns indices of users at uniform steps/intervals.
    
    Parameters
    ----------
    steps : list
        Step size along x and y dimensions. The list should have 2 elements only.
        Examples:
        [1,1] = indices of all users
        [1,2] = same number of users across x, one every two users along y. Results: half the users.
        [2,2] = takes one user every 2 users (step=2), along x and y. Results: 1/4 the total users
    n_rows : int
        Number of rows in the generated dataset. Necessary for indexing users.
    users_per_row : int
        Number of users per row in the generated dataset. Necessary for indexing users.
        
    Returns
    -------
    data : list
        List of undersampled indices.
    """
    cols = np.arange(users_per_row, step=sampling_div[0])
    rows = np.arange(n_rows, step=sampling_div[1])
    uniform_idxs = np.array([j + i*users_per_row for i in rows for j in cols])
    
    return uniform_idxs


class LinearPath():
    """Creates a linear path of features"""
    def __init__(self, deepmimo_dataset, first_pos, last_pos, res=1, n_steps=False, filter_repeated=True):
        """
        Creates a linear path of features. 

        Parameters
        ----------
        deepmimo_dataset : DeepMIMO dataset
            A dataset generated from DeepMIMO.generate_data(parameters).
        first_pos : numpy.ndarray
            [x, y, (z)] position of the start of the linear path.
        last_pos : numpy.ndarray
            [x, y, (z)] position of the end of the linear path.
        res : float, optional
            Resolution [in meters]. The same as providing n_steps. The default is 1.
        n_steps : int, optional
            Number of positions / samples between first and last position (including).
            If False, either uses the resolution parameter <res>. The default is False.
        filter_repeated : bool, optional
            Whether to eliminated repeated positions. 
            Repeated positions happen when a path is oversampled or the closest
            dataste position repeats for a set of path positions. 
            If True, the repeated positions are removed. The default is True.

        Returns
        -------
        None.

        """
        
        if len(first_pos) == 2: # if not given, assume z-coordinate = 0
            first_pos = np.concatenate((first_pos,[0]))
            last_pos = np.concatenate((last_pos,[0]))
            
        self.first_pos = first_pos
        self.last_pos = last_pos
        
        self.dataset = deepmimo_dataset if type(deepmimo_dataset) != list else deepmimo_dataset[0]
        self._set_idxs_pos_res_steps(res, n_steps, filter_repeated)
        self._copy_data_from_dataset()
        self._extract_features()
        
    def _set_idxs_pos_res_steps(self, res, n_steps, filter_repeated):
        dataset_pos = self.dataset['user']['location']
        if not n_steps:
            data_res = np.linalg.norm(dataset_pos[0] - dataset_pos[1])
            if res < data_res and filter_repeated:
                print(f'Changing resolution to {data_res} to eliminate repeated positions')
                res = data_res
                
            self.n = int(np.linalg.norm(self.first_pos - self.last_pos) / res)
        else:
            self.n = n_steps
        
        xs = np.linspace(self.first_pos[0], self.last_pos[0], self.n).reshape((-1,1))
        ys = np.linspace(self.first_pos[1], self.last_pos[1], self.n).reshape((-1,1))
        zs = np.linspace(self.first_pos[2], self.last_pos[2], self.n).reshape((-1,1))
        
        interpolated_pos = np.hstack((xs,ys,zs))
        idxs = np.array([np.argmin(np.linalg.norm(dataset_pos - pos, axis=1)) 
                         for pos in interpolated_pos])
        
        if filter_repeated:
            # soft: removes adjacent repeated only
            idxs = np.concatenate(([idxs[0]], idxs[1:][(idxs[1:]-idxs[:-1]) != 0]))
            
            if filter_repeated == 'hard':
                # hard: removes all repeated
                idxs = np.unique(idxs)
            
            self.n = len(idxs)
    
        self.idxs = idxs
        self.pos = dataset_pos[idxs]
    
    def _copy_data_from_dataset(self):
        self.feature_names = ['LoS', 'pathloss', 'distance']
        
        self.LoS = self.dataset['user']['LoS'][self.idxs]
        self.pathloss = self.dataset['user']['pathloss'][self.idxs]
        self.distance = self.dataset['user']['distance'][self.idxs]
        self.paths = self.dataset['user']['paths'][self.idxs]
        self.channel = self.dataset['user']['channel'][self.idxs]
        
    def _extract_features(self):
        # Main path features
        self.path_features = ['DoD_phi', 'DoD_theta', 'DoA_phi', 'DoA_theta', 
                              'ToA', 'phase', 'power']
        self.feature_names += ['main_path_' + var for var in self.path_features]
        for feat in self.path_features:
            setattr(self, f'main_path_{feat}', 
                    np.array([self.paths[i][feat][0] for i in range(self.n)]))#self.idxs]))
        
        # Other features
        self.feature_names += ['pwr_ratio_main_path', 'total_power']
        self.total_power = np.array([np.sum(self.paths[i]['power']) for i in range(self.n)])
        self.pwr_ratio_main_path = np.array([self.main_path_power[i] / np.sum(self.paths[i]['power'])
                                             if self.LoS[i] != -1 else np.nan for i in range(self.n)])

    def get_feature_names(self):
        return self.feature_names
    

def get_idxs_with_limits(data_pos, **limits):
    """
    Returns indices of positions that satisfy the given coordinate limits.
    
    Parameters
    ----------
    data_pos : numpy.ndarray
        Array with all positions. Dimensions: [n_positions, 2 or 3]
    **limits : dict
        Arbitrary combination of limits as keyword arguments.
        Supported keys: x_min, x_max, y_min, y_max, z_min, z_max
        Example: x_min=10, y_max=50, z_min=-2
        
    Returns
    -------
    numpy.ndarray
        Indices of positions that satisfy all specified limits.
    """
    valid_limits = {'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'}
    if not all(key in valid_limits for key in limits):
        raise ValueError(f"Invalid limit key. Supported limits are: {valid_limits}")
    
    # Start with all indices as valid
    valid_idxs = np.arange(len(data_pos))
    
    # Apply each limit sequentially
    coord_map = {'x': 0, 'y': 1, 'z': 2}
    for limit_name, limit_value in limits.items():
        coord = limit_name.split('_')[0]  # Extract 'x', 'y', or 'z'
        is_min = limit_name.endswith('min')
        
        if coord_map[coord] >= data_pos.shape[1]:
            raise ValueError(f"Cannot apply {coord} limit to {data_pos.shape[1]}D positions")
            
        if is_min:
            mask = data_pos[valid_idxs, coord_map[coord]] >= limit_value
        else:  # is_max
            mask = data_pos[valid_idxs, coord_map[coord]] <= limit_value
            
        valid_idxs = valid_idxs[mask]
    
    return valid_idxs