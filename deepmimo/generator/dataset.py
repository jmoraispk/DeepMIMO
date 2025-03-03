"""
Dataset module for DeepMIMO.

This module provides the Dataset class for managing DeepMIMO datasets,
including channel matrices, path information, and metadata.
"""

# Standard library imports
import inspect
from typing import Dict, Optional, Any, List

# Third-party imports
import numpy as np

# Base utilities
from ..general_utilities import DotDict
from .. import consts as c
from ..info import info

# Channel generation
from .channel import generate_MIMO_channel, ChannelGenParameters, validate_ch_gen_params

# Antenna patterns and geometry
from .ant_patterns import AntennaPattern
from .geometry import (
    rotate_angles_batch,
    apply_FoV_batch,
    array_response_batch,
    ant_indices
)

# Utilities
from .utils import dbw2watt

# Parameters that should remain consistent across datasets in a MacroDataset
SHARED_PARAMS = [
    c.SCENE_PARAM_NAME,           # Scene object
    c.MATERIALS_PARAM_NAME,       # MaterialList object
    c.LOAD_PARAMS_PARAM_NAME,     # Loading parameters
    c.RT_PARAMS_PARAM_NAME,       # Ray-tracing parameters
]

class Dataset(DotDict):
    """Class for managing DeepMIMO datasets.
    
    This class provides an interface for accessing dataset attributes including:
    - Channel matrices
    - Path information (angles, powers, delays)
    - Position information
    - Metadata
    
    Attributes can be accessed using both dot notation (dataset.channel) 
    and dictionary notation (dataset['channel']).
    
    Primary (Static) Attributes:
        power: Path powers in dBm
        phase: Path phases in degrees
        delay: Path delays in seconds (i.e. propagation time)
        aoa_az/aoa_el: Angles of arrival (azimuth/elevation)
        aod_az/aod_el: Angles of departure (azimuth/elevation)
        rx_pos: Receiver positions
        tx_pos: Transmitter position
        inter: Path interaction indicators
        inter_pos: Path interaction positions
        
    Secondary (Computed) Attributes:
        power_linear: Path powers in linear scale
        channel: MIMO channel matrices
        num_paths: Number of paths per user
        pathloss: Path loss in dB
        distances: Distances between TX and RXs
        los: Line of sight status for each receiver
        pwr_ant_gain: Powers with antenna patterns applied
        aoa_az_rot/aoa_el_rot: Rotated angles of arrival based on antenna orientation
        aod_az_rot/aod_el_rot: Rotated angles of departure based on antenna orientation
        aoa_az_rot_fov/aoa_el_rot_fov: Field of view filtered angles of arrival
        aod_az_rot_fov/aod_el_rot_fov: Field of view filtered angles of departure
        fov_mask: Field of view mask
        
    Common Aliases:
        ch, pwr, rx_loc, pl, dist, n_paths, etc.
        (See aliases dictionary for complete mapping)
    """
    
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize dataset with optional data.
        
        Args:
            data: Initial dataset dictionary. If None, creates empty dataset.
        """
        super().__init__(data or {})

    def compute_channels(self, params: Optional[ChannelGenParameters] = None) -> np.ndarray:
        """Compute MIMO channel matrices for all users.
        
        This is the main public method for computing channel matrices. It handles all the
        necessary preprocessing steps including:
        - Antenna pattern application
        - Field of view filtering
        - Array response computation
        - OFDM processing (if enabled)
        
        The computed channel will be cached and accessible as dataset.channel
        or dataset['channel'] after this call.
        
        Args:
            params: Channel generation parameters. If None, uses default parameters.
                   See ChannelGenParameters class for details.
            
        Returns:
            numpy.ndarray: MIMO channel matrix with shape [n_users, n_rx_ant, n_tx_ant, n_subcarriers]
                          if freq_domain=True, otherwise [n_users, n_rx_ant, n_tx_ant, n_paths]
        """
        channel = self._compute_channels(params)
        self['channel'] = channel  # Explicitly cache the result
        return channel

    def __getattr__(self, key: str) -> Any:
        """Enable dot notation access."""
        try:
            return super().__getitem__(key)
        except KeyError:
            return self._resolve_key(key)

    def __getitem__(self, key: str) -> Any:
        """Get an item from the dataset, computing it if necessary."""
        try:
            return super().__getitem__(key)
        except KeyError:
            return self._resolve_key(key)

    def _resolve_key(self, key: str) -> Any:
        """Resolve a key through the lookup chain.
        
        Order of operations:
        1. Check if key is an alias and resolve it first
        2. Try direct access with resolved key
        3. Try computing the attribute if it's computable
        
        Args:
            key: The key to resolve
            
        Returns:
            The resolved value
            
        Raises:
            KeyError if key cannot be resolved
        """
        # First check if it's an alias and resolve it
        resolved_key = self._aliases.get(key, key)
        if resolved_key != key:
            key = resolved_key
            try:
                # Try direct access with resolved key
                return self._data[key]  # Access underlying dictionary directly
            except KeyError:
                pass  # Continue to computation check
            
        if key in self._computed_attributes:
            compute_method_name = self._computed_attributes[key]
            compute_method = getattr(self, compute_method_name)
            value = compute_method()
            # Cache the result
            if isinstance(value, dict):
                self.update(value)  # Uses DotDict's update which stores in _data
                return self._data[key]  # Return the specific key requested
            else:
                self[key] = value  # Only store in _data
                return value
        
        raise KeyError(key)

    def _compute_num_paths(self) -> np.ndarray:
        """Compute number of valid paths for each user."""
        max_paths = self.aoa_az.shape[-1]
        nan_count_matrix = np.isnan(self.aoa_az).sum(axis=1)
        return max_paths - nan_count_matrix

    def _compute_n_ue(self) -> int:
        """Return the number of UEs/receivers in the dataset."""
        return self.rx_pos.shape[0]

    def _compute_num_interactions(self) -> np.ndarray:
        """Compute number of interactions for each path of each user."""
        result = np.zeros_like(self.inter, dtype=int)
        non_zero = self.inter > 0
        result[non_zero] = np.floor(np.log10(self.inter[non_zero])).astype(int) + 1
        return result

    def _compute_distances(self) -> np.ndarray:
        """Compute Euclidean distances between receivers and transmitter."""
        return np.linalg.norm(self.rx_pos - self.tx_pos, axis=1)

    def compute_pathloss(self, coherent: bool = True) -> np.ndarray:
        """Compute path loss in dB, assuming 0 dBm transmitted power.
        
        Args:
            coherent (bool): Whether to use coherent sum. Defaults to True
        
        Returns:
            numpy.ndarray: Path loss in dB
        """
        # Convert powers to linear scale
        powers_linear = 10 ** (self.power / 10)  # mW
        phases_rad = np.deg2rad(self.phase)
        
        # Sum complex path gains
        complex_gains = np.sqrt(powers_linear).astype(np.complex64)
        if coherent:
            complex_gains *= np.exp(1j * phases_rad)
        total_power = np.abs(np.sum(complex_gains, axis=1))**2
        
        # Convert back to dB
        return -10 * np.log10(total_power)

    def _compute_channels(self, params: Optional[ChannelGenParameters] = None) -> np.ndarray:
        """Internal method to compute MIMO channel matrices.
        
        This is an internal implementation method. Users should use compute_channel() instead.
        
        Args:
            params: Channel generation parameters
            
        Returns:
            numpy.ndarray: MIMO channel matrix
        """
        if params is None:
            params = ChannelGenParameters()
        else:
            validate_ch_gen_params(params, self.n_ue)
        
        # Store params directly in dictionary
        self.ch_params = params
        
        np.random.seed(1001)
        
        # Compute array response product
        array_response_product = self._compute_array_response_product()
        
        n_paths_to_gen = params.num_paths
        
        return generate_MIMO_channel(
            array_response_product=array_response_product[..., :n_paths_to_gen],
            powers=self.power_linear_ant_gain[..., :n_paths_to_gen],
            delays=self.delay[..., :n_paths_to_gen],
            phases=self.phase[..., :n_paths_to_gen],
            ofdm_params=params.ofdm,
            freq_domain=params.freq_domain
        )

    def compute_los(self) -> np.ndarray:
        """Calculate Line of Sight status (1: LoS, 0: NLoS, -1: No paths) for each receiver.

        Uses the interaction codes defined in consts.py:
            INTERACTION_LOS = 0: Line-of-sight (direct path)
            INTERACTION_REFLECTION = 1: Reflection
            INTERACTION_DIFFRACTION = 2: Diffraction
            INTERACTION_SCATTERING = 3: Scattering
            INTERACTION_TRANSMISSION = 4: Transmission

        Returns:
            numpy.ndarray: LoS status array, shape (n_users, n_paths) 
        """
        result = np.full(self.inter.shape[0], -1)
        has_paths = self.num_paths > 0
        result[has_paths] = 0
        
        first_path = self.inter[:, 0]
        los_mask = first_path == c.INTERACTION_LOS
        result[los_mask & has_paths] = 1
        
        return result

    def _compute_power_linear_ant_gain(self, tx_ant_params: Optional[Dict[str, Any]] = None,
                               rx_ant_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Compute received power with antenna patterns applied.
        
        Args:
            tx_ant_params (Optional[Dict[str, Any]]): Transmitter antenna parameters. If None, uses stored params.
            rx_ant_params (Optional[Dict[str, Any]]): Receiver antenna parameters. If None, uses stored params.
            
        Returns:
            np.ndarray: Powers with antenna pattern applied, shape [n_users, n_paths]
        """
        # Use stored channel parameters if none provided
        if tx_ant_params is None:
            tx_ant_params = self.ch_params[c.PARAMSET_ANT_BS]
        if rx_ant_params is None:
            rx_ant_params = self.ch_params[c.PARAMSET_ANT_UE]
            
        # Create antenna pattern object
        antennapattern = AntennaPattern(tx_pattern=tx_ant_params[c.PARAMSET_ANT_RAD_PAT],
                                      rx_pattern=rx_ant_params[c.PARAMSET_ANT_RAD_PAT])
        
        # Get FoV filtered angles and apply antenna patterns in batch
        return antennapattern.apply_batch(power=self[c.PWR_LINEAR_PARAM_NAME],
                                        aoa_theta=self[c.AOA_EL_FOV_PARAM_NAME],
                                        aoa_phi=self[c.AOA_AZ_FOV_PARAM_NAME], 
                                        aod_theta=self[c.AOD_EL_FOV_PARAM_NAME],
                                        aod_phi=self[c.AOD_AZ_FOV_PARAM_NAME])

    def _compute_rotated_angles(self, tx_ant_params: Optional[Dict[str, Any]] = None, 
                              rx_ant_params: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """Compute rotated angles for all users in batch.
        
        Args:
            tx_ant_params: Dictionary containing transmitter antenna parameters. If None, uses stored params.
            rx_ant_params: Dictionary containing receiver antenna parameters. If None, uses stored params.
            
        Returns:
            Dictionary containing the rotated angles for all users
        """
        # Use stored channel parameters if none provided
        if tx_ant_params is None:
            tx_ant_params = self.ch_params.bs_antenna
        if rx_ant_params is None:
            rx_ant_params = self.ch_params.ue_antenna
            
        # Rotate angles for all users at once
        aod_theta_rot, aod_phi_rot = rotate_angles_batch(
            rotation=tx_ant_params[c.PARAMSET_ANT_ROTATION],
            theta=self[c.AOD_EL_PARAM_NAME],
            phi=self[c.AOD_AZ_PARAM_NAME])
        
        aoa_theta_rot, aoa_phi_rot = rotate_angles_batch(
            rotation=rx_ant_params[c.PARAMSET_ANT_ROTATION],
            theta=self[c.AOA_EL_PARAM_NAME],
            phi=self[c.AOA_AZ_PARAM_NAME])
        
        return {
            c.AOD_EL_ROT_PARAM_NAME: aod_theta_rot,
            c.AOD_AZ_ROT_PARAM_NAME: aod_phi_rot,
            c.AOA_EL_ROT_PARAM_NAME: aoa_theta_rot,
            c.AOA_AZ_ROT_PARAM_NAME: aoa_phi_rot
        }

    def _compute_fov(self, bs_params: Optional[Dict[str, Any]] = None, 
                     ue_params: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        """Compute field of view filtered angles for all users.
        
        This function applies field of view constraints to the rotated angles
        and stores both the filtered angles and the mask in the dataset.
        Optimizes computation by skipping mask generation when FoV is full.
        
        Args:
            bs_params: Base station antenna parameters including FoV. If None, uses stored params.
            ue_params: User equipment antenna parameters including FoV. If None, uses stored params.
            
        Returns:
            Dict: Dictionary containing FoV filtered angles and mask
        """
        # Use stored channel parameters if none provided
        if bs_params is None:
            bs_params = self.ch_params.bs_antenna
        if ue_params is None:
            ue_params = self.ch_params.ue_antenna
            
        # Get rotated angles from dataset
        aod_theta = self[c.AOD_EL_ROT_PARAM_NAME]  # [n_users, n_paths]
        aod_phi = self[c.AOD_AZ_ROT_PARAM_NAME]    # [n_users, n_paths]
        aoa_theta = self[c.AOA_EL_ROT_PARAM_NAME]  # [n_users, n_paths]
        aoa_phi = self[c.AOA_AZ_ROT_PARAM_NAME]    # [n_users, n_paths]
        
        # Get FoV parameters
        tx_fov = bs_params[c.PARAMSET_ANT_FOV]  # [horizontal, vertical]
        rx_fov = ue_params[c.PARAMSET_ANT_FOV]  # [horizontal, vertical]
        
        # Skip operations if fov is full
        bs_full_fov = (tx_fov[0] >= 360 and tx_fov[1] >= 180)
        ue_full_fov = (rx_fov[0] >= 360 and rx_fov[1] >= 180)
        
        if bs_full_fov and ue_full_fov:
            return {
                c.FOV_MASK_PARAM_NAME: None,
                c.AOD_EL_FOV_PARAM_NAME: aod_theta,
                c.AOD_AZ_FOV_PARAM_NAME: aod_phi,
                c.AOA_EL_FOV_PARAM_NAME: aoa_theta,
                c.AOA_AZ_FOV_PARAM_NAME: aoa_phi
            }
        
        # Initialize mask as all True
        fov_mask = np.ones_like(aod_theta, dtype=bool)
        
        # Check if BS FoV is limited
        if not bs_full_fov:
            tx_mask = apply_FoV_batch(tx_fov, aod_theta, aod_phi)  # [n_users, n_paths]
            fov_mask = np.logical_and(fov_mask, tx_mask)
    
        # Check if UE FoV is limited
        if not ue_full_fov:
            rx_mask = apply_FoV_batch(rx_fov, aoa_theta, aoa_phi)  # [n_users, n_paths]
            fov_mask = np.logical_and(fov_mask, rx_mask)
        
        return {
            c.FOV_MASK_PARAM_NAME: fov_mask,
            c.AOD_EL_FOV_PARAM_NAME: np.where(fov_mask, aod_theta, np.nan),
            c.AOD_AZ_FOV_PARAM_NAME: np.where(fov_mask, aod_phi, np.nan),
            c.AOA_EL_FOV_PARAM_NAME: np.where(fov_mask, aoa_theta, np.nan),
            c.AOA_AZ_FOV_PARAM_NAME: np.where(fov_mask, aoa_phi, np.nan)
        }

    def _compute_single_array_response(self, ant_params: Dict, theta: np.ndarray, 
                                       phi: np.ndarray) -> np.ndarray:
        """Internal method to compute array response for a single antenna array.
        
        Args:
            ant_params: Antenna parameters dictionary
            theta: Elevation angles array
            phi: Azimuth angles array
            
        Returns:
            Array response matrix
        """
        # Use attribute access for antenna parameters
        kd = 2 * np.pi * ant_params.spacing
        ant_ind = ant_indices(ant_params[c.PARAMSET_ANT_SHAPE])  # tuple complications..
        
        return array_response_batch(ant_ind=ant_ind, theta=theta, phi=phi, kd=kd)

    def _compute_array_response_product(self) -> np.ndarray:
        """Internal method to compute product of TX and RX array responses.
        
        Returns:
            Array response product matrix
        """
        # Get antenna parameters from ch_params
        tx_ant_params = self.ch_params.bs_antenna
        rx_ant_params = self.ch_params.ue_antenna
        
        # Compute individual responses
        array_response_TX = self._compute_single_array_response(
            tx_ant_params, self[c.AOD_EL_FOV_PARAM_NAME], self[c.AOD_AZ_FOV_PARAM_NAME])
            
        array_response_RX = self._compute_single_array_response(
            rx_ant_params, self[c.AOA_EL_FOV_PARAM_NAME], self[c.AOA_AZ_FOV_PARAM_NAME])
        
        # Compute product with proper broadcasting
        # [n_users, M_rx, M_tx, n_paths]
        return array_response_RX[:, :, None, :] * array_response_TX[:, None, :, :]

    def _compute_power_linear(self) -> np.ndarray:
        """Internal method to compute linear power from power in dBm"""
        return dbw2watt(self.power) 

    def _compute_grid_info(self) -> Dict[str, np.ndarray]:
        """Internal method to compute grid size and spacing information from receiver positions.
        
        Returns:
            Dict containing:
                grid_size: Array with [x_size, y_size] - number of points in each dimension
                grid_spacing: Array with [x_spacing, y_spacing] - spacing between points in meters
        """
        x_positions = np.unique(self.rx_pos[:, 0])
        y_positions = np.unique(self.rx_pos[:, 1])
        
        grid_size = np.array([len(x_positions), len(y_positions)])
        grid_spacing = np.array([
            np.mean(np.diff(x_positions)),
            np.mean(np.diff(y_positions))
        ])
        
        return {
            'grid_size': grid_size,
            'grid_spacing': grid_spacing
        }

    def _is_valid_grid(self) -> bool:
        """Check if the dataset has a valid grid structure.
        
        A valid grid means that:
        1. The total number of points in the grid matches the number of receivers
        2. The receivers are arranged in a regular grid pattern
        
        Returns:
            bool: True if dataset has valid grid structure, False otherwise
        """
        # Check if total grid points match number of receivers
        grid_points = np.prod(self.grid_size)
        
        return grid_points == self.n_ue

    def subset(self, idxs: np.ndarray) -> 'Dataset':
        """Create a new dataset containing only the selected indices.
        
        Args:
            idxs: Array of indices to include in the new dataset
            
        Returns:
            Dataset: A new dataset containing only the selected indices
        """
        # Create a new dataset with initial data
        initial_data = {}
        
        # Copy shared parameters that should remain consistent across datasets
        for param in SHARED_PARAMS:
            if hasattr(self, param):
                initial_data[param] = getattr(self, param)
            
        # Directly set n_ue
        initial_data['n_ue'] = len(idxs)
        
        # Create new dataset with initial data
        new_dataset = Dataset(initial_data)
        
        # Copy all attributes
        for attr, value in self.to_dict().items():
            # skip private and already handled attributes
            if not attr.startswith('_') and attr not in SHARED_PARAMS + ['n_ue']:
                print(f"attr: {attr}")
                if isinstance(value, np.ndarray) and value.shape[0] == self.n_ue:
                    # Copy and index arrays with UE dimension
                    print(f"value.shape: {value.shape}")
                    setattr(new_dataset, attr, value[idxs])
                else:
                    # Copy other attributes as is
                    setattr(new_dataset, attr, value)
                
        return new_dataset

    def get_active_idxs(self) -> np.ndarray:
        """Return indices of active users.
        
        Returns:
            Array of indices of active users
        """
        return np.where(self.num_paths > 0)[0]

    def get_uniform_idxs(self, steps: List[int]) -> np.ndarray:
        """Return indices of users at uniform intervals.
        
        Args:
            steps: List of sampling steps for each dimension [x_step, y_step]
            
        Returns:
            Array of indices for uniformly sampled users
            
        Raises:
            ValueError: If dataset does not have a valid grid structure
        """
        
        grid_size = self.grid_size  # [x_size, y_size] = [n_cols, n_rows]
        
        # Check if dataset has valid grid structure
        if not self._is_valid_grid():
            print(f"Warning. Grid_size: {grid_size} = {np.prod(grid_size)} users != {self.n_ue} users in rx_pos")
            if steps == [1, 1]:
                idxs = np.arange(self.n_ue)
            else:
                raise ValueError("Dataset does not have a valid grid structure. Cannot perform uniform sampling.")
        else:
            # Get indices of users at uniform intervals
            cols = np.arange(grid_size[0], step=steps[0])
            rows = np.arange(grid_size[1], step=steps[1])
            idxs = np.array([j + i*grid_size[0] for i in rows for j in cols])
        
        return idxs

    # Dictionary mapping attribute names to their computation methods
    # (in order of computation)
    _computed_attributes = {
        'num_paths': '_compute_num_paths',
        'n_ue': '_compute_n_ue',
        'num_interactions': '_compute_num_interactions',
        'distances': '_compute_distances',
        'pathloss': 'compute_pathloss',
        'channel': '_compute_channels',
        'los': 'compute_los',
        
        # Power linear
        'power_linear': '_compute_power_linear',
        
        # Rotated angles
        'aoa_az_rot': '_compute_rotated_angles',
        'aoa_el_rot': '_compute_rotated_angles', 
        'aod_az_rot': '_compute_rotated_angles',
        'aod_el_rot': '_compute_rotated_angles',
        'array_response_product': '_compute_array_response_product',
        
        # Field of view
        'fov': '_compute_fov',
        'fov_mask': '_compute_fov',
        'aoa_az_rot_fov': '_compute_fov',
        'aoa_el_rot_fov': '_compute_fov',
        'aod_az_rot_fov': '_compute_fov',
        'aod_el_rot_fov': '_compute_fov',
        
        # Power with antenna gain
        'power_linear_ant_gain': '_compute_power_linear_ant_gain',
        
        # Grid information
        'grid_size': '_compute_grid_info',
        'grid_spacing': '_compute_grid_info'
    }

    # Dictionary of common aliases for dataset attributes
    _aliases = {
        # Channel aliases
        'ch': 'channel',
        'chs': 'channel',
        'channels': 'channel',

        # Channel parameters aliases
        'channel_params': 'ch_params',
        
        # Power aliases
        'pwr': 'power',
        'powers': 'power',
        'pwr_lin': 'power_linear',
        'pwr_ant_gain': 'power_linear_ant_gain',
        
        # Position aliases
        'rx_loc': 'rx_pos',
        'rx_position': 'rx_pos',
        'rx_locations': 'rx_pos',
        'tx_loc': 'tx_pos',
        'tx_position': 'tx_pos',
        'tx_locations': 'tx_pos',
        
        # Pathloss aliases
        'pl': 'pathloss',
        'path_loss': 'pathloss',
        
        # Distance aliases
        'dist': 'distances',
        'distance': 'distances',
        'dists': 'distances',
        
        # Angle aliases
        'aoa_phi': 'aoa_az',
        'aoa_theta': 'aoa_el',
        'aod_phi': 'aod_az',
        'aod_theta': 'aod_el',
        
        # Path count aliases
        'n_paths': 'num_paths',
        
        # Time of arrival aliases
        'toa': 'delay',
        'time_of_arrival': 'delay',

        # Interaction aliases
        'bounce_type': 'inter',
        'interactions': 'inter',
        'bounce_pos': 'inter_pos',
        'interaction_positions': 'inter_pos',
        'interaction_locations': 'inter_pos'
    }

    
    def __dir__(self):
        """Return list of valid attributes including computed ones."""
        # Include standard attributes, computed attributes, and aliases
        return list(set(
            list(super().__dir__()) + 
            list(self._computed_attributes.keys()) + 
            list(self._aliases.keys())
        ))

    def info(self, param_name: str | None = None) -> None:
        """Display help information about DeepMIMO dataset parameters.
        
        Args:
            param_name: Name of the parameter to get info about.
                       If None or 'all', displays information for all parameters.
                       If the parameter name is an alias, shows info for the resolved parameter.
        """
        # If it's an alias, resolve it first
        if param_name in self._aliases:
            resolved_name = self._aliases[param_name]
            print(f"'{param_name}' is an alias for '{resolved_name}'")
            param_name = resolved_name
            
        info(param_name)

class MacroDataset:
    """A container class that holds multiple Dataset instances and propagates operations to all children.
    
    This class acts as a simple wrapper around a list of Dataset objects. When any attribute
    or method is accessed on the MacroDataset, it automatically propagates that operation
    to all contained Dataset instances. If the MacroDataset contains only one dataset,
    it will return single value instead of a list with a single element.
    """
    
    # Methods that should only be called on the first dataset
    SINGLE_ACCESS_METHODS = {
        'info',  # Parameter info should only be shown once
    }
    
    # Methods that should be propagated to children - automatically populated from Dataset methods
    PROPAGATE_METHODS = {
        name for name, _ in inspect.getmembers(Dataset, predicate=inspect.isfunction)
        if not name.startswith('__')  # Skip dunder methods
    }
    
    def __init__(self, datasets=None):
        """Initialize with optional list of Dataset instances.
        
        Args:
            datasets: List of Dataset instances. If None, creates empty list.
        """
        self.datasets = datasets if datasets is not None else []
        
    def get_single(self, key):
        """Get a single value from the first dataset for shared parameters.
        
        Args:
            key: Key to get value for
            
        Returns:
            Single value from first dataset if key is in SHARED_PARAMS,
            otherwise returns list of values from all datasets
        """
        if not self.datasets:
            raise IndexError("MacroDataset is empty")
        return self.datasets[0][key]
        
    def __getattr__(self, name):
        """Propagate any attribute/method access to all datasets.
        
        If the attribute is a method in PROPAGATE_METHODS, call it on all children.
        If the attribute is in SHARED_PARAMS, return from first dataset.
        If there is only one dataset, return single value instead of lists.
        Otherwise, return list of results from all datasets.
        """
        # Check if it's a method we should propagate
        if name in self.PROPAGATE_METHODS:
            if name in self.SINGLE_ACCESS_METHODS:
                # For single access methods, only call on first dataset
                def single_method(*args, **kwargs):
                    return getattr(self.datasets[0], name)(*args, **kwargs)
                return single_method
            else:
                # For normal methods, propagate to all datasets
                def propagated_method(*args, **kwargs):
                    results = [getattr(dataset, name)(*args, **kwargs) for dataset in self.datasets]
                    return results[0] if len(results) == 1 else results
                return propagated_method
            
        # Handle shared parameters
        if name in SHARED_PARAMS:
            return self.get_single(name)
            
        # Default: propagate to all datasets
        results = [getattr(dataset, name) for dataset in self.datasets]
        return results[0] if len(results) == 1 else results
        
    def __getitem__(self, idx):
        """Get dataset at specified index if idx is integer, otherwise propagate to all datasets.
        
        Args:
            idx: Integer index to get specific dataset, or string key to get attribute from all datasets
            
        Returns:
            Dataset instance if idx is integer,
            single value if idx is in SHARED_PARAMS or if there is only one dataset,
            or list of results if idx is string and there are multiple datasets
        """
        if isinstance(idx, (int, slice)):
            return self.datasets[idx]
        if idx in SHARED_PARAMS:
            return self.get_single(idx)
        results = [dataset[idx] for dataset in self.datasets]
        return results[0] if len(results) == 1 else results
        
    def __setitem__(self, key, value):
        """Set item on all contained datasets.
        
        Args:
            key: Key to set
            value: Value to set
        """
        for dataset in self.datasets:
            dataset[key] = value
        
    def __len__(self):
        """Return number of contained datasets."""
        return len(self.datasets)
        
    def append(self, dataset):
        """Add a dataset to the collection.
        
        Args:
            dataset: Dataset instance to add
        """
        self.datasets.append(dataset)
        
        