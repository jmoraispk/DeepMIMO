"""
Dataset module for DeepMIMO.

This module provides the Dataset class for managing DeepMIMO datasets,
including channel matrices, path information, and metadata.
"""

# Standard library imports
from typing import Dict, Optional, Any

# Third-party imports
import numpy as np

# Base utilities
from ...general_utilities import DotDict
from ... import consts as c

# Channel generation
from .channel import generate_MIMO_channel, ChannelGenParameters

# Antenna patterns and geometry
from .ant_patterns import AntennaPattern
from .geometry import (
    rotate_angles_batch,
    apply_FoV_batch,
    array_response_batch,
    ant_indices
)

# Utilities
from .utils import dbm2watt

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
        toa: Time of arrival for each path
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

    def _compute_num_paths(self) -> np.ndarray:
        """Compute number of valid paths for each user."""
        max_paths = self.aoa_az.shape[-1]
        nan_count_matrix = np.isnan(self.aoa_az).sum(axis=1)
        return max_paths - nan_count_matrix

    def _compute_num_interactions(self) -> np.ndarray:
        """Compute number of interactions for each path of each user."""
        result = np.zeros_like(self.inter, dtype=int)
        non_zero = self.inter > 0
        result[non_zero] = np.floor(np.log10(self.inter[non_zero])).astype(int) + 1
        return result

    def _compute_distances(self) -> np.ndarray:
        """Compute Euclidean distances between receivers and transmitter."""
        return np.linalg.norm(self.rx_pos - self.tx_pos, axis=1)

    def _compute_pathloss(self, coherent: bool = True) -> np.ndarray:
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
        """Compute MIMO channel matrices for all users.
        
        Args:
            params (ChannelGenParameters, optional): Channel generation parameters
            
        Returns:
            numpy.ndarray: MIMO channel matrix
        """
        if params is None:
            params = ChannelGenParameters()
        
        # Store params for use by other compute functions
        self.ch_params = params
        
        np.random.seed(1001)
        
        # Compute array response product
        array_response_product = self._compute_array_response_product()
        
        return generate_MIMO_channel(
            array_response_product=array_response_product,
            powers=self.power_linear_ant_gain,
            toas=self.toa,
            phases=self.phase,
            ofdm_params=params.OFDM,
            freq_domain=params.OFDM_channels
        )

    def _compute_los(self) -> np.ndarray:
        """Calculate Line of Sight status (1: LoS, 0: NLoS, -1: No paths) for each receiver.

        Uses the interaction codes defined in consts.py:
            INTERACTION_LOS = 0: Line-of-sight (direct path)
            INTERACTION_REFLECTION = 1: Reflection
            INTERACTION_DIFFRACTION = 2: Diffraction
            INTERACTION_TRANSMISSION = 3: Transmission
            INTERACTION_SCATTERING = 4: Scattering
        
        Returns:
            numpy.ndarray: LoS status array, shape (n_users, n_paths) 
        """
        result = np.full(self.inter.shape[0], -1)
        has_paths = np.any(self.inter > 0, axis=1)
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
            tx_ant_params = self.ch_params[c.PARAMSET_ANT_BS]
        if rx_ant_params is None:
            rx_ant_params = self.ch_params[c.PARAMSET_ANT_UE]
            
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
            bs_params = self.ch_params[c.PARAMSET_ANT_BS]
        if ue_params is None:
            ue_params = self.ch_params[c.PARAMSET_ANT_UE]
            
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

    def _compute_single_array_response(self, ant_params: Dict, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Compute array response for a single antenna array.
        
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
        """Compute product of TX and RX array responses.
        
        Returns:
            Array response product matrix
        """
        # Get antenna parameters from ch_params
        tx_ant_params = self.ch_params.bs_antenna
        rx_ant_params = self.ch_params.ue_antenna
        
        # Compute individual responses
        array_response_TX = self._compute_single_array_response(
            tx_ant_params, self.aod_el_rot_fov, self.aod_az_rot_fov)
            
        array_response_RX = self._compute_single_array_response(
            rx_ant_params, self.aoa_el_rot_fov, self.aoa_az_rot_fov)
        
        # Compute product with proper broadcasting
        # [n_users, M_rx, M_tx, n_paths]
        return array_response_RX[:, :, None, :] * array_response_TX[:, None, :, :]

    def _compute_power_linear(self) -> np.ndarray:
        """Compute linear power from power in dBm"""
        return dbm2watt(self.power) 

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
        print(f"\nTrying to get '{key}'")
        
        # First check if it's an alias and resolve it
        resolved_key = self.aliases.get(key, key)
        if resolved_key != key:
            print(f"Found alias: {key} -> {resolved_key}")
            key = resolved_key
            try:
                # Try direct access with resolved key
                print(f"Trying direct access for '{key}'")
                return self._data[key]  # Access underlying dictionary directly
            except KeyError as e:
                # Then check if it's a computable attribute
                print(f"Direct access failed: {str(e)}")
            
        if key in self._computed_attributes:
            compute_method_name = self._computed_attributes[key]
            print(f"Found compute method: {compute_method_name}")
            compute_method = getattr(self, compute_method_name)
            print(f"Computing value for '{key}'")
            value = compute_method()
            # Cache the result
            if isinstance(value, dict):
                print("Caching dictionary result")
                self.update(value)  # Uses DotDict's update which stores in _data
            else:
                print("Caching scalar result")
                self[key] = value  # Only store in _data
            return value

    def __getattr__(self, key: str) -> Any:
        """Enable dot notation access."""
        try:
            return super().__getitem__(key)
        except KeyError:
            return self._resolve_key(key)

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary access."""
        try:
            return super().__getitem__(key)  # Try direct dictionary access first
        except KeyError:
            return self._resolve_key(key)  # Fall back to resolution chain

    # Dictionary mapping attribute names to their computation methods
    _computed_attributes = {
        'num_paths': '_compute_num_paths',
        'num_interactions': '_compute_num_interactions',
        'distances': '_compute_distances',
        'pathloss': '_compute_pathloss',
        'channel': '_compute_channels',
        'los': '_compute_los',
        'power_linear': '_compute_power_linear',
        'power_linear_ant_gain': '_compute_power_linear_ant_gain',
        'fov': '_compute_fov',
        'array_response_product': '_compute_array_response_product',
        'aoa_az_rot': '_compute_rotated_angles',
        'aoa_el_rot': '_compute_rotated_angles',
        'aod_az_rot': '_compute_rotated_angles',
        'aod_el_rot': '_compute_rotated_angles',
        'aoa_az_rot_fov': '_compute_fov',
        'aoa_el_rot_fov': '_compute_fov',
        'aod_az_rot_fov': '_compute_fov',
        'aod_el_rot_fov': '_compute_fov',
        'fov_mask': '_compute_fov'  
    }

    # Dictionary of common aliases for dataset attributes
    aliases = {
        # Channel aliases
        'ch': 'channel',
        'chs': 'channel',
        'channels': 'channel',
        
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
        'time_of_arrival': 'toa',
        
        # Interaction aliases
        'interactions': 'inter',
        'interaction_positions': 'inter_pos',
        'interaction_locations': 'inter_pos'
    }

    
    def __dir__(self):
        """Return list of valid attributes including computed ones."""
        # Include standard attributes, computed attributes, and aliases
        return list(set(
            list(super().__dir__()) + 
            list(self._computed_attributes.keys()) + 
            list(self.aliases.keys())
        ))
        