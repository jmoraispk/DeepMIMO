"""
Antenna patterns module for DeepMIMO.

This module provides antenna radiation pattern modeling capabilities for the DeepMIMO
dataset generator. It implements various antenna types and their radiation patterns,
including:
- Omnidirectional patterns
- Half-wave dipole patterns

The module provides a unified interface for applying antenna patterns to signal power
calculations in MIMO channel generation.
"""

# Third-party imports
import numpy as np

# Local imports
from ... import consts as c


def pattern_isotropic(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Calculate isotropic antenna pattern (unity gain in all directions).
    
    Args:
        theta (np.ndarray): Theta angles in radians.
        phi (np.ndarray): Phi angles in radians.
        
    Returns:
        float: Unit gain (1.0) regardless of input angles.
    """
    return 1.


def pattern_halfwave_dipole(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Calculate half-wave dipole antenna pattern.
    
    This function implements the theoretical radiation pattern of a half-wave
    dipole antenna, including its characteristic figure-8 shape.
    The pattern follows the formula: G(θ) = 1.643 * [cos(π/2 * cos(θ))]²/sin(θ)
    where θ is measured from the dipole axis.
    
    Reference: Balanis, C.A. "Antenna Theory: Analysis and Design", 4th Edition
    
    Args:
        theta (np.ndarray): Theta angles in radians.
        phi (np.ndarray): Phi angles in radians.
        
    Returns:
        np.ndarray: Antenna gain pattern for given angles.
    """
    max_gain = 1.643  # Half-wave dipole maximum directivity
    
    # Convert to numpy array if not already
    theta = np.asarray(theta)
    
    # Initialize pattern array
    pattern = np.zeros_like(theta, dtype=np.float64)
    
    # Handle valid angles (not near 0 or π)
    valid_angles = (np.abs(np.sin(theta)) > 1e-10)
    
    # Calculate the pattern using the standard dipole formula
    # Pre-compute terms for better performance
    theta_valid = theta[valid_angles]
    sin_theta = np.sin(theta_valid)
    cos_term = np.cos(np.pi/2 * np.cos(theta_valid))
    
    # Apply the formula: G(θ) = max_gain * [cos(π/2 * cos(θ))]²/sin²(θ)
    pattern[valid_angles] = max_gain * (cos_term**2 / sin_theta)
    
    return pattern


# Pattern registry mapping pattern names to their functions
PATTERN_REGISTRY = {
    'isotropic': pattern_isotropic,
    'halfwave-dipole': pattern_halfwave_dipole
}


class AntennaPattern:
    """Class for handling antenna radiation patterns.
    
    This class manages the radiation patterns for both TX and RX antennas,
    providing a unified interface for pattern application in signal power
    calculations.
    
    Attributes:
        tx_pattern_fn (Optional[Callable]): Function implementing TX antenna pattern.
        rx_pattern_fn (Optional[Callable]): Function implementing RX antenna pattern.
    """

    def __init__(self, tx_pattern: str, rx_pattern: str) -> None:
        """Initialize antenna patterns for transmitter and receiver.
        
        Args:
            tx_pattern (str): Transmitter antenna pattern type from PARAMSET_ANT_RAD_PAT_VALS.
            rx_pattern (str): Receiver antenna pattern type from PARAMSET_ANT_RAD_PAT_VALS.
        
        Raises:
            NotImplementedError: If specified pattern type is not supported.
        """
        self.tx_pattern_fn = self._load_pattern(tx_pattern, "TX")
        self.rx_pattern_fn = self._load_pattern(rx_pattern, "RX")

    def _load_pattern(self, pattern: str, pattern_type: str) -> callable:
        """Load antenna pattern function from registry.
        
        Args:
            pattern (str): Pattern type from PARAMSET_ANT_RAD_PAT_VALS.
            pattern_type (str): Either "TX" or "RX" for error messages.
            
        Returns:
            callable: Pattern function to use.
            
        Raises:
            NotImplementedError: If pattern is not supported or not implemented.
        """
        if pattern not in c.PARAMSET_ANT_RAD_PAT_VALS:
            raise NotImplementedError(f"The given '{pattern}' antenna radiation pattern is not applicable for {pattern_type}.")
        if pattern not in PATTERN_REGISTRY:
            raise NotImplementedError(f"The pattern '{pattern}' is defined but not implemented for {pattern_type}.")
        return PATTERN_REGISTRY[pattern]

    def apply(self, power: np.ndarray, doa_theta: np.ndarray, doa_phi: np.ndarray,
             dod_theta: np.ndarray, dod_phi: np.ndarray) -> np.ndarray:
        """Apply antenna patterns to input power.
        
        This function applies both TX and RX antenna patterns to modify the input
        power values based on arrival and departure angles.
        
        Args:
            power (np.ndarray): Input power values.
            doa_theta (np.ndarray): Direction of arrival theta angles in radians.
            doa_phi (np.ndarray): Direction of arrival phi angles in radians.
            dod_theta (np.ndarray): Direction of departure theta angles in radians.
            dod_phi (np.ndarray): Direction of departure phi angles in radians.
            
        Returns:
            np.ndarray: Modified power values after applying antenna patterns.
        """
        pattern = self.tx_pattern_fn(dod_theta, dod_phi) * self.rx_pattern_fn(doa_theta, doa_phi)
        return power * pattern

    def apply_batch(self, power: np.ndarray, doa_theta: np.ndarray, doa_phi: np.ndarray,
                   dod_theta: np.ndarray, dod_phi: np.ndarray) -> np.ndarray:
        """Apply antenna patterns to powers in batch.
        
        Args:
            power (np.ndarray): Powers array with shape (n_users, n_paths)
            doa_theta (np.ndarray): Direction of arrival elevation angles (n_users, n_paths)
            doa_phi (np.ndarray): Direction of arrival azimuth angles (n_users, n_paths)
            dod_theta (np.ndarray): Direction of departure elevation angles (n_users, n_paths)
            dod_phi (np.ndarray): Direction of departure azimuth angles (n_users, n_paths)
            
        Returns:
            np.ndarray: Modified powers with antenna patterns applied (n_users, n_paths)
        """
        # Reshape inputs to 2D if they're 1D
        if power.ndim == 1:
            power = power.reshape(1, -1)
            doa_theta = doa_theta.reshape(1, -1)
            doa_phi = doa_phi.reshape(1, -1)
            dod_theta = dod_theta.reshape(1, -1)
            dod_phi = dod_phi.reshape(1, -1)
            
        pattern = self.tx_pattern_fn(dod_theta, dod_phi) * self.rx_pattern_fn(doa_theta, doa_phi)
        return power * pattern
