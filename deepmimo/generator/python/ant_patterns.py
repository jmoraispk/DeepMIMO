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
        # Initialize TX Pattern
        if tx_pattern in c.PARAMSET_ANT_RAD_PAT_VALS:
            if tx_pattern == c.PARAMSET_ANT_RAD_PAT_VALS[0]:
                self.tx_pattern_fn = None
            else:
                tx_pattern = tx_pattern.replace('-', '_')
                tx_pattern = 'pattern_' + tx_pattern
                self.tx_pattern_fn = globals()[tx_pattern]
        else:
            raise NotImplementedError(f"The given '{tx_pattern}' antenna radiation pattern is not applicable.")

        # Initialize RX Pattern
        if rx_pattern in c.PARAMSET_ANT_RAD_PAT_VALS:
            if rx_pattern == c.PARAMSET_ANT_RAD_PAT_VALS[0]:
                self.rx_pattern_fn = None
            else:
                rx_pattern = rx_pattern.replace('-', '_')
                rx_pattern = 'pattern_' + rx_pattern
                self.rx_pattern_fn = globals()[rx_pattern]
        else:
            raise NotImplementedError(f"The given '{rx_pattern}' antenna radiation pattern is not applicable.")

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
        pattern = 1.
        if self.tx_pattern_fn is not None:
            pattern *= self.tx_pattern_fn(dod_theta, dod_phi)
        if self.rx_pattern_fn is not None:
            pattern *= self.rx_pattern_fn(doa_theta, doa_phi)
            
        return power * pattern


def pattern_halfwave_dipole(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Calculate half-wave dipole antenna pattern.
    
    This function implements the theoretical radiation pattern of a half-wave
    dipole antenna, including its characteristic figure-8 shape.
    
    Args:
        theta (np.ndarray): Theta angles in radians.
        phi (np.ndarray): Phi angles in radians.
        
    Returns:
        np.ndarray: Antenna gain pattern for given angles.
    """
    max_gain = 1.6409223769  # Half-wave dipole maximum directivity
    theta_nonzero = theta.copy()
    zero_idx = theta_nonzero == 0
    theta_nonzero[zero_idx] = 1e-4  # Approximation of 0 at limit
    pattern = max_gain * np.cos((np.pi/2)*np.cos(theta))**2 / np.sin(theta)**2
    pattern[zero_idx] = 0
    return pattern

