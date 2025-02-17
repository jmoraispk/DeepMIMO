"""
Ray Tracing Parameters Module.

This module provides the base class for ray tracing parameters used across different
ray tracing engines (Wireless Insite, Sionna, etc.). It defines common parameters
and functionality while allowing engine-specific extensions.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional
from pathlib import Path

@dataclass
class RayTracingParameters:
    """Base class for ray tracing parameters.
    
    This class defines common parameters across different ray tracing engines.
    Each specific engine (Wireless Insite, Sionna, etc.) should extend this class
    with its own parameters and methods.
    
    Note: All parameters are required to allow child classes to add their own required
    parameters. Default values are set in __post_init__.
    """
    # Frequency (determines material properties)
    frequency: float  # Center frequency in Hz
    
    # Ray tracing interaction settings
    max_depth: int  # Maximum number of interactions (reflections + diffractions + scattering)
    max_reflections: int  # Maximum number of reflections
    los: bool  # Line of sight
    reflection: bool  # Reflections
    diffraction: bool  # Diffraction
    scattering: bool  # Scattering
    
    # Engine info
    raytracer_name: str  # Name of ray tracing engine (from constants)
    raytracer_version: str  # Version of ray tracing engine
    
    # Raw parameters storage
    raw_params: Dict  # Store original parameters from engine
    
    def __post_init__(self):
        """Set default values for parameters if not explicitly provided."""
        # Set defaults for interaction flags
        if not hasattr(self, 'los'):
            self.los = True
        if not hasattr(self, 'reflection'):
            self.reflection = True
        if not hasattr(self, 'diffraction'):
            self.diffraction = False
        if not hasattr(self, 'scattering'):
            self.scattering = False
            
        # Set defaults for engine info
        if not hasattr(self, 'raytracer_version'):
            self.raytracer_version = ''
            
        # Initialize raw parameters if not provided
        if not hasattr(self, 'raw_params'):
            self.raw_params = {}
    
    def to_dict(self) -> Dict:
        """Convert parameters to dictionary format.
        
        Returns:
            Dictionary containing all parameters
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, params_dict: Dict, raw_params: Optional[Dict] = None) -> 'RayTracingParameters':
        """Create RayTracingParameters from a dictionary.
        
        Args:
            params_dict: Dictionary containing parameter values
            raw_params: Optional dictionary containing original engine parameters
            
        Returns:
            RayTracingParameters object
        """
        # Store raw parameters if provided
        if raw_params is not None:
            params_dict['raw_params'] = raw_params
        return cls(**params_dict)
    
    @classmethod
    def read_parameters(cls, load_folder: str | Path) -> 'RayTracingParameters':
        """Read parameters from a folder.
        
        This is an abstract method that should be implemented by each engine-specific
        subclass to read parameters in the appropriate format.
        
        Args:
            load_folder: Path to folder containing parameter files
            
        Returns:
            RayTracingParameters object
        
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Must be implemented by subclass") 