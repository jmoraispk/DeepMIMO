"""
General utility functions and classes for the DeepMIMO dataset generation.

This module provides utility functions and classes for handling printing,
file naming, string ID generation, and dictionary utilities used across 
the DeepMIMO toolkit.
"""

import numpy as np
import scipy.io
from pprint import pformat
from typing import Dict, Any, TypeVar, Mapping, Optional
from . import consts as c
from .scene import Scene

K = TypeVar('K', bound=str)
V = TypeVar('V')

class DotDict(Mapping[K, V]):
    """A dictionary subclass that supports dot notation access to nested dictionaries.
    
    This class allows accessing dictionary items using both dictionary notation (d['key'])
    and dot notation (d.key). It automatically converts nested dictionaries to DotDict
    instances to maintain dot notation access at all levels.
    
    Example:
        >>> d = DotDict({'a': 1, 'b': {'c': 2}})
        >>> d.a
        1
        >>> d.b.c
        2
        >>> d['b']['c']
        2
        >>> list(d.keys())
        ['a', 'b']
    """
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize DotDict with a dictionary.
    
        Args:
            dictionary: Dictionary to convert to DotDict
        """
        # Store protected attributes in a set
        self._data = {}
        if data:
            for key, value in data.items():
                if isinstance(value, dict):
                    self._data[key] = DotDict(value)
                else:
                    self._data[key] = value

    def __getattr__(self, key: str) -> Any:
        """Enable dot notation access to dictionary items."""
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key: str, value: Any) -> None:
        """Enable dot notation assignment."""
        if key == '_data':
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dictionary-style assignment."""
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        self._data[key] = value

    def update(self, other: Dict[str, Any]) -> None:
        """Update the dictionary with elements from another dictionary."""
        # Convert any nested dicts to DotDicts first
        processed = {
            k: DotDict(v) if isinstance(v, dict) and not isinstance(v, DotDict) else v 
            for k, v in other.items()
        }
        self._data.update(processed)

    def __len__(self) -> int:
        """Return the length of the underlying data dictionary."""
        return len(self._data)
        
    def __iter__(self):
        """Return an iterator over the data dictionary keys."""
        return iter(self._data)
        
    def __dir__(self):
        """Return list of valid attributes."""
        return list(set(
            list(super().__dir__()) + 
            list(self._data.keys())
        ))
        
    @property
    def shape(self):
        """Return shape of the first array-like value in the dictionary."""
        for val in self._data.values():
            if hasattr(val, 'shape'):
                return val.shape
        return None
        
    @property
    def size(self):
        """Return size of the first array-like value in the dictionary."""
        for val in self._data.values():
            if hasattr(val, 'size'):
                return val.size
        return None

    def keys(self):
        """Return dictionary keys."""
        return self._data.keys()

    def values(self):
        """Return dictionary values."""
        return self._data.values()

    def items(self):
        """Return dictionary items as (key, value) pairs."""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get value for key, returning default if key doesn't exist."""
        return self._data.get(key, default)
            
    def to_dict(self) -> Dict:
        """Convert DotDict back to a regular dictionary.
        
        Returns:
            dict: Regular dictionary representation
        """
        result = {}
        for key, value in self._data.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self) -> str:
        """Return string representation of dictionary."""
        return pformat(self._data)

class PrintIfVerbose:
    """A callable class that conditionally prints messages based on verbosity setting.
    
    Args:
        verbose (bool): Flag to control whether messages should be printed.
    """
    
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
    
    def __call__(self, message: str) -> None:
        """Print the message if verbose mode is enabled.
        
        Args:
            message (str): The message to potentially print.
        """
        if self.verbose:
            print(message)

def get_txrx_str_id(tx_set_idx: int, tx_idx: int, rx_set_idx: int) -> str:
    """Generate a standardized string identifier for TX-RX combinations.
    
    Args:
        tx_set_idx (int): Index of the transmitter set.
        tx_idx (int): Index of the transmitter within its set.
        rx_set_idx (int): Index of the receiver set.
    
    Returns:
        str: Formatted string identifier in the form 't{tx_set_idx}_tx{tx_idx}_r{rx_set_idx}'.
    """
    return f't{tx_set_idx:03}_tx{tx_idx:03}_r{rx_set_idx:03}'

def get_mat_filename(key: str, tx_set_idx: int, tx_idx: int, rx_set_idx: int) -> str:
    """Generate a .mat filename for storing DeepMIMO data.
    
    Args:
        key (str): The key identifier for the data type.
        tx_set_idx (int): Index of the transmitter set.
        tx_idx (int): Index of the transmitter within its set.
        rx_set_idx (int): Index of the receiver set.
    
    Returns:
        str: Complete filename with .mat extension.
    """
    str_id = get_txrx_str_id(tx_set_idx, tx_idx, rx_set_idx)
    return f'{key}_{str_id}.mat'

def load_mat_file_as_dict(file_path: str) -> Dict[str, Any]:
    """Load MATLAB .mat file as Python dictionary.
    
    Args:
        file_path (str): Path to .mat file to load
        
    Returns:
        dict: Dictionary containing loaded MATLAB data
        
    Raises:
        ValueError: If file cannot be loaded
    """
    mat_data = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    return {key: mat_struct_to_dict(value) for key, value in mat_data.items()
            if not key.startswith('__')}

def mat_struct_to_dict(mat_struct: Any) -> Dict[str, Any]:
    """Convert MATLAB structure to Python dictionary.
    
    This function recursively converts MATLAB structures and arrays to
    Python dictionaries and numpy arrays.

    Args:
        mat_struct (any): MATLAB structure to convert
        
    Returns:
        dict: Dictionary containing converted data
    """
    if isinstance(mat_struct, scipy.io.matlab.mat_struct):
        result = {}
        for field in mat_struct._fieldnames:
            result[field] = mat_struct_to_dict(getattr(mat_struct, field))
        return result
    elif isinstance(mat_struct, np.ndarray):
        # Process arrays recursively in case they contain mat_structs
        try:
            # First try to convert directly to numpy array
            return np.array([mat_struct_to_dict(item) for item in mat_struct])
        except ValueError:
            # If that fails due to inhomogeneous shapes, return as list instead
            return [mat_struct_to_dict(item) for item in mat_struct]
    return mat_struct  # Return the object as is for other types


def summary(scen_name: str) -> None:
    """Print a summary of the dataset."""
    
    # Read params.mat and provide TXRX summary, total number of tx & rx, scene size, 
    # and other relevant parameters, computed/extracted from the all dicts, not just rt_params

    scen_folder = c.SCENARIOS_FOLDER + '/' + scen_name

    mat_file = f'./{scen_folder}/{c.PARAMS_FILENAME}.mat'

    params_dict = load_mat_file_as_dict(mat_file)[c.PARAMS_FILENAME]
    rt_params = params_dict[c.RT_PARAMS_PARAM_NAME]
    scene_params = params_dict[c.SCENE_PARAM_NAME]
    material_params = params_dict[c.MATERIALS_PARAM_NAME]
    txrx_params = params_dict[c.TXRX_PARAM_NAME]

    print("\n" + "="*50)
    print(f"DeepMIMO {scen_name} Scenario Summary")
    print("="*50)

    print("\n[Ray-Tracing Configuration]")

    print(f"- Ray-tracer: {rt_params[c.RT_PARAM_RAYTRACER]} "
        f"v{rt_params[c.RT_PARAM_RAYTRACER_VERSION]}")
    print(f"- Frequency: {rt_params[c.RT_PARAM_FREQUENCY]/1e9:.1f} GHz")

    print("\n[Ray-tracing parameters]")

    # Interaction limits
    print("\nMain interaction limits")
    print(f"- Max path depth: {rt_params[c.RT_PARAM_PATH_DEPTH]}")
    print(f"- Max reflections: {rt_params[c.RT_PARAM_MAX_REFLECTIONS]}")
    print(f"- Max diffractions: {rt_params[c.RT_PARAM_MAX_DIFFRACTIONS]}")
    print(f"- Max scatterings: {rt_params[c.RT_PARAM_MAX_SCATTERINGS]}")
    print(f"- Max transmissions: {rt_params[c.RT_PARAM_MAX_TRANSMISSIONS]}")

    # Diffuse scattering settings
    print("\nDiffuse Scattering")
    is_diffuse_enabled = (rt_params[c.RT_PARAM_MAX_SCATTERINGS] > 0)
    print(f"- Diffuse scattering: {'Enabled' if is_diffuse_enabled else 'Disabled'}")
    print(f"- Diffuse reflections: {rt_params[c.RT_PARAM_DIFFUSE_REFLECTIONS]}")
    print(f"- Diffuse diffractions: {rt_params[c.RT_PARAM_DIFFUSE_DIFFRACTIONS]}")
    print(f"- Diffuse transmissions: {rt_params[c.RT_PARAM_DIFFUSE_TRANSMISSIONS]}")
    print(f"- Final interaction only: {rt_params[c.RT_PARAM_DIFFUSE_FINAL_ONLY]}")
    print(f"- Random phases: {rt_params[c.RT_PARAM_DIFFUSE_RANDOM_PHASES]}")

    # Terrain settings
    print("\nTerrain")
    print(f"- Terrain reflection: {rt_params[c.RT_PARAM_TERRAIN_REFLECTION]}")
    print(f"- Terrain diffraction: {rt_params[c.RT_PARAM_TERRAIN_DIFFRACTION]}")
    print(f"- Terrain scattering: {rt_params[c.RT_PARAM_TERRAIN_SCATTERING]}")

    # Ray casting settings
    print("\nRay Casting Settings")
    print(f"- Number of rays: {rt_params[c.RT_PARAM_NUM_RAYS]:,}")
    print(f"- Casting method: {rt_params[c.RT_PARAM_RAY_CASTING_METHOD]}")
    print(f"- Casting range (az): {rt_params[c.RT_PARAM_RAY_CASTING_RANGE_AZ]:.1f}°")
    print(f"- Casting range (el): {rt_params[c.RT_PARAM_RAY_CASTING_RANGE_EL]:.1f}°")
    print(f"- Synthetic array: {rt_params[c.RT_PARAM_SYNTHETIC_ARRAY]}")

    print("\n[Scene]")
    # Get scene object counts using Scene class method
    scene = Scene.from_data(scene_params, scen_folder)
    label_counts = scene.count_objects_by_label()
    objects_summary = ', '.join(f'{count} {label}' for label, count in label_counts.items())

    # Count faces from scene metadata
    normal_faces = sum(len(obj.faces) for obj in scene.objects)

    print(f"- Number of scenes: {scene_params[c.SCENE_PARAM_NUMBER_SCENES]}")
    print(f"- Total objects: {scene_params[c.SCENE_PARAM_N_OBJECTS]} ({objects_summary})")
    print(f"- Vertices: {scene_params[c.SCENE_PARAM_N_VERTICES]}")
    print(f"- Faces: {normal_faces:,} (decomposed into {scene_params[c.SCENE_PARAM_N_TRIANGULAR_FACES]:,} triangular faces)")

    # Get scene boundaries from scene bounding box
    bbox = scene.bounding_box
    print("\nBoundaries:")
    print(f"- X: {bbox.x_min:.2f}m to {bbox.x_max:.2f}m (width: {bbox.width:.2f}m)")
    print(f"- Y: {bbox.y_min:.2f}m to {bbox.y_max:.2f}m (length: {bbox.length:.2f}m)")
    print(f"- Z: {bbox.z_min:.2f}m to {bbox.z_max:.2f}m (height: {bbox.height:.2f}m)")
    print(f"- Area: {bbox.width * bbox.length:,.2f}m²")

    print("\n[Materials]")
    print(f"Total materials: {len(material_params)}")
    for _, mat_props in material_params.items():
        print(f"\n{mat_props[c.MATERIALS_PARAM_NAME_FIELD]}:")
        print(f"- Permittivity: {mat_props[c.MATERIALS_PARAM_PERMITTIVITY]:.2f}")
        print(f"- Conductivity: {mat_props[c.MATERIALS_PARAM_CONDUCTIVITY]:.2f} S/m")
        print(f"- Scattering model: {mat_props[c.MATERIALS_PARAM_SCATTERING_MODEL]}")
        print(f"- Scattering coefficient: {mat_props[c.MATERIALS_PARAM_SCATTERING_COEF]:.2f}")
        print(f"- Cross-polarization coefficient: {mat_props[c.MATERIALS_PARAM_CROSS_POL_COEF]:.2f}")

    print("\n[TX/RX Configuration]")

    # Sum total number of receivers and transmitters
    n_rx = sum(set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS] for set_info in txrx_params.values()
            if set_info[c.TXRX_PARAM_IS_RX])
    n_tx = sum(set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS] for set_info in txrx_params.values()
            if set_info[c.TXRX_PARAM_IS_TX])
    print(f"Total number of receivers: {n_rx}")
    print(f"Total number of transmitters: {n_tx}")

    for set_name, set_info in txrx_params.items():
        print(f"\n{set_name} ({set_info[c.TXRX_PARAM_NAME_FIELD]}):")
        role = []
        if set_info[c.TXRX_PARAM_IS_TX]: role.append("TX")
        if set_info[c.TXRX_PARAM_IS_RX]: role.append("RX")
        print(f"- Role: {' & '.join(role)}")
        print(f"- Total points: {set_info[c.TXRX_PARAM_NUM_POINTS]:,}")
        print(f"- Active points: {set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS]:,}")
        print(f"- Antennas per point: {set_info[c.TXRX_PARAM_NUM_ANT]}")
        print(f"- Dual polarization: {set_info[c.TXRX_PARAM_DUAL_POL]}")

    print(f"\n[Version]")
    print(f"- DeepMIMO Version: {params_dict[c.VERSION_PARAM_NAME]}")


