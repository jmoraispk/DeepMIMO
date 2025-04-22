"""
General utility functions and classes for the DeepMIMO dataset generation.

This module provides utility functions and classes for handling printing,
file naming, string ID generation, and dictionary utilities used across 
the DeepMIMO toolkit.
"""

import numpy as np
from pprint import pformat
from typing import Dict, Any, TypeVar, Mapping, Optional
from . import consts as c
import os
from tqdm import tqdm
import zipfile
import json
from .config import config

K = TypeVar("K", bound=str)
V = TypeVar("V")

# Headers for HTTP requests
HEADERS = {
    'User-Agent': 'DeepMIMO-Python/1.0',
    'Accept': '*/*'
}

def check_scen_name(scen_name: str) -> None:
    """Check if a scenario name is valid.
    
    Args:
        scen_name (str): The scenario name to check
    
    """
    if np.any([char in scen_name for char in c.SCENARIO_NAME_INVALID_CHARS]):
        raise ValueError(f"Invalid scenario name: {scen_name}.\n"
                         f"Contains one of the following invalid characters: {c.SCENARIO_NAME_INVALID_CHARS}")
    return 

def get_scenarios_dir() -> str:
    """Get the absolute path to the scenarios directory.
    
    This directory contains the extracted scenario folders ready for use.
    
    Returns:
        str: Absolute path to the scenarios directory
    """
    return os.path.join(os.getcwd(), config.get('scenarios_folder'))

def get_scenario_folder(scenario_name: str) -> str:
    """Get the absolute path to a specific scenario folder.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        str: Absolute path to the scenario folder
    """
    check_scen_name(scenario_name)
    return os.path.join(get_scenarios_dir(), scenario_name)

def get_params_path(scenario_name: str) -> str:
    """Get the absolute path to a scenario's params file.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        str: Absolute path to the scenario's params file
    """
    check_scen_name(scenario_name)
    return os.path.join(get_scenario_folder(scenario_name), f'{c.PARAMS_FILENAME}.json')

def save_dict_as_json(output_path: str, data_dict: Dict[str, Any]) -> None:
    """Save dictionary as JSON, handling NumPy arrays and other non-JSON types.
    
    Args:
        output_path: Path to save JSON file
        data_dict: Dictionary to save
    """
    numpy_handler = lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)
    with open(output_path, 'w') as f:
        json.dump(data_dict, f, indent=2, default=numpy_handler)

def load_dict_from_json(file_path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Dictionary containing loaded data
    """
    with open(file_path, 'r') as f:
        return json.load(f)

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
        if key == "_data":
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

    def __delitem__(self, key: str) -> None:
        """Enable dictionary-style deletion."""
        del self._data[key]

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
        return list(set(list(super().__dir__()) + list(self._data.keys())))

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

    def deepcopy(self) -> 'DotDict':
        """Create a deep copy of the DotDict instance.
        
        This method creates a completely independent copy of the DotDict,
        including nested dictionaries and numpy arrays. This ensures that
        modifications to the copy won't affect the original.
        
        Returns:
            DotDict: A deep copy of this instance
        """
        result = {}
        for key, value in self._data.items():
            if isinstance(value, DotDict):
                result[key] = value.deepcopy()
            elif isinstance(value, dict):
                result[key] = DotDict(value).deepcopy()
            elif isinstance(value, np.ndarray):
                result[key] = value.copy()
            else:
                result[key] = value
        return type(self)(result)  # Use the same class type as self

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
    return f"t{tx_set_idx:03}_tx{tx_idx:03}_r{rx_set_idx:03}"


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
    return f"{key}_{str_id}.mat"


def summary(scen_name: str, print_summary: bool = True) -> Optional[str]:
    """Print a summary of the dataset."""
    # Initialize empty string to collect output
    summary_str = ""

    # Read params.mat and provide TXRX summary, total number of tx & rx, scene size,
    # and other relevant parameters, computed/extracted from the all dicts, not just rt_params

    params_json_path = get_params_path(scen_name)

    params_dict = load_dict_from_json(params_json_path)
    rt_params = params_dict[c.RT_PARAMS_PARAM_NAME]
    scene_params = params_dict[c.SCENE_PARAM_NAME]
    material_params = params_dict[c.MATERIALS_PARAM_NAME]
    txrx_params = params_dict[c.TXRX_PARAM_NAME]

    summary_str += "\n" + "=" * 50 + "\n"
    summary_str += f"DeepMIMO {scen_name} Scenario Summary\n"
    summary_str += "=" * 50 + "\n"

    summary_str += "\n[Ray-Tracing Configuration]\n"
    summary_str += (
        f"- Ray-tracer: {rt_params[c.RT_PARAM_RAYTRACER]} "
        f"v{rt_params[c.RT_PARAM_RAYTRACER_VERSION]}\n"
    )
    summary_str += f"- Frequency: {rt_params[c.RT_PARAM_FREQUENCY]/1e9:.1f} GHz\n"

    summary_str += "\n[Ray-tracing parameters]\n"

    # Interaction limits
    summary_str += "\nMain interaction limits\n"
    summary_str += f"- Max path depth: {rt_params[c.RT_PARAM_PATH_DEPTH]}\n"
    summary_str += f"- Max reflections: {rt_params[c.RT_PARAM_MAX_REFLECTIONS]}\n"
    summary_str += f"- Max diffractions: {rt_params[c.RT_PARAM_MAX_DIFFRACTIONS]}\n"
    summary_str += f"- Max scatterings: {rt_params[c.RT_PARAM_MAX_SCATTERING]}\n"
    summary_str += f"- Max transmissions: {rt_params[c.RT_PARAM_MAX_TRANSMISSIONS]}\n"

    # Diffuse scattering settings
    summary_str += "\nDiffuse Scattering\n"
    is_diffuse_enabled = rt_params[c.RT_PARAM_MAX_SCATTERING] > 0
    summary_str += f"- Diffuse scattering: {'Enabled' if is_diffuse_enabled else 'Disabled'}\n"
    if is_diffuse_enabled:
        summary_str += f"- Diffuse reflections: {rt_params[c.RT_PARAM_DIFFUSE_REFLECTIONS]}\n"
        summary_str += f"- Diffuse diffractions: {rt_params[c.RT_PARAM_DIFFUSE_DIFFRACTIONS]}\n"
        summary_str += f"- Diffuse transmissions: {rt_params[c.RT_PARAM_DIFFUSE_TRANSMISSIONS]}\n"
        summary_str += f"- Final interaction only: {rt_params[c.RT_PARAM_DIFFUSE_FINAL_ONLY]}\n"
        summary_str += f"- Random phases: {rt_params[c.RT_PARAM_DIFFUSE_RANDOM_PHASES]}\n"

    # Terrain settings
    summary_str += "\nTerrain\n"
    summary_str += f"- Terrain reflection: {rt_params[c.RT_PARAM_TERRAIN_REFLECTION]}\n"
    summary_str += f"- Terrain diffraction: {rt_params[c.RT_PARAM_TERRAIN_DIFFRACTION]}\n"
    summary_str += f"- Terrain scattering: {rt_params[c.RT_PARAM_TERRAIN_SCATTERING]}\n"

    # Ray casting settings
    summary_str += "\nRay Casting Settings\n"
    summary_str += f"- Number of rays: {rt_params[c.RT_PARAM_NUM_RAYS]:,}\n"
    summary_str += f"- Casting method: {rt_params[c.RT_PARAM_RAY_CASTING_METHOD]}\n"
    summary_str += f"- Casting range (az): {rt_params[c.RT_PARAM_RAY_CASTING_RANGE_AZ]:.1f}°\n"
    summary_str += f"- Casting range (el): {rt_params[c.RT_PARAM_RAY_CASTING_RANGE_EL]:.1f}°\n"
    summary_str += f"- Synthetic array: {rt_params[c.RT_PARAM_SYNTHETIC_ARRAY]}\n"

    # Scene
    summary_str += "\n[Scene]\n"
    summary_str += f"- Number of scenes: {scene_params[c.SCENE_PARAM_NUMBER_SCENES]}\n"
    summary_str += f"- Total objects: {scene_params[c.SCENE_PARAM_N_OBJECTS]:,}\n"
    summary_str += f"- Vertices: {scene_params[c.SCENE_PARAM_N_VERTICES]:,}\n"
    summary_str += f"- Faces: {scene_params[c.SCENE_PARAM_N_FACES]:,}\n"
    summary_str += f"- Triangular faces: {scene_params[c.SCENE_PARAM_N_TRIANGULAR_FACES]:,}\n"

    # Materials
    summary_str += "\n[Materials]\n"
    summary_str += f"Total materials: {len(material_params)}\n"
    for _, mat_props in material_params.items():
        summary_str += f"\n{mat_props[c.MATERIALS_PARAM_NAME_FIELD]}:\n"
        summary_str += f"- Permittivity: {mat_props[c.MATERIALS_PARAM_PERMITTIVITY]:.2f}\n"
        summary_str += f"- Conductivity: {mat_props[c.MATERIALS_PARAM_CONDUCTIVITY]:.2f} S/m\n"
        summary_str += f"- Scattering model: {mat_props[c.MATERIALS_PARAM_SCATTERING_MODEL]}\n"
        summary_str += f"- Scattering coefficient: {mat_props[c.MATERIALS_PARAM_SCATTERING_COEF]:.2f}\n"
        summary_str += f"- Cross-polarization coefficient: {mat_props[c.MATERIALS_PARAM_CROSS_POL_COEF]:.2f}\n"

    # TX/RX
    summary_str += "\n[TX/RX Configuration]\n"

    # Sum total number of receivers and transmitters
    n_rx = sum(
        set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS]
        for set_info in txrx_params.values()
        if set_info[c.TXRX_PARAM_IS_RX]
    )
    n_tx = sum(
        set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS]
        for set_info in txrx_params.values()
        if set_info[c.TXRX_PARAM_IS_TX]
    )
    summary_str += f"Total number of receivers: {n_rx}\n"
    summary_str += f"Total number of transmitters: {n_tx}\n"

    for set_name, set_info in txrx_params.items():
        summary_str += f"\n{set_name} ({set_info[c.TXRX_PARAM_NAME_FIELD]}):\n"
        role = []
        if set_info[c.TXRX_PARAM_IS_TX]:
            role.append("TX")
        if set_info[c.TXRX_PARAM_IS_RX]:
            role.append("RX")
        summary_str += f"- Role: {' & '.join(role)}\n"
        summary_str += f"- Total points: {set_info[c.TXRX_PARAM_NUM_POINTS]:,}\n"
        summary_str += f"- Active points: {set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS]:,}\n"
        summary_str += f"- Antennas per point: {set_info[c.TXRX_PARAM_NUM_ANT]}\n"
        summary_str += f"- Dual polarization: {set_info[c.TXRX_PARAM_DUAL_POL]}\n"

    # GPS Bounding Box
    if rt_params[c.RT_PARAM_GPS_BBOX] != (0,0,0,0):
        summary_str += "\n[GPS Bounding Box]\n"
        summary_str += f"- Min latitude: {rt_params[c.RT_PARAM_GPS_BBOX][0]:.2f}\n"
        summary_str += f"- Min longitude: {rt_params[c.RT_PARAM_GPS_BBOX][1]:.2f}\n"
        summary_str += f"- Max latitude: {rt_params[c.RT_PARAM_GPS_BBOX][2]:.2f}\n"
        summary_str += f"- Max longitude: {rt_params[c.RT_PARAM_GPS_BBOX][3]:.2f}\n"

    # Print summary
    if print_summary:
        print(summary_str)
        return None
    
    return summary_str


def zip(folder_path: str) -> str:
    """Create zip archive of folder contents.

    This function creates a zip archive containing all files and subdirectories in the 
    specified folder. The archive is created in the same directory as the folder with
    '.zip' appended to the folder name. The directory structure is preserved in the zip.

    Args:
        folder_path (str): Path to folder to be zipped

    Returns:
        Path to the created zip file
    """
    zip_path = folder_path + ".zip"
    
    # Get all files and folders recursively
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Get full path of file
            file_path = os.path.join(root, file)
            # Get relative path from the base folder for preserving structure
            rel_path = os.path.relpath(file_path, os.path.dirname(folder_path))
            all_files.append((file_path, rel_path))

    # Create a zip file
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path, rel_path in tqdm(all_files, desc="Compressing", unit="file"):
            zipf.write(file_path, rel_path)

    return zip_path


def unzip(path_to_zip: str) -> str:
    """Extract a zip file to its parent directory.

    This function extracts the contents of a zip file to the directory
    containing the zip file.

    Args:
        path_to_zip (str): Path to the zip file to extract.

    Raises:
        zipfile.BadZipFile: If zip file is corrupted.
        OSError: If extraction fails due to file system issues.

    Returns:
        Path to the extracted folder
    """
    extracted_path = path_to_zip.replace(".zip", "")
    with zipfile.ZipFile(path_to_zip, "r") as zip_ref:
        files = zip_ref.namelist()
        for file in tqdm(files, desc="Extracting", unit="file"):
            zip_ref.extract(file, extracted_path)

    return extracted_path


def compare_two_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> bool:
    """Compare two dictionaries for equality.
            
    This function performs a deep comparison of two dictionaries, handling
    nested dictionaries.
    
    Args:
        dict1 (dict): First dictionary to compare
        dict2 (dict): Second dictionary to compare

    Returns:
        set: Set of keys in dict1 that are not in dict2
    """
    additional_keys = dict1.keys() - dict2.keys()
    for key, item in dict1.items():
        if isinstance(item, dict):
            if key in dict2:
                additional_keys = additional_keys | compare_two_dicts(dict1[key], dict2[key])
    return additional_keys


def get_available_scenarios() -> list:
    """Get a list of all available scenarios in the scenarios directory.
    
    Returns:
        list: List of scenario names (folder names in the scenarios directory)
    """
    scenarios_dir = get_scenarios_dir()
    if not os.path.exists(scenarios_dir):
        return []
    
    # Get all subdirectories in the scenarios folder
    scenarios = [f for f in os.listdir(scenarios_dir) 
                if os.path.isdir(os.path.join(scenarios_dir, f))]
    return sorted(scenarios)

