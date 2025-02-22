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
import os
import requests
from tqdm import tqdm
import hashlib
import zipfile
import shutil

K = TypeVar("K", bound=str)
V = TypeVar("V")

# Headers for HTTP requests
HEADERS = {
    'User-Agent': 'DeepMIMO-Python/1.0',
    'Accept': '*/*'
}

def get_scenarios_dir() -> str:
    """Get the absolute path to the scenarios directory.
    
    This directory contains the extracted scenario folders ready for use.
    
    Returns:
        str: Absolute path to the scenarios directory
    """
    return os.path.join(os.getcwd(), c.SCENARIOS_FOLDER)

def get_downloads_dir() -> str:
    """Get the absolute path to the downloads directory.
    
    This directory contains downloaded scenario ZIP files before extraction.
    
    Returns:
        str: Absolute path to the downloads directory
    """
    return os.path.join(os.getcwd(), c.SCENARIOS_DOWNLOAD_FOLDER)

def get_scenario_folder(scenario_name: str) -> str:
    """Get the absolute path to a specific scenario folder.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        str: Absolute path to the scenario folder
    """
    return os.path.join(get_scenarios_dir(), scenario_name)

def get_params_path(scenario_name: str) -> str:
    """Get the absolute path to a scenario's params.mat file.
    
    Args:
        scenario_name: Name of the scenario
        
    Returns:
        str: Absolute path to the scenario's params.mat file
    """
    return os.path.join(get_scenario_folder(scenario_name), f'{c.PARAMS_FILENAME}.mat')

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

    @property
    def shape(self):
        """Return shape of the first array-like value in the dictionary."""
        for val in self._data.values():
            if hasattr(val, "shape"):
                return val.shape
        return None

    @property
    def size(self):
        """Return size of the first array-like value in the dictionary."""
        for val in self._data.values():
            if hasattr(val, "size"):
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
    return {
        key: mat_struct_to_dict(value)
        for key, value in mat_data.items()
        if not key.startswith("__")
    }


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


def summary(scen_name: str, print_summary: bool = True) -> Optional[str]:
    """Print a summary of the dataset."""
    # Initialize empty string to collect output
    summary_str = ""

    # Read params.mat and provide TXRX summary, total number of tx & rx, scene size,
    # and other relevant parameters, computed/extracted from the all dicts, not just rt_params

    scen_folder = c.SCENARIOS_FOLDER + "/" + scen_name

    mat_file = f"./{scen_folder}/{c.PARAMS_FILENAME}.mat"

    params_dict = load_mat_file_as_dict(mat_file)[c.PARAMS_FILENAME]
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

    summary_str += "\n[Scene]\n"
    summary_str += f"- Number of scenes: {scene_params[c.SCENE_PARAM_NUMBER_SCENES]}\n"
    summary_str += f"- Total objects: {scene_params[c.SCENE_PARAM_N_OBJECTS]}\n"
    # TODO: Put object label summary into scene dict (no. buildings, trees, etc)
    summary_str += f"- Vertices: {scene_params[c.SCENE_PARAM_N_VERTICES]}\n"
    # TODO: Put normal face count into scene dict
    normal_faces = 332
    summary_str += (
        f"- Faces: {normal_faces:,} (decomposed into {scene_params[c.SCENE_PARAM_N_TRIANGULAR_FACES]:,} triangular faces)\n"
    )

    # Get scene boundaries from scene bounding box
    # TODO: print this into the scene dict
    # bbox = scene.bounding_box
    # print("\nBoundaries:")
    # print(f"- X: {bbox.x_min:.2f}m to {bbox.x_max:.2f}m (width: {bbox.width:.2f}m)")
    # print(f"- Y: {bbox.y_min:.2f}m to {bbox.y_max:.2f}m (length: {bbox.length:.2f}m)")
    # print(f"- Z: {bbox.z_min:.2f}m to {bbox.z_max:.2f}m (height: {bbox.height:.2f}m)")
    # print(f"- Area: {bbox.width * bbox.length:,.2f}m²")

    summary_str += "\n[Materials]\n"
    summary_str += f"Total materials: {len(material_params)}\n"
    for _, mat_props in material_params.items():
        summary_str += f"\n{mat_props[c.MATERIALS_PARAM_NAME_FIELD]}:\n"
        summary_str += f"- Permittivity: {mat_props[c.MATERIALS_PARAM_PERMITTIVITY]:.2f}\n"
        summary_str += f"- Conductivity: {mat_props[c.MATERIALS_PARAM_CONDUCTIVITY]:.2f} S/m\n"
        summary_str += f"- Scattering model: {mat_props[c.MATERIALS_PARAM_SCATTERING_MODEL]}\n"
        summary_str += (
            f"- Scattering coefficient: {mat_props[c.MATERIALS_PARAM_SCATTERING_COEF]:.2f}\n"
        )
        summary_str += (
            f"- Cross-polarization coefficient: {mat_props[c.MATERIALS_PARAM_CROSS_POL_COEF]:.2f}\n"
        )

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

    summary_str += f"\n[Version]\n"
    summary_str += f"- DeepMIMO Version: {params_dict[c.VERSION_PARAM_NAME]}\n"
    
    if print_summary:
        print(summary_str)
        return None
    
    return summary_str


# Upload and Download Scenarios

def _dm_upload_api_call(link: str, file: str, key: str) -> Optional[Dict]:
    """Upload file to server endpoint with progress bar."""
    try:
        # Get file info first
        filename = os.path.basename(file)
        file_size = os.path.getsize(file)

        # First check if file exists on B2
        check_response = requests.get(
            "https://dev.deepmimo.net/api/b2/check-filename",
            params={"filename": filename},
            headers={"Authorization": f"Bearer {key}"},
        )
        check_response.raise_for_status()

        if check_response.json().get("exists"):
            print(f"Error: File {filename} already exists on B2")
            return None

        # Get upload authorization
        auth_response = requests.get(
            "https://dev.deepmimo.net/api/b2/authorize-upload",
            headers={"Authorization": f"Bearer {key}"},
        )
        auth_response.raise_for_status()
        auth_data = auth_response.json()

        if not auth_data.get("uploadUrl") or not auth_data.get("authorizationToken"):
            print("Error: Invalid authorization response")
            return None

        # Calculate file hash
        sha1 = hashlib.sha1()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha1.update(chunk)
        file_hash = sha1.hexdigest()

        # Upload file to B2
        print(f"Uploading {filename} to B2...")
        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading")
        
        class ProgressFileReader:
            def __init__(self, file_path, progress_bar):
                self.file_path = file_path
                self.progress_bar = progress_bar
                self.file_object = open(file_path, 'rb')
                self.len = os.path.getsize(file_path)
                self.bytes_read = 0

            def read(self, size=-1):
                data = self.file_object.read(size)
                self.bytes_read += len(data)
                self.progress_bar.n = self.bytes_read
                self.progress_bar.refresh()
                return data

            def close(self):
                self.file_object.close()

        try:
            progress_reader = ProgressFileReader(file, pbar)
            
            upload_response = requests.post(
                auth_data["uploadUrl"],
                headers={
                    "Authorization": auth_data["authorizationToken"],
                    "X-Bz-File-Name": filename,
                    "Content-Type": "application/zip",
                    "X-Bz-Content-Sha1": file_hash,
                    "Content-Length": str(file_size),
                },
                data=progress_reader
            )
            upload_response.raise_for_status()
        finally:
            progress_reader.close()
            pbar.close()

        # Get the proper download URL from the server
        download_response = requests.get(
            "https://dev.deepmimo.net/api/b2/download-url",
            params={"filename": filename},
            headers={"Authorization": f"Bearer {key}"},
        )
        download_response.raise_for_status()

        return {
            "downloadUrl": download_response.json()["downloadUrl"],
            "fileId": upload_response.json().get("fileId"),
        }

    except requests.exceptions.RequestException as e:
        print(f"API call failed: {str(e)}")
        if hasattr(e.response, "text"):
            print(f"Server response: {e.response.text}")
        return None
    except Exception as e:
        print(f"Upload failed: {str(e)}")
        return None


def _process_params_data(params_dict: Dict) -> Dict:
    """Process params.mat data into submission format - used in DeepMIMO database.

    Args:
        params_dict: Dictionary containing parsed params.mat data

    Returns:
        Processed parameters in submission format
    """
    params = params_dict.get("params", {})
    rt_params = params.get("rt_params", {})
    txrx_sets = params.get("txrx", {})
    scene_params = params.get("scene", {})

    # Convert frequency from Hz to GHz
    frequency = float(rt_params.get("frequency", 3.5e9)) / 1e9

    # Count total Tx and Rx
    num_tx = (
        sum(
            set_info.get("num_active_points", 0)
            for set_info in txrx_sets.values()
            if set_info.get("is_tx")
        )
        or 1
    )
    num_rx = (
        sum(
            set_info.get("num_active_points", 0)
            for set_info in txrx_sets.values()
            if set_info.get("is_rx")
        )
        or 1
    )

    return {
        "primaryParameters": {
            "bands": {
                "sub6": frequency >= 0 and frequency < 6,
                "mmW": frequency >= 6 and frequency <= 100,
                "subTHz": frequency > 100,
            },
            "numRx": num_rx,
            "maxReflections": rt_params.get("max_reflections", 1),
            "raytracerName": rt_params.get("raytracer_name", "Insite"),
            "environment": "outdoor",
        },
        "advancedParameters": {
            "dmVersion": params.get("version", "4.0.0a"),
            "numTx": num_tx,
            "multiRxAnt": any(
                set_info.get("num_ant", 0) > 1
                for set_info in txrx_sets.values()
                if set_info.get("is_rx")
            ),
            "multiTxAnt": any(
                set_info.get("num_ant", 0) > 1
                for set_info in txrx_sets.values()
                if set_info.get("is_tx")
            ),
            "dualPolarization": any(
                set_info.get("dual_pol", False) for set_info in txrx_sets.values()
            ),
            "BS2BS": any(
                set_info.get("is_tx") and set_info.get("is_rx")
                for set_info in txrx_sets.values()
            ) or None,
            "pathDepth": rt_params.get("max_path_depth", None),
            "diffraction": bool(rt_params.get("max_diffractions", 0)),
            "scattering": bool(rt_params.get("max_scattering", 0)),
            "transmission": bool(rt_params.get("max_transmissions", 0)),
            "numRays": rt_params.get("num_rays", 1000000),
            "city": None,
            "digitalTwin": False,
            "dynamic": scene_params.get("num_scenes", 1) > 1
        }
    }


def _generate_key_components2(summary_str: str) -> Dict:
    """Generate key components sections from summary string.

    Args:
        summary_str: Summary string from scenario containing sections in [Section Name] format
                    followed by their descriptions

    Returns:
        Dictionary containing sections with their names and HTML-formatted descriptions
    """
    def create_section(name: str, desc_lines: list) -> dict:
        """Helper to create a section with consistent HTML formatting."""
        return {
            "name": name,
            "description": f"""
                <div class="section-content">
                    {' '.join(line.replace('- ', '<li>').replace(':', '</li>') 
                             for line in desc_lines)}
                </div>
            """
        }

    html_dict = {"sections": []}
    current_section = None
    current_description = []
    
    for line in [l.strip() for l in summary_str.split('\n') if l.strip()]:
        if line.startswith('[') and line.endswith(']'):
            if current_section:
                html_dict["sections"].append(create_section(current_section, current_description))
            current_section = line[1:-1]
            current_description = []
        elif current_section:
            current_description.append(line)
    
    # Add the last section if exists
    if current_section:
        html_dict["sections"].append(create_section(current_section, current_description))

    return html_dict


def _generate_key_components(params_dict: Dict) -> Dict:
    """Generate key components sections from params data.

    Args:
        params_dict: Dictionary containing parsed params.mat data

    Returns:
        Key components sections for submission
    """
    params = params_dict.get("params", {})
    rt_params = params.get("rt_params", {})
    txrx_sets = params.get("txrx", {})
    scene_params = params.get("scene", {})

    frequency = float(rt_params.get("frequency", 3.5e9)) / 1e9

    return {
        "sections": [
            {
                "name": "Ray-Tracing Configuration",
                "description": f"""
                    <p><strong>Ray-tracer:</strong> {rt_params.get(c.RT_PARAM_RAYTRACER, 'Unknown')} 
                    v{rt_params.get(c.RT_PARAM_RAYTRACER_VERSION, 'Unknown')}</p>
                    <p><strong>Frequency:</strong> {frequency:.1f} GHz</p>
                """,
            },
            {
                "name": "Ray-tracing parameters",
                "description": f"""
                    <h4>Main interaction limits:</h4>
                    <ul>
                        <li>Max path depth: {rt_params.get(c.RT_PARAM_PATH_DEPTH, 0)}</li>
                        <li>Max reflections: {rt_params.get(c.RT_PARAM_MAX_REFLECTIONS, 0)}</li>
                        <li>Max diffractions: {rt_params.get(c.RT_PARAM_MAX_DIFFRACTIONS, 0)}</li>
                        <li>Max scatterings: {rt_params.get(c.RT_PARAM_MAX_SCATTERING, 0)}</li>
                        <li>Max transmissions: {rt_params.get(c.RT_PARAM_MAX_TRANSMISSIONS, 0)}</li>
                    </ul>
                    <h4>Ray Casting Settings:</h4>
                    <ul>
                        <li>Number of rays: {rt_params.get(c.RT_PARAM_NUM_RAYS, 0):,}</li>
                        <li>Casting method: {rt_params.get(c.RT_PARAM_RAY_CASTING_METHOD, 'Unknown')}</li>
                        <li>Casting range (az): {rt_params.get(c.RT_PARAM_RAY_CASTING_RANGE_AZ, 0):.1f}°</li>
                        <li>Casting range (el): {rt_params.get(c.RT_PARAM_RAY_CASTING_RANGE_EL, 0):.1f}°</li>
                    </ul>
                """,
            },
            {
                "name": "Scene",
                "description": f"""
                    <ul>
                        <li>Number of scenes: {scene_params.get(c.SCENE_PARAM_NUMBER_SCENES, 1)}</li>
                        <li>Total objects: {scene_params.get(c.SCENE_PARAM_N_OBJECTS, 0)}</li>
                        <li>Vertices: {scene_params.get(c.SCENE_PARAM_N_VERTICES, 0)}</li>
                        <li>Faces: {scene_params.get(c.SCENE_PARAM_N_TRIANGULAR_FACES, 0)} triangular faces</li>
                    </ul>
                """,
            },
            {
                "name": "TX/RX Configuration",
                "description": f"""
                    <p><strong>Total number of receivers:</strong> {sum(set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS] 
                        for set_info in txrx_sets.values() if set_info[c.TXRX_PARAM_IS_RX])}</p>
                    <p><strong>Total number of transmitters:</strong> {sum(set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS] 
                        for set_info in txrx_sets.values() if set_info[c.TXRX_PARAM_IS_TX])}</p>
                    <h4>TX/RX Sets:</h4>
                    {"".join(f'''
                        <div class="txrx-set">
                            <h5>{set_name} ({set_info[c.TXRX_PARAM_NAME_FIELD]})</h5>
                            <ul>
                                <li>Role: {' & '.join(filter(None, ['TX' if set_info[c.TXRX_PARAM_IS_TX] else '', 
                                                                  'RX' if set_info[c.TXRX_PARAM_IS_RX] else '']))}</li>
                                <li>Total points: {set_info[c.TXRX_PARAM_NUM_POINTS]:,}</li>
                                <li>Active points: {set_info[c.TXRX_PARAM_NUM_ACTIVE_POINTS]:,}</li>
                                <li>Antennas per point: {set_info[c.TXRX_PARAM_NUM_ANT]}</li>
                                <li>Dual polarization: {set_info[c.TXRX_PARAM_DUAL_POL]}</li>
                            </ul>
                        </div>
                    ''' for set_name, set_info in txrx_sets.items())}
                """,
            },
            {
                "name": "Version",
                "description": f"""
                    <p><strong>DeepMIMO Version:</strong> {params.get(c.VERSION_PARAM_NAME, 'Unknown')}</p>
                """,
            },
        ]
    }


def upload(scenario_name: str, key: str) -> str:
    """Upload a DeepMIMO scenario to the server.

    Args:
        scenario_name: Path to scenario ZIP file
        key: Upload authorization key

    Returns:
        Download URL if successful, None otherwise
    """
    scen_folder = get_scenario_folder(scenario_name)
    params_path = get_params_path(scenario_name)

    print(f"Processing scenario: {scenario_name}")

    try:
        print("Parsing scenario parameters...")
        params_dict = load_mat_file_as_dict(params_path)
        print("✓ Parameters parsed successfully")
    except Exception as e:
        print(f"Error: Failed to parse parameters - {str(e)}")
        return None
    
    try:
        # Process parameters and generate submission data
        processed_params = _process_params_data(params_dict)
        key_components = _generate_key_components(params_dict)
    except Exception as e:
        print(f"Error: Failed to generate key components - {str(e)}")
        return None

    submission_data = {
        "title": scenario_name.replace("_", " ").replace("-", " ").title(),
        "linkName": scenario_name.replace("_", " ").replace("-", " ").title(),
        "subMenu": "v4",
        "description": f"A scenario for {scenario_name}",
        "details": None,
        "images": [],
        "keyComponents": key_components["sections"],
        "download": [],
        "features": processed_params["primaryParameters"],
        "advancedParameters": processed_params["advancedParameters"],
    }

    # Zip scenario
    zip_path = zip(scen_folder)

    try:
        print("Uploading to storage...")
        upload_result = _dm_upload_api_call("https://dev.deepmimo.net/api/b2/authorize-upload", 
                                            zip_path, key)
    except Exception as e:
        print(f"Error: Failed to upload to storage - {str(e)}")

    if not upload_result or "downloadUrl" not in upload_result:
        raise RuntimeError("Failed to upload to B2")
    print("✓ Upload successful")
    
    submission_data["download"] = [
        {
            "version": f"v{processed_params['advancedParameters']['dmVersion']}",
            "description": "Initial version",
            "zip": f"{upload_result['downloadUrl']}",
            "folder": "",
            "fileId": upload_result.get("fileId"),
        }
    ]

    print("Creating website submission...")
    try:
        response = requests.post(
            "https://dev.deepmimo.net/api/submissions",
            json={"type": "scenario", "content": submission_data},
            headers={"Authorization": f"Bearer {key}"},
        )
        response.raise_for_status()
        print("✓ Submission created successfully")

        result = submission_data["download"][0]["zip"]
    except Exception as e:
        print(f"Error: Upload failed - {str(e)}")
        result = None

    print('Thank you for your submission!')
    print('Head over to deepmimo.net/dashboard?tab=submissions to add any additional details. ')
    print('The admins have been notified and will get to it ASAP.')
    
    return result

def _download_url(scenario_name: str) -> str:
    """Get the download URL for a DeepMIMO scenario.

    Args:
        scenario_name: Name of the scenario ZIP file

    Returns:
        Public URL for downloading the scenario

    Raises:
        ValueError: If scenario name is invalid
        RuntimeError: If server returns error
    """
    if not scenario_name.endswith(".zip"):
        scenario_name += ".zip"

    try:
        # Get download URL from server
        response = requests.get(
            "https://dev.deepmimo.net/api/b2/download-url",
            params={"filename": scenario_name},
        )

        if response.status_code != 200:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        download_url = response.json().get("downloadUrl")
        if not download_url:
            raise RuntimeError("Invalid response from server")

        return download_url

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to get scenario URL: {str(e)}")


def download(scenario_name: str, output_dir: str = None) -> Optional[str]:
    """Download a DeepMIMO scenario from B2 storage.

    Args:
        scenario_name: Name of the scenario
        output_dir: Directory to save file (defaults to current directory)

    Returns:
        Path to downloaded file if successful, None otherwise
    """
    scenarios_dir = get_scenarios_dir()
    download_dir = get_downloads_dir()

    # TODO: when adding new scenario versions, change this check to read the version number
    #       and ask for compatibility with the current version of DeepMIMO.
    #       This may require downloading the zip again.
    # Check if file already exists in scenarios folder
    if os.path.exists(get_scenario_folder(scenario_name)):
        print(f'Scenario "{scenario_name}" already exists in {scenarios_dir}')
        return None

    try:
        # Get download URL using existing helper
        url = _download_url(scenario_name)
    except Exception as e:
        print(f"Error: Failed to get download URL - {str(e)}")
        return None
    
    output_path = os.path.join(download_dir, f"{scenario_name}_downloaded.zip")

    # Check if file already exists in download folder
    if not os.path.exists(output_path):
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        print(f"Downloading scenario '{scenario_name}'")
        try:
            response = requests.get(url, stream=True, headers=HEADERS)
            response.raise_for_status()

            # Get total file size for progress bar
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with open(output_path, "wb") as file:
                with tqdm(total=total_size, unit="B", unit_scale=True, 
                        unit_divisor=1024, dynamic_ncols=True) as progress_bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            progress_bar.update(len(chunk))

            print(f"✓ Downloaded to {output_path}")

        except Exception as e:
            print(f"Download failed: {str(e)}")
            return None
    else: # Extract the zip if it exists, don't download again
        print(f'Output path "{output_path}" already exists')
    
    # Unzip downloaded scenario
    unzipped_folder = unzip(output_path)

    # Move unzipped folder to scenarios folder
    unzipped_folder_without_suffix = unzipped_folder.replace('_downloaded', '')
    os.rename(unzipped_folder, unzipped_folder_without_suffix)
    shutil.move(unzipped_folder_without_suffix, scenarios_dir)
    print(f"✓ Unzipped and moved to {scenarios_dir}")

    print(f"✓ Scenario '{scenario_name}' ready to use!")

    return output_path

# Zip and Unzip

def zip(folder_path: str) -> str:
    """Create zip archive of folder contents.

    This function creates a zip archive containing all files in the specified
    folder. The archive is created in the same directory as the folder with
    '.zip' appended to the folder name.

    Args:
        folder_path (str): Path to folder to be zipped

    Returns:
        Path to the created zip file
    """
    files_in_folder = os.listdir(folder_path)
    file_full_paths = [os.path.join(folder_path, file) for file in files_in_folder]

    zip_path = folder_path + ".zip"
    # Create a zip file
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path in tqdm(file_full_paths, desc="Compressing", unit="file"):
            zipf.write(file_path, os.path.basename(file_path))

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


def zip_folder(folder_path: str) -> str:
    """Create a zip file from a folder.

    Args:
        folder_path: Path to folder to zip

    Returns:
        Path to created zip file

    Raises:
        ValueError: If folder doesn't exist
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder not found: {folder_path}")

    # Get absolute path and base name
    abs_path = os.path.abspath(folder_path)
    base_name = os.path.basename(abs_path.rstrip("/\\"))

    # Create zip in parent directory
    zip_path = os.path.join(os.path.dirname(abs_path), f"{base_name}.zip")

    # Remove existing zip if it exists
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # Create zip file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Walk through directory
        for root, dirs, files in os.walk(folder_path):
            # Calculate path relative to folder_path
            rel_path = os.path.relpath(root, folder_path)

            # Add each file
            for file in files:
                # Get full file path
                file_path = os.path.join(root, file)
                # Get archive path (relative to zip root)
                archive_path = os.path.join(base_name, rel_path, file)
                # Add file to zip
                zipf.write(file_path, archive_path)

    return zip_path

