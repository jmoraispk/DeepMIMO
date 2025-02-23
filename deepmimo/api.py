"""
API utility functions for the DeepMIMO dataset generation.

This module provides functions for uploading and downloading DeepMIMO scenarios
from the DeepMIMO server.
"""

import os
import requests
import hashlib
from tqdm import tqdm
from typing import Dict, Any, Optional
from . import consts as c
from .general_utilities import (
    get_scenarios_dir,
    get_downloads_dir,
    get_scenario_folder,
    get_params_path,
    load_mat_file_as_dict,
    zip,
    unzip
)

# Headers for HTTP requests
HEADERS = {
    'User-Agent': 'DeepMIMO-Python/1.0',
    'Accept': '*/*'
}

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
    params = params_dict.get(c.PARAMS_FILENAME, {})
    rt_params = params.get(c.RT_PARAMS_PARAM_NAME, {})
    txrx_sets = params.get(c.TXRX_PARAM_NAME, {})
    scene_params = params.get(c.SCENE_PARAM_NAME, {})

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


def upload(scenario_name: str, key: str, description: Optional[str] = None,
           details: Optional[list[str]] = None) -> str:
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
        "title": scenario_name,
        "linkName": scenario_name,
        "subMenu": "v4",
        "description": description if description else f"A scenario for {scenario_name}",
        "details": details,
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
    scenario_folder = get_scenario_folder(scenario_name)
    
    # TODO: when adding new scenario versions, change this check to read the version number
    #       and ask for compatibility with the current version of DeepMIMO.
    #       This may require downloading the zip again.
    # Check if file already exists in scenarios folder
    if os.path.exists(scenario_folder):
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
        print(f'Scenario zip file "{output_path}" already exists.')
    
    # Unzip downloaded scenario
    unzipped_folder = unzip(output_path)

    # Move unzipped folder to scenarios folder
    unzipped_folder_without_suffix = unzipped_folder.replace('_downloaded', '')
    os.rename(unzipped_folder, unzipped_folder_without_suffix)
    shutil.move(unzipped_folder_without_suffix, scenario_folder)
    print(f"✓ Unzipped and moved to {scenarios_dir}")

    print(f"✓ Scenario '{scenario_name}' ready to use!")

    return output_path 