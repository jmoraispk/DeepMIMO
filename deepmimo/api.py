"""
API utility functions for the DeepMIMO dataset generation.

This module provides functions for uploading and downloading DeepMIMO scenarios
from the DeepMIMO server.
"""

import os
import shutil
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
    load_dict_from_json,
    summary,
    zip,
    unzip
)

# Headers for HTTP requests
HEADERS = {
    'User-Agent': 'DeepMIMO-Python/1.0',
    'Accept': '*/*'
}

def _dm_upload_api_call(link: str, file: str, key: str) -> Optional[Dict]:
    """Upload a file to the DeepMIMO API server.
    
    Args:
        link (str): API endpoint URL
        file (str): Path to file to upload
        key (str): API authentication key
        
    Returns:
        Optional[Dict]: Response data containing filename and fileId if successful,
                       None if upload fails
        
    Notes:
        Uses chunked upload with progress bar for large files.
        Handles file upload only, no longer returns direct download URLs.
    """
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

        return {
            "filename": filename,
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


def _process_params_data(params_dict: Dict, extra_metadata: Optional[Dict] = None) -> Dict:
    """Process params.mat data into submission format - used in DeepMIMO database.

    Args:
        params_dict: Dictionary containing parsed params.mat data
        extra_metadata: Optional dictionary with additional metadata fields

    Returns:
        Processed parameters in submission format
    """
    rt_params = params_dict.get(c.RT_PARAMS_PARAM_NAME, {})
    txrx_sets = params_dict.get(c.TXRX_PARAM_NAME, {})
    scene_params = params_dict.get(c.SCENE_PARAM_NAME, {})

    # Convert frequency from Hz to GHz
    frequency = float(rt_params.get("frequency", 3.5e9)) / 1e9

    # Count total Tx and Rx
    num_tx = sum(set_info.get("num_active_points", 0)
                 for set_info in txrx_sets.values()
                 if set_info.get("is_tx")) or 1
    
    num_rx = sum(set_info.get("num_active_points", 0)
                 for set_info in txrx_sets.values()
                 if set_info.get("is_rx")) or 1

    raytracer_map = {
        c.RAYTRACER_NAME_WIRELESS_INSITE: "Insite",
        c.RAYTRACER_NAME_SIONNA: "Sionna",
        c.RAYTRACER_NAME_AODT: "AODT",
    }

    # Create base parameter dictionaries
    primary_params = {
        "bands": {
            "sub6": frequency >= 0 and frequency < 6,
            "mmW": frequency >= 6 and frequency <= 100,
            "subTHz": frequency > 100,
        },
        "numRx": num_rx,
        "maxReflections": rt_params.get("max_reflections", 1),
        "raytracerName": raytracer_map.get(rt_params.get("raytracer_name"), "Insite"),
        "environment": "outdoor",
    }

    advanced_params = {
        "dmVersion": params_dict.get("version", "4.0.0a"),
        "numTx": num_tx,
        "multiRxAnt": any(set_info.get("num_ant", 0) > 1 for set_info in txrx_sets.values()
                           if set_info.get("is_rx")),
        "multiTxAnt": any(set_info.get("num_ant", 0) > 1 for set_info in txrx_sets.values()
                           if set_info.get("is_tx")),
        "dualPolarization": any(set_info.get("dual_pol", False)
                                for set_info in txrx_sets.values()),
        "BS2BS": any(set_info.get("is_tx") and set_info.get("is_rx")
                     for set_info in txrx_sets.values()) or None,
        "pathDepth": rt_params.get("max_path_depth", None),
        "diffraction": bool(rt_params.get("max_diffractions", 0)),
        "scattering": bool(rt_params.get("max_scattering", 0)),
        "transmission": bool(rt_params.get("max_transmissions", 0)),
        "numRays": rt_params.get("num_rays", 1000000),
        "city": None,
        "digitalTwin": False,
        "dynamic": scene_params.get("num_scenes", 1) > 1,
        "bbCoords": None
    }

    # Override with extra metadata if provided
    if extra_metadata:
        for param in extra_metadata:
            if param in primary_params:
                primary_params[param] = extra_metadata[param]
            elif param in advanced_params:
                advanced_params[param] = extra_metadata[param]

    return {
        "primaryParameters": primary_params,
        "advancedParameters": advanced_params
    }


def _generate_key_components(summary_str: str) -> Dict:
    """Generate key components sections from summary string.

    Args:
        summary_str: Summary string from scenario containing sections in [Section Name] format
                    followed by their descriptions

    Returns:
        Dictionary containing sections with their names and HTML-formatted descriptions
    """
    html_dict = {"sections": []}
    current_section = None
    current_lines = []
    
    for line in summary_str.split('\n'):
        line = line.strip()
        if not line or line.startswith('='):  # Skip empty lines and separator lines
            continue
            
        if line.startswith('[') and line.endswith(']'):
            # Process previous section if it exists
            if current_section:
                html_dict["sections"].append(_format_section(current_section, current_lines))
            
            # Start new section
            current_section = line[1:-1]
            current_lines = []
        elif current_section:
            current_lines.append(line)
    
    # Add the final section
    if current_section:
        html_dict["sections"].append(_format_section(current_section, current_lines))

    return html_dict

def _format_section(name: str, lines: list) -> dict:
    """Format a section's content into proper HTML with consistent styling.

    Args:
        name: Section name
        lines: List of content lines for the section

    Returns:
        Formatted section dictionary with name and HTML description
    """
    # Group content by subsections (lines starting with newline)
    subsections = []
    current_subsection = []
    
    for line in lines:
        if line and not line.startswith('-'):  # New subsection header
            if current_subsection:
                subsections.append(current_subsection)
            current_subsection = [line]
        elif line:  # Content line
            current_subsection.append(line)
    
    if current_subsection:
        subsections.append(current_subsection)

    # Build HTML content
    html_parts = []
    for subsection in subsections:
        if len(subsection) == 1:  # Single line - use paragraph
            html_parts.append(f"<p>{subsection[0]}</p>")
        else:  # Multiple lines - use header and list
            header = subsection[0]
            items = [line[2:] for line in subsection[1:]]  # Remove "- " prefix
            
            html_parts.append(f"<h4>{header}</h4>")
            html_parts.append("<ul>")
            html_parts.extend(f"<li>{item}</li>" for item in items)
            html_parts.append("</ul>")

    return {
        "name": name,
        "description": f"""
            <div class="section-content">
                {''.join(html_parts)}
            </div>
        """
    }

def upload(scenario_name: str, key: str, description: Optional[str] = None,
           details: Optional[list[str]] = None, extra_metadata: Optional[dict] = None, 
           skip_zip: bool = False) -> str:
    """Upload a DeepMIMO scenario to the server.

    Args:
        scenario_name: Scenario name
        key: Upload authorization key
        description: Optional description of the scenario. This is the subtitle.
        details: Optional list of details about the scenario. This is the detail boxes.
        extra_metadata: Optional dictionary containing additional metadata fields
                        (environment, digitalTwin, city, bbCoords, etc.)
            dict contents = {
                'digitalTwin': <boolean>,
                'environment': <enum> of 'indoor' or 'outdoor' (all lowercase),
                'bbCoords': {
                    "minLat": <float>,
                    "minLon": <float>,
                    "maxLat": <float>,
                    "maxLon": <float>
                }
                'city': <string>
            }
            This variable can be used to assign manual metadata to the scenario.
            Through this, submissions can be 100% automated (via code).

        skip_zip: Skip zipping the scenario folder if True

    Returns:
        Scenario name if successful, None otherwise
    """
    scen_folder = get_scenario_folder(scenario_name)
    params_path = get_params_path(scenario_name)

    print(f"Processing scenario: {scenario_name}")

    try:
        print("Parsing scenario parameters...")
        params_dict = load_dict_from_json(params_path)
        print("✓ Parameters parsed successfully")
    except Exception as e:
        print(f"Error: Failed to parse parameters - {str(e)}")
        return None
    
    try:
        # Process parameters and generate submission data
        processed_params = _process_params_data(params_dict, extra_metadata)
        key_components = _generate_key_components(summary(scenario_name, print_summary=False))
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
    zip_path = scen_folder + ".zip" if skip_zip else zip(scen_folder)

    try:
        print("Uploading to storage...")
        upload_result = _dm_upload_api_call("https://dev.deepmimo.net/api/b2/authorize-upload", 
                                            zip_path, key)
    except Exception as e:
        print(f"Error: Failed to upload to storage - {str(e)}")

    if not upload_result or "filename" not in upload_result:
        raise RuntimeError("Failed to upload to B2")
    print("✓ Upload successful")
    
    submission_data["download"] = [
        {
            "version": f"v{processed_params['advancedParameters']['dmVersion']}",
            "description": "Initial version",
            "zip": upload_result["filename"],  # Store only the filename, not the URL
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


        submission_result = scenario_name
  
        print('Thank you for your submission!')
        print('Head over to deepmimo.net/dashboard?tab=submissions to monitor it.')
        print('The admins have been notified and will get to it ASAP.')
    
    except Exception as e:
        print(f"Error: Upload failed - {str(e)}")
        submission_result = None

    return submission_result


def _download_url(scenario_name: str) -> str:
    """Get the secure download endpoint URL for a DeepMIMO scenario.

    Args:
        scenario_name: Name of the scenario ZIP file

    Returns:
        Secure URL for downloading the scenario through the API endpoint

    Raises:
        ValueError: If scenario name is invalid
        RuntimeError: If server returns error
    """
    if not scenario_name.endswith(".zip"):
        scenario_name += ".zip"

    # Return the secure download endpoint URL with the filename as a parameter
    return f"https://dev.deepmimo.net/api/download/secure?filename={scenario_name}"


def download(scenario_name: str, output_dir: Optional[str] = None) -> Optional[str]:
    """Download a DeepMIMO scenario from B2 storage.

    Args:
        scenario_name: Name of the scenario
        output_dir: Directory to save file (defaults to current directory)

    Returns:
        Path to downloaded file if successful, None otherwise
    """
    scenarios_dir = get_scenarios_dir()
    download_dir = output_dir if output_dir else get_downloads_dir()
    scenario_folder = get_scenario_folder(scenario_name)
    
    # DEV NOTE: when adding new scenario versions, change this check to read the version number
    #           and ask for compatibility with the current version of DeepMIMO.
    #           This may require downloading the zip again if the version is not compatible.

    # Check if file already exists in scenarios folder
    if os.path.exists(scenario_folder):
        print(f'Scenario "{scenario_name}" already exists in {scenarios_dir}')
        return None

    # Get secure download URL using existing helper
    url = _download_url(scenario_name)
    
    output_path = os.path.join(download_dir, f"{scenario_name}_downloaded.zip")

    # Check if file already exists in download folder
    if not os.path.exists(output_path):
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)

        print(f"Downloading scenario '{scenario_name}'")
        try:
            # Get download token and redirect URL
            resp = requests.get(url, headers=HEADERS)
            resp.raise_for_status()
            token_data = resp.json()
            
            if "error" in token_data:
                print(f"Server error: {token_data.get('error')}")
                return None
            
            # Get and format redirect URL
            redirect_url = token_data.get('redirectUrl')
            if not redirect_url:
                print("Error: Missing redirect URL")
                return None
                
            if not redirect_url.startswith('http'):
                redirect_url = f"{url.split('/api/')[0]}{redirect_url}"
            
            # Download the file
            download_resp = requests.get(redirect_url, stream=True, headers=HEADERS)
            download_resp.raise_for_status()
            total_size = int(download_resp.headers.get("content-length", 0))

            with open(output_path, "wb") as file, \
                 tqdm(total=total_size, unit='B', unit_scale=True, 
                      unit_divisor=1024, desc="Downloading") as pbar:
                for chunk in download_resp.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✓ Downloaded to {output_path}")

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)  # Clean up partial download
            return None
    else: # Extract the zip if it exists, don't download again
        print(f'Scenario zip file "{output_path}" already exists.')
    
    # Unzip downloaded scenario
    unzipped_folder = unzip(output_path)

    # Move unzipped folder to scenarios folder
    unzipped_folder_without_suffix = unzipped_folder.replace('_downloaded', '')
    os.makedirs(scenarios_dir, exist_ok=True)
    os.rename(unzipped_folder, unzipped_folder_without_suffix)
    shutil.move(unzipped_folder_without_suffix, scenario_folder)
    print(f"✓ Unzipped and moved to {scenarios_dir}")

    print(f"✓ Scenario '{scenario_name}' ready to use!")

    return output_path 

def search(query: Dict) -> Optional[Dict]:
    """
    Search for scenarios in the DeepMIMO database.

    Args:
        query: Dictionary containing search parameters from the following list:
            - bands: List[str] - Array of frequency bands ['sub6', 'mmW', 'subTHz']
            - raytracerName: str - Raytracer name or 'all'
            - environment: str - 'indoor', 'outdoor', or 'all'
            - numTx: Dict - Numeric range filter {'min': number, 'max': number}
            - numRx: Dict - Numeric range filter {'min': number, 'max': number}
            - pathDepth: Dict - Numeric range filter {'min': number, 'max': number}
            - maxReflections: Dict - Numeric range filter {'min': number, 'max': number}
            - numRays: Dict - Numeric range filter {'min': number, 'max': number}
            - multiRxAnt: bool - Boolean filter or 'all' to ignore
            - multiTxAnt: bool - Boolean filter or 'all' to ignore
            - dualPolarization: bool - Boolean filter or 'all' to ignore
            - BS2BS: bool - Boolean filter or 'all' to ignore
            - dynamic: bool - Boolean filter or 'all' to ignore
            - diffraction: bool - Boolean filter or 'all' to ignore
            - scattering: bool - Boolean filter or 'all' to ignore
            - transmission: bool - Boolean filter or 'all' to ignore
            - digitalTwin: bool - Boolean filter or 'all' to ignore
            - city: str - City name text filter
            - bbCoords: Dict - Bounding box coordinates {'minLat': float, 'minLon': float, 'maxLat': float, 'maxLon': float}
    
    Returns:
        Dict containing count and list of matching scenario names if successful, None otherwise
    """
    try:
        response = requests.post('https://dev.deepmimo.net/api/search/scenarios', json=query)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if hasattr(e.response, 'text'):
            try:
                error_data = e.response.json()
                print(f"Server error details: {error_data.get('message', e.response.text)}")
            except:
                print(f"Server response: {e.response.text}")
        return None
    except requests.exceptions.ConnectionError:
        print("Error: Connection failed. Please check your internet connection and try again.")
        return None
    except requests.exceptions.Timeout:
        print("Error: Request timed out. Please try again later.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {str(e)}")
        return None
    except ValueError as e:
        print(f"Error parsing response: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

