"""
Wireless InSite Ray Tracing Pipeline.

This module provides a complete pipeline for generating electromagnetic simulation scenarios
using OpenStreetMap data and running ray tracing simulations with Wireless InSite.
It handles the entire process from OSM data extraction to DeepMIMO dataset generation.
"""

#%% Imports
from WI_interface.XmlGenerator import XmlGenerator
from WI_interface.SetupEditor import SetupEditor, RayTracingParam
from WI_interface.TxRxEditor import TxRxEditor
from WI_interface.TerrainEditor import TerrainEditor
from convert_ply2city import convert_to_city_file
from geo_utils import convert_GpsBBox2CartesianBBox, convert_Gps2RelativeCartesian
import pandas as pd
import numpy as np
import subprocess
import os
from dataclasses import fields
from typing import Dict, List, Tuple, Any

import sys
sys.path.append("C:/Users/jmora/Documents/GitHub/DeepMIMO")
import deepmimo as dm  # type: ignore

#from .. import deepmimo as dm 

#%% Resources and Constants

# Paths
OSM_ROOT = "C:/Users/jmora/Downloads/osm_root"
BLENDER_PATH = "C:/Program Files/Blender Foundation/Blender 3.6/blender-launcher.exe"
BLENDER_SCRIPT_PATH = "./blender_osm_export.py"

# Wireless InSite
WI_ROOT = "C:/Program Files/Remcom/Wireless InSite 4.0.0"
WI_EXE = os.path.join(WI_ROOT, "bin/calc/wibatch.exe")
WI_MAT = os.path.join(WI_ROOT, "materials")
WI_LIC = "C:/Users/jmora/Documents/GitHub/DeepMIMO/executables/wireless insite"
WI_VERSION = "4.0.1"

# Material paths
BUILDING_MATERIAL_PATH = os.path.join(WI_MAT, "ITU Concrete 3.5 GHz.mtl")
ROAD_MATERIAL_PATH = os.path.join(WI_MAT, "Asphalt_1GHz.mtl")
TERRAIN_MATERIAL_PATH = os.path.join(WI_MAT, "ITU Wet earth 3.5 GHz.mtl")

# Ray-tracing parameters
UE_HEIGHT = 1.5  # meters
BS_HEIGHT = 20  # meters
GRID_SPACING = 1.0  # meters

POS_PREC = 4


#%% Functions
def create_directory_structure(base_path: str, rt_params: Dict[str, Any]) -> Tuple[str, str]:
    """Create folders for the scenario generations with a names based on parameters.
    
    Args:
        base_path (str): Base path for the scenario
        rt_params (Dict[str, Any]): Ray tracing parameters
        
    Returns:
        Tuple[str, str]: Paths to the insite directory and study area directory
    """
    
    # Format folder name with key parameters
    folder_name = (f"insite2_{rt_params['carrier_freq']/1e9:.1f}GHz_"
                   f"{rt_params['max_reflections']}R_{rt_params['max_diffractions']}D_{rt_params['max_diffractions']}D")
    insite_path = os.path.join(osm_folder, folder_name)
    os.makedirs(insite_path, exist_ok=True)

    insite_path = os.path.join(base_path, folder_name)
    study_area_path = os.path.join(insite_path, "study_area")

    # Create directories
    for path in [insite_path, study_area_path]:
        os.makedirs(path, exist_ok=True)

    return insite_path, study_area_path

def run_command(command: List[str], description: str) -> None:
    """Run a shell command and stream output in real-time.
    
    Args:
        command (List[str]): Command to run
        description (str): Description of the command for logging
    """
    print(f"\n🚀 Starting: {description}...\n")
    print('running command: ', ' '.join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

    # Stream the output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end="")  # Print each line as it arrives

    process.stdout.close()
    process.wait()

    print(f"\n✅ {description} completed!\n")

def read_rt_configs(row: pd.Series) -> Dict[str, Any]:
    """Read ray tracing configurations from a pandas Series.
    
    Args:
        row (pd.Series): Row from the parameters DataFrame
        
    Returns:
        Dict[str, Any]: Ray tracing parameters
    """
    bs_lats = np.array(row['bs_lat'].split(',')).astype(np.float32)
    bs_lons = np.array(row['bs_lon'].split(',')).astype(np.float32)
    carrier_freq = row['freq (ghz)'] * 1e9
    diffraction = scattering = bool(row['ds_enable'])

    rt_params = {
        'name': row['name'],
        'city': row['city'],
        'min_lat': row['min_lat'],
        'min_lon': row['min_lon'],
        'max_lat': row['max_lat'],
        'max_lon': row['max_lon'],
        'bs_lats': bs_lats,
        'bs_lons': bs_lons,
        'carrier_freq': carrier_freq,

        # Ray-tracing parameters -> Efficient if they match the dataclass in SetupEditor.py
        'max_reflections': row['n_reflections'],
        'diffraction': diffraction,
        'scattering': scattering,
        'max_paths': row['max_paths'],
        'ray_spacing': row['ray_spacing'],
        'max_transmissions': row['max_transmissions'],
        'max_diffractions': row['max_diffractions'],
        'ds_enable': row['ds_enable'],
        'ds_max_reflections': row['ds_max_reflections'],
        'ds_max_transmissions': row['ds_max_transmissions'],
        'ds_max_diffractions': row['ds_max_diffractions'],
        'ds_final_interaction_only': row['ds_final_interaction_only'],
        'conform_to_terrain': row['conform_to_terrain'] == 1
    }
    return rt_params

def gen_tx_pos(rt_params: Dict[str, Any]) -> np.ndarray:
    """Generate transmitter positions from GPS coordinates.
    
    Args:
        rt_params (Dict[str, Any]): Ray tracing parameters
        
    Returns:
        List[List[float]]: Transmitter positions in Cartesian coordinates
    """
    num_bs = len(rt_params['bs_lats'])
    print(f"Number of BSs: {num_bs}")
    bs_pos = []
    for bs_lat, bs_lon in zip(rt_params['bs_lats'], rt_params['bs_lons']):
        bs_cartesian = convert_Gps2RelativeCartesian(bs_lat, bs_lon, 
                                                     rt_params['origin_lat'], 
                                                     rt_params['origin_lon'])
        bs_pos.append([bs_cartesian[0], bs_cartesian[1], BS_HEIGHT])
    return np.round(np.array(bs_pos), POS_PREC)

def gen_rx_pos(row: pd.Series, osm_folder: str) -> np.ndarray:
    """Generate receiver positions from GPS coordinates.
    
    Args:
        row (pd.Series): Row from the parameters DataFrame
        osm_folder (str): Path to the OSM folder
        
    Returns:
        np.ndarray: Receiver positions in Cartesian coordinates
    """
    with open(os.path.join(osm_folder, 'osm_gps_origin.txt'), "r") as f:
        origin_lat, origin_lon = map(float, f.read().split())
    print(f"origin_lat: {origin_lat}, origin_lon: {origin_lon}")

    user_grid = generate_user_grid(row, origin_lat, origin_lon)
    print(f"User grid shape: {user_grid.shape}")
    return np.round(user_grid, POS_PREC) 

def generate_user_grid(row: pd.Series, origin_lat: float, origin_lon: float) -> np.ndarray:
    """Generate user grid in Cartesian coordinates.
    
    Args:
        row (pd.Series): Row from the parameters DataFrame
        origin_lat (float): Origin latitude in degrees
        origin_lon (float): Origin longitude in degrees
        
    Returns:
        np.ndarray: User grid positions in Cartesian coordinates
    """
    min_lat, min_lon = row['min_lat'], row['min_lon']
    max_lat, max_lon = row['max_lat'], row['max_lon']
    xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(
        min_lat, min_lon, 
        max_lat, max_lon, 
        origin_lat, origin_lon)
    grid_x = np.arange(xmin, xmax + GRID_SPACING, GRID_SPACING)
    grid_y = np.arange(ymin, ymax + GRID_SPACING, GRID_SPACING)
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    grid_z = np.zeros_like(grid_x) + UE_HEIGHT
    return np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=-1) 

def call_blender1(rt_params: Dict[str, Any], osm_folder: str) -> None:
    """Process OSM extraction directly without calling a separate script.
    
    Args:
        rt_params (Dict[str, Any]): Ray tracing parameters
        osm_folder (str): Path to the OSM folder
    """
    
    # Extract coordinates from rt_params
    minlat = rt_params['min_lat']
    minlon = rt_params['min_lon']
    maxlat = rt_params['max_lat']
    maxlon = rt_params['max_lon']
    
    # Check if the folder already exists
    if os.path.exists(osm_folder):
        print(f"⏩ Folder '{osm_folder}' already exists. Skipping OSM extraction.")
        return
    
    # Validate paths
    if not os.path.exists(BLENDER_PATH):
        raise FileNotFoundError(f"❌ Blender executable not found at {BLENDER_PATH}")
        
    if not os.path.exists(BLENDER_SCRIPT_PATH):
        raise FileNotFoundError(f"❌ Blender script not found at {BLENDER_SCRIPT_PATH}")
    
    # Build command to run Blender
    command = [
        BLENDER_PATH, 
        "--background", 
        "--python", 
        BLENDER_SCRIPT_PATH, 
        "--", 
        "--minlat", str(minlat), 
        "--minlon", str(minlon), 
        "--maxlat", str(maxlat), 
        "--maxlon", str(maxlon),
        "--output", str(osm_folder)  # Pass the output folder to the Blender script
    ]
    
    # Run the command
    run_command(command, "OSM Extraction")

def insite_raytrace(osm_folder: str, tx_pos: np.ndarray, rx_pos: np.ndarray, **rt_params: Any) -> str:
    """Run Wireless InSite ray tracing simulation.
    
    This function sets up the simulation environment, generates the necessary files,
    and runs the ray tracing simulation. It creates both human-readable text files
    (.setup, .txrx, .ter, .city) and the XML file that is actually used by Wireless InSite.
    
    The text files are saved for reference and compatibility with the converter, but
    the XML file is what is actually used in the simulation. The text files are important
    because they are human-readable and can be used with the Wireless InSite UI for
    verification and debugging.
    
    Args:
        osm_folder (str): Path to the OSM folder
        tx_pos (np.ndarray): Transmitter positions
        rx_pos (np.ndarray): Receiver positions
        **rt_params (Any): Ray tracing parameters
        
    Returns:
        str: Path to the insite directory
    """
    insite_path, study_area_path = create_directory_structure(osm_folder, rt_params)
    
    # Create buildings.city & roads.city files
    bldgs_city = convert_to_city_file(osm_folder, insite_path, "buildings", BUILDING_MATERIAL_PATH)
    roads_city = convert_to_city_file(osm_folder, insite_path, "roads", ROAD_MATERIAL_PATH)
    TERRAIN_TEMPLATE = "newTerrain.ter"

    xmin_pad, ymin_pad, xmax_pad, ymax_pad = convert_GpsBBox2CartesianBBox(
        rt_params['min_lat'], rt_params['min_lon'], rt_params['max_lat'], rt_params['max_lon'],
        rt_params['origin_lat'], rt_params['origin_lon'], pad=30
    ) # pad makes the box larger

    # Create terrain file (.ter)
    terrain_editor = TerrainEditor()
    terrain_editor.set_vertex(xmin=xmin_pad, ymin=ymin_pad, xmax=xmax_pad, ymax=ymax_pad)
    terrain_editor.set_material(TERRAIN_MATERIAL_PATH)
    terrain_editor.save(os.path.join(insite_path, TERRAIN_TEMPLATE))

    # Configure Tx/Rx (.txrx)
    txrx_editor = TxRxEditor()

    # TX (BS)
    for b_idx, pos in enumerate(tx_pos):
        txrx_editor.add_txrx(
            txrx_type="points",
            is_transmitter=True,
            is_receiver=True,
            pos=pos,
            name=f"BS{b_idx+1}",
            conform_to_terrain=True
        )

    # RX (UEs)
    # txrx_editor.add_txrx(
    #         txrx_type="points",
    #         is_transmitter=False,
    #         is_receiver=True,
    #         pos=rx_pos,
    #         name="user_grid",
    #         conform_to_terrain=rt_params['conform_to_terrain']
    #     )
    grid_side = [xmax_pad - xmin_pad - 60 + GRID_SPACING, ymax_pad - ymin_pad - 60 + GRID_SPACING]
    txrx_editor.add_txrx(
        txrx_type="grid",
        is_transmitter=False,
        is_receiver=True,
        pos=[xmin_pad +30+1e-3, ymin_pad+30, rt_params['ue_height']],
        name="UE_grid",
        grid_side=grid_side,
        grid_spacing=GRID_SPACING,
        conform_to_terrain=rt_params['conform_to_terrain']
    )
    txrx_editor.save(os.path.join(insite_path, "insite.txrx"))

    # Create setup file (.setup)
    scenario = SetupEditor(insite_path)
    scenario.set_carrierFreq(rt_params['carrier_freq'])
    scenario.set_bandwidth(rt_params['bandwidth'])
    study_area_vertex = np.array([[xmin_pad, ymin_pad, 0],
                                  [xmax_pad, ymin_pad, 0],
                                  [xmax_pad, ymax_pad, 0],
                                  [xmin_pad, ymax_pad, 0]])
    scenario.set_study_area(zmin=-3, zmax=20, all_vertex=study_area_vertex)

    # Get ray tracing parameter names from the dataclass
    rt_param_names = {field.name for field in fields(RayTracingParam)}
    rt_params_filtered = {k: v for k, v in rt_params.items() if k in rt_param_names}
    scenario.set_ray_tracing_param(rt_params_filtered)
    scenario.set_txrx("insite.txrx")
    scenario.add_feature(TERRAIN_TEMPLATE, "terrain")
    scenario.add_feature(bldgs_city, "city")
    scenario.add_feature(roads_city, "road")
    scenario.save("insite") # insite.setup

    # Generate XML file (.xml) - What Wireless InSite executable actually uses
    xml_generator = XmlGenerator(insite_path, scenario, txrx_editor, version=int(WI_VERSION[0]))
    xml_generator.update()
    xml_path = os.path.join(insite_path, "insite.study_area.xml")
    xml_generator.save(xml_path)

    license_info = ["-set_licenses", WI_LIC] if WI_VERSION.startswith("4") else []
    
    # Run Wireless InSite using the XML file
    command = [WI_EXE, "-f", xml_path, "-out", study_area_path, "-p", "insite"] + license_info
    print('running command: ', ' '.join(command))
    subprocess.run(command, check=True)
    
    return insite_path

#%% Main execution
if __name__ == "__main__":
    df = pd.read_csv('C:/Users/jmora/Documents/GitHub/DeepMIMO/pipeline_dev/params.csv')

    for index, row in df.iterrows():
        print(f"\n{'='*50}")
        print(f"STARTING SCENARIO {index+1}/{len(df)}: {row['name']}")
        print(f"{'='*50}\n")
        
        print("PHASE 1: Reading ray tracing configurations...")
        rt_params = read_rt_configs(row)
        rt_params['ue_height'] = UE_HEIGHT     # HARD-CODED
        rt_params['bandwidth'] = 10e6  # HARD-CODED
        print("✓ Configuration loaded successfully")
        
        osm_folder = os.path.join(OSM_ROOT, rt_params['name'])
        
        print("\nPHASE 2: Running OSM extraction...")
        call_blender1(rt_params, osm_folder)
        print("✓ OSM extraction completed")
        
        # Read origin coordinates
        with open(os.path.join(osm_folder, 'osm_gps_origin.txt'), "r") as f:
            rt_params['origin_lat'], rt_params['origin_lon'] = map(float, f.read().split())

        # Generate RX and TX positions
        print("\nPHASE 3: Generating transmitter and receiver positions...")
        rx_pos = gen_rx_pos(row, osm_folder)  # N x 3 (N ~ 20k)
        tx_pos = gen_tx_pos(rt_params)  # M x 3 (M ~ 3)
        print(f"✓ Generated {len(tx_pos)} transmitter positions and {len(rx_pos)} receiver positions")

        # Ray Tracing
        print("\nPHASE 4: Running Wireless InSite ray tracing...")
        insite_rt_path = insite_raytrace(osm_folder, tx_pos, rx_pos, **rt_params)
        print(f"✓ Ray tracing completed. Results saved to: {insite_rt_path}")

        # Convert to DeepMIMO
        print("\nPHASE 5: Converting to DeepMIMO format...")
        dm.config('wireless_insite_version', WI_VERSION)
        scen_insite = dm.convert(insite_rt_path)
        print("✓ Conversion to DeepMIMO completed")

        # Test Conversion
        print("\nTesting DeepMIMO conversion...")
        dataset = dm.load(scen_insite)[0]
        print("✓ DeepMIMO conversion test completed")
        dataset.plot_coverage(dataset.los)
        dataset.plot_coverage(dataset.pwr[:,0])

        print('\n✓ SCENARIO COMPLETED SUCCESSFULLY!')
        print('--------------------')
        break

    p = dataset.pwr[:, 0]

    print(rx_pos)
    print(dataset.rx_pos)


# %%
