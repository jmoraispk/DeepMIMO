#%% Imports
from generate_city.generate_city import generate_city
from WI_interface.XmlGenerator import XmlGenerator
from WI_interface.SetupEditor import SetupEditor
from WI_interface.TxRxEditor import TxRxEditor
from WI_interface.TerrainEditor import TerrainEditor
from geo_utils import convert_GpsBBox2CartesianBBox, convert_Gps2RelativeCartesian
import pandas as pd
import numpy as np
import subprocess
import os

import sys
sys.path.append("C:/Users/jmora/Documents/GitHub/DeepMIMO")
import deepmimo as dm  # type: ignore


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


#%% Functions
def create_directory_structure(base_path, rt_params):
    """Create necessary directories for the scenario with a professional folder name based on parameters."""
    
    # Format folder name with key parameters
    folder_name = (f"insite_{rt_params['carrier_freq']/1e9:.1f}GHz_{rt_params['bandwidth']/1e6:.0f}MHz_"
                   f"{rt_params['max_paths']}paths_{rt_params['max_reflections']}ref_{rt_params['max_transmissions']}trans_{rt_params['max_diffractions']}diff")
    
    insite_path = os.path.join(base_path, folder_name)
    study_area_path = os.path.join(insite_path, "study_area")

    # Create directories
    for path in [insite_path, study_area_path]:
        os.makedirs(path, exist_ok=True)

    return insite_path, study_area_path

def get_grid_info(xmin, ymin, xmax, ymax, grid_spacing):
    """Calculate the grid layout and extract available rows and users per row."""
    # Create grid
    x_coords = np.arange(xmin, xmax + grid_spacing, grid_spacing)
    y_coords = np.arange(ymin, ymax + grid_spacing, grid_spacing)
    # Indices of rows and number of users per row
    row_indices = np.arange(len(y_coords) - 1)
    users_per_row = len(x_coords) - 1  # Each row has the same number of users
    return row_indices, users_per_row

def run_command(command, description):
    """Run a shell command and stream output in real-time."""
    print(f"\n🚀 Starting: {description}...\n")
    print('running command: ', ' '.join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

    # Stream the output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end="")  # Print each line as it arrives

    process.stdout.close()
    process.wait()

    print(f"\n✅ {description} completed!\n")

def read_rt_configs(row):
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
        'ds_final_interaction_only': row['ds_final_interaction_only']
    }
    return rt_params

def gen_tx_pos(rt_params):
    num_bs = len(rt_params['bs_lats'])
    print(f"Number of BSs: {num_bs}")
    bs_pos = [[convert_Gps2RelativeCartesian(rt_params['bs_lats'][i], rt_params['bs_lons'][i], rt_params['origin_lat'], rt_params['origin_lon'])[0],
                convert_Gps2RelativeCartesian(rt_params['bs_lats'][i], rt_params['bs_lons'][i], rt_params['origin_lat'], rt_params['origin_lon'])[1], 
                BS_HEIGHT]
                for i in range(num_bs)]
    return bs_pos

def gen_rx_pos(row, osm_folder):
    with open(os.path.join(osm_folder, 'osm_gps_origin.txt'), "r") as f:
        origin_lat, origin_lon = map(float, f.read().split())
    print(f"origin_lat: {origin_lat}, origin_lon: {origin_lon}")

    user_grid = generate_user_grid(row, origin_lat, origin_lon)
    print(f"User grid shape: {user_grid.shape}")
    return user_grid

def generate_user_grid(row, origin_lat, origin_lon):
    """Generate user grid in Cartesian coordinates."""
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

def call_blender1(rt_params, osm_folder):
    """Process OSM extraction directly without calling a separate script."""
    
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

def insite_raytrace(osm_folder, tx_pos, rx_pos, **rt_params):

    insite_path, study_area_path = create_directory_structure(osm_folder, rt_params)
    
    # Generate city features (creates roads.city and buildings.city)
    generate_city(osm_folder + os.sep, insite_path + os.sep, 
                  building_mtl_path=BUILDING_MATERIAL_PATH, 
                  road_mtl_path=ROAD_MATERIAL_PATH)
    
    # xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(
    #     rt_params['min_lat'], rt_params['min_lon'], rt_params['max_lat'], rt_params['max_lon'],
    #     rt_params['origin_lat'], rt_params['origin_lon'], pad=0
    # )

    xmin_pad, ymin_pad, xmax_pad, ymax_pad = convert_GpsBBox2CartesianBBox(
        rt_params['min_lat'], rt_params['min_lon'], rt_params['max_lat'], rt_params['max_lon'],
        rt_params['origin_lat'], rt_params['origin_lon'], pad=30
    ) # pad makes the box larger

    folder_name = (f"insite_{rt_params['carrier_freq']/1e9:.1f}GHz_{rt_params['bandwidth']/1e6:.0f}MHz_"
                   f"{rt_params['max_paths']}paths_{rt_params['max_reflections']}ref_{rt_params['max_transmissions']}trans_{rt_params['max_diffractions']}diff")
    insite_path = os.path.join(osm_folder, folder_name)

    terrain_editor = TerrainEditor()
    terrain_editor.set_vertex(xmin=xmin_pad, ymin=ymin_pad, xmax=xmax_pad, ymax=ymax_pad)
    terrain_editor.set_material(TERRAIN_MATERIAL_PATH)
    terrain_editor.save(os.path.join(insite_path, "newTerrain.ter"))

    # Configure Tx/Rx
    txrx_editor = TxRxEditor()
    for b_idx, pos in enumerate(tx_pos):
        txrx_editor.add_txrx(
            txrx_type="points",
            is_transmitter=True,
            is_receiver=True,
            pos=pos,
            name=f"BS{b_idx+1}"
        )

    grid_side = [xmax_pad - xmin_pad, ymax_pad - ymin_pad]
    grid_spacing = rt_params['ray_spacing'] 
    txrx_editor.add_txrx(
        txrx_type="grid",
        is_transmitter=False,
        is_receiver=True,
        pos=[xmin_pad, ymin_pad, rt_params['ue_height']],
        name="UE_grid",
        grid_side=grid_side,
        grid_spacing=grid_spacing
    )
    txrx_editor.save(os.path.join(insite_path, "insite.txrx"))

    # Calculate grid info
    # row_indices, users_per_row = get_grid_info(xmin, ymin, xmax, ymax, grid_spacing)
    
    # Create setup file
    scenario = SetupEditor(str(insite_path))
    scenario.set_carrierFreq(rt_params['carrier_freq'])
    scenario.set_bandwidth(rt_params['bandwidth'])
    scenario.set_study_area(
        zmin=-3,
        zmax=20,
        all_vertex=np.array([
            [xmin_pad, ymin_pad, 0],
            [xmax_pad, ymin_pad, 0],
            [xmax_pad, ymax_pad, 0],
            [xmin_pad, ymax_pad, 0]
        ])
    )
    scenario.set_ray_tracing_param(
        rt_params['max_paths'],
        rt_params['ray_spacing'],
        rt_params['max_reflections'],
        rt_params['max_transmissions'],
        rt_params['max_diffractions'],
        rt_params['ds_enable'],
        rt_params['ds_max_reflections'],
        rt_params['ds_max_transmissions'],
        rt_params['ds_max_diffractions'],
        rt_params['ds_final_interaction_only']
    )
    scenario.set_txrx("/insite.txrx")
    scenario.add_feature("newTerrain.ter", "terrain")
    scenario.add_feature('buildings.city', "city")
    scenario.add_feature('roads.city', "road")
    scenario.save("/insite") # insite

    # Generate XML and run simulation
    xml_generator = XmlGenerator(insite_path, "/insite.setup", version=int(WI_VERSION[0]))
    xml_generator.update()
    xml_path = os.path.join(insite_path, "insite.study_area.xml")
    xml_generator.save(xml_path)

    license_info = ["-set_licenses", WI_LIC] if WI_VERSION.startswith("4") else []
    
    # Run Wireless InSite
    command = [WI_EXE, "-f", xml_path, "-out", study_area_path, "-p", "insite"] + license_info
    print('running command: ', ' '.join(command))
    subprocess.run(command, check=True)
    
    return insite_path

#%% Main

df = pd.read_csv('C:/Users/jmora/Documents/GitHub/DeepMIMO/pipeline_dev/params.csv')

for index, row in df.iterrows():
    print(f"\n{'='*50}")
    print(f"STARTING SCENARIO {index+1}/{len(df)}: {row['name']}")
    print(f"{'='*50}\n")
    
    print("PHASE 1: Reading ray tracing configurations...")
    rt_params = read_rt_configs(row)
    rt_params['ue_height'] = 2     # HARD-CODED
    rt_params['bandwidth'] = 10e6  # HARD-CODED
    print("✓ Configuration loaded successfully")
    
    print("\nPHASE 2: Setting up paths and directories...")
    bbox_folder = f"bbox_{rt_params['min_lat']}_{rt_params['min_lon']}_{rt_params['max_lat']}_{rt_params['max_lon']}"
    osm_folder = os.path.join(OSM_ROOT, rt_params['name'])
    
    print("\nPHASE 3: Running OSM extraction...")
    call_blender1(rt_params, osm_folder)
    print("✓ OSM extraction completed")
    
    # Read origin coordinates
    with open(os.path.join(osm_folder, 'osm_gps_origin.txt'), "r") as f:
        rt_params['origin_lat'], rt_params['origin_lon'] = map(float, f.read().split())

	# Generate RX and TX positions
    print("\nPHASE 5: Generating transmitter and receiver positions...")
    rx_pos = gen_rx_pos(row, osm_folder)  # N x 3 (N ~ 20k)
    tx_pos = gen_tx_pos(rt_params)  # M x 3 (M ~ 3)
    print(f"✓ Generated {len(tx_pos)} transmitter positions and {len(rx_pos)} receiver positions")

	# Ray Tracing
    print("\nPHASE 6: Running Wireless InSite ray tracing...")
    insite_rt_path = insite_raytrace(osm_folder, tx_pos, rx_pos, **rt_params)
    print(f"✓ Ray tracing completed. Results saved to: {insite_rt_path}")

	# Convert to DeepMIMO
    print("\nPHASE 7: Converting to DeepMIMO format...")
    dm.config('wireless_insite_version', WI_VERSION)
    scen_insite = dm.convert(insite_rt_path)
    print("✓ Conversion to DeepMIMO completed")

	# Test Conversion
    print("\nPHASE 8: Testing DeepMIMO conversion...")
    dataset_insite = dm.load(scen_insite)[0]
    print("✓ DeepMIMO conversion test completed")

    print('\n✓ SCENARIO COMPLETED SUCCESSFULLY!')
    print('--------------------')
    break
