import sys
print("\n".join(sys.path))

import pandas as pd
import deepmimo as dm
import subprocess
import os
from utils.geo_utils import convert_GpsBBox2CartesianBBox, convert_Gps2RelativeCartesian
from constants import PROJ_ROOT, GRID_SPACING, UE_HEIGHT, BS_HEIGHT, BLENDER_PATH
import numpy as np
from datetime import datetime as dt 

df = pd.read_csv('params.csv')

def run_command(command, description):
    """Run a shell command and stream output in real-time."""
    print(f"\n🚀 Starting: {description}...\n")
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")

    # Stream the output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end="")  # Print each line as it arrives

    process.stdout.close()
    process.wait()

    print(f"\n✅ {description} completed!\n")

def read_rt_configs(row):
    scene_name = row['scenario_name']
    min_lat = row['min_lat']
    min_lon = row['min_lon']
    max_lat = row['max_lat']
    max_lon = row['max_lon']
    bs_lats = np.array(row['bs_lat'].split(',')).astype(np.float32)
    bs_lons = np.array(row['bs_lon'].split(',')).astype(np.float32)
    carrier_freq = row['freq (ghz)'] * 1e9
    n_reflections = row['n_reflections']
    diffraction = bool(row['diffraction'])
    scattering = bool(row['scattering'])

    rt_params = {
        'scene_name': scene_name,
        'min_lat': min_lat,
        'min_lon': min_lon,
        'max_lat': max_lat,
        'max_lon': max_lon,
        'bs_lats': bs_lats,
        'bs_lons': bs_lons,
        'carrier_freq': carrier_freq,
        'n_reflections': n_reflections,
        'diffraction': diffraction,
        'scattering': scattering,
    }
    return rt_params


def gen_tx_pos(rt_params):
    num_bs = len(rt_params['bs_lats'])
    print(f"Number of BSs: {num_bs}")
    bs_pos = [[convert_Gps2RelativeCartesian(rt_params['bs_lats'][i], rt_params['bs_lons'][i], origin_lat, origin_lon)[0],
                convert_Gps2RelativeCartesian(rt_params['bs_lats'][i], rt_params['bs_lons'][i], origin_lat, origin_lon)[1], 
                BS_HEIGHT]
                for i in range(num_bs)]
    return bs_pos

def gen_rx_pos(rt_params):
    with open(os.path.join(osm_folder, rt_params['scene_name'], "osm_gps_origin.txt"), "r") as f:
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


def call_blender1(rt_params):
    osm_command = [
        "python", "run_osm_extraction.py",
        "--minlat", str(rt_params['min_lat']), "--minlon", str(rt_params['min_lon']),
        "--maxlat", str(rt_params['max_lat']), "--maxlon", str(rt_params['max_lon'])
    ]
    run_command(osm_command, "OSM Extraction")

def call_blender2():
    pass
    # Run Blender
    # command = [
    #     BLENDER_PATH,
    #     '-b',
    #     '-P',
    #     os.path.join(PROJ_ROOT, 'scene_builder.py'),
    #     '--',
    #     scene_name,
    #     str(min_lat),
    #     str(min_lon),
    #     str(max_lat),
    #     str(max_lon),
    #     osm_folder,
    #     time_str
    # ]
    # try:
    #     result = subprocess.run(command, capture_output=True, text=True, check=True)
    #     print("Blender output:")
    #     print(result.stdout)
    # except subprocess.CalledProcessError as e:
    #     print("Errors:", e.stderr)

for index, row in df.iterrows():
	# TODO1: read_rt_configs()
    rt_params = read_rt_configs(row) # dict(n_reflections, diffraction, scattering, ...)
    
    
    # TODO2: call_blender1()
    call_blender1(rt_params)

    # TODO5: call_blender2()
    # call_blender1(rt_params)
	# osm_path = call_blender1(rt_params['gps_bbox'], outputs=['insite', 'sionna'])

    # Identify the paths --
    time_str = dt.now().strftime("%m-%d-%Y_%HH%MM%SS")
    osm_folder = os.path.join(PROJ_ROOT, f"bbox_{rt_params['min_lat']}_{rt_params['min_lon']}_{rt_params['max_lat']}_{rt_params['max_lon']}".replace(".", "-"))
    csv_path = os.path.join(PROJ_ROOT, 'params.csv')

    with open(os.path.join('osm_exports', osm_folder, 'osm_gps_origin.txt'), "r") as f:
        origin_lat, origin_lon = map(float, f.read().split())
    print(f"origin_lat: {origin_lat}, origin_lon: {origin_lon}")

    user_grid = generate_user_grid(row, origin_lat, origin_lon)
    print(f"User grid shape: {user_grid.shape}")

	
	# TODO3: gen_positins()
	# Generate XY user grid and BS positions
    rx_pos = gen_rx_pos(rt_params)  # N x 3 (N ~ 20k)
    tx_pos = gen_tx_pos(rt_params)  # M x 3 (M ~ 3)

	# TODO4: insite_raytrace()
	# Ray Tracing
    insite_rt_path = dm.insite_raytrace(osm_folder, tx_pos, rx_pos, **rt_params)
	
    
	# Convert to DeepMIMO
    # scen_insite = dm.convert(insite_rt_path)

	# Test Conversion
    # dataset_insite = dm.load(scen_insite)
    



for index, row in df.iterrows():

    # STEP 1: replace the CSV logic
    # python scenario_generator.py --minlat 64.11029 --minlon -21.90496 --maxlat 64.11197 --maxlon -21.90077 --bs 64.11118,-21.90184 --ue_height 1.5 --bs_height 15 --max_path 25 --max_reflections 3 --max_diffractions 2

    command = ['python', 'scenario_generator.py',
            '--minlat', row['min_lat'], 
            '--minlon', row['min_lon'],
            '--maxlat', row['max_lat'],
            '--maxlon', row['max_lon'],
            '--bs', row['bs_lat'], row['bs_lon'],
            '--ue_height', '1.5', 
            '--bs_height', '15', 
            '--max_path', '25', 
            '--max_reflections', '3', 
            '--max_diffractions', '2']

    subprocess.run(command, capture_output=True, text=True, check=True)


    # STEP 2: replace the CSV logic
    rt_params = read_rt_configs(row)
    run_insite(**rt_params)

    # STEP 3: replace the CSV logic
    blender_path = call_blender1()
    run_insite(blender_path, **rt_params)