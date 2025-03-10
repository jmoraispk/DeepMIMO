import os 
import subprocess 
import numpy as np
import pandas as pd 
from datetime import datetime as dt 
from constants import PROJ_ROOT, GRID_SPACING, UE_HEIGHT, BS_HEIGHT, BLENDER_PATH
from utils.geo_utils import convert_GpsBBox2CartesianBBox, convert_Gps2RelativeCartesian

import tensorflow as tf
from ray_tracer import RayTracer

tf.random.set_seed(1)
gpus = tf.config.list_physical_devices('GPU')
print("TensorFlow sees GPUs:" if gpus else "No GPUs found.", [gpu.name for gpu in gpus] if gpus else "")

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

if __name__ == '__main__':
    # Setup run directory
    time_str = dt.now().strftime("%m-%d-%Y_%HH%MM%SS")
    osm_folder = os.path.join(PROJ_ROOT, 'all_runs', f'run_{time_str}')
    csv_path = os.path.join(PROJ_ROOT, 'params.csv')

    # Read positions from CSV
    df = pd.read_csv(csv_path)

    # Save scenes folder path
    with open(os.path.join(PROJ_ROOT, 'scenes_folder.txt'), 'w') as fp:
        fp.write(osm_folder + '\n')

    # Run Blender and Sionna for each scenario
    n_rows = df.index.stop
    for row_idx in range(0, n_rows):
        row = df.iloc[row_idx]
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

        # Run Blender
        command = [
            BLENDER_PATH,
            '-b',
            '-P',
            os.path.join(PROJ_ROOT, 'scene_builder.py'),
            '--',
            scene_name,
            str(min_lat),
            str(min_lon),
            str(max_lat),
            str(max_lon),
            osm_folder,
            time_str
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print("Blender output:")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("Errors:", e.stderr)

        # Run Sionna
        with open(os.path.join(osm_folder, scene_name, "osm_gps_origin.txt"), "r") as f:
            origin_lat, origin_lon = map(float, f.read().split())
        print(f"origin_lat: {origin_lat}, origin_lon: {origin_lon}")

        user_grid = generate_user_grid(row, origin_lat, origin_lon)
        print(f"User grid shape: {user_grid.shape}")

        num_bs = len(bs_lats)
        print(f"Number of BSs: {num_bs}")
        bs_pos = [[convert_Gps2RelativeCartesian(bs_lats[i], bs_lons[i], origin_lat, origin_lon)[0],
                   convert_Gps2RelativeCartesian(bs_lats[i], bs_lons[i], origin_lat, origin_lon)[1], 
                   BS_HEIGHT]
                   for i in range(num_bs)]

        ray_tracer = RayTracer(osm_folder)
        ray_tracer.run(scene_name, bs_pos, user_grid, carrier_freq, n_reflections, diffraction, scattering)
        break
        # exit()