"""
Steps to run a pipeline:

1. pip install deepmimo

2. Setup Blender
   That's how we currently fetch OSM data and convert it to a format
	that can be used by a ray tracer.

3. Install dependencies
   - a ray tracer: 
	- Wireless InSite (3.3.x or 4.0.x)
	  + pip install lxml plyfile
	- Sionna (0.19.1)
	  + pip install ...
   - pip install utm

4. Change parameters in params.csv and in this file

5. Run the pipeline
   - python pipeline_runner.py
"""


#%% Imports
import pandas as pd
import os
import numpy as np
from deepmimo.pipelines.utils.pipeline_utils import call_blender, get_origin_coords, load_params_from_row

# import sys
# sys.path.append("C:/Users/jmora/Documents/GitHub/DeepMIMO")
import deepmimo as dm  # type: ignore

from deepmimo.pipelines.TxRxPlacement import gen_rx_grid, gen_tx_pos

# from deepmimo.pipelines.wireless_insite.insite_raytracer import raytrace_insite
from deepmimo.pipelines.sionna_rt.sionna_raytracer import raytrace_sionna

# Paths
# Windows versions
# OSM_ROOT = "C:/Users/jmora/Downloads/osm_root"
# BLENDER_PATH = "C:/Program Files/Blender Foundation/Blender 3.6/blender-launcher.exe"

# Linux versions
OSM_ROOT = "/mnt/c/Users/jmora/Downloads/osm_root"
BLENDER_PATH = "/home/joao/blender-3.6.0-linux-x64/blender"

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

#%% Step 1: (Optional) Generate CSV with GPS coordinates for map and basestation placement

print('not implemented yet')
# TODO:
# - Configure cell size to be ~80m longer in x and y compared to NY. Maybe 200 x 400m. (2x4)

COUNTER = 5
#%% Step 2: Iterate over rows of CSV file to extract the map, create TX/RX positions, and run RT

df = pd.read_csv('./pipeline_dev/params.csv')

# TODO: 
# For Sionna GPU definition: 
# gpu_num = 0 # Use "" to use the CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

# Parameters
p = {
	# Scenario parameters (to be loaded from CSV)
	'name': None,
	'city': None,
	'min_lat': None,
	'min_lon': None,
	'max_lat': None,
	'max_lon': None,
	'bs_lats': None, 
	'bs_lons': None,
	'bs_heights': None,

	# User placement parameters
	'ue_height': 1.5,
	'grid_spacing': 1,
	'pos_prec': 4, # Decimal places for coordinates

	# Paths required by Wireless InSite
	'wi_exe': WI_EXE,
	'wi_lic': WI_LIC,
	'wi_version': WI_VERSION,
	'building_material': BUILDING_MATERIAL_PATH,
	'road_material': ROAD_MATERIAL_PATH,
	'terrain_material': TERRAIN_MATERIAL_PATH,

	# Sionna specific parameters
	'batch_size': 15,  # Number of users to compute at a time
	# (heuristic: 1.5 per GB of GPU VRAM, if using scattering, else 5-10 users per GB)

	# Ray-tracing parameters -> Efficient if they match the dataclass in SetupEditor.py
	'carrier_freq': 3.5e9,  # Hz
	'bandwidth': 10e6,  # Hz
	'max_reflections': 5,
	'max_paths': 10,
	'ray_spacing': 0.25,  # m
	'max_transmissions': 0,
	'max_diffractions': 0,
	'ds_enable': False,
	'ds_max_reflections': 2,
	'ds_max_transmissions': 0,
	'ds_max_diffractions': 1,
	'ds_final_interaction_only': True,
	'conform_to_terrain': False  # Whether to conform the terrain to the ray tracing grid
	                             # (if True, positions have added the terrain height)
}

for index, row in df.iterrows():
	print(f"\n{'='*50}\nSTARTING SCENARIO {index+1}/{len(df)}: {row['name']}\n{'='*50}")

	# RT Phase 1: Load GPS coordinates from CSV
	load_params_from_row(row, p)

	# RT Phase 2: Extract OSM data
	# COUNTER += 1
	osm_folder = os.path.join(OSM_ROOT, row['name']) + f'_{COUNTER}'
	call_blender(p['min_lat'], p['min_lon'], p['max_lat'], p['max_lon'],
			     osm_folder, # Output folder to the Blender script
				 BLENDER_PATH, 
				 outputs=['sionna']) # List of outputs to generate
	p['origin_lat'], p['origin_lon'] = get_origin_coords(osm_folder)

	
	# RT Phase 3: Generate RX and TX positions
	rx_pos = gen_rx_grid(p)  # N x 3 (N ~ 100k)
	tx_pos = gen_tx_pos(p)   # M x 3 (M ~ 3)
	
	# Optional: Round positions (visually *way* better)
	rx_pos = np.round(rx_pos, p['pos_prec'])
	tx_pos = np.round(tx_pos, p['pos_prec'])

	# RT Phase 4: Run Wireless InSite ray tracing
	# rt_path = raytrace_insite(osm_folder, tx_pos, rx_pos, **p)
	
	rt_path = raytrace_sionna(osm_folder, tx_pos, rx_pos, **p)

	# RT Phase 5: Convert to DeepMIMO format
	dm.config('wireless_insite_version', WI_VERSION)
	dm.config('sionna_version', '0.19.1')
	scen_name = dm.convert(rt_path, overwrite=True)

	# RT Phase 6: Test Conversion
	dataset = dm.load(scen_name)[0]
	dataset.plot_coverage(dataset.los)
	dataset.plot_coverage(dataset.pwr[:, 0])
	break

# %%
