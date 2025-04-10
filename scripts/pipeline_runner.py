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
from deepmimo.pipelines.pipeline_utils import call_blender1, get_origin_coords
from deepmimo.pipelines.pipeline_consts import (OSM_ROOT, BLENDER_PATH, BLENDER_SCRIPT_PATH)

# import sys
# sys.path.append("C:/Users/jmora/Documents/GitHub/DeepMIMO")
import deepmimo as dm  # type: ignore

from deepmimo.pipelines.TxRxPlacement import gen_rx_grid, gen_tx_pos

from deepmimo.pipelines.wireless_insite.insite_raytracer import raytrace_insite

#%%

df = pd.read_csv('C:/Users/jmora/Documents/GitHub/DeepMIMO/pipeline_dev/params.csv')

for index, row in df.iterrows():
	print(f"\n{'='*50}\nSTARTING SCENARIO {index+1}/{len(df)}: {row['name']}\n{'='*50}")
	
	rt_params = {
        'name': row['name'],
        'city': row['city'],
        'min_lat': row['min_lat'],
        'min_lon': row['min_lon'],
        'max_lat': row['max_lat'],
        'max_lon': row['max_lon'],
        'bs_lats': np.array(row['bs_lat'].split(',')).astype(np.float32),
        'bs_lons': np.array(row['bs_lon'].split(',')).astype(np.float32),
        'carrier_freq': 3.5e9, # GHz

		# Paths
		'osm_folder': os.path.join(OSM_ROOT, row['name']), # Output folder to the Blender script

		# User placement parameters
		'ue_height': 1.5,
		'bs_height': 20,
		'bandwidth': 10e6,
		'grid_spacing': 1,
		'pos_prec': 4,

        # Ray-tracing parameters -> Efficient if they match the dataclass in SetupEditor.py
        'max_reflections': row['n_reflections'],
        'diffraction': False,
        'scattering': False,
        'max_paths': row['max_paths'],
        'ray_spacing': row['ray_spacing'],
        'max_transmissions': row['max_transmissions'],
        'max_diffractions': row['max_diffractions'],
        'ds_enable': row['ds_enable'],
        'ds_max_reflections': row['ds_max_reflections'],
        'ds_max_transmissions': row['ds_max_transmissions'],
        'ds_max_diffractions': row['ds_max_diffractions'],
        'ds_final_interaction_only': row['ds_final_interaction_only'],
        'conform_to_terrain': False
    }

	print("\nPHASE 2: Running OSM extraction...")
	call_blender1(rt_params, # minlat, minlon, maxlat, maxlon
			      rt_params['osm_folder'], # Output folder to the Blender script
				  BLENDER_PATH, 
				  BLENDER_SCRIPT_PATH)
	# osm_path = call_blender2(rt_params['gps_bbox'], outputs=['insite', 'sionna'])
	print("✓ OSM extraction completed")
	
	# Read origin coordinates
	rt_params['origin_lat'], rt_params['origin_lon'] = get_origin_coords(rt_params['osm_folder'])
	

	# Generate RX and TX positions
	print("\nPHASE 3: Generating transmitter and receiver positions...")
	rx_pos = gen_rx_grid(rt_params)  # N x 3 (N ~ 20k)
	tx_pos = gen_tx_pos(rt_params)   # M x 3 (M ~ 3)
	print(f"✓ Generated {len(tx_pos)} transmitter positions and {len(rx_pos)} receiver positions")

	# Optional: Round positions (visually *way* better)
	rx_pos = np.round(rx_pos, rt_params['pos_prec'])
	tx_pos = np.round(tx_pos, rt_params['pos_prec'])

	# Ray Tracing
	print("\nPHASE 4: Running Wireless InSite ray tracing...")
	insite_rt_path = raytrace_insite(rt_params['osm_folder'], tx_pos, rx_pos, **rt_params)
	print(f"✓ Ray tracing completed. Results saved to: {insite_rt_path}")

	break
	# Convert to DeepMIMO
	print("\nPHASE 5: Converting to DeepMIMO format...")
	dm.config('wireless_insite_version', "4.0.1")
	scen_insite = dm.convert(insite_rt_path)
	print("✓ Conversion to DeepMIMO completed")

	# Test Conversion
	print("\nTesting DeepMIMO conversion...")
	dataset = dm.load(scen_insite)[0]
	print("✓ DeepMIMO conversion test completed")
	dataset.plot_coverage(dataset.los)
	break

