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
from deepmimo.pipelines.pipeline_utils import call_blender1, read_rt_configs
from deepmimo.pipelines.pipeline_consts import (OSM_ROOT, BLENDER_PATH, BLENDER_SCRIPT_PATH)

# import sys
# sys.path.append("C:/Users/jmora/Documents/GitHub/DeepMIMO")
import deepmimo as dm  # type: ignore

from deepmimo.pipelines.TxRxPlacement import gen_rx_pos, gen_tx_pos

from deepmimo.pipelines.wireless_insite.insite_raytracer import raytrace_insite

#%%

df = pd.read_csv('C:/Users/jmora/Documents/GitHub/DeepMIMO/pipeline_dev/params.csv')

for index, row in df.iterrows():
	print(f"\n{'='*50}")
	print(f"STARTING SCENARIO {index+1}/{len(df)}: {row['name']}")
	print(f"{'='*50}\n")
	
	print("PHASE 1: Reading ray tracing configurations...")
	rt_params = read_rt_configs(row)
	rt_params['ue_height'] = 1.5     # HARD-CODED
	rt_params['bs_height'] = 20      # HARD-CODED
	rt_params['bandwidth'] = 10e6    # HARD-CODED
	rt_params['grid_spacing'] = 1    # HARD-CODED
	print("✓ Configuration loaded successfully")
	
	osm_folder = os.path.join(OSM_ROOT, rt_params['name'])
	
	print("\nPHASE 2: Running OSM extraction...")
	call_blender1(rt_params, osm_folder, BLENDER_PATH, BLENDER_SCRIPT_PATH)
	print("✓ OSM extraction completed")
	
	# Read origin coordinates
	with open(os.path.join(osm_folder, 'osm_gps_origin.txt'), "r") as f:
		rt_params['origin_lat'], rt_params['origin_lon'] = map(float, f.read().split())

	# Generate RX and TX positions
	print("\nPHASE 3: Generating transmitter and receiver positions...")
	rx_pos = gen_rx_pos(rt_params, osm_folder)  # N x 3 (N ~ 20k)
	tx_pos = gen_tx_pos(rt_params)        # M x 3 (M ~ 3)
	print(f"✓ Generated {len(tx_pos)} transmitter positions and {len(rx_pos)} receiver positions")

	# Ray Tracing
	print("\nPHASE 4: Running Wireless InSite ray tracing...")
	insite_rt_path = raytrace_insite(osm_folder, tx_pos, rx_pos, **rt_params)
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

#%%

df = pd.read_csv('scenarios.csv')

for index, row in df.iterrows():
	
	rt_params = dm.read_rt_configs(row)  # dict(n_reflections, diffraction, scattering, ...)

	osm_path = dm.pipelines.call_blender(rt_params['gps_bbox'], outputs=['insite', 'sionna'])
	
	# Generate XY user grid and BS positions
	rx_pos = dm.pipelines.gen_rx_pos(row)  # N x 3 (N ~ 20k)
	tx_pos = dm.pipelines.gen_tx_pos(row)  # M x 3 (M ~ 3)

	# Ray Tracing
	insite_rt_path = dm.pipelines.insite_raytrace(osm_path, tx_pos, rx_pos, **rt_params)
	sionna_rt_path = dm.pipelines.sionna_raytrace(osm_path, tx_pos, rx_pos, **rt_params)
	
	# Convert to DeepMIMO
	scen_insite = dm.convert(insite_rt_path)
	scen_sionna = dm.convert(sionna_rt_path)

	# Test Conversion
	dataset_insite = dm.load(scen_insite)
	dataset_sionna = dm.load(scen_sionna)

#%%
