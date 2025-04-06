import pandas as pd
import deepmimo as dm

df = pd.read_csv('scenarios.csv')

for row in df:
	
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