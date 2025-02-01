# DeepMIMO specs
This folder contains the files that specify how each DeepMIMO version organizes files.

## DeepMIMOv4 Spec

1. dataset[scene][tx][...]
    1. ['rx_loc'] is N x 3
    2. ['tx_loc'] is 1 x 3
    2. ['chs'] is N x N_ant_pairs. N_ant_pairs is an index that refers to tx (h,v,p) ↔ rx (h,v,p)
    3. ['aoa_az'] | ['aod_az'] | ['aoa_el'] | ['aod_el'] | ['toa'] | ['phase'] | ['power'] are N x MAX_PATHS
    4. ['inter'] is N x MAX_PATHS
        1 = reflection. 11 = 2 reflections. 2 = diffraction. 3 = scatering. 4 = transmission. 0 = LoS. -1 = no path
    5. ['inter_loc'] is N x MAX_PATHS x 3
    6. ['building_faces'] is N_building_tri_faces x 9 (xyz of vertex 1, xyz of vertex 2, xyz of vertex 3)
       Note that each face is triangular.
    7. ['building_materials'] N_building_tri_faces x 1
    8. Similar to building faces and materials, there can be matrices for the ground, moving objects and vegetation.
    7. New format storage: 
        1. 8 matrices of N x MAX_PATHS
            (4 angles → aoa_azi, aoa_el, aod_azi, aod_el)
            (4 others → toa, power, phase, inter)
        2. 1 matrix of N x MAX_PATHS x 3 → interactions locations
        3. 1 matrix of N x 3 (rx_loc.mat)
        4. 1 matrix of M x 3 (tx_loc.mat)
        5. 1 params.mat

Design principles:
- The dataset does not include information that be easily derived. 
E.g. number of interactions per path = dataset[scene][tx]['inter'] // 10 + 1
E.g. count nans: number of paths = dataset[scene][tx]['toa']
- Each dataset keeps constant a number of things: 
    - all TXs/RXs have the same number and type of antennas
    - 
- If there is a single element in a list, there is no point having a list.
  (we flatten datset[scene][tx] to dataset if it's a single-scene and single-tx dataset)
- All .mat files are matrices and as low dimension as possible. The only non-matrix
  is a struct/dictionairy called params.mat.
- To the user, the only visible differences of which ray tracer was used, will be 
  explicit in params.mat (ray tracer and version). Everything else should be transparent
  to the ray tracer.


We have these in helper functions.

Files in stored: 
- scene_X_tx_Y.mat
- rx_loc.mat
- tx_loc.mat
- params.mat

NOTE: an array that is 100k x 20 occupies ~7 MB of memory. 

## DeepMIMOv3 Spec

N+1 files, N = number of RX-TX pairs enabled

Rx-Tx pair i (BS1_BS.mat or BS3_UE_0-1024.mat)                                               -> NOTE: why not BS1_BS1?
    - channels: 1 x N cells
      Each cell: is a 1 x #P array, #P = num_paths
        AoA_phi
        AoA_theta
        AoD_phi
        AoD_theta
        ToA
        num_paths
        phase
        power 
    - rx_locs: N x 5 = N x [x_real, y_real, z_real, ]
    - tx_loc: 1x3 = [x_real, y_real, z_real]

    Note: x|y|z_real can be lat|long|alt in case the scenarios comes from OSM

params.mat
    - 'version': 2,
    - 'carrier_freq': 28e9,
    - 'transmit_power': 0.0, 
    - 'user_grids': np.array([[1, 411, 321]], dtype=float), # num samples in z, x, y    || -> NOTE: why not xyz?
    - 'num_BS': 1,
    - 'dual_polar_available': 0,
    - 'doppler_available': 0

## DeepMIMOv2 Spec

Needs inspection... 
