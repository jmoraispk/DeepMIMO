# -*- coding: utf-8 -*-

#%%
import numpy as np
import deepmimo as dm
import matplotlib.pyplot as plt

scen_name = 'asu_campus_3p5'
dm.download(scen_name)
dataset = dm.load(scen_name)

#%% LOAD: Detailed example

scen_name = 'city_0_newyork_3p5'

tx_sets_dict = {1: [0]}  # Load first points from set 1
rx_sets_dict = {4: np.arange(10)}  # Load first 10 points from set 4

dataset1 = dm.load(
    scen_name,
    tx_sets=tx_sets_dict,
    rx_sets=rx_sets_dict,
    matrices=['aoa_az', 'aoa_el', 'inter_pos', 'inter'],
    max_paths=10
)

# Example 2: Load all points of specific TX/RX sets using lists
dataset2 = dm.load(scen_name, tx_sets=[1], rx_sets=[2])

# Example 3: Load all TX/RX sets (default)
dataset3 = dm.load(scen_name, tx_sets='all', rx_sets='all')

#%% SCENARIO INFORMATION

dm.summary('asu_campus')

#%% SCENARIO INFORMATION: Transmitters and Receivers

# Get all available TX-RX pairs
txrx_sets = dm.get_txrx_sets(scen_name)
pairs = dm.get_txrx_pairs(txrx_sets)

print(txrx_sets)
print(pairs)

#%% SCENARIO INFORMATION: Ray Tracing Parameters

# Get all available scenarios
scenarios = dm.get_available_scenarios()
print(f"Found {len(scenarios)} scenarios\n")

for scen_name in scenarios:
      params_json_path = dm.get_params_path(scen_name)

      # Skip if params file doesn't exist
      if not os.path.exists(params_json_path):
          print(f"Skipping {scen_name} - no params file found")
          continue

      params_dict = dm.load_dict_from_json(params_json_path)
      rt_params = params_dict[dm.consts.RT_PARAMS_PARAM_NAME]

      # Calculate sums
      max_reflections = rt_params[dm.consts.RT_PARAM_MAX_REFLECTIONS]
      max_diffractions = rt_params[dm.consts.RT_PARAM_MAX_DIFFRACTIONS]
      total_interactions = max_reflections + max_diffractions

      print(f"\nScenario: {scen_name}")
      print(f"Max Reflections: {max_reflections}")
      print(f"Max Diffractions: {max_diffractions}")
      print(f"Total Interactions: {total_interactions}")


#%% VISUALIZATION: Coverage Maps

main_keys = ['aoa_az', 'aoa_el', 'aod_az', 'aod_el', 'delay', 'power', 'phase',
             'los', 'distances', 'num_paths']
for key in main_keys:
    plt_var = dataset[key][:,0] if dataset[key].ndim == 2 else dataset[key]
    dm.plot_coverage(dataset.rx_pos, plt_var, bs_pos=dataset.tx_pos.T, title=key)

#3D version
plot_coverage(data['user']['location'], data['user'][var_name], bs_pos=bs_pos,
              bs_ori=bs_ori, title=var_name, cbar_title=var_name,
              proj_3D=False, scat_sz=0.1)

"""## Rays"""

dm.plot_rays(dataset.rx_pos[10], dataset.tx_pos[0],
             dataset.inter_pos[10], dataset.inter[10],
             proj_3D=True, color_by_type=True)

# 2D and 3D

"""## Scene"""

# Plot the full scene
dataset.scene.plot()

# Plot the scene with triangular faces
scene.plot(mode='tri_faces')

"""## Path Plots

Percentage of Power in the First Path
"""

# Percentage of the power in first path
pwr_in_first_path = [data['user']['paths'][u]['power'][0] / np.sum(data['user']['paths'][u]['power'])
                     if data['user']['LoS'][u] != -1 else np.nan for u in range(n_users)]

plot_coverage(data['user']['location'], pwr_in_first_path, bs_pos=bs_pos,
              proj_3D=False, title='Percentage of power in 1st path',
              cbar_title='Percentage of power [%]')

"""# Channel Generation"""

#%% Channel Generation Example
print("\nChannel Generation Example")
print("-" * 50)

# Create channel parameters with all options
ch_params = dm.ChannelGenParameters()

# Base station antenna parameters
ch_params.bs_antenna.rotation = np.array([30, 40, 30])  # [az, el, pol] in degrees
ch_params.bs_antenna.fov = np.array([360, 180])         # [az, el] in degrees
ch_params.bs_antenna.shape = np.array([8, 8])           # [horizontal, vertical] elements
ch_params.bs_antenna.spacing = 0.5                      # Element spacing in wavelengths

# User equipment antenna parameters
ch_params.ue_antenna.rotation = np.array([0, 0, 0])  # [az, el, pol] in degrees
ch_params.ue_antenna.fov = np.array([120, 180])      # [az, el] in degrees
ch_params.ue_antenna.shape = np.array([4, 4])        # [horizontal, vertical] elements
ch_params.ue_antenna.spacing = 0.5                   # Element spacing in wavelengths

# # Channel computation parameters
ch_params.freq_domain = True     # Whether to compute frequency domain channels
ch_params.bandwidth = 1e6        # Bandwidth in Hz
ch_params.num_subcarriers = 64   # Number of subcarriers

# Generate channels
dataset.compute_channels(ch_params)
dataset.channel.shape

dm.info('ch_params')

"""Below is a brief summary for the parameters. For more details, see [DeepMIMOv3 page](https://www.deepmimo.net/versions/deepmimo-v3-python/).

| Parameter              | Default Value          | Description                                                                           |
|------------------------|------------------------|---------------------------------------------------------------------------------------|
| **enable_doppler**     | 0                          | Enable Doppler shift (if available in the scenario)                                |
| **enable_dual_polar**  | 0                          | Enable dual Dual-polarized antennas (if available)                |
| **num_paths**          | 5                          | Number of maximum paths                                                            |
| **OFDM_channels**      | 1                          | Generate OFDM (True) or time domain channels (False)                               |
| | | |
| **OFDM**               |                            | OFDM parameters (only applies if OFDM_channels is True)                            |
|   - **subcarriers**    | 512                        | Total number of subcarriers                                                        |
|   - **selected_subcarriers** | [0]                  | Subcarriers to be generated                                                        |
|   - **bandwidth**      | 0.05                       | Bandwidth                                                                          |
|   - **RX_filter**      | 0                          | Receive filter                                                                     |
| | | |
| **bs_antenna**/**ue_antenna**         |                            | BS/UE antenna properties                                            |
|   - **FoV**            | [360, 180]                 | Antenna FoV Horizontal in [0,360] and vertical in [0, 180]                         |
|   - **radiation_pattern** | isotropic              | Radiation pattern applied to the antenna, in ['isotropic', 'halfwave-dipole']       |
|   - **rotation**       | [0, 0, 0]                  | Rotation of the antenna - in compliance with 38.901                                |
|   - **shape**          | [8, 4]                     | UPA panel shape in the shape of (horizontal elements, vertical elements)           |
|   - **spacing**        | 0.5                        | Antenna spacing                                                                    |
"""

channels = dataset.compute_channels()

# Visualize channel magnitude response (NOTE: requires at >1 subcarriers and antenna)
user_idx = np.where(dataset.n_paths > 0)[0][0]
plt.imshow(np.abs(np.squeeze(channels[user_idx]).T))
plt.title('Channel Magnitude Response')
plt.xlabel('TX Antennas')
plt.ylabel('Subcarriers')

"""## Time Domain

## Frequency Domain (OFDM)
"""

# Visualize power discarding statistics
fig, axes = dm.plot_power_discarding(dataset)
plt.show()

"""## Comparison"""

scen_name = 'asu_campus'

tx_sets = {1: [0]}
rx_sets = {2: 'all'}

load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': 25}
dataset = dm.load(scen_name, **load_params)

# Create channel generation parameters
ch_params = dm.ChannelGenParameters()

ch_params.num_paths = 5
ch_params.ofdm.subcarriers = 64
ch_params.ue_antenna.shape = np.array([1,1])

# Compute channels in frequency domain first
ch_params.freq_domain = True
dataset.compute_channels(ch_params)
fd_channels = dataset.channel.copy()  # Save frequency domain channels

# Print average channel magnitudes
print("\nChannel magnitude analysis:")
print(f"FD channel average magnitude: {np.mean(np.abs(fd_channels)):.2e}")

# Now compute in time domain and apply FFT
ch_params.freq_domain = False
dataset.compute_channels(ch_params)
td_channels = dataset.channel.copy()  # Save time domain channels
print(f"TD->FD channel average magnitude: {np.mean(np.abs(td_channels)):.2e}")

# Apply FFT to time domain channels
# Note: FFT size should match number of subcarriers for fair comparison
n_fft = ch_params.ofdm.subcarriers
td_to_fd_channels = np.fft.fft(td_channels, n=n_fft, axis=-1)

# Compare the results
diff = np.abs(fd_channels - td_to_fd_channels)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

# Compute normalized differences per user
user_norms_fd = np.mean(np.abs(fd_channels), axis=(1,2,3))  # Average magnitude per user
user_norms_td = np.mean(np.abs(td_to_fd_channels), axis=(1,2,3))
relative_diff = np.abs(user_norms_fd - user_norms_td) / np.maximum(user_norms_fd, 1e-10)  # Avoid division by zero

print("\nPer-user relative differences:")
print(f"Maximum relative difference: {np.max(relative_diff):.2e}")
print(f"Mean relative difference: {np.mean(relative_diff):.2e}")
print(f"Median relative difference: {np.median(relative_diff):.2e}")

print("\nAbsolute differences:")
print(f"Maximum difference: {max_diff:.2e}")
print(f"Mean difference: {mean_diff:.2e}")

"""# Basic Dataset Operations

## Line-of-Sight Status
"""

active_mask = dataset.num_paths > 0
print(f"\nNumber of active positions: {np.sum(active_mask)}")
print(f"Number of inactive positions: {np.sum(~active_mask)}")

# Create scatter plot showing active vs inactive positions
plt.figure(figsize=(12, 8))
plt.scatter(dataset.rx_pos[~active_mask, 0], dataset.rx_pos[~active_mask, 1],
           alpha=0.5, s=3, c='red', label='Inactive')
plt.scatter(dataset.rx_pos[active_mask, 0], dataset.rx_pos[active_mask, 1],
           alpha=0.5, s=3, c='green', label='Active')
plt.legend()
plt.show()

dm.plot_coverage(dataset['rx_pos'], dataset.los != -1)

"""## Distances"""

dataset.distance

"""## Number of Paths"""

dataset.num_paths

"""## Pathloss"""



"""## Number of Interactions"""



"""## Field-of-View (FoV)"""

parameters = DeepMIMOv3.default_params()
parameters['scenario'] = 'asu_campus1'
parameters['ue_antenna']['shape'] = np.array([1, 1])
parameters['bs_antenna']['shape'] = np.array([8, 1])
parameters['bs_antenna']['FoV'] = np.array([180, 180])
parameters['bs_antenna']['rotation'] = np.array([0, 0, 0]) # +x rotation

bs_location = dataset[0]['location']
LoS_status = dataset[0]['user']['LoS']
user_locations = dataset[0]['user']['location']
plot_LoS_status(bs_location, user_locations, LoS_status)

parameters['bs_antenna']['rotation'] = np.array([0, 0, 90])

parameters['bs_antenna']['rotation'] = np.array([0, 0, -135])
parameters['bs_antenna']['FoV'] = np.array([90, 180])



"""# Scene & Materials"""

print("\nScene and Materials Example")
print("-" * 50)

# Load a scenario
dataset = dm.load('simple_street_canyon_test')
scene = dataset.scene

# 1. Basic scene information
print("\n1. Scene Overview:")
print(f"Total objects: {len(scene.objects)}")

# Get objects by category
buildings = scene.get_objects('buildings')
terrain = scene.get_objects('terrain')
vegetation = scene.get_objects('vegetation')

print(f"Buildings: {len(buildings)}")
print(f"Terrain: {len(terrain)}")
print(f"Vegetation: {len(vegetation)}")

# 2. Materials and Filtering
print("\n2. Materials and Filtering:")
materials = dataset.materials

# Get materials used by buildings
building_materials = buildings.get_materials()
print(f"Materials used in buildings: {building_materials}")

# Different ways to filter objects
print("\nFiltering examples:")

# Filter by label only
buildings = scene.get_objects(label='buildings')
print(f"Buildings: {len(buildings)}")

# Filter by material only
material_idx = building_materials[0]
objects_with_material = scene.get_objects(material=material_idx)
print(f"Objects with material {material_idx}: {len(objects_with_material)}")

# Filter by both label and material
buildings_with_material = scene.get_objects(label='buildings', material=material_idx)
print(f"Buildings with material {material_idx}: {len(buildings_with_material)}")

# Print material properties
material = materials[material_idx]
print(f"\nMaterial {material_idx} properties:")
print(f"- Name: {material.name}")
print(f"- Permittivity: {material.permittivity}")
print(f"- Conductivity: {material.conductivity}")

# 3. Object Properties
print("\n3. Object Properties:")
building = buildings[0]
print(f"Building faces: {len(building.faces)}")
print(f"Building height: {building.height:.2f}m")
print(f"Building volume: {building.volume:.2f}m³")
print(f"Building footprint area: {building.footprint_area:.2f}m²")

# 4. Bounding Boxes
print("\n4. Bounding Boxes:")
bb = buildings.bounding_box
print(f"Buildings bounding box:")
print(f"- Width (X): {bb.width:.2f}m")
print(f"- Length (Y): {bb.length:.2f}m")
print(f"- Height (Z): {bb.height:.2f}m")

"""# User Sampling & Dataset Subset

## Dataset Trimming
"""

print("\nActive Users and Dataset Subsetting (Trimming) Example")
print("-" * 50)

# Get indices of active users (those with paths)
active_idxs = dataset.get_active_idxs()
print(f"Original dataset has {dataset.n_ue} UEs")
print(f"Found {len(active_idxs)} active UEs")

# Create new dataset with only active users
active_dataset = dataset.subset(active_idxs)
print(f"New dataset has {active_dataset.n_ue} UEs")

active_dataset.scene.plot()

dm.plot_coverage(active_dataset.rx_pos, active_dataset.aoa_az[:,0],
                 bs_pos=active_dataset.tx_pos.T)

"""## Uniform"""

idxs = dataset.get_uniform_idxs([4,4])
dm.plot_coverage(dataset.rx_pos[idxs], dataset.aoa_az[idxs, 0], bs_pos=dataset.tx_pos.T)

"""## Linear"""

# LINEAR PATHS: 2D plots across positions (with Class)

# Get the closest dataset positions for a given path
linpath1 = LinearPath(data, [100, 90], [-50, 90], n_steps=75)
linpath2 = LinearPath(data, [100, 80], [-50, 80], n_steps=75)
linpath3 = LinearPath(data, [ 30,  0], [ 30, 150], n_steps=75)

plot_coverage(data['user']['location'], data['user']['LoS'], bs_pos=bs_pos, bs_ori=bs_ori,
              # title='Pathloss with positions', cbar_title='pathloss [dB]')
              title='LoS with positions', cbar_title='LoS status')

plt.scatter(linpath1.pos[:,0], linpath1.pos[:,1], c='blue', label='path1', s=4, lw=.1)
plt.scatter(linpath2.pos[:,0], linpath2.pos[:,1], c='cyan', label='path2', s=4, lw=.1)
plt.scatter(linpath3.pos[:,0], linpath3.pos[:,1], c='red',  label='path3', s=4, lw=.1)
plt.legend()

# TODO: change make this general (able to plot linear paths on top of coverage map)

# LINEAR PATHS: feature variation across linear path

for var_name in linpath1.get_feature_names():
    plt.figure(dpi=200)
    plt.plot(getattr(linpath1, var_name), ls='-',  c='blue', label='path1',marker='*', markersize=7)
    plt.plot(getattr(linpath2, var_name), ls='-.', c='cyan', label='path2',marker='s', markerfacecolor='none')
    plt.plot(getattr(linpath3, var_name), ls='--', c='red',  label='path3',marker='o', markerfacecolor='w')
    plt.xlabel('position index')
    plt.ylabel(f'{var_name}')
    plt.grid()
    plt.legend()
    plt.show()

"""## In XYZ boxes"""

# ZONE SAMPLING: Random sampling positions from two areas

idxs_A = get_idxs_in_xy_box(data['user']['location'],
                            x_min=-100, x_max=-60, y_min=0, y_max=40)

idxs_B = get_idxs_in_xy_box(data['user']['location'],
                            x_min= 125, x_max=165, y_min=0, y_max=40)

n = 50
np.random.seed(11)
datasetA_idxs = np.random.choice(idxs_A, size=n)
datasetB_idxs = np.random.choice(idxs_A, size=n)
datasetC_idxs = np.random.choice(idxs_B, size=n)

# ZONE SAMPLING: Plot boxes

plot_coverage(data['user']['location'], best_beams, bs_pos=bs_pos, bs_ori=bs_ori,
              title='Dataset zones on Optimum Beams', cbar_title='Optimum Beam Index')

plt.scatter(data['user']['location'][idxs_A,0], data['user']['location'][idxs_A,1],
            label='box A', s=2, lw=.1, alpha=.1)
plt.scatter(data['user']['location'][idxs_B,0], data['user']['location'][idxs_B,1],
            label='box B', s=2, lw=.1, alpha=.1)

# ZONE SAMPLING: Plot distributions on x, y, beams and beamformed power

hist_params = {'alpha': 0.5, 'bins': 8, 'zorder':2}

# dist on x
plt.figure(dpi=200)
plt.hist(data['user']['location'][datasetA_idxs,0], **hist_params, label='A')
plt.hist(data['user']['location'][datasetB_idxs,0], **hist_params, label='B')
plt.hist(data['user']['location'][datasetC_idxs,0], **hist_params, label='C')
plt.title('Position (x-coordinate) distribution')
plt.xlabel('x-coordinate')
plt.grid()
plt.show()

# dist on y
plt.figure(dpi=200)
plt.hist(data['user']['location'][datasetA_idxs,1], **hist_params, label='A')
plt.hist(data['user']['location'][datasetB_idxs,1], **hist_params, label='B')
plt.hist(data['user']['location'][datasetC_idxs,1], **hist_params, label='C')
plt.title('Position (y-coordinate) distribution')
plt.xlabel('y-coordinate')
plt.grid()
plt.show()

# best beam
plt.figure(dpi=200)
plt.hist(best_beams[datasetA_idxs], **hist_params, label='A')
plt.hist(best_beams[datasetB_idxs], **hist_params, label='B')
plt.hist(best_beams[datasetC_idxs], **hist_params, label='C')
plt.title('Best beam distributions')
plt.xlabel('Best beam index')
plt.grid()
plt.show()

# received power on the best beam
plt.figure(dpi=200)
plt.hist(max_bf_pwr[datasetA_idxs], **hist_params, label='A')
plt.hist(max_bf_pwr[datasetB_idxs], **hist_params, label='B')
plt.hist(max_bf_pwr[datasetC_idxs], **hist_params, label='C')
plt.title('Beamformed Power distributions')
plt.xlabel('Received power on best beam[dBm]')
plt.grid()
plt.show()

"""# Beamforming

## Computing Received Power with TX Beamforming
"""

# Compute Received power in different Beams and Bands
n_beams = 25
n_subbands = 10

# Setup Beamformers
beam_angles = np.around(np.linspace(-60, 60, n_beams), 2)

# n_beams x n_ant = 25 x 64
F1 = np.array([steering_vec(parameters['bs_antenna']['shape'], phi=azi,
                            spacing=parameters['bs_antenna']['spacing']).squeeze()
               for azi in beam_angles])

# Assuming 1 dBm transmit power
full_dbm = np.zeros((n_beams, n_subbands, n_users), dtype=float)
for ue_idx in tqdm(range(n_users), desc='Computing the channel for each user'):
    if data['user']['LoS'][ue_idx] == -1:
        full_dbm[:,:,ue_idx] = np.nan
    else:
        chs = F1 @ data['user']['channel'][ue_idx]
        full_linear = np.abs(np.mean(chs.squeeze().reshape((n_beams, n_subbands, -1)), axis=-1))
        full_dbm[:,:,ue_idx] = np.around(20*np.log10(full_linear) + 30, 1)

"""## Plot Received Power for each Beam"""

# Plot Coverage on each Beam and each Subband

for beam_idx in [12]:#range(n_beams):
    for subband_idx in range(n_subbands):
        rcv_pwr = full_dbm[beam_idx, subband_idx]

        title = f'Beam = {beam_idx} ({beam_angles[beam_idx]:.1f}º) | Subband = {subband_idx}'

        plot_coverage(data['user']['location'], rcv_pwr, bs_pos=bs_pos,
                      bs_ori=bs_ori, title=title, lims=[-180, -60])
        break
    break

"""## Plot Best beam for each position

"""

# Average the power on each subband and get the index of the beam that delivers max pwr
best_beams = np.argmax(np.mean(full_dbm,axis=1), axis=0)
best_beams = best_beams.astype(float)
best_beams[np.isnan(full_dbm[0,0,:])] = np.nan

plot_coverage(data['user']['location'], best_beams, bs_pos=bs_pos, bs_ori=bs_ori,
              title= 'Best Beams', cbar_title='Best beam index')

max_bf_pwr = np.max(np.mean(full_dbm,axis=1), axis=0) # assumes grid of beams!
plot_coverage(data['user']['location'], max_bf_pwr, bs_pos=bs_pos, bs_ori=bs_ori,
              title= 'Best Beamformed Power (assuming GoB) ')



"""# Convert Dataset to DeepMIMO
asdfadsf

## From Wireless InSite
"""

!wget -O asu_campus_insite_source.zip "https://www.dropbox.com/scl/fi/unldvnar22cuxjh7db2rf/ASU_Campus1.zip?rlkey=rs2ofv3pt4ctafs2zi3vwogrh&dl=0"
!unzip asu_campus_insite_source.zip

# rt_folder = ''
scen_name = dm.convert(rt_folder, scenario_name='prototype')
dataset_converted = dm.load(scen_name)

"""## From Sionna RT
Sionna is a bit more complicated because it doesn't have standard saving methods. Because of that, we use DeepMIMO exporter for Sionna, that saves the Scene, Path and computation parameters.

Below is an example of ray tracing a simple scene in Sionna and converting it to DeepMIMO.
"""

!pip install sionna



