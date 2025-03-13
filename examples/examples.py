# -*- coding: utf-8 -*-
#%%
import deepmimo as dm

#%% LOAD: Simple
import os
import numpy as np
import deepmimo as dm
import matplotlib.pyplot as plt

scen_name = 'asu_campus_3p5'
dm.download(scen_name)
dataset = dm.load(scen_name)[0]

#%% LOAD: Detailed 

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

# Like the information present in the scenario webpage
dm.summary('asu_campus')

#%% SCENARIO INFORMATION: Transmitters and Receivers

# There are paths between a transmitter and a set of receiver set. 
# There can also be transmitter sets with multiple transmitters, but we decouple
# the transmitters from the sets for keeping smaller, consistent matrices.

# Get all available TX-RX pairs
txrx_sets = dm.get_txrx_sets(scen_name)
pairs = dm.get_txrx_pairs(txrx_sets)

print(txrx_sets)
print(pairs)

dm.print_available_txrx_pair_ids(scen_name)

dataset.txrx


#%% SCENARIO INFORMATION: Ray Tracing Parameters

# This information is present in the scenario table and can be used to search and filter.
# (soon in dm.search())

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
             'los', 'distance', 'num_paths']
for key in main_keys:
    plt_var = dataset[key][:,0] if dataset[key].ndim == 2 else dataset[key]
    dm.plot_coverage(dataset.rx_pos, plt_var, bs_pos=dataset.tx_pos.T, title=key)

#%%  VISUALIZATION: Coverage Maps (3D)

#3D version
dm.plot_coverage(dataset.rx_pos, dataset['los'], bs_pos=dataset.tx_pos.T,
                bs_ori=dataset.tx_ori, title='LoS', cbar_title='LoS status',
                proj_3D=True, scat_sz=0.1)

#%% VISUALIZATION: Rays

dm.plot_rays(dataset.rx_pos[10], dataset.tx_pos[0],
             dataset.inter_pos[10], dataset.inter[10],
             proj_3D=False, color_by_type=True)

# 2D and 3D

#%% VISUALIZATION: Path Plots (1) Power in main path

# Percentage of the power in first path
pwr_in_first_path = dataset.lin_pwr[:, 0] / np.nansum(dataset.lin_pwr, axis=-1)

dm.plot_coverage(dataset.rx_pos, pwr_in_first_path, bs_pos=dataset.tx_pos.T,
                title='Percentage of power in 1st path', cbar_title='Percentage of power [%]')

#%% VISUALIZATION: Path Plots (2) Number of interactions in main path

dm.plot_coverage(dataset.rx_pos, dataset.num_interactions[:,0], bs_pos=dataset.tx_pos.T,
                title='Number of interactions in 1st path', cbar_title='Number of interactions')

#%% VISUALIZATION: Path Plots (3) First interaction in main path

first_bounce_codes = [code[0] if code else '' for code in dataset.inter_str[:,0]] # 'n', '2', '1', ...

unique_first_bounces = ['n', '', 'R', 'D', 'S']

coded_data = np.array([unique_first_bounces.index(code) for code in first_bounce_codes])

viridis_colors = plt.cm.viridis(np.linspace(0, 1, 4))  # Get 4 colors from viridis

dm.plot_coverage(dataset.rx_pos, coded_data,
                 bs_pos=dataset.tx_pos.T, scat_sz=5.5,
                 title='Type of first bounce of first path',
                 cmap=['white'] + viridis_colors.tolist(), # white for 'n'
                 cbar_labels=['None', 'LoS', 'R', 'D', 'S'])

#%% VISUALIZATION: Path Plots (4) Bounce profile in main path

# Full bounce profile visualization
unique_profiles = np.unique(dataset.inter_str[:,0])
print(f"\nUnique bounce profiles found: {unique_profiles}")

# Create mapping for full profiles
profile_to_idx = {profile: idx for idx, profile in enumerate(unique_profiles)}
full_profile_data = np.array([profile_to_idx[profile] for profile in dataset.inter_str])

# Create colormap with white for no interaction and viridis colors for the rest
n_profiles = len(unique_profiles)
viridis = plt.cm.viridis(np.linspace(0, 1, n_profiles - 1))  # Get colors for the rest

# Create decoded labels for the colorbar
profile_labels = ['-'.join(p) if p else 'LoS' for p in unique_profiles]

# Plot the full bounce profiles
dm.plot_coverage(dataset.rx_pos, full_profile_data,
                 bs_pos=dataset.tx_pos.T, scat_sz=5.5,
                 title='Full bounce profile of first path',
                 cmap=viridis.tolist() + ['white'],
                 cbar_labels=profile_labels)

#%% CHANNEL GENERATION: Parameters

# Create channel parameters with all options
ch_params = dm.ChannelGenParameters()

# Antenna parameters

# Base station antenna parameters
ch_params.bs_antenna.rotation = np.array([0, 0, 0])  # [az, el, pol] in degrees
ch_params.bs_antenna.fov = np.array([360, 180])      # [az, el] in degrees
ch_params.bs_antenna.shape = np.array([4, 4])        # [horizontal, vertical] elements
ch_params.bs_antenna.spacing = 0.5                   # Element spacing in wavelengths

# User equipment antenna parameters
ch_params.ue_antenna.rotation = np.array([0, 0, 0])  # [az, el, pol] in degrees
ch_params.ue_antenna.fov = np.array([360, 180])      # [az, el] in degrees
ch_params.ue_antenna.shape = np.array([1, 1])        # [horizontal, vertical] elements
ch_params.ue_antenna.spacing = 0.5                   # Element spacing in wavelengths

# Channel parameters
ch_params.freq_domain = True  # Whether to compute frequency domain channels
ch_params.num_paths = 10      # Number of paths

# OFDM parameters
ch_params.ofdm.bandwidth = 10e6                         # Bandwidth in Hz
ch_params.ofdm.num_subcarriers = 512                    # Number of subcarriers
ch_params.ofdm.selected_subcarriers = np.arange(1)      # Which subcarriers to generate
ch_params.ofdm.rx_lpf = 0                               # Receive Low Pass / ADC Filter

# Generate channels
dataset.compute_channels(ch_params)
dataset.channel.shape

#%% CHANNEL GENERATION: Parameters (2)
dm.info('ch_params')

#%% CHANNEL GENERATION: Time Domain (possibly for later?)

# Channel computation parameters
ch_params.freq_domain = False     # Whether to compute frequency domain channels

dataset.compute_channels(ch_params)
dataset.channel.shape

# TODO:  PLOT CIRs using the delays of each path

#%% CHANNEL GENERATION: Frequency Domain
ch_params = dm.ChannelGenParameters()

ch_params.num_paths = 5
ch_params.ofdm.bandwidth = 50e6
ch_params.ofdm.selected_subcarriers = np.arange(64)      # Which subcarriers to generate

channels = dataset.compute_channels(ch_params)

# Visualize channel magnitude response (NOTE: requires at >1 subcarriers and antennas)
user_idx = np.where(dataset.n_paths > 0)[0][0]
plt.imshow(np.abs(np.squeeze(channels[user_idx]).T))
plt.title('Channel Magnitude Response')
plt.xlabel('TX Antennas')
plt.ylabel('Subcarriers')
plt.show()

# NOTE: show the case of when there are too few subcarriers

#%% CHANNEL GENERATION: Frequency Domain (2) CIR
pwrs = dataset.power[user_idx]

plt.figure(dpi=200)
plt.stem(dataset.delay[user_idx]*10**6, pwrs, basefmt='none')
plt.xlabel('Time of arrival [us]')
plt.ylabel('Power per path [dBW]')
plt.grid()
plt.show()

#%% CHANNEL GENERATION: Frequency Domain (3) [LATER] Compare with Time Domain

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

#%% [LATER] CHANNEL GENERATION:

dm.plot_power_discarding(dataset)

def convert_channel_angle_delay(channel):
    """
    Requires a channel where:
        - The last dimension (-1) is subcarriers (frequency)
        - The second from last dimension (-2) is antennas (space)
        - Returns a channel like: ... x angle x delay 
    """
    # Inside FFT Conversion to Angle Domain: fft + shift across antennas (axis = 2)
    # Outside IFFT Conversion from Frequency (subcarriers) to Delay Domain (axis = 3)
    # Using singles leads to an error of 1e-14. The usual value of one channel entry is 1e-7. csingles are ok.
    return np.fft.ifft(np.fft.fftshift(np.fft.fft(channel, axis=-2), -2), axis=-1).astype(np.csingle)


def compute_DoA(channel, N, norm_ant_spacing=0.5, method_subcarriers='sum'):
    """
    expects the channel from one user in the form of "n_rx=1, n_tx, n_subcarriers"
    N = number of antenna elements
    norm_ant_spacing = element distance in wavelengths
    """
    # Create an array of bin indices
    n = np.arange(-N/2, N/2, 1)  # Adjusted for zero-centered FFT
    
    # Principle of DoA: the signal arriving from angle theta will be captured by each
    # antenna and have a constant phase difference across the elements. This phase difference
    # is proportional to the spacing between elements and the sin(angle of arrival). 
    # This phase shift will manifest itself as a (spatial) frequency when we take an FFT. 
    # Then we only need to see which bin (or frequency) has the most power and
    # convert that frequency to the angle of arrival. 
    
    # Calculate angles from bin indices
    theta = np.arcsin(n / (N * norm_ant_spacing))
    
    # Convert angles from radians to degrees
    theta_degrees = np.degrees(theta)
    
    # Assuming fft_results is your FFT output array
    if method_subcarriers == 'sum':
        f = np.sum
    elif method_subcarriers == 'mean':
        f = np.mean
    ch_ang = f(channel, axis=-1).squeeze()
    fft_results = np.fft.fftshift(np.fft.fft(ch_ang))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(theta_degrees, np.abs(fft_results))
    plt.title('FFT Output vs. Angle')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
    
    print(f'main direction of arrival = {theta_degrees[np.argmax(np.abs(fft_results))]:.2f}')


def plot_ang_delay(ch, n_ant=32, NC=32, title='', label_axis=True, bandwidth=50e6, spacing=.5):
    f, ax = plt.subplots(dpi=300)
    ax.imshow(np.squeeze(np.abs(ch))[:,:NC])
    # plt.imshow(np.squeeze(np.abs(ch2))[i][:,:50])
    #, extent=[0, 50, angles[0], angles[-1]]) # change limits!
    
    plt.title(title)
    plt.ylabel('angle bins')
    plt.xlabel('delay bins')
    
    if label_axis:
        # X-Axis
        n_xtickstep = 4
        plt.xlabel('delays bins [us]')
        delay_idxs = np.arange(NC)
        delay_labels = delay_idxs / (bandwidth) * 1e6
        ax.set_xticks(delay_idxs[::n_xtickstep])
        ax.set_xticklabels([f'{label:.1f}' for label in delay_labels[::n_xtickstep]])
        
        # Y-Axis
        n_ytickstep = 4
        plt.ylabel('angle bins [º]')
        # Create an array of bin indices
        n = np.arange(-n_ant/2, n_ant/2, 1)  # Adjusted for zero-centered FFT
        # Calculate angles from bin indices
        ang_degrees = np.degrees(np.arcsin(n / (n_ant * spacing)))
        ax.set_yticks(np.arange(n_ant)[::n_ytickstep])
        ax.set_yticklabels([f'{label:.0f}' for label in ang_degrees[::n_ytickstep]])
        
    plt.show()


#%% BASIC OPERATIONS: Line-of-Sight Status

active_mask = dataset.num_paths > 0
print(f"\nNumber of active positions: {np.sum(active_mask)}")
print(f"Number of inactive positions: {np.sum(~active_mask)}")

# Create scatter plot showing active vs inactive positions
plt.figure(figsize=(8, 6))
plt.scatter(dataset.rx_pos[~active_mask, 0], dataset.rx_pos[~active_mask, 1],
           alpha=0.5, s=1, c='red', label='Inactive')
plt.scatter(dataset.rx_pos[active_mask, 0], dataset.rx_pos[active_mask, 1],
           alpha=0.5, s=1, c='green', label='Active')
plt.legend()
plt.show()

dm.plot_coverage(dataset['rx_pos'], dataset.los != -1)

#%% BASIC OPERATIONS: Pathloss

non_coherent_pathloss = dataset.compute_pathloss(coherent=False)
coherent_pathloss = dataset.compute_pathloss(coherent=True) # default

dataset.plot_coverage(non_coherent_pathloss, title='Non-Coherent pathloss')
dataset.plot_coverage(coherent_pathloss, title='Coherent pathloss')

#%% BASIC OPERATIONS: Implicit Computations

# Implicit and lazy computations
# Functions are public when arguments are needed

# Public compute functions
dataset.channels          # calls dataset.compute_channels()
dataset.pathloss          # calls dataset.compute_pathloss()

# Hidden compute functions
dataset.distance           # calls dataset._compute_distances()
dataset.num_paths          # calls dataset._compute_num_paths()
dataset.num_interactions   # calls dataset._compute_num_interactions()
dataset.los                # calls dataset._compute_los()
dataset.n_ue               # calls dataset._compute_n_ue()
dataset.grid_size          # calls dataset._compute_grid_info()
dataset.grid_spacing       # calls dataset._compute_grid_info()

#%% BASIC OPERATIONS: Aliases

checks = [
    dataset.pwr is dataset.power,
    dataset.pl is dataset.pathloss,
    dataset.ch is dataset.channels,
    dataset.ch_params is dataset.channel_params,
    dataset.n_paths is dataset.num_paths,
    dataset.n_ue is dataset.n_ue,
    dataset.grid_size is dataset.grid_size,
    dataset.grid_spacing is dataset.grid_spacing,
]

for check in checks:
    print(check)

# add alias?

# dataset.tx_ori == dataset.bs_ori

#%% BASIC OPERATIONS: Attribute Access
for var_name in ['pl', 'rx_pos', 'aoa_az', 'channel', ]:
    a = dataset[var_name]
    b = getattr(dataset, var_name)
    print(f"dataset['{var_name}'] == dataset.{var_name}: {a is b}")

#%% BASIC OPERATIONS: Antenna Rotations (1) Azimuth

params = dm.ChannelGenParameters()
dataset.set_channel_params(params)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Define 3 different rotations to show
rotations = [np.array([0, 0, 0]),     # Facing +x
             np.array([0, 0, 180]),   # Facing -x
             np.array([0, 0, -135])]  # Facing 45º between -x and -y

titles = ['Orientation along +x (0°)', 
          'Orientation along -x (180°)', 
          'Orientation at 45º between -x and -y (-135°)']

# Plot each azimuth rotation
for i, (rot, title) in enumerate(zip(rotations, titles)):
    # Update channel parameters with new rotation
    dataset.ch_params.bs_antenna.rotation = rot
    
    # Create coverage plot in current subplot
    dm.plot_coverage(dataset.rx_pos, dataset.los, 
                     bs_pos=dataset.tx_pos.T, bs_ori=dataset.tx_ori,
                     ax=axes[i], title=title, cbar_title='LoS status')

plt.tight_layout()
plt.show()

#%% BASIC OPERATIONS: Antenna Rotations (2) Elevation

params = dm.ChannelGenParameters()
dataset.set_channel_params(params)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), subplot_kw={'projection': '3d'})

# Define 3 different rotations to show
rotations = [np.array([0,  0, -180]),   # Facing -x
             np.array([0, 30, -180]),   # Facing 30º below -x in XZ plane
             np.array([0, 60, -180])]   # Facing 60º below -x in XZ plane

titles = ['Orientation along -x (180°)', 
          'Orientation at 30º between -x and -z', 
          'Orientation at 60º between -x and -z']

# Plot each azimuth rotation
for i, (rot, title) in enumerate(zip(rotations, titles)):
    # Update channel parameters with new rotation
    dataset.ch_params.bs_antenna.rotation = rot
    
    # Create coverage plot in current subplot
    dataset.plot_coverage(dataset.los, proj_3D=True, ax=axes[i],
                          title=title, cbar_title='LoS status')
    axes[i].view_init(elev=5, azim=-90)  # Set view to xz plane to see tilt
    axes[i].set_yticks([])  # Remove y-axis ticks to unclutter the plot

plt.tight_layout()
plt.show()


#%% [LATER] ADVANCED OPERATIONS: Antenna Field-of-View (FoV)

params = dm.ChannelGenParameters()
params['bs_antenna']['rotation'] = np.array([0, 0, -135])
dataset.set_channel_params(params)

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), tight_layout=True)

# Define 3 FoV
fovs = [np.array([180, 180]),   # Facing -x
        np.array([ 90, 180]),   # Facing 30º below -x in XZ plane
        np.array([ 60, 180])]   # Facing 60º below -x in XZ plane

titles = ['FoV = 180°', 
          'FoV = 90°', 
          'FoV = 60°']

# Plot each azimuth rotation
for i, (fov, title) in enumerate(zip(fovs, titles)):
    # Update channel parameters with new rotation
    print(f"Iteration {i}: Setting FoV to {fov}")
    dataset.apply_fov(bs_fov=fov)

    dataset.plot_coverage(dataset.los, ax=axes[i], title=title, cbar_title='LoS status')

# Note, when applying fov, several cached values will be invalidated, like the los and channels

#%% SCENE & MATERIALS

print("\nScene and Materials Example")
print("-" * 50)

scene = dataset.scene

# 1. Basic scene information
print("\nScene Overview:")
print(f"- Total objects: {len(scene.objects)}")

# Get objects by category
buildings = scene.get_objects('buildings')
terrain = scene.get_objects('terrain')
vegetation = scene.get_objects('vegetation')

print(f"- Buildings: {len(buildings)}")
print(f"- Terrain: {len(terrain)}")
print(f"- Vegetation: {len(vegetation)}")

# 2. Materials and Filtering
materials = dataset.materials

# Get materials used by buildings
building_materials = buildings.get_materials()
print(f"\nMaterials used in buildings: {building_materials}")

# Different ways to filter objects
print("\nFiltering examples:")

# Filter by label only
buildings = scene.get_objects(label='buildings')
print(f"- Buildings: {len(buildings)}")

# Filter by material only
material_idx = building_materials[0]
objects_with_material = scene.get_objects(material=material_idx)
print(f"- Objects with material {material_idx}: {len(objects_with_material)}")

# Filter by both label and material
buildings_with_material = scene.get_objects(label='buildings', material=material_idx)
print(f"- Buildings with material {material_idx}: {len(buildings_with_material)}")

# Print material properties
material = materials[material_idx]
print(f"\nMaterial {material_idx} properties:")
print(f"- Name: {material.name}")
print(f"- Permittivity: {material.permittivity}")
print(f"- Conductivity: {material.conductivity}")

# 3. Object Properties
print("\nObject Properties:")
building = buildings[0]
print(f"- Building faces: {len(building.faces)}")
print(f"- Building height: {building.height:.2f}m")
print(f"- Building volume: {building.volume:.2f}m³")
print(f"- Building footprint area: {building.footprint_area:.2f}m²")

# 4. Bounding Boxes
print("\nBuildings Bounding Box:")
bb = buildings.bounding_box
print(f"- Width (X): {bb.width:.2f}m")
print(f"- Length (Y): {bb.length:.2f}m")
print(f"- Height (Z): {bb.height:.2f}m")

#%% USER SAMPLING: Dataset Trimming

# For sampling users, we always have to find first the indices of the users we want to keep
# Then, we can use them to index particular matrix, or the entire dataset -> subset() method

print("\nActive Users and Dataset Subsetting (Trimming) Example")
print("-" * 50)

# Get indices of active users (those with paths)
active_idxs = dataset.get_active_idxs()
print(f"Original dataset has {dataset.n_ue} UEs")
print(f"Found {len(active_idxs)} active UEs")

# Create new dataset with only active users
dataset_t = dataset.subset(active_idxs)
print(f"New dataset has {dataset_t.n_ue} UEs")

dataset_t.plot_coverage(dataset_t.aoa_az[:,0])

#%% USER SAMPLING: Uniform

idxs = dataset.get_uniform_idxs([4,4])
dm.plot_coverage(dataset.rx_pos[idxs], dataset.aoa_az[idxs, 0], bs_pos=dataset.tx_pos.T)

#%% USER SAMPLING: Linear (1) 2D plots across positions (with Class)

# Get the closest dataset positions for a given path
idxs1 = dm.LinearPath(dataset.rx_pos, [100, 90], [-50,  90], n_steps=75).idxs
idxs2 = dm.LinearPath(dataset.rx_pos, [100, 80], [-50,  80], n_steps=75).idxs
idxs3 = dm.LinearPath(dataset.rx_pos, [ 30,  0], [ 30, 150], n_steps=75).idxs

dataset.plot_coverage(dataset.los,title='LoS with positions', cbar_title='LoS status')

plt.scatter(dataset.rx_pos[idxs1,0], dataset.rx_pos[idxs1,1], c='blue', label='path1', s=6, lw=.1)
plt.scatter(dataset.rx_pos[idxs2,0], dataset.rx_pos[idxs2,1], c='cyan', label='path2', s=6, lw=.1)
plt.scatter(dataset.rx_pos[idxs3,0], dataset.rx_pos[idxs3,1], c='red',  label='path3', s=6, lw=.1)
plt.legend()

#%% USER SAMPLING: Linear (2) feature variation across linear path

for var_name in ['los', 'pathloss', 'delay']:
    plt.figure(dpi=200)
    data = dataset[var_name] if var_name != 'delay' else dataset[var_name][:,0]
    plt.plot(data[idxs1], ls='-',  c='blue', label='path1',marker='*', markersize=7)
    plt.plot(data[idxs2], ls='-.', c='cyan', label='path2',marker='s', markerfacecolor='none')
    plt.plot(data[idxs3], ls='--', c='red',  label='path3',marker='o', markerfacecolor='w')
    plt.xlabel('Position index')
    plt.ylabel(f'{var_name}')
    plt.grid()
    plt.legend()
    plt.show()

#%% USER SAMPLING: In rectangular zones

idxs_A = dm.get_idxs_with_limits(dataset.rx_pos, x_min=-100, x_max=-60, y_min=0, y_max=40)

idxs_B = dm.get_idxs_with_limits(dataset.rx_pos, x_min= 125, x_max=165, y_min=0, y_max=40)

# Plot boxes 
dataset.plot_coverage(dataset.aoa_az[:,0])

plt.scatter(dataset.rx_pos[idxs_A,0], dataset.rx_pos[idxs_A,1],
            label='box A', s=2, lw=.1, alpha=.3)
plt.scatter(dataset.rx_pos[idxs_B,0], dataset.rx_pos[idxs_B,1],
            label='box B', s=2, lw=.1, alpha=.3)
plt.title('Dataset zones on AoA Azimuth [º]')

#%% USER SAMPLING: In rectangular zones (2) Feature distributions
def plot_feat_dist(data_A, data_B, feat_name):
    """Plot histograms of coordinate distributions for two datasets.
    
    Args:
        data_A: Array of coordinates for dataset A
        data_B: Array of coordinates for dataset B
    """
    hist_params = {'alpha': 0.5, 'bins': 8, 'zorder':2}

    # dist on x
    plt.figure(dpi=200)
    plt.hist(data_A, **hist_params, label='A')
    plt.hist(data_B, **hist_params, label='B')
    plt.title(f'{feat_name} distribution')
    plt.xlabel(f'{feat_name}')
    plt.grid()
    plt.show()

plot_feat_dist(dataset.rx_pos[idxs_A, 0], dataset.rx_pos[idxs_B, 0], 'x (m)')
plot_feat_dist(dataset.rx_pos[idxs_A, 1], dataset.rx_pos[idxs_B, 1], 'y (m)')
plot_feat_dist(dataset.aoa_az[idxs_A, 0], dataset.aoa_az[idxs_B, 0], 'AoA Azimuth [º]')

#%% BEAMFORMING: Received Power with TX Beamforming
ch_params = dm.ChannelGenParameters()
ch_params.bs_antenna.rotation = np.array([0, 0, -135])
dataset.compute_channels(ch_params)

n_beams = 25

beam_angles = np.around(np.linspace(-60, 60, n_beams), 2)

# F1 is [n_beams, n_ant]
F1 = np.array([dm.steering_vec(dataset.ch_params.bs_antenna.shape, phi=azi,
                               spacing=dataset.ch_params.bs_antenna.spacing).squeeze()
               for azi in beam_angles])

recv_bf_pwr_dbm = np.zeros((dataset.n_ue, n_beams)) * np.nan
mean_amplitude = np.abs(F1 @ dataset.channel[dataset.los != -1]).mean(axis=1).mean(axis=-1)
# Avg over rx antennas and subcarriers, respectively
recv_bf_pwr_dbm[dataset.los != -1] = np.around(20*np.log10(mean_amplitude) + 30, 1)

#%% BEAMFORMING: Received Power with TX Beamforming (2) Plotting

fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300, tight_layout=True)

for plt_idx, beam_idx in enumerate([8, 12, 16]):
    dataset.plot_coverage(recv_bf_pwr_dbm[:, beam_idx], ax=axes[plt_idx], lims=[-180, -60],
                          title=f'Beam # {beam_idx} ({beam_angles[beam_idx]:.1f}º)')

#%% BEAMFORMING: Received Power with TX Beamforming (3) Plotting Best Beam

# Average the power on each subband and get the index of the beam that delivers max pwr
best_beams = np.argmax(recv_bf_pwr_dbm, axis=1).astype(float)
best_beams[np.isnan(recv_bf_pwr_dbm[:, 0])] = np.nan

dm.plot_coverage(dataset.rx_pos, best_beams, bs_pos=dataset.tx_pos.T, bs_ori=dataset.tx_ori,
                 title= 'Best Beams', cbar_title='Best beam index')

#%% BEAMFORMING: Received Power with TX Beamforming (4) Plotting Max Beamformed Receive Power

max_bf_pwr = np.max(recv_bf_pwr_dbm, axis=1) # assumes grid of beams!
dm.plot_coverage(dataset.rx_pos, max_bf_pwr, bs_pos=dataset.tx_pos.T, bs_ori=dataset.tx_ori,
              title= 'Best Beamformed Power (assuming GoB) ')

#%% [LATER] BEAMFORMING: (integrating beam angles and beamforming in dataset?)

def get_beam_angles(fov, n_beams=None, beam_res=None):
    """
    3 ways of computing beam angles:
        1- given the codebook size (n_beams) and fov --- compute resolution
        2- given codebook size and resolution        --- computes range (not done)
        3- given range and resolution                --- computes codebook size
    """
    
    if n_beams:
        angs = np.linspace(-fov/2, fov/2, n_beams)
    elif beam_res:
        angs = np.arange(-fov/2, fov/2+.001, beam_res)
    else:
        raise Exception('Not enough information to compute beam angles.')
    
    return angs

#%% CONVERTING DATASETS: From Wireless InSite

# !wget -O asu_campus_p2m.zip "https://www.dropbox.com/s/lgzw8am5v5qz06v/asu_campus_p2m.zip?e=1&st=pcon8w9l&dl=1"
# !unzip asu_campus_p2m.zip

rt_folder = 'asu_campus'
scen_name_insite = dm.convert(rt_folder, scenario_name='asu_campus_insite')
dataset_converted = dm.load(scen_name_insite)


#%% CONVERTING DATASETS: From Sionna RT (1) Install Sionna

# !pip install sionna

#%% CONVERTING DATASETS: From Sionna RT (2) Ray tracing

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, DirectivePattern

def compute_array_combinations(arrays):
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))

def gen_user_grid(box_corners, steps, box_offsets=None):
    """
    box_corners is = [bbox_min_corner, bbox_max_corner]
    steps = [x_step, y_step, z_step]
    """

    # Sample the ranges of coordinates
    ndim = len(box_corners[0])
    dim_ranges = []
    for dim in range(ndim):
        if steps[dim]:
            dim_range = np.arange(box_corners[0][dim], box_corners[1][dim], steps[dim])
        else:
            dim_range = np.array([box_corners[0][dim]]) # select just the first limit
        
        dim_ranges.append(dim_range + box_offsets[dim] if box_offsets else 0)
    
    pos = compute_array_combinations(dim_ranges)
    print(f'Total positions generated: {pos.shape[0]}')
    return pos


def create_base_scene(scene_path, center_frequency):
    scene = load_scene(scene_path)
    scene.frequency = center_frequency
    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")
    
    scene.rx_array = scene.tx_array
    scene.synthetic_array = True
    
    return scene



# Save dict with compute path params to export later
my_compute_path_params = dict(
    max_depth=5,
    num_samples=1e6,
    scattering=False,
    diffraction=False
)
carrier_freq = 3.5 * 1e9  # Hz

tx_pos = [-33, 11, 32.03]

# 0- Create/Fetch scene and get buldings in the scene
scene = create_base_scene(sionna.rt.scene.simple_street_canyon,
                          center_frequency=carrier_freq)

# 1- Compute TX position
print('Computing BS position')
scene.add(Transmitter(name="tx", position=tx_pos, orientation=[0,0,0]))

# 2- Compute RXs positions
print('Computing UEs positions')
d = 10
rxs = gen_user_grid(box_corners=[(-d, -d, 0), (d, d, 0)], steps=[2, 2, 0], box_offsets=[0, 0, 2])

# 3- Add the first batch of receivers to the scene
n_rx = len(rxs)
n_rx_in_scene = 5  # to compute in parallel
print(f'Adding users to the scene ({n_rx_in_scene} at a time)')
for rx_idx in range(n_rx_in_scene):
    scene.add(Receiver(name=f"rx_{rx_idx}", position=rxs[rx_idx], orientation=[0,0,0]))

# 4- Enable scattering in the radio materials
if my_compute_path_params['scattering']:
    for rm in scene.radio_materials.values():
        rm.scattering_coefficient = 1/np.sqrt(3) # [0,1]
        rm.scattering_pattern = DirectivePattern(alpha_r=10)

# 5- Compute the paths for each set of receiver positions
path_list = []
n_rx_remaining = n_rx
for x in tqdm(range(int(n_rx / n_rx_in_scene)+1), desc='Path computation'):
    if n_rx_remaining > 0:
        n_rx_remaining -= n_rx_in_scene
    else:
        break
    if x != 0:
        # modify current RXs in scene
        for rx_idx in range(n_rx_in_scene):
            if rx_idx + n_rx_in_scene*x < n_rx:
                scene.receivers[f'rx_{rx_idx}'].position = rxs[rx_idx + n_rx_in_scene*x]
            else:
                # remove the last receivers in the scene
                scene.remove(f'rx_{rx_idx}')
    
    paths = scene.compute_paths(**my_compute_path_params)
    
    paths.normalize_delays = False  # sum min_tau to tau, or tau of 1st path is always = 0
    
    path_list.append(paths)

#%% CONVERTING DATASETS: From Sionna RT (3) Convert to DeepMIMO

from deepmimo.converter.sionna_rt import sionna_exporter

save_folder = 'sionna_folder/'
sionna_exporter.export_to_deepmimo(scene, path_list, my_compute_path_params, save_folder)

scen_name_sionna = dm.convert(save_folder)

dataset_sionna = dm.load(scen_name_sionna)

dm.plot_coverage(dataset_sionna.rx_pos, dataset_sionna.los, bs_pos=dataset_sionna.tx_pos.T)

#%% UPLOADING DATASETS

dm.upload(scen_name_insite, key='')
dm.upload(scen_name_sionna, key='')
