# Visualization

The visualization module provides tools for visualizing DeepMIMO datasets, including coverage maps, ray paths, and channel characteristics.

## Core Functions

### plot_coverage()
Create coverage map visualization for user positions.

```python
import deepmimo as dm

# Plot coverage with default settings
dm.plot_coverage(rxs, cov_map)

# Customize visualization
fig, ax, cbar = dm.plot_coverage(
    rxs,                    # User positions (N×3)
    cov_map,               # Coverage values
    dpi=100,               # Plot resolution
    figsize=(6, 4),        # Figure size
    cbar_title='Power',    # Colorbar title
    title=True,            # Show title
    scat_sz=0.5,          # Marker size
    bs_pos=bs_position,    # Base station position
    bs_ori=bs_orientation, # Base station orientation
    legend=True,          # Show legend
    proj_3D=False,        # 2D/3D projection
    cmap='viridis'        # Color map
)
```

### plot_rays()
Plot ray paths between transmitter and receiver with interaction points.

```python
# Plot ray paths
fig, ax = dm.plot_rays(
    rx_loc,          # Receiver location
    tx_loc,          # Transmitter location
    inter_pos,       # Interaction positions
    inter,           # Interaction types
    figsize=(10, 8), # Figure size
    dpi=100,         # Plot resolution
    proj_3D=True,    # 3D projection
    color_by_type=True  # Color by interaction type
)
```

### plot_power_discarding()
Analyze and visualize power discarding due to path delays.

```python
# Analyze power discarding
fig, ax = dm.plot_power_discarding(
    dataset,
    trim_delay=None  # Use OFDM symbol duration
)
```

## Visualization Settings

### Coverage Map Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dpi` | int | 100 | Plot resolution |
| `figsize` | tuple | (6,4) | Figure dimensions |
| `cbar_title` | str | '' | Colorbar title |
| `title` | bool/str | False | Plot title |
| `scat_sz` | float | 0.5 | Marker size |
| `legend` | bool | False | Show legend |
| `proj_3D` | bool | False | 3D projection |
| `equal_aspect` | bool | False | Equal axis scaling |
| `cmap` | str | 'viridis' | Color map |

### Ray Path Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `figsize` | tuple | (10,8) | Figure dimensions |
| `dpi` | int | 100 | Plot resolution |
| `proj_3D` | bool | True | 3D projection |
| `color_by_type` | bool | False | Color by interaction |

### Interaction Colors

| Type | Color | Description |
|------|-------|-------------|
| 0 | green | Line-of-sight |
| 1 | red | Reflection |
| 2 | orange | Diffraction |
| 3 | blue | Scattering |
| 4 | purple | Transmission |
| -1 | gray | Unknown |

## Module Structure

```
visualization.py
  ├── plot_coverage() (Coverage maps)
  ├── plot_rays() (Ray paths)
  ├── plot_power_discarding() (Power analysis)
  └── _create_colorbar() (Helper function)
```

## Dependencies

The visualization module depends on:
- `matplotlib` for plotting
- `numpy` for numerical operations
- `tqdm` for progress bars

## Import Paths

```python
# Main functions
from deepmimo.generator.visualization import (
    plot_coverage,
    plot_rays,
    plot_power_discarding
)
```

## Best Practices

1. Coverage Maps
   - Use appropriate color maps for data type
   - Add colorbar and title for clarity
   - Consider 2D vs 3D based on data

2. Ray Paths
   - Use 3D for complex environments
   - Color by interaction type for analysis
   - Add legend for interaction types

3. Performance
   - Adjust DPI and figure size for balance
   - Use appropriate marker sizes
   - Consider memory for large datasets

## Examples

### Coverage Maps

```python
import deepmimo as dm
import numpy as np

# Load dataset
dataset = dm.load('O1_60')[0]

# Basic coverage plot
dataset.plot_coverage(dataset.los, title='Line of Sight Status')

# 3D coverage plot with base station
dm.plot_coverage(dataset.rx_pos, dataset.los, 
                bs_pos=dataset.tx_pos.T,
                bs_ori=dataset.tx_ori, 
                title='LoS', 
                cbar_title='LoS status',
                proj_3D=True)

# Plot multiple features
features = ['aoa_az', 'aoa_el', 'aod_az', 'aod_el', 
           'delay', 'power', 'phase', 'los', 'num_paths']

for key in features:
    plt_var = dataset[key][:,0] if dataset[key].ndim == 2 else dataset[key]
    dataset.plot_coverage(plt_var, title=key)
```

### Ray Path Visualization

```python
# Plot rays for a specific user with line-of-sight
los_user_idx = np.where(dataset.los == 1)[0][0]
dataset.plot_rays(los_user_idx, proj_3D=True)

# Analyze power in main path
pwr_in_first_path = dataset.lin_pwr[:, 0] / np.nansum(dataset.lin_pwr, axis=-1) * 100

dm.plot_coverage(dataset.rx_pos, pwr_in_first_path, 
                bs_pos=dataset.tx_pos.T,
                title='Percentage of power in 1st path', 
                cbar_title='Percentage of power [%]')
```

### Path Analysis

```python
# Plot number of interactions in main path
dm.plot_coverage(dataset.rx_pos, 
                dataset.num_interactions[:,0], 
                bs_pos=dataset.tx_pos.T,
                title='Number of interactions in 1st path', 
                cbar_title='Number of interactions')

# Analyze first interaction type
first_bounce_codes = [code[0] if code else '' for code in dataset.inter_str[:,0]]
unique_first_bounces = ['n', '', 'R', 'D', 'S']
coded_data = np.array([unique_first_bounces.index(code) for code in first_bounce_codes])

dm.plot_coverage(dataset.rx_pos, coded_data,
                bs_pos=dataset.tx_pos.T, 
                scat_sz=5.5,
                title='Type of first bounce of first path',
                cmap=['white'] + plt.cm.viridis(np.linspace(0, 1, 4)).tolist(),
                cbar_labels=['None', 'LoS', 'R', 'D', 'S'])
```

### Power Analysis

```python
# Analyze power discarding due to delay
dm.plot_power_discarding(dataset)

# Compare coherent vs non-coherent pathloss
non_coherent_pl = dataset.compute_pathloss(coherent=False)
coherent_pl = dataset.compute_pathloss(coherent=True)

_, axes = plt.subplots(1, 2, figsize=(12, 5))
dataset.plot_coverage(non_coherent_pl, 
                     title='Non-Coherent pathloss', 
                     ax=axes[0])
dataset.plot_coverage(coherent_pl, 
                     title='Coherent pathloss', 
                     ax=axes[1])
```

### Data Export

```python
# Export coverage data for external tools
dm.export_xyz_csv(
    data=dataset,               
    z_var=dataset.power,        # Values to visualize
    outfile='coverage.csv',     # Output filename
    google_earth=True           # Convert to geo coordinates
)
```

## See Also
- [Dataset](../objects/dataset.md) - Dataset documentation
- [Scene](../objects/scene.md) - Scene visualization 