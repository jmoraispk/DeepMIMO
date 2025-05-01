# Visualization

The visualization module provides comprehensive tools for visualizing DeepMIMO datasets.

## Core Functions

```{eval-rst}
.. autofunction:: deepmimo.generator.visualization.plot_coverage

.. autofunction:: deepmimo.generator.visualization.plot_rays

.. autofunction:: deepmimo.generator.visualization.export_xyz_csv

.. autofunction:: deepmimo.generator.visualization.plot_power_discarding
```

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

## Best Practices

1. Memory Management:
   ```python
   # For large datasets, plot in batches
   for batch in dataset.iter_batches(100):
       batch.plot_coverage(power[batch.indices])
   ```

2. Publication Quality:
   ```python
   # High-quality output
   dataset.plot_coverage(
       power,
       dpi=300,
       figsize=(10, 8),
       equal_aspect=True,
       tight=True
   )
   ```

3. Interactive Plots:
   ```python
   # Enable interactive features
   dataset.plot_coverage(
       power,
       proj_3D=True,
       legend=True,
       cbar_labels=custom_labels
   )
   ```

## See Also
- [Dataset](../objects/dataset.md) - Dataset documentation
- [Scene](../objects/scene.md) - Scene visualization 