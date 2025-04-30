# Visualization

The visualization module provides comprehensive tools for visualizing DeepMIMO datasets.

## Coverage Maps

Create detailed coverage maps with customizable parameters:

```python
import deepmimo as dm
import numpy as np

# Create coverage map
fig, ax, cbar = dm.plot_coverage(
    rxs=user_positions,          # Shape: (n_users, 3)
    cov_map=received_power,      # Values to visualize
    bs_pos=base_station_pos,     # Optional: BS location
    bs_ori=base_station_ori,     # Optional: BS orientation
    proj_3D=True,               # 3D visualization
    cmap='viridis'              # Color scheme
)
```

## Ray Path Visualization

Visualize ray paths and interaction points:

```python
# Plot ray paths for a specific user
fig, ax = dm.plot_rays(
    rx_loc=user_position,        # Receiver location
    tx_loc=transmitter_pos,      # Transmitter location
    inter_pos=interaction_pos,   # Interaction points
    inter=interaction_types,     # Interaction types
    color_by_type=True,         # Color code by interaction type
    proj_3D=True                # 3D visualization
)
```

## Data Export

Export visualization data to external tools:

```python
# Export to CSV for external tools
dm.export_xyz_csv(
    data=dataset,               # DeepMIMO dataset
    z_var=values_to_export,     # Values to export
    outfile='coverage.csv',     # Output filename
    google_earth=True           # Convert to geo coordinates
)
```

## Features

The visualization module supports:
- Multiple color schemes via matplotlib colormaps
- 2D and 3D projections
- Adjustable figure sizes and DPI
- Custom axis limits and scaling
- Categorical and continuous color bars
- Geographic coordinate transformation
- Export capabilities for external tools

## Examples

### Coverage Analysis
```python
# Load dataset
dataset = dm.load('scenario_name')[0]

# Compute and plot coverage
power = dataset.compute_power()
dataset.plot_coverage(
    power,
    title="Power Coverage Map",
    colorbar_label="Power (dBm)",
    proj_3D=True
)
```

### Ray Tracing Visualization
```python
# Plot rays for specific user
dataset.plot_rays(
    user_idx=0,
    max_paths=10,
    show_interactions=True,
    color_by_power=True
)
```

### Custom Plots
```python
# Create custom visualization
fig, ax = plt.subplots(figsize=(10, 8))
dataset.plot_scene(ax=ax)
dataset.plot_coverage(power, ax=ax, alpha=0.5)
dataset.plot_rays(user_idx=0, ax=ax)
plt.show()
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
       bbox_inches='tight',
       save_path='coverage_map.png'
   )
   ```

3. Interactive Plots:
   ```python
   # Enable interactive features
   dataset.plot_scene(
       interactive=True,
       show_controls=True
   )
   ```

## See Also
- [Dataset](../objects/dataset.md) - Dataset documentation
- [Scene](../objects/scene.md) - Scene visualization 