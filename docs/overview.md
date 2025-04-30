# DeepMIMO Overview

DeepMIMO is organized into three main modules and several supporting components, designed to provide a complete workflow for MIMO channel modeling and dataset generation.

## Core Modules

### A. Convert Module
The Convert module handles the transformation of ray-tracing data from various sources into the standardized DeepMIMO format.

```python
import deepmimo as dm

# Convert from different sources
scen_name_insite = dm.convert(rt_folder, scenario_name='my_scenario')
scen_name_sionna = dm.convert(sionna_folder)

# Load converted dataset
dataset = dm.load(scen_name_insite)
```

### B. Database API
The Database API provides tools for accessing, sharing, and discovering DeepMIMO datasets.

```python
# Download a specific dataset
scen_name = 'asu_campus_3p5'
dm.download(scen_name)

# Load dataset
dataset = dm.load(scen_name)

# Get available scenarios
scenarios = dm.get_available_scenarios()
print(f"Found {len(scenarios)} scenarios")

# Upload your dataset (requires authentication)
dm.upload(scen_name, key='your_key')
```

### C. Generator
The Generator module provides tools for loading, computing, and visualizing MIMO channels.

```python
# Load a dataset
dataset = dm.load(scen_name)

# Configure channel parameters
ch_params = dm.ChannelGenParameters()
ch_params.bs_antenna.rotation = np.array([0, 0, 0])  # [az, el, pol] in degrees
ch_params.bs_antenna.shape = np.array([8, 1])        # [horizontal, vertical] elements
ch_params.bs_antenna.spacing = 0.5                   # Element spacing in wavelengths

# Compute channels
dataset.set_channel_params(ch_params)
channels = dataset.compute_channels()

# Visualize
dataset.plot_coverage(dataset.los, title='Line of Sight Status')
dataset.plot_rays(user_idx=100, proj_3D=True)
```

## Objects Hierarchy

DeepMIMO uses a clear object hierarchy to organize data:

```
MacroDataset
└── Dataset
    └── Scene
        ├── Materials
        ├── Rays
        └── Channels
```

- **MacroDataset**: Container for multiple datasets
- **Dataset**: Single scenario with consistent parameters
- **Scene**: Specific configuration of transmitters and receivers
- **Materials**: Physical properties of surfaces
- **Matrix**: Convenient interface for data manipulation (coming soon)

## Integrations (Coming Soon)

DeepMIMO will integrate with popular wireless communication tools:
- Sionna
- NeoRadium
- MATLAB 5G Toolbox

## Pipelines (Coming Soon)

The Pipelines feature will provide end-to-end workflows for common use cases. Contact us if you're interested in collaborating on pipeline development.

## Getting Help

Remember that you can use the `dm.info()` function to get detailed information about any concept:

```python
# Get information about angles convention
dm.info('angles')

# Learn about channel computation
dm.info('compute_channels')

# Understand dataset structure
dm.info('Dataset')

# Get scenario summary
dm.summary('asu_campus_3p5')
```

For more examples and tutorials:
1. Check the [DeepMIMO Manual](https://colab.research.google.com/drive/1U-e2rLDJYW-VcbJ7C3H2dqC625JJH5FF) notebook
2. Watch our [Video Tutorials](https://deepmimo.net/tutorials)