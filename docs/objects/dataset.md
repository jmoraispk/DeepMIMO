# DeepMIMO Dataset Objects

This document describes the two main dataset classes in DeepMIMO: `Dataset` and `MacroDataset`.

## Overview

DeepMIMO's dataset hierarchy consists of:
1. `MacroDataset`: Top-level container managing multiple datasets
2. `Dataset`: Individual dataset containing channel, path, and position information

A typical usage pattern starts with loading a scenario:

```python
import deepmimo as dm

# Load scenario - returns a MacroDataset
macro_dataset = dm.load('scenario_name')

# Access individual dataset
dataset = macro_dataset[0]  # First dataset
```

## MacroDataset

### Description
The `MacroDataset` class is the top-level container in DeepMIMO's object hierarchy. It manages multiple datasets and provides unified access to their data. It acts as a simple wrapper around a list of `Dataset` objects, automatically propagating operations to all contained datasets.

### Key Features

#### Dataset Management
- Container for multiple `Dataset` objects
- Unified interface for multi-dataset operations
- Consistent parameter handling across datasets
- Automatic propagation of operations to all child datasets

#### Access Methods
```python
# Access individual datasets
dataset = macro_dataset[0]  # First dataset
datasets = macro_dataset[1:3]  # Slice of datasets

# Iterate over datasets
for dataset in macro_dataset:
    print(f"Dataset has {len(dataset)} users")
```

#### Batch Operations
```python
# Compute channels for all datasets
channels = macro_dataset.compute_channels()

# Apply operations across datasets
macro_dataset.normalize()
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_datasets` | int | Number of contained datasets |
| `parameters` | dict | Shared parameters across datasets |
| `datasets` | list | List of contained Dataset objects |

### Best Practices

1. Use MacroDataset for:
   - Multi-scenario analysis
   - Batch processing
   - Comparative studies

2. Memory Management:
   ```python
   # Load only needed datasets
   macro_dataset = dm.load('scenario_name',
                          tx_sets={1: [0, 1]},  # Specific TX points
                          rx_sets={2: 'all'})   # All RX in set 2
   ```

3. Parameter Sharing:
   ```python
   # Set parameters once for all datasets
   macro_dataset.set_channel_params(ch_params)
   ```

## Dataset

### Description
The `Dataset` class represents a single dataset within DeepMIMO, containing transmitter, receiver, and channel information for a specific scenario configuration. It provides comprehensive functionality for managing and processing wireless channel data.

### Key Features

#### Data Management
- Storage and access of transmitter (TX) and receiver (RX) locations
- Channel data organization and computation
- Path information management (angles, powers, delays)
- Parameter management for the specific dataset

#### Access Methods
```python
# Access transmitter data
tx_locations = dataset.tx_locations
n_tx = len(dataset.tx_locations)

# Access receiver data
rx_locations = dataset.rx_locations
n_rx = len(dataset.rx_locations)

# Access channel data
channels = dataset.channels  # If already computed
```

#### Channel Operations
```python
# Compute channels
channels = dataset.compute_channels()

# Apply processing
dataset.normalize_channels()
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tx_locations` | ndarray | Transmitter locations (N×3) |
| `rx_locations` | ndarray | Receiver locations (M×3) |
| `channels` | ndarray | Channel matrices |
| `parameters` | dict | Dataset-specific parameters |
| `power` | ndarray | Path powers in dBm |
| `phase` | ndarray | Path phases in degrees |
| `delay` | ndarray | Path delays in seconds |
| `aoa_az/aoa_el` | ndarray | Angles of arrival (azimuth/elevation) |
| `aod_az/aod_el` | ndarray | Angles of departure (azimuth/elevation) |
| `inter` | ndarray | Path interaction indicators |
| `inter_pos` | ndarray | Path interaction positions |

### Core Functionality

#### Channel Generation
```python
# Configure channel parameters
ch_params = dm.ChannelGenParameters()
ch_params.bs_antenna.shape = np.array([8, 1])

# Compute channels
channels = dataset.compute_channels(ch_params)
```

#### Path Analysis
```python
# Access path information
pathloss = dataset.compute_pathloss()
los_status = dataset.los  # Line of sight status
num_paths = dataset.num_paths  # Paths per user
```

#### Field of View (FoV) Operations
```python
# Apply FoV filtering
dataset.apply_fov(bs_fov=[120, 60],  # [horizontal, vertical] in degrees
                 ue_fov=[360, 180])  # Full sphere for UE
```

#### Visualization
```python
# Plot coverage
coverage = dataset.compute_coverage()
dataset.plot_coverage(coverage)

# Plot rays for specific user
dataset.plot_rays(user_idx=0)
```

### Best Practices

1. Memory Management:
   ```python
   # Clear channel data when no longer needed
   dataset.clear_channels()
   ```

2. Parameter Configuration:
   ```python
   # Set dataset-specific parameters
   dataset.set_channel_params(ch_params)
   ```

3. Data Processing:
   ```python
   # Process channels in batches
   for batch in dataset.channel_batches(batch_size=100):
       process_batch(batch)
   ```

4. Grid Operations:
   ```python
   # Get uniform sampling of users
   uniform_idxs = dataset.get_uniform_idxs([2, 2])  # [x_step, y_step]
   subset = dataset.subset(uniform_idxs)
   ```

## See Also
- [Channel Parameters](channel_params.md) - Channel parameter configuration
- [Channel Generation](../api/generator.md) - Channel generation details 