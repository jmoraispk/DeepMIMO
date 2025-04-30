# DeepMIMO Objects

DeepMIMO uses a hierarchical object system to organize data and functionality. This section provides detailed information about each object type.

## Object Hierarchy

1. **MacroDataset**: Top-level container managing multiple datasets
   - Unified interface for multi-dataset operations
   - Parameter sharing across datasets
   - Batch processing capabilities

2. **Dataset**: Individual dataset containing TX/RX data
   - Channel computation and storage
   - Location management
   - Coverage analysis

3. **ChannelParameters**: Configuration for channel generation
   - Antenna configurations
   - Frequency settings
   - Processing options

## Object Relationships

```
MacroDataset
├── Dataset 1
│   ├── Channel Parameters
│   ├── TX Locations
│   └── RX Locations
├── Dataset 2
│   ├── Channel Parameters
│   ├── TX Locations
│   └── RX Locations
└── ...
```

## Quick Links

- [Dataset Documentation](dataset.md)
- [Channel Parameters Documentation](channel_params.md)

## Common Operations

```python
import deepmimo as dm

# Create and configure parameters
params = dm.ChannelParameters()
params.carrier_freq = 28e9

# Load datasets
macro_dataset = dm.load('scenario_name')
dataset = macro_dataset[0]

# Compute channels
channels = dataset.compute_channels(params)
```

## Core Objects

### [MacroDataset](dataset.md)
Container for multiple datasets. Useful when working with multiple scenarios or configurations.

```python
macro_dataset = dm.load('scenario_name')  # Returns MacroDataset
dataset = macro_dataset[0]  # Access first Dataset
```

### [Dataset](dataset.md)
Single scenario with consistent parameters. Main interface for most operations.

```python
# Access dataset attributes
print(dataset.rx_pos)  # Receiver positions
print(dataset.tx_pos)  # Transmitter positions
print(dataset.los)     # Line of sight status

# Compute channels
channels = dataset.compute_channels()

# Visualize
dataset.plot_coverage(dataset.power[:,0])
```

### [ChannelParameters](channel_params.md)
Configuration for channel generation and processing.

```python
# Configure parameters
params = dm.ChannelParameters()
params.carrier_freq = 28e9
params.bandwidth = 100e6
params.num_subcarriers = 64
```

## Best Practices

1. Use attribute access when possible:
   ```python
   # Good
   los_status = dataset.los
   
   # Also works, but less readable
   los_status = dataset['los']
   ```

2. Take advantage of lazy evaluation:
   ```python
   # Channels are computed only when needed
   dataset.channels  # Triggers computation
   ```

3. Use object methods for operations:
   ```python
   # Let objects handle complex operations
   dataset.compute_pathloss(coherent=True)
   ```

4. Chain visualization calls:
   ```python
   # Objects handle plot configuration
   dataset.plot_coverage(dataset.power[:,0],
                        title="Power Coverage",
                        proj_3D=True)
   ```

For more examples and detailed usage, see the [Quick Start Guide](../quickstart.md). 