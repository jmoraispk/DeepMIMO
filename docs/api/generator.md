# Generator Module

The generator module is the core component of DeepMIMO that handles the generation of MIMO channel datasets from ray-tracing data.

## Core Functions

```{eval-rst}
.. autofunction:: deepmimo.generate

.. autofunction:: deepmimo.load
```

## Dataset Classes

```{eval-rst}
.. autoclass:: deepmimo.generator.dataset.Dataset
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: deepmimo.generator.dataset.MacroDataset
   :members:
   :undoc-members:
   :show-inheritance:
```

## Channel Generation

```{eval-rst}
.. autoclass:: deepmimo.generator.channel.ChannelGenParameters
   :members:
   :undoc-members:
   :show-inheritance:
```

## Antenna Patterns

```{eval-rst}
.. autoclass:: deepmimo.generator.ant_patterns.AntennaPattern
   :members:
   :undoc-members:
   :show-inheritance:
```

## Geometry Functions

```{eval-rst}
.. autofunction:: deepmimo.generator.geometry.steering_vec

.. autofunction:: deepmimo.generator.geometry._rotate_angles_batch

.. autofunction:: deepmimo.generator.geometry._apply_FoV_batch

.. autofunction:: deepmimo.generator.geometry._array_response_batch

.. autofunction:: deepmimo.generator.geometry._ant_indices
```

## Examples

### Basic Dataset Generation

```python
import deepmimo as dm

# Load a scenario
dataset = dm.load('O1_60')

# Configure channel parameters
params = dm.ChannelGenParameters()
params.carrier_freq = 28e9
params.num_paths = 10

# Generate channels
channels = dataset.compute_channels(params)
```

### Advanced Channel Generation

```python
# Configure antenna arrays
params = dm.ChannelGenParameters()
params.bs_antenna.shape = [8, 1]     # 8-element ULA at base station
params.bs_antenna.spacing = 0.5      # Half-wavelength spacing
params.bs_antenna.rotation = [0,0,0] # No rotation

# Configure OFDM parameters
params.ofdm.bandwidth = 100e6        # 100 MHz bandwidth
params.ofdm.subcarriers = 1024       # 1024 subcarriers

# Generate frequency-domain channels
params.freq_domain = True
channels = dataset.compute_channels(params)
```

### Working with MacroDatasets

```python
# Load multiple base stations
macro_dataset = dm.load('city_scenario')

# Apply same parameters to all datasets
for dataset in macro_dataset:
    dataset.set_channel_params(params)
    
# Get channels for all base stations
all_channels = macro_dataset.channels
```

# Channel Generator

The channel generator module provides functionality for computing MIMO channels.

## Core Classes

### `ChannelGenerator`
Main class for channel generation.

```python
from deepmimo.generator import ChannelGenerator

generator = ChannelGenerator()
channels = generator.compute(dataset, params)
```

## Configuration

### Parameters
- `antenna_config`: Antenna array configuration
- `frequency_config`: Frequency domain settings
- `processing_config`: Signal processing options

## Methods

### `compute(dataset, params)`
Compute channels for a dataset.

### `compute_batch(dataset, params, batch_size)`
Compute channels in batches to manage memory.

## Examples

```python
import deepmimo as dm
from deepmimo.generator import ChannelGenerator

# Create generator
generator = ChannelGenerator()

# Load dataset
dataset = dm.load('scenario_name')[0]

# Configure parameters
params = dm.ChannelParameters()
params.carrier_freq = 28e9

# Compute channels
channels = generator.compute(dataset, params)
```

Generator Module

The generator module is the core component of DeepMIMO that handles the generation of MIMO channel datasets from ray-tracing data.

Core Components

Dataset Management

   :undoc-members:
   :show-inheritance:

   The Dataset class is the primary container for DeepMIMO data, providing:
* Channel matrices and path information storage
* Automatic computation of derived quantities
* Field of view and antenna pattern application
* Grid-based sampling capabilities

Core Processing

   :undoc-members:
   :show-inheritance:

   Core functionality for:
* Dataset generation and scenario management
* Ray-tracing data loading and processing
* Channel computation and parameter validation
* Multi-user MIMO channel generation

Channel Generation

   :undoc-members:
   :show-inheritance:

   Channel generation components including:
* Channel parameter management (ChannelGenParameters)
* OFDM processing and path verification
* MIMO channel matrix computation

Geometry and Array Response

   :undoc-members:
   :show-inheritance:

   Geometric computations for:
* Antenna array response calculation
* Angle rotation and transformation
* Field of view filtering
* Array indexing utilities

Antenna Patterns

   :undoc-members:
   :show-inheritance:

   Antenna pattern functionality:
* Standard antenna pattern implementations
* Pattern application to path gains
* Custom pattern support

## Visualization

See [Visualization](visualization.md) for details on plotting and visualization tools.

## Utilities

See [Utility Functions](utils.md) for helper functions and tools.

## Integrations

See [Integrations](integrations.md) for details on external tool integration.
