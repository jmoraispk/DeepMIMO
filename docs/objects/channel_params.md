# Channel Parameters

The `ChannelParameters` class provides configuration options for channel generation in DeepMIMO simulations.

## Overview

Channel parameters control how wireless channels are computed between transmitters and receivers, including antenna configurations, frequency settings, and processing options.

```python
import deepmimo as dm

# Create channel parameters
params = dm.ChannelParameters()

# Configure parameters
params.carrier_freq = 28e9  # 28 GHz
params.bandwidth = 100e6    # 100 MHz
```

## Parameter Categories

### Antenna Configuration

```python
# Base station antenna configuration
params.bs_antenna.shape = [8, 4]  # 8x4 antenna array
params.bs_antenna.spacing = [0.5, 0.5]  # Half-wavelength spacing

# User equipment antenna configuration
params.ue_antenna.shape = [2, 2]  # 2x2 antenna array
params.ue_antenna.spacing = [0.5, 0.5]
```

### Frequency Settings

```python
# Carrier frequency and bandwidth
params.carrier_freq = 28e9  # 28 GHz
params.bandwidth = 100e6    # 100 MHz
params.num_subcarriers = 64  # OFDM subcarriers
```

### Processing Options

```python
# Channel processing configuration
params.los_only = False  # Include multipath
params.apply_pathloss = True
params.normalize_channels = True
```

## Properties

| Category | Parameter | Type | Description | Default |
|----------|-----------|------|-------------|---------|
| Antenna | `bs_antenna.shape` | ndarray | BS antenna array dimensions | [1, 1] |
| | `ue_antenna.shape` | ndarray | UE antenna array dimensions | [1, 1] |
| | `bs_antenna.spacing` | ndarray | BS element spacing (λ) | [0.5, 0.5] |
| | `ue_antenna.spacing` | ndarray | UE element spacing (λ) | [0.5, 0.5] |
| Frequency | `carrier_freq` | float | Carrier frequency (Hz) | 28e9 |
| | `bandwidth` | float | System bandwidth (Hz) | 100e6 |
| | `num_subcarriers` | int | Number of OFDM subcarriers | 1 |
| Processing | `los_only` | bool | Consider only LOS paths | False |
| | `apply_pathloss` | bool | Apply pathloss to channels | True |
| | `normalize_channels` | bool | Normalize channel matrices | False |

## Methods

### `copy()`
Create a deep copy of the parameters.

### `validate()`
Validate parameter consistency.

### `to_dict()`
Convert parameters to dictionary format.

### `from_dict(dict_params)`
Load parameters from dictionary.

## Example Usage

### Basic Configuration
```python
params = dm.ChannelParameters()

# Configure antenna arrays
params.bs_antenna.shape = [8, 4]
params.ue_antenna.shape = [2, 2]

# Set frequency parameters
params.carrier_freq = 28e9
params.bandwidth = 100e6
params.num_subcarriers = 64

# Configure processing
params.los_only = False
params.apply_pathloss = True
```

### Advanced Usage
```python
# Create parameters for multiple scenarios
params_list = []
for freq in [28e9, 60e9]:
    params = dm.ChannelParameters()
    params.carrier_freq = freq
    params_list.append(params)

# Save/load parameters
params.save('channel_config.json')
loaded_params = dm.ChannelParameters.load('channel_config.json')
```

## Best Practices

1. Parameter Validation:
   ```python
   # Always validate parameters before use
   params.validate()
   ```

2. Configuration Management:
   ```python
   # Save configurations for reproducibility
   params.save('config.json')
   ```

3. Parameter Sharing:
   ```python
   # Share parameters across datasets
   for dataset in macro_dataset:
       dataset.set_channel_params(params)
   ```

## See Also
- [Dataset](dataset.md) - Dataset documentation
- [Generator](../api/generator.md) - Channel generation details 