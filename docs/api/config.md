# Data Converter

The converter module handles conversion of ray-tracing data from various formats.

## Supported Formats

### InSite Format
Convert Remcom InSite ray-tracing data.

```python
from deepmimo.converter import convert_insite

dataset = convert_insite('path/to/insite/data')
```

### Sionna Format
Convert Sionna ray-tracing data.

```python
from deepmimo.converter import convert_sionna

dataset = convert_sionna('path/to/sionna/data')
```

## Configuration

This module provides configuration management for DeepMIMO through a singleton configuration instance.

## Global Configuration Interface

The configuration interface is provided through a singleton instance `deepmimo.config`:

```{eval-rst}
.. autoclass:: deepmimo.config.DeepMIMOConfig
   :members: set, get, reset, print_config, get_all
   :undoc-members:
   :show-inheritance:
```

## Basic Usage

### Getting Configuration Values

```python
import deepmimo as dm

# Using the function-like interface
version = dm.config('ray_tracer_version')

# Using the get method
version = dm.config.get('ray_tracer_version')

# Get all values as a dictionary
all_config = dm.config.get_all()
for key, value in all_config.items():
    print(f"{key}: {value}")
```

### Setting Configuration Values

```python
# Using the function-like interface with positional arguments
dm.config('ray_tracer_version', '4.0.0')

# Using the function-like interface with keyword arguments
dm.config(use_gpu=True, gpu_device_id=1)

# Using the set method
dm.config.set('ray_tracer_version', '4.0.0')

# Adding custom parameters
dm.config.set('my_custom_param', 'my_value')
```

### Managing Configuration

```python
# Print current configuration
dm.config.print_config()

# Reset to defaults
dm.config.reset()
```

## Default Configuration Parameters

### Ray Tracing Parameters
- `wireless_insite_version`: Default Wireless InSite version
- `sionna_version`: Default Sionna version
- `aodt_version`: Default AODT version
- `ray_tracer_version`: The version of the ray tracer to use (default: '3.0.0')

### Computation Settings
- `use_gpu`: Whether to use GPU acceleration (default: False)
- `gpu_device_id`: GPU device ID to use (default: 0)

### File Paths
- `scenarios_folder`: Folder containing scenario files (default: 'deepmimo_scenarios')

## Example Usage

```python
import deepmimo as dm

# Configure ray tracing settings
dm.config(
    ray_tracer_version='4.0.0',
    use_gpu=True,
    gpu_device_id=0
)

# Set custom scenario path
dm.config.set('scenarios_folder', '/path/to/scenarios')

# Print configuration
dm.config.print_config()

# Reset if needed
dm.config.reset()
```

See the `examples/config_example.py` file for a complete example of how to use the configuration system.

## Methods

### `convert_insite(path, **options)`
Convert InSite data to DeepMIMO format.

### `convert_sionna(path, **options)`
Convert Sionna data to DeepMIMO format.

## Examples

```python
import deepmimo as dm
from deepmimo.converter import convert_insite

# Convert InSite data
dataset = convert_insite('path/to/insite',
                        antenna_pattern='isotropic',
                        frequency=28e9)

# Save converted data
dm.save_dataset(dataset, 'converted_data.h5')
```

