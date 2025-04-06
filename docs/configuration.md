# DeepMIMO Configuration System

The DeepMIMO configuration system provides a way to set and retrieve global configuration values that can be accessed from anywhere in the code. This is useful for setting parameters like ray tracer version, GPU usage, and other global settings.

## Basic Usage

### Importing the Configuration

```python
import deepmimo
```

### Getting Configuration Values

There are several ways to get configuration values:

```python
# Using the function-like interface
version = deepmimo.config('ray_tracer_version')

# Using the get method
version = deepmimo.config.get('ray_tracer_version')
```

### Setting Configuration Values

There are several ways to set configuration values:

```python
# Using the function-like interface with positional arguments
deepmimo.config('ray_tracer_version', '4.0.0')

# Using the function-like interface with keyword arguments
deepmimo.config(use_gpu=True, gpu_device_id=1)

# Using the set method
deepmimo.config.set('ray_tracer_version', '4.0.0')
```

### Printing All Configuration Values

```python
# Using the function-like interface
deepmimo.config()

# Using the print_config method
deepmimo.config.print_config()
```

### Resetting to Defaults

```python
deepmimo.config.reset()
```

### Getting All Configuration Values as a Dictionary

```python
all_config = deepmimo.config.get_all()
for key, value in all_config.items():
    print(f"{key}: {value}")
```

## Available Configuration Parameters

The following configuration parameters are available by default:

### Ray Tracing Parameters
- `ray_tracer_version`: The version of the ray tracer to use (default: '3.0.0')
- `use_gpu`: Whether to use GPU for computations (default: False)
- `gpu_device_id`: The ID of the GPU device to use (default: 0)

## Adding Custom Configuration Parameters

You can add custom configuration parameters by simply setting them:

```python
deepmimo.config.set('my_custom_param', 'my_value')
```

If the parameter doesn't exist, it will be added to the configuration.

## Example

See the `examples/config_example.py` file for a complete example of how to use the configuration system. 