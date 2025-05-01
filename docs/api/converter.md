# Data Converter

The converter module handles conversion of ray-tracing data from various formats into the standardized DeepMIMO format.

## Core Functions

```{eval-rst}
.. autofunction:: deepmimo.converter.convert
```

## Supported Ray Tracers

DeepMIMO supports converting data from the following ray tracers:

### Wireless InSite
[Remcom Wireless InSite](https://www.remcom.com/wireless-insite-em-propagation-software) is a commercial ray-tracing software that provides detailed propagation modeling.

### Sionna RT
[Sionna RT](https://nvlabs.github.io/sionna/) is an open-source ray tracer built on top of the Sionna deep learning framework.

### AODT
AODT (Automotive Obstruction Detection and Tracking) is a specialized ray tracer for vehicular scenarios.

## Usage Examples

```python
import deepmimo as dm

# Convert Wireless InSite data
scenario = dm.convert('path/to/insite/data',
                     antenna_pattern='isotropic',
                     frequency=28e9)

# Convert Sionna data
scenario = dm.convert('path/to/sionna/data',
                     overwrite=True)

# Convert AODT data
scenario = dm.convert('path/to/aodt/data')
```

## File Format Guidelines

For detailed information about the expected file formats and organization for each ray tracer, please refer to the {doc}`../raytracing_guidelines` guide.

## Configuration

### Parameters
- `input_format`: Source data format
- `output_format`: Target data format (default: DeepMIMO)
- `options`: Format-specific options

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

