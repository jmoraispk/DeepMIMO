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

