# Converter

The converter module provides functionality to automatically detect and convert raytracing data from various supported formats into a standardized DeepMIMO format.

## Core Function

### convert()
Convert raytracing data to DeepMIMO format.

```python
import deepmimo as dm

# Convert raytracing data
scenario = dm.convert(
    path_to_rt_folder,  # Path to raytracing data
    **conversion_params  # Additional parameters
)
```

## Supported Formats

### Wireless InSite
```python
# Convert Wireless InSite project
scenario = dm.convert('path/to/insite_project')

# Required files:
# - *.setup (Project setup)
# - *.paths (Ray paths)
# - *.points (Receiver points)
```

### Sionna RT
```python
# Convert Sionna RT data
scenario = dm.convert('path/to/sionna_data')

# Required files:
# - *.pkl (Scene data)
# - *.json (Parameters)
```

### AODT
```python
# Convert AODT data
scenario = dm.convert('path/to/aodt_data')

# Required files:
# - *.aodt (Ray data)
```

## Module Structure

```
converter/
  ├── converter.py (Main converter)
  ├── converter_utils.py (Helper functions)
  ├── aodt/ (AODT format)
  │   └── aodt_converter.py
  ├── sionna_rt/ (Sionna RT format)
  │   └── sionna_converter.py
  └── wireless_insite/ (Wireless InSite format)
      └── insite_converter.py
```

## Dependencies

The converter module depends on:
- `scene.py` for physical world representation
- `materials.py` for material properties
- `general_utils.py` for utility functions

## Import Paths

```python
# Main function
from deepmimo.converter import convert

# Format-specific converters
from deepmimo.converter.aodt import aodt_rt_converter
from deepmimo.converter.sionna_rt import sionna_rt_converter
from deepmimo.converter.wireless_insite import insite_rt_converter
```

## Conversion Process

1. Format Detection
   ```python
   # Automatic format detection based on file extensions
   if '.aodt' in files:
       converter = aodt_rt_converter
   elif '.pkl' in files:
       converter = sionna_rt_converter
   elif '.setup' in files:
       converter = insite_rt_converter
   ```

2. Data Loading
   ```python
   # Load raytracing data
   rt_data = converter.load_data(path_to_rt_folder)
   ```

3. Standardization
   ```python
   # Convert to standard format
   scenario = converter.standardize(rt_data)
   ```

4. Validation
   ```python
   # Validate converted data
   converter.validate(scenario)
   ```

## Best Practices

1. Data Organization
   - Keep raytracing files in dedicated folders
   - Use consistent naming conventions
   - Maintain original file structure

2. Memory Management
   - Convert large datasets in chunks
   - Clean up temporary files
   - Monitor memory usage

3. Error Handling
   - Validate input files before conversion
   - Handle missing or corrupt data gracefully
   - Provide informative error messages

## File Format Guidelines

For detailed information about the expected file formats and organization for each ray tracer, please refer to the {doc}`