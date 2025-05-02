# Integrations

DeepMIMO integrates with various external tools and frameworks to enhance its capabilities.

## Sionna Integration

[Sionna](https://github.com/NVlabs/sionna) is an open-source GPU-accelerated library for link-level simulations.

### Data Conversion
```python
import deepmimo as dm
from deepmimo.integrations import sionna

# Convert DeepMIMO dataset to Sionna format
sionna_dataset = sionna.to_sionna(deepmimo_dataset)

# Convert Sionna dataset to DeepMIMO format
deepmimo_dataset = sionna.from_sionna(sionna_dataset)
```

### Parameter Mapping
```python
# Map DeepMIMO parameters to Sionna
sionna_params = sionna.map_parameters(deepmimo_params)

# Map Sionna parameters to DeepMIMO
deepmimo_params = sionna.map_parameters(sionna_params, reverse=True)
```

### Channel Generation
```python
# Generate channels using Sionna's GPU acceleration
channels = sionna.generate_channels(
    dataset,
    params,
    batch_size=1000,
    device='cuda'
)
```

## MATLAB 5G Toolbox

(coming soon - workitem not started - feel free to contribute)

## NeoRadium

[NeoRadium](https://github.com/InterDigitalInc/NeoRadium) is an open-source GPU-accelerated library for link-level simulations.

(coming soon - workitem not started - feel free to contribute)