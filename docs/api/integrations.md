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

## Machine Learning Frameworks

### PyTorch Integration
```python
import torch
from deepmimo.integrations import torch as dm_torch

# Convert to PyTorch tensors
channels_tensor = dm_torch.to_tensor(channels)
dataset_tensor = dm_torch.DatasetTensor(dataset)

# Create PyTorch DataLoader
dataloader = dm_torch.create_dataloader(
    dataset,
    batch_size=32,
    shuffle=True
)
```

### TensorFlow Integration
```python
import tensorflow as tf
from deepmimo.integrations import tensorflow as dm_tf

# Convert to TensorFlow tensors
channels_tf = dm_tf.to_tensor(channels)
dataset_tf = dm_tf.DatasetTensor(dataset)

# Create TensorFlow Dataset
tf_dataset = dm_tf.create_dataset(
    dataset,
    batch_size=32,
    shuffle=True
)
```

## Ray Tracing Tools

### Wireless InSite
```python
from deepmimo.integrations import insite

# Convert InSite project to DeepMIMO
dataset = insite.convert_project('path/to/project')

# Export DeepMIMO dataset to InSite
insite.export_dataset(dataset, 'path/to/export')
```

### Sionna RT
```python
from deepmimo.integrations import sionna_rt

# Convert Sionna RT scene to DeepMIMO
dataset = sionna_rt.convert_scene('path/to/scene')

# Export DeepMIMO dataset to Sionna RT
sionna_rt.export_dataset(dataset, 'path/to/export')
```

## Best Practices

1. Memory Management:
   ```python
   # Use batch processing for large datasets
   for batch in dataset.iter_batches(1000):
       tensor_batch = dm_torch.to_tensor(batch)
       process_batch(tensor_batch)
   ```

2. GPU Acceleration:
   ```python
   # Enable GPU acceleration
   channels = sionna.generate_channels(
       dataset,
       params,
       device='cuda',
       mixed_precision=True
   )
   ```

3. Format Conversion:
   ```python
   # Preserve metadata during conversion
   dataset_converted = sionna.to_sionna(
       dataset,
       preserve_metadata=True,
       copy_arrays=False
   )
   ```

## See Also
- [Dataset](../objects/dataset.md) - Dataset documentation
- [Channel Parameters](../objects/channel_params.md) - Parameter configuration 