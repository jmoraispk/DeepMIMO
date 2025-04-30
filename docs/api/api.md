# API Reference

This section provides detailed documentation for DeepMIMO's public API.

## Core Functions

### `load(scenario_name, **kwargs)`
Load a scenario and return a MacroDataset.

### `download(scenario_name)`
Download a pre-configured scenario.

### `info(topic)`
Get help about a specific topic.

## Configuration

### `set_config(key, value)`
Set global configuration options.

### `get_config(key)`
Get current configuration value.

## Data Management

### `save_dataset(dataset, filename)`
Save a dataset to disk.

### `load_dataset(filename)`
Load a dataset from disk.

## Examples

```python
import deepmimo as dm

# Load a scenario
dataset = dm.load('O1_60')

# Get help
dm.info('channel_parameters')

# Configure globally
dm.set_config('cache_dir', '/path/to/cache')
```

Database API

This module provides functionality for interfacing with the DeepMIMO database and managing scenarios.

Remote Repository Interface

   :undoc-members:
   :show-inheritance:

   Functions for interacting with the DeepMIMO remote repository:
* Upload scenarios to the DeepMIMO database
* Download scenarios from the database
* Version control and caching
* Authentication and authorization

Basic Usage

``python
    import deepmimo as dm
``

    # Download a scenario from the database (implicitely called in ....)
    dm.download('asu_campus_3p5')

    # Upload your own scenario (requires API key)
    dm.upload('my_scenario', 'your-api-key',
             details=['Custom scenario at 3.5 GHz'])

To upload scenarios, you need an API key. You can obtain one by:

1. Go to "Contribute"
2. Create an account on the DeepMIMO website
3. Generate an API key from the dashboard: https://dev.deepmimo.net/dashboard?tab=uploadKey
