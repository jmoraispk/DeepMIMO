# Overview

DeepMIMO is organized into three main modules and several supporting components, designed to provide a complete workflow for MIMO channel modeling and dataset generation.

## Core Modules

### Generator
The generator module is the heart of DeepMIMO, responsible for:
- Dataset generation and management
- Channel computation
- Parameter validation
- Multi-user MIMO channel generation

Entry point: `deepmimo.generate()`, `deepmimo.load()`

### Converter
Handles DeepMIMO conversion from different ray tracing formats:
- Wireless InSite
- Sionna RT
- AODT

Entry point: `deepmimo.convert()`

### Scene
Physical world representation including:
- Buildings, terrain, vegetation
- Material properties
- Geometric computations

Entry point: `deepmimo.Scene`, `deepmimo.PhysicalElement`

### Materials
Material property management:
- Electromagnetic properties
- Scattering characteristics
- Material database

Entry point: `deepmimo.Material`, `deepmimo.MaterialList`

### Visualization
Visualization tools for:
- Coverage maps
- Ray paths
- Channel characteristics

Entry point: `deepmimo.plot_coverage()`, `deepmimo.plot_rays()`

### Integrations
Integration with external tools:
- Sionna
- Machine learning frameworks
- Ray tracing tools

Entry point: `deepmimo.integrations`

### Pipelines
Pre-built workflows for common tasks:
- Blender/OSM export
- Wireless InSite automation
- Sionna RT integration

Entry point: `deepmimo.pipelines`

### Utilities
Helper functions for:
- File management
- Data conversion
- Geometry calculations

Entry point: `deepmimo.utils`

## Import Structure

```python
import deepmimo as dm

# Core functionality
dataset = dm.generate(scenario_name)  # Generate dataset
dataset = dm.load(scenario_name)      # Load existing dataset

# Scene management
scene = dm.Scene()
scene.add_object(dm.PhysicalElement(...))

# Material handling
materials = dm.MaterialList()
materials.add_materials([dm.Material(...)])

# Visualization
dm.plot_coverage(dataset)
dm.plot_rays(dataset)

# Conversion
dm.convert(path_to_rt_folder)

# Utilities
dm.get_available_scenarios()
dm.get_params_path(scenario_name)
```

## Module Dependencies

The following diagram shows the main dependencies between DeepMIMO modules:

```
generator
  ├── core.py
  │   ├── channel.py (Channel generation)
  │   ├── dataset.py (Dataset management)
  │   └── visualization.py (Plotting)
  └── utils.py (Helper functions)

converter
  ├── converter.py (Main converter)
  ├── wireless_insite/
  ├── sionna_rt/
  └── aodt/

scene.py (Physical world)
  └── materials.py (Material properties)

integrations/
  ├── sionna_adapter.py
  └── matlab/

pipelines/
  ├── blender_osm_export.py
  ├── wireless_insite/
  └── sionna_rt/
```

Each module is designed to be self-contained while working seamlessly with others through well-defined interfaces.

## Getting Help

Remember that you can use the `dm.info()` function to get detailed information about any concept:

```python
# Get information about angles convention
dm.info('angles')

# Learn about channel computation
dm.info('compute_channels')

# Understand dataset structure
dm.info('Dataset')

# Get scenario summary
dm.summary('asu_campus_3p5')
```

For more examples and tutorials:
1. Check the [DeepMIMO Manual](https://colab.research.google.com/drive/1U-e2rLDJYW-VcbJ7C3H2dqC625JJH5FF) notebook
2. Watch our [Video Tutorials](https://deepmimo.net/tutorials)