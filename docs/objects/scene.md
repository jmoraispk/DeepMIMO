# Scene

The `Scene` class represents the physical environment configuration in DeepMIMO, including objects, materials, and ray paths.

## Overview

A Scene represents a physical environment with the following structure:
- A scene contains PhysicalElements
- Each PhysicalElement has a BoundingBox and several Faces
- Each Face consists of multiple triangular faces with:
  - Counter-clockwise vertex ordering
  - Surface normal pointing outward (following right-hand rule)
  - Associated material properties

A Scene contains all the geometric and physical properties of the environment:
- Building geometries
- Material properties
- Terrain information
- Ray interaction points

```python
import deepmimo as dm

# Load dataset
dataset = dm.load('scenario_name')[0]

# Access scene
scene = dataset.scene
```

## Key Features

### Object Management
```python
# Get objects by category
buildings = scene.get_objects('buildings')
terrain = scene.get_objects('terrain')
vegetation = scene.get_objects('vegetation')

# Access object properties
for building in buildings:
    print(f"Material: {building.material}")
    print(f"Vertices: {building.vertices}")
```

### Ray Path Analysis
```python
# Get ray paths for a specific user
rays = scene.get_rays(user_idx=0)

# Analyze ray interactions
for ray in rays:
    print(f"Path length: {ray.length}")
    print(f"Interactions: {ray.interactions}")
```

### Visualization
```python
# Plot entire scene
scene.plot()

# Plot with specific options
scene.plot(
    mode='tri_faces',          # Show triangulated faces
    color_by_material=True,    # Color code by material
    show_terrain=True         # Include terrain
)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `objects` | dict | Dictionary of scene objects by category |
| `materials` | list | List of material definitions |
| `bounds` | ndarray | Scene boundaries (min/max coordinates) |
| `center` | ndarray | Scene center point |

## Methods

### `get_objects(category)`
Get objects of a specific category (buildings, terrain, etc.).

### `get_rays(user_idx)`
Get ray paths for a specific user.

### `plot(**kwargs)`
Visualize the scene with various options.

### `get_material(material_id)`
Get material properties by ID.

## Best Practices

1. Scene Analysis:
   ```python
   # Get scene statistics
   print(f"Scene dimensions: {scene.bounds}")
   print(f"Number of buildings: {len(scene.get_objects('buildings'))}")
   ```

2. Efficient Ray Access:
   ```python
   # Get rays for multiple users
   user_rays = [scene.get_rays(i) for i in user_indices]
   ```

3. Visualization:
   ```python
   # Create publication-ready plots
   scene.plot(
       dpi=300,
       show_axes=True,
       grid=True,
       save_path='scene.png'
   )
   ```

## See Also
- [Materials](materials.md) - Material properties documentation
- [Dataset](dataset.md) - Parent dataset documentation 