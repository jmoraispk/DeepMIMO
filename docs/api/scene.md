# Scene

The scene module provides classes for representing and managing physical objects in a wireless environment, including buildings, terrain, vegetation, and other structures that affect wireless propagation.

## Core Classes

### Scene
The `Scene` class represents a complete physical environment containing multiple objects.

```python
import deepmimo as dm

# Create a new scene
scene = dm.Scene()

# Add objects
scene.add_object(building)
scene.add_objects([tree1, tree2])

# Get objects by category
buildings = scene.get_objects(label='buildings')
metal_objects = scene.get_objects(material=1)  # material_id = 1

# Export scene data
metadata = scene.export_data('path/to/folder')

# Load scene from data
scene = dm.Scene.from_data('path/to/folder')
```

#### Visualization
```python
# 3D visualization
scene.plot(mode='faces')  # Use convex hull representation
scene.plot(mode='tri_faces')  # Use triangular representation

# 2D top-down view
scene.plot(proj_2d=True)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `objects` | List[PhysicalElement] | List of objects in the scene |
| `bounding_box` | BoundingBox | Scene's bounding box |
| `visualization_settings` | dict | Visualization parameters |

### PhysicalElement
The `PhysicalElement` class represents individual physical objects in the scene.

```python
# Create a physical object
element = dm.PhysicalElement(
    faces=faces,  # List of Face objects
    object_id=1,  # Unique identifier
    label='buildings',  # Object category
    name='Building 1'  # Optional name
)

# Access properties
height = element.height
volume = element.volume
position = element.position
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `faces` | List[Face] | Object's faces |
| `height` | float | Object height |
| `volume` | float | Object volume |
| `position` | ndarray | Center position |
| `bounding_box` | BoundingBox | Object's bounding box |
| `footprint_area` | float | Ground projection area |

### Face
The `Face` class represents a single face (surface) of a physical object.

```python
# Create a face
face = dm.Face(
    vertices=vertices,  # Array of vertex coordinates
    material_idx=1  # Material index
)

# Access properties
normal = face.normal
area = face.area
centroid = face.centroid
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `vertices` | ndarray | Face vertices |
| `normal` | ndarray | Face normal vector |
| `area` | float | Face area |
| `centroid` | ndarray | Face center point |
| `material_idx` | int | Material index |

### PhysicalElementGroup
The `PhysicalElementGroup` class manages collections of physical objects.

```python
# Create a group
group = dm.PhysicalElementGroup(objects)

# Filter objects
buildings = group.get_objects(label='buildings')
metal = group.get_objects(material=1)

# Access objects
first = group[0]
subset = group[1:3]
```

## Object Categories

The scene module defines standard categories for physical objects:

| Category | Description | Example Objects |
|----------|-------------|-----------------|
| `buildings` | Building structures | Houses, offices |
| `terrain` | Ground surfaces | Ground, hills |
| `vegetation` | Plant life | Trees, bushes |
| `floorplans` | Indoor layouts | Walls, rooms |
| `objects` | Other items | Cars, signs |

## Module Structure

```
scene.py
  ├── BoundingBox (3D bounds)
  ├── Face (Surface representation)
  ├── PhysicalElement (Individual objects)
  ├── PhysicalElementGroup (Object collections)
  └── Scene (Complete environment)
```

## Dependencies

The scene module depends on:
- `materials.py` for material properties
- `general_utils.py` for utility functions
- NumPy for geometric computations
- Matplotlib for visualization

## Import Paths

```python
# Main classes
from deepmimo.scene import (
    Scene,
    PhysicalElement,
    PhysicalElementGroup,
    Face
)

# Constants
from deepmimo.scene import (
    CAT_BUILDINGS,
    CAT_TERRAIN,
    CAT_VEGETATION,
    CAT_FLOORPLANS,
    CAT_OBJECTS
)
``` 