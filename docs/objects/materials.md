# Materials

The `Materials` class manages material properties and electromagnetic characteristics in DeepMIMO scenes.

## Overview

Materials define how electromagnetic waves interact with objects in the scene:
- Dielectric properties
- Surface roughness
- Reflection/transmission characteristics

```python
import deepmimo as dm

# Load dataset
dataset = dm.load('scenario_name')[0]

# Access materials
materials = dataset.scene.materials
```

## Key Features

### Material Properties
```python
# Access material properties
for material in materials:
    print(f"Name: {material.name}")
    print(f"Permittivity: {material.permittivity}")
    print(f"Conductivity: {material.conductivity}")
```

### Electromagnetic Parameters
```python
# Get electromagnetic properties
material = materials[0]
print(f"Relative permittivity: {material.er}")
print(f"Loss tangent: {material.tan_delta}")
print(f"Surface roughness: {material.roughness}")
```

### Material Assignment
```python
# Get objects with specific material
concrete_objects = scene.get_objects_by_material('concrete')

# Check material of an object
building = scene.get_objects('buildings')[0]
material = materials.get_material(building.material_id)
```

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | Material name |
| `er` | complex | Complex relative permittivity |
| `conductivity` | float | Electrical conductivity (S/m) |
| `roughness` | float | Surface roughness (m) |
| `thickness` | float | Material thickness (m) |

## Methods

### `get_material(material_id)`
Get material by ID.

### `get_material_by_name(name)`
Get material by name.

### `compute_reflection_coefficient(angle, frequency)`
Compute reflection coefficient for given angle and frequency.

### `compute_transmission_coefficient(angle, frequency)`
Compute transmission coefficient for given angle and frequency.

## Standard Materials

DeepMIMO includes several standard materials:

| Material | εr (2.4 GHz) | Conductivity (S/m) |
|----------|--------------|-------------------|
| Concrete | 4.5 - j0.2 | 0.01 |
| Glass | 6.0 - j0.05 | 0.001 |
| Metal | 1.0 - j∞ | 1e7 |
| Wood | 2.1 - j0.02 | 0.001 |

## Best Practices

1. Material Selection:
   ```python
   # Use standard materials when possible
   material = materials.get_material_by_name('concrete')
   ```

2. Custom Materials:
   ```python
   # Define custom material properties
   custom_material = {
       'name': 'custom',
       'er': 3.0 - 0.1j,
       'conductivity': 0.005,
       'roughness': 0.001
   }
   ```

3. Material Analysis:
   ```python
   # Analyze material impact
   for freq in frequencies:
       R = material.compute_reflection_coefficient(angle, freq)
       print(f"Reflection at {freq/1e9} GHz: {abs(R):.2f}")
   ```

## See Also
- [Scene](scene.md) - Parent scene documentation
- [Dataset](dataset.md) - Dataset documentation 