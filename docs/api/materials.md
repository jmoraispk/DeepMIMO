# Materials

The materials module provides classes for representing and managing material properties in wireless environments, including electromagnetic and scattering characteristics.

## Core Classes

### Material
The `Material` class represents a single material with its electromagnetic and scattering properties.

```python
import deepmimo as dm

# Create a material
material = dm.Material(
    id=1,
    name='Concrete',
    permittivity=4.5,
    conductivity=0.02,
    scattering_coefficient=0.2,
    roughness=0.001  # meters
)
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | int | Unique identifier |
| `name` | str | Material name |
| `permittivity` | float | Relative permittivity |
| `conductivity` | float | Conductivity (S/m) |
| `scattering_coefficient` | float | Fraction of incident fields scattered (0-1) |
| `cross_polarization_coefficient` | float | Fraction of scattered field cross-polarized (0-1) |
| `roughness` | float | Surface roughness (m) |
| `thickness` | float | Material thickness (m) |

#### Scattering Models

The Material class supports different scattering models:

| Model | Description | Parameters |
|-------|-------------|------------|
| `SCATTERING_NONE` | No scattering | None |
| `SCATTERING_LAMBERTIAN` | Lambertian scattering | `scattering_coefficient` |
| `SCATTERING_DIRECTIVE` | Directive scattering | `alpha_r`, `alpha_i`, `lambda_param` |

```python
# Configure directive scattering
material.scattering_model = Material.SCATTERING_DIRECTIVE
material.alpha_r = 4.0  # Forward lobe width
material.alpha_i = 4.0  # Backward lobe width
material.lambda_param = 0.5  # Forward/backward ratio
```

### MaterialList
The `MaterialList` class manages collections of materials and provides database functionality.

```python
# Create material list
materials = dm.MaterialList()

# Add materials
materials.add_materials([concrete, glass, metal])

# Access materials
concrete = materials[0]  # By index
metal_objects = materials[1:3]  # Slice

# Export/import
materials_dict = materials.to_dict()
materials = dm.MaterialList.from_dict(materials_dict)
```

#### Material Management

```python
# Iterate over materials
for material in materials:
    print(f"{material.name}: εᵣ={material.permittivity}")

# Get material indices
indices = materials.get_materials()

# Filter duplicates
materials._filter_duplicates()
```

## Default Materials

The module provides default materials commonly used in wireless propagation:

| Material | εᵣ | σ (S/m) | Scattering |
|----------|-----|---------|------------|
| Perfect Electric Conductor | ∞ | ∞ | None |
| Concrete | 4.5 | 0.02 | Lambertian |
| Glass | 6.0 | 0.004 | Directive |
| Wood | 2.1 | 0.002 | Lambertian |
| Metal | 1.0 | 1e7 | None |

## Module Structure

```
materials.py
  ├── Material (Single material)
  └── MaterialList (Material collection)
```

## Dependencies

The materials module depends on:
- `dataclasses` for class definitions
- `typing` for type hints
- NumPy for numerical operations

## Import Paths

```python
# Main classes
from deepmimo.materials import (
    Material,
    MaterialList
)

# Scattering models
from deepmimo.materials import (
    SCATTERING_NONE,
    SCATTERING_LAMBERTIAN,
    SCATTERING_DIRECTIVE
)
```

## Best Practices

1. Material Properties
   - Use realistic values from measurements when available
   - Consider frequency dependence for wideband simulations
   - Document sources of material parameters

2. Scattering Models
   - Use Lambertian for rough surfaces
   - Use Directive for smoother surfaces
   - Adjust parameters based on measurements if available

3. Performance
   - Reuse common materials across objects
   - Filter duplicates in large material lists
   - Cache material computations when possible 