"""
DeepMIMO Python Package.
"""

from .generator.python_old.generator import generate_data as generate_old
from .generator.python_old.params import Parameters as Parameters_old

# Core functionality
from .generator.python.core import (
    generate,
    load_scenario,
)
from .generator.python.dataset import Dataset

# Visualization
from .generator.python import visualization

# Utilities
from .generator.python.utils import (
    dbm2watt,
    uniform_sampling,
    LinearPath,
    get_idxs_with_limits
)

# Channel parameters
from .generator.python.channel import ChannelGenParameters

from .converter.converter import convert
from .info import info

# Physical world representation
from .scene import (
    Building,
    Face,
    Terrain,
    Vegetation,
    PhysicalObject,
    Scene,
    BuildingsGroup,
    TerrainGroup,
    VegetationGroup
)

__all__ = [
    # Core functionality
    'generate',
    'convert', 
    'info',
    'load_scenario',
    'Dataset',
    'ChannelGenParameters',
    
    # Utilities
    'LinearPath',
    'uniform_sampling',
    'dbm2watt',
    'get_idxs_with_limits',
    
    # Physical world representation
    'Building',
    'Face',
    'Terrain',
    'Vegetation',
    'PhysicalObject',
    'Scene',
    'BuildingsGroup',
    'TerrainGroup',
    'VegetationGroup',
]
