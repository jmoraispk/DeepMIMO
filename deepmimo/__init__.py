"""
DeepMIMO Python Package.
"""

from .generator.python_old.generator import generate_data as generate_old
from .generator.python_old.params import Parameters as Parameters_old

# Core functionality
from .generator.python.core import (
    generate,
    load_scenario,
    compute_channels,
    compute_num_paths,
    compute_num_interactions,
    compute_pathloss,
    compute_distances,
    compute_los,
    compute_rotated_angles
)

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

from .converter.converter import create_scenario
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
    'create_scenario', 
    'info',
    'load_scenario',
    'compute_channels',
    'compute_num_paths',
    'compute_num_interactions',
    'compute_pathloss',
    'compute_distances',
    'compute_los',
    'compute_rotated_angles',
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
