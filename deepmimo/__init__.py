"""
DeepMIMO Python Package.
"""

__version__ = "4.0.0a2"

# Core functionality
from .generator.core import (
    generate,
    load,
)
from .generator.dataset import Dataset

# Visualization
from .generator.visualization import (
    plot_coverage,
    plot_rays,
    plot_power_discarding,
)

# Utilities
from .generator.utils import (
    uniform_sampling,
    LinearPath,
    get_idxs_with_limits
)

# Channel parameters
from .generator.channel import ChannelGenParameters

from .converter.converter import convert
from .info import info
from .general_utilities import (
    summary,
    get_available_scenarios,
    get_params_path,
    load_dict_from_json,
)

from .api import (
    upload,
    download,
)

# Physical world representation
from .scene import (
    Face,
    PhysicalElement,
    PhysicalElementGroup,
    Scene
)

# Import immediate modules
from . import consts
from . import general_utilities

__all__ = [
    # Core functionality
    'generate',
    'convert', 
    'info',
    'load',
    'Dataset',
    'ChannelGenParameters',
    
    # Visualization
    'plot_coverage',
    'plot_rays',
    
    # Utilities
    'LinearPath',
    'uniform_sampling',
    'get_idxs_with_limits',
    
    # Physical world representation
    'Face',
    'PhysicalElement',
    'PhysicalElementGroup',
    'Scene',
    
    # General utilities
    'summary',
    'upload',
    'download',
    'get_available_scenarios',
    'get_params_path',
    'load_dict_from_json',
    
    # Constants
    'consts',
    'general_utilities',
]
