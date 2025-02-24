"""
DeepMIMO Python Package.
"""

__version__ = "4.0.0a2"

# Core functionality
from .generator.python.core import (
    generate,
    load_scenario,
)
from .generator.python.dataset import Dataset

# Visualization
from .generator.python.visualization import (
    plot_coverage,
    plot_rays,
)

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
from .general_utilities import (
    summary,
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
    'load_scenario',
    'Dataset',
    'ChannelGenParameters',
    
    # Visualization
    'plot_coverage',
    'plot_rays',
    
    # Utilities
    'LinearPath',
    'uniform_sampling',
    'dbm2watt',
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
    
    # Constants
    'consts',
    'general_utilities',
]
