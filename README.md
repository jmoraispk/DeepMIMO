# DeepMIMO
DeepMIMO Toolchain: Bridging RayTracers and 5G Simulators

**[Goal]** Enabling large-scale AI benchmarking using site-specific wireless raytracing datasets.

**[How]** Converting the outputs of the best wireless propagation ray tracers to a distributable format readable by the best simulation toolboxes. 

**[Why]** To revolutionize AI-driven wireless research by making ray tracing data easy to access, share, and benchmark.

## Project Structure
```
deepmimo/
├── api.py                 # Main API interface for dataset management
├── scene.py               # Scene management and generation
├── consts.py              # Constants and configurations
├── info.py                # Dataset information and metadata
├── materials.py           # Material properties
├── txrx.py                # Transmitter and receiver models
├── rt_params.py           # Ray tracing parameters
├── general_utilities.py   # Utility functions
├── converter/             # Ray tracer output converters
│   ├── aodt/              # AODT converter
│   ├── sionna_rt/         # Sionna RT converter
│   ├── wireless_insite/   # Wireless Insite converter
│   ├── converter.py       # Base converter class
│   └── converter_utils.py # Converter utilities
└── generator/             # Dataset generators
    ├── core.py            # Core generation functionality
    ├── dataset.py         # Dataset class and management
    ├── channel.py         # Channel generation
    ├── geometry.py        # Geometric calculations
    ├── ant_patterns.py    # Antenna pattern definitions
    ├── visualization.py   # Visualization tools
    ├── utils.py           # Generator utilities
    └── sionna_adapter.py  # Sionna integration
```

## Installation

### Basic Installation
```bash
pip install deepmimo
```

### Development Installation
```bash
git clone https://github.com/DeepMIMO/DeepMIMO.git
cd DeepMIMO
pip install -e .
```

## Usage Examples

### Basic Dataset Generation
```python
import deepmimo as dm

# Print summary of available datasets
dm.summary('asu_campus_3p5')

# Generate a dataset
dataset = dm.generate('asu_campus_3p5')

# Access channel parameters
channel = dataset.channel
```

### Converting Ray Tracer Outputs
```python
import deepmimo as dm

# Convert Wireless Insite output
converter = dm.convert('path_to_ray_tracing_output')
```

### Uploading and Downloading Datasets
```python
import deepmimo as dm

# Upload a dataset to the DeepMIMO server
dm.upload('my_scenario', 'your-api-key')

# Download a dataset
dm.download('asu_campus_3p5')
```

## Building Documentation

| Step    | Command                                           | Description                       |
|---------|---------------------------------------------------|-----------------------------------|
| Install | `pip install sphinx nbsphinx furo`                | Install dependencies              |
| Build   | `cd docs`<br>`sphinx-build -b html . _build/html` | Generate HTML documentation       |
| Serve   | `cd docs/_build/html`<br>`python -m http.server`  | View docs at http://localhost:8000|

## Contributing

We welcome contributions to DeepMIMO! Here's how you can help:

1. **Converter Development**
   - Add support for new ray tracers
   - Improve existing converter implementations
   - Add validation and error handling

2. **Generator Enhancements**
   - Add new dataset generation features
   - Improve performance and memory efficiency
   - Add new visualization tools

3. **Documentation**
   - Improve existing documentation
   - Add new examples and tutorials
   - Create guides for common use cases

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

We aim to review and respond to pull requests within 24 hours.

## Citation

If you use this software, please cite it as:

```bibtex
@misc{alkhateeb2019deepmimo,
      title={DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and Massive MIMO Applications}, 
      author={Ahmed Alkhateeb},
      year={2019},
      eprint={1902.06435},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/1902.06435}, 
}
```
