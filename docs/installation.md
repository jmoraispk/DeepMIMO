# Installation

```{toctree}
:hidden:

self
```

You can install DeepMIMO using pip:

```bash
pip install deepmimo
```

## Requirements

DeepMIMO requires Python 3.10 or later. The main dependencies are:
* matplotlib (>= 3.8.2)
* numpy (>= 1.19.5)
* scipy (>= 1.6.2)
* tqdm (>= 4.59.0)
* pandas (>= 2.0.0)
* h5py (>= 3.8.0)
* pyyaml (>= 6.0.0)

## Optional Dependencies

For specific features, you may need additional packages:

### Ray Tracing Visualization
* plotly (>= 5.13.0)
* dash (>= 2.9.0)

### Advanced Data Processing
* scikit-learn (>= 1.0.0)
* networkx (>= 3.0.0)

## Development Installation

For development installation, clone the repository and install in editable mode:

```bash
git clone https://github.com/DeepMIMO/DeepMIMO.git
cd DeepMIMO
pip install -e .
```

For development, you'll also want to install the test dependencies:

```bash
pip install -e ".[dev]"
```

## Documentation Dependencies

To build the documentation locally, you'll need:

```bash
# Install Sphinx and basic extensions
pip install sphinx nbsphinx

# Install themes
pip install furo sphinx-rtd-theme
```

## Troubleshooting

Common installation issues:

1. **Version Conflicts**: If you encounter version conflicts, try creating a new virtual environment:
   
   ```bash
   python -m venv deepmimo-env
   source deepmimo-env/bin/activate  # On Windows: deepmimo-env\Scripts\activate
   pip install deepmimo
   ```

2. **Missing Dependencies**: If you get errors about missing dependencies, install them manually:
   
   ```bash
   pip install matplotlib numpy scipy tqdm pandas h5py pyyaml
   ```

3. **Build Issues**: If you encounter build issues on Windows, ensure you have the latest pip and setuptools:
   
   ```bash
   python -m pip install --upgrade pip setuptools wheel 
   ```