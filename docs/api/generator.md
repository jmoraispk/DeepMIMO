# Generator

The generator module is the core of DeepMIMO. This module takes ray tracing scenarios saved in the DeepMIMO format, and generates channels. 

Below is an ascii diagram of how the simulations from the ray tracers are converted into DeepMIMO scenarios (by the converter module, following the DeepMIMO SPEC), and then loaded and used to generate channels (with the generator module).
```c++

+-----------------+     +-------------------+    +-------------------+
| WIRELESS INSITE |     |     SIONNA_RT     |    |       AODT        |
+--------+--------+     +---------+---------+    +---------+---------+
         |                        |                        |
         +------------------------+------------------------+
                                  |
                                  v
                         +------------------+
                         |   dm.convert()   |
                         +--------+---------+
                                  v
                         +------------------+
                         |    DEEPMIMO      |
                         |    SCENARIOS     |
                         +--------+---------+
                                  v
                      +-------------------------+
                      |   dataset = dm.load()   |
                      +-----------+-------------+
                                  v
                    +-----------------------------+
                    | dataset.compute_channels()  |
                    +-------------+---------------+
                                  v
                         +------------------+
                         |  dataset.plot()  |
                         +------------------+
```

Dependencies of the Generator Module:
```
generator/
  ├── core.py (Main generation functions)
  ├── channel.py (Channel computation)
  ├── dataset.py (Dataset classes)
  |    ├── geometry.py (Antenna array functions)
  |    └── ant_patterns.py (Antenna patterns)
  ├── visualization.py (Plotting functions)
  └── utils.py (Helper functions)
```

Additionally, the generator module depends on:
- `scene.py` for physical world representation
- `materials.py` for material properties
- `general_utils.py` for utility functions
- `api.py` for scenario management


## Dataset
The `Dataset` class represents a single dataset within DeepMIMO, containing transmitter, receiver, and channel information for a specific scenario configuration.

```python
import deepmimo as dm

# Load a dataset
dataset = dm.load('scenario_name')

# Access transmitter data
tx_locations = dataset.tx_locations
n_tx = len(dataset.tx_locations)

# Access receiver data
rx_locations = dataset.rx_locations
n_rx = len(dataset.rx_locations)

# Access channel data
channels = dataset.channels  # If already computed
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tx_locations` | ndarray | Transmitter locations (N×3) |
| `rx_locations` | ndarray | Receiver locations (M×3) |
| `channels` | ndarray | Channel matrices |
| `parameters` | dict | Dataset-specific parameters |
| `power` | ndarray | Path powers in dBm |
| `phase` | ndarray | Path phases in degrees |
| `delay` | ndarray | Path delays in seconds |
| `aoa_az/aoa_el` | ndarray | Angles of arrival (azimuth/elevation) |
| `aod_az/aod_el` | ndarray | Angles of departure (azimuth/elevation) |
| `inter` | ndarray | Path interaction indicators |
| `inter_pos` | ndarray | Path interaction positions |

### Channel Generation
```python
# Configure channel parameters
ch_params = dm.ChannelGenParameters()
ch_params.bs_antenna.shape = np.array([8, 1])

# Compute channels
channels = dataset.compute_channels(ch_params)
```

### Path Analysis
```python
# Access path information
pathloss = dataset.compute_pathloss()
los_status = dataset.los  # Line of sight status
num_paths = dataset.num_paths  # Paths per user
```

## MacroDataset
The `MacroDataset` class is a container for managing multiple datasets, providing unified access to their data.

```python
# Access individual datasets
dataset = macro_dataset[0]  # First dataset
datasets = macro_dataset[1:3]  # Slice of datasets

# Iterate over datasets
for dataset in macro_dataset:
    print(f"Dataset has {len(dataset)} users")

# Batch operations
channels = macro_dataset.compute_channels()
```

## ChannelGenParameters
The `ChannelGenParameters` class manages parameters for MIMO channel generation.

```python
import deepmimo as dm

# Load a scenario
dataset = dm.load('O1_60')

# Instantiate channel parameters
params = dm.ChannelGenParameters()

# Configure BS antenna array
params.bs_antenna.shape = np.array([8, 1])  # 8x1 array
params.bs_antenna.spacing = 0.5  # Half-wavelength spacing
params.bs_antenna.rotation = np.array([0, 0, 0])  # No rotation

# Configure UE antenna array
params.ue_antenna.shape = np.array([1, 1])  # Single antenna
params.ue_antenna.spacing = 0.5
params.ue_antenna.rotation = np.array([0, 0, 0])

# Configure OFDM parameters
params.ofdm.subcarriers = 512  # Number of subcarriers
params.ofdm.bandwidth = 10e6  # 10 MHz bandwidth

# Generate frequency-domain channels
params.freq_domain = True
channels = dataset.compute_channels(params)
```

```{eval-rst}
.. autoclass:: deepmimo.generator.channel.ChannelGenParameters
   :members:
   :undoc-members:
   :show-inheritance:
```
| Parameter | Default Value | Description |
|-----------|--------------|-------------|
| `bs_antenna.shape` | [8, 1] | BS antenna array dimensions |
| `bs_antenna.spacing` | 0.5 | BS antenna spacing (wavelengths) |
| `bs_antenna.rotation` | [0, 0, 0] | BS rotation angles (degrees) |
| `ue_antenna.shape` | [1, 1] | UE antenna array dimensions |
| `ue_antenna.spacing` | 0.5 | UE antenna spacing (wavelengths) |
| `ue_antenna.rotation` | [0, 0, 0] | UE rotation angles (degrees) |
| `ofdm.subcarriers` | 512 | Number of OFDM subcarriers |
| `ofdm.bandwidth` | 10e6 | OFDM bandwidth (Hz) |

## Load

Load a DeepMIMO scenario.

```python
dataset = dm.load(
    scen_name,  # Scenario name
    tx_sets={1: [0, 1]},  # Specific TX points
    rx_sets={2: 'all'},   # All RX in set 2
    matrices=['power', 'delay']  # Matrices to load
)
```

## Generate

A wrapper to Load and Compute channels. 

```python
dataset = dm.generate(
    scen_name,  # Scenario name
    load_params={},  # Parameters for loading the scenario
    ch_gen_params={}  # Parameters for channel generation
)
```

