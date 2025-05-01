# Utilities

```{eval-rst}
.. autofunction:: deepmimo.generator.generator_utils.dbw2watt

.. autofunction:: deepmimo.generator.generator_utils.get_uniform_idxs

.. autoclass:: deepmimo.generator.generator_utils.LinearPath
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: deepmimo.generator.generator_utils.get_idxs_with_limits

.. autofunction:: deepmimo.general_utils.save_dict_as_json

.. autofunction:: deepmimo.general_utils.load_dict_from_json

.. autoclass:: deepmimo.general_utils.DotDict
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: deepmimo.general_utils.get_scenarios_dir

.. autofunction:: deepmimo.general_utils.get_scenario_folder

.. autofunction:: deepmimo.general_utils.get_params_path

.. autofunction:: deepmimo.general_utils.get_available_scenarios
```

## Examples

```python
# Power conversion
power_w = dm.dbw2watt(power_dbw)

# Linear path sampling
path = dm.LinearPath(
    rx_pos=dataset.rx_pos,
    first_pos=np.array([0, 0, 0]),
    last_pos=np.array([10, 0, 0]),
    res=0.5
)
path_idxs = path.idxs

# Position filtering
valid_idxs = dm.get_idxs_with_limits(
    dataset.rx_pos,
    x_min=0, x_max=10,
    y_min=-5, y_max=5
)

# File operations
config = {'scenario': 'O1_60', 'params': {'frequency': 60e9}}
dm.save_dict_as_json('config.json', config)
params = dm.load_dict_from_json('params.json')

# Dot notation access
params = dm.DotDict(params)
freq = params.params.frequency

# Scenario management
scenarios_dir = dm.get_scenarios_dir()
available_scenarios = dm.get_available_scenarios()
scenario_folder = dm.get_scenario_folder('O1_60')
params_path = dm.get_params_path('O1_60')
```

## Best Practices

1. Memory Efficiency:
   ```python
   # Process large datasets in chunks
   chunk_size = 1000
   for i in range(0, len(dataset), chunk_size):
       chunk = dataset[i:i+chunk_size]
       # Process chunk...
   ```

2. File Handling:
   ```python
   # Use context managers for safe file operations
   with open('config.json', 'w') as f:
       dm.save_dict_as_json(f, config)
   ```

3. Error Handling:
   ```python
   try:
       data = dm.load_dict_from_json('config.json')
   except FileNotFoundError:
       print("Config file not found")
   except Exception as e:
       print(f"Error loading file: {e}")
   ```

## Beamforming Functions

### steering_vec
```python
def steering_vec(array_shape, phi)
```
Generates a steering vector for beamforming based on array shape and azimuth angle.

**Parameters:**
- `array_shape` : array_like
  - Shape of the antenna array [horizontal, vertical] elements
- `phi` : float
  - Azimuth angle in degrees

**Returns:**
- `ndarray`
  - Steering vector for beamforming

## User Sampling Functions

### get_uniform_idxs
```python
def get_uniform_idxs(grid_shape)
```
Gets indices for uniform sampling of users in a grid pattern.

**Parameters:**
- `grid_shape` : tuple
  - Shape of the grid [rows, columns]

**Returns:**
- `ndarray`
  - Indices of uniformly sampled positions

### LinearPath
```python
class LinearPath:
    def __init__(self, positions, start_point, end_point, n_steps=75)
```
Class for sampling users along a linear path between two points.

**Parameters:**
- `positions` : ndarray
  - Array of all available positions
- `start_point` : array_like
  - Starting point coordinates [x, y]
- `end_point` : array_like
  - Ending point coordinates [x, y]
- `n_steps` : int, optional
  - Number of points to sample along the path (default: 75)

**Attributes:**
- `idxs` : ndarray
  - Indices of positions along the linear path

### get_idxs_with_limits
```python
def get_idxs_with_limits(positions, x_min=None, x_max=None, y_min=None, y_max=None)
```
Gets indices of positions within specified rectangular boundaries.

**Parameters:**
- `positions` : ndarray
  - Array of positions to filter
- `x_min` : float, optional
  - Minimum x-coordinate
- `x_max` : float, optional
  - Maximum x-coordinate
- `y_min` : float, optional
  - Minimum y-coordinate
- `y_max` : float, optional
  - Maximum y-coordinate

**Returns:**
- `ndarray`
  - Indices of positions within the specified boundaries

## Dataset Operations

### get_active_idxs
```python
def get_active_idxs(dataset)
```
Gets indices of active users (those with valid paths) in the dataset.

**Parameters:**
- `dataset` : DeepMIMODataset
  - Dataset to analyze

**Returns:**
- `ndarray`
  - Indices of active users

### plot_coverage
```python
def plot_coverage(positions, values, bs_pos=None, bs_ori=None, title='', cbar_title='', 
                 ax=None, proj_3D=False, cmap=None, lims=None, scat_sz=None)
```
Plots coverage map of values over positions.

**Parameters:**
- `positions` : ndarray
  - Array of positions [n_points, 2/3]
- `values` : ndarray
  - Values to plot at each position
- `bs_pos` : ndarray, optional
  - Base station position
- `bs_ori` : ndarray, optional
  - Base station orientation
- `title` : str, optional
  - Plot title
- `cbar_title` : str, optional
  - Colorbar title
- `ax` : matplotlib.axes, optional
  - Axes to plot on
- `proj_3D` : bool, optional
  - Whether to create a 3D projection
- `cmap` : str or list, optional
  - Colormap to use
- `lims` : list, optional
  - Color limits [min, max]
- `scat_sz` : float, optional
  - Scatter plot point size

## Power Calculations

### compute_pathloss
```python
def compute_pathloss(dataset, coherent=True)
```
Computes pathloss from the dataset.

**Parameters:**
- `dataset` : DeepMIMODataset
  - Dataset to compute pathloss from
- `coherent` : bool, optional
  - Whether to compute coherent (True) or non-coherent (False) pathloss

**Returns:**
- `ndarray`
  - Computed pathloss values

## Field of View Operations

### apply_fov
```python
def apply_fov(dataset, bs_fov=None, ue_fov=None)
```
Applies field of view constraints to the dataset.

**Parameters:**
- `dataset` : DeepMIMODataset
  - Dataset to apply FoV to
- `bs_fov` : array_like, optional
  - Base station field of view [azimuth, elevation] in degrees
- `ue_fov` : array_like, optional
  - User equipment field of view [azimuth, elevation] in degrees

**Note:** Calling without parameters resets FoV constraints. 