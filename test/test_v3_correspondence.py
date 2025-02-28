"""
Test script to verify correspondence between DeepMIMO v3 and v4 implementations.

This script loads and processes the same scenario using both v3 and v4 implementations,
then compares the results to ensure consistency between versions across different
parameter combinations.
"""
#%%
import time
import numpy as np
import deepmimo as dm
from pprint import pprint
from typing import Dict, Any, List, Tuple

# Import V3 functions
from deepmimo_old_v3.generator.python.generator import generate_data as generate_old
from deepmimo_old_v3.generator.python.params import Parameters as Parameters_old
from deepmimo_old_v3.converter.wireless_insite.insite_converter_v3 import insite_rt_converter_v3

# Define parameter combinations to test
PARAM_COMBINATIONS = {
    'num_paths': [5, 10, 25],
    'subcarriers': [64, 512],
    'selected_subcarriers': [np.arange(1), np.arange(3)*3],
    'antenna_shape': [np.array([1,1]), np.array([3,2])],
    'freq_domain': [True, False],
    'bs_rotation': [
        None, 
        np.array([30,40,30]),  # Single rotation for all users
        'per_user'  # Special value to indicate per-user rotation
    ],
    'bs_fov': [None, np.array([140, 120])],
    'ue_fov': [None, np.array([90, 80])]
}

def convert_scenario(rt_folder: str, use_v3: bool = False) -> str:
    """Convert a Wireless Insite scenario to DeepMIMO format.
    
    Args:
        rt_folder (str): Path to the ray tracing folder
        use_v3 (bool): Whether to use v3 converter. Defaults to False.
        
    Returns:
        str: Name of the converted scenario
    """
    if use_v3:
        # Set parameters based on scenario
        if 'asu_campus' in rt_folder:
            old_params_dict = {'num_bs': 1, 'user_grid': [1, 411, 321], 'freq': 3.5e9} # asu
        else:
            old_params_dict = {'num_bs': 1, 'user_grid': [1, 91, 61], 'freq': 3.5e9} # simple canyon

    # Convert to unix path
    rt_folder = rt_folder.replace('\\', '/')

    # Get scenario name
    scen_name = os.path.basename(rt_folder)

    # Convert using appropriate converter
    if use_v3:
        return insite_rt_converter_v3(rt_folder, None, None, old_params_dict, scen_name)
    else:
        return dm.convert(rt_folder, overwrite=True, scenario_name=scen_name, vis_scene=True)

def generate_per_user_rotations(num_users: int) -> np.ndarray:
    """Generate random rotations for each user.
    
    Args:
        num_users (int): Number of users to generate rotations for
        
    Returns:
        np.ndarray: Array of shape (num_users, 3) with random rotations
    """
    np.random.seed(42)  # For reproducibility
    # Use a more controlled range: [0, 45] degrees for each angle
    return np.random.uniform(0, 45, (num_users, 3))

def generate_v4_dataset(scen_name: str, params: Dict[str, Any]):
    """Generate dataset using DeepMIMO v4.
    
    Args:
        scen_name (str): Name of the scenario to load
        params (Dict[str, Any]): Dictionary of parameters to use
        
    Returns:
        Dataset object from v4
    """
    print("\nGenerating V4 dataset with params:", params)
    start_time = time.time()
    
    # Load parameters
    tx_sets = {1: [0]}
    rx_sets = {2: 'all'}
    load_params = {'tx_sets': tx_sets, 'rx_sets': rx_sets, 'max_paths': params['num_paths']}
    
    # Load dataset
    dataset = dm.load(scen_name, **load_params)
    
    # Configure channel parameters
    ch_params = dm.ChannelGenParameters()
    ch_params.num_paths = params['num_paths']
    ch_params.ofdm.subcarriers = params['subcarriers']
    ch_params.ue_antenna.shape = params['antenna_shape']
    
    if params['selected_subcarriers'] is not None:
        ch_params.ofdm.selected_subcarriers = params['selected_subcarriers']
    if params['freq_domain'] is not None:
        ch_params.freq_domain = params['freq_domain']
    if params['bs_rotation'] is not None:
        if isinstance(params['bs_rotation'], str) and params['bs_rotation'] == 'per_user':
            # Generate per-user rotations
            num_users = len(dataset.rx_pos)
            rotations = generate_per_user_rotations(num_users)
            # Validate rotation shape
            if not (len(rotations.shape) == 2 and rotations.shape[1] == 3):
                raise ValueError("Per-user rotations must be a (N,3) array")
            ch_params.bs_antenna.rotation = rotations
        else:
            # Validate single rotation shape
            rotation = np.array(params['bs_rotation'])
            if not (len(rotation.shape) == 1 and rotation.shape[0] == 3):
                raise ValueError("Single rotation must be a 3D vector")
            ch_params.bs_antenna.rotation = rotation
    
    # Compute channels
    dataset.compute_channels(ch_params)
    
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    return dataset

def generate_v3_dataset(scen_name: str, params: Dict[str, Any]):
    """Generate dataset using DeepMIMO v3.
    
    Args:
        scen_name (str): Name of the scenario to load
        params (Dict[str, Any]): Dictionary of parameters to use
        
    Returns:
        Dataset from v3
    """
    print("\nGenerating V3 dataset with params:", params)
    start_time = time.time()
    
    # Configure parameters
    ch_params = Parameters_old(scen_name)
    ch_params['user_rows'] = np.arange(1)
    ch_params['num_paths'] = params['num_paths']
    ch_params['ofdm']['subcarriers'] = params['subcarriers']
    ch_params['ue_antenna']['shape'] = params['antenna_shape']
    
    if params['selected_subcarriers'] is not None:
        ch_params['ofdm']['selected_subcarriers'] = params['selected_subcarriers']
    if params['freq_domain'] is not None:
        ch_params['freq_domain'] = params['freq_domain']
    if params['bs_rotation'] is not None:
        if isinstance(params['bs_rotation'], str) and params['bs_rotation'] == 'per_user':
            # Generate per-user rotations
            # Note: We need to load the dataset first to get num_users
            temp_dataset = dm.load(scen_name, tx_sets={1: [0]}, rx_sets={2: 'all'})
            num_users = len(temp_dataset.rx_pos)
            rotations = generate_per_user_rotations(num_users)
            print("\nWARNING: V3 doesn't support per-user rotations. Using first rotation for all users.")
            # For v3, we need to ensure it's a single 3D vector
            ch_params['bs_antenna']['rotation'] = rotations[0]  # Use first user's rotation
        else:
            # Ensure rotation is a 3D vector
            rotation = np.array(params['bs_rotation'])
            if not (len(rotation.shape) == 1 and rotation.shape[0] == 3):
                raise ValueError("For v3, BS antenna rotation must be a 3D vector")
            ch_params['bs_antenna']['rotation'] = rotation
           
    # Generate dataset
    dataset = generate_old(ch_params)
    
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    return dataset

def compare_datasets(dataset_v4, dataset_v3, user_idx: int = 10) -> Tuple[float, int]:
    """Compare channel outputs between v3 and v4 datasets.
    
    Args:
        dataset_v4: V4 dataset
        dataset_v3: V3 dataset
        user_idx (int): Index of user to compare. Defaults to 10.
        
    Returns:
        Tuple[float, int]: Maximum absolute difference and number of non-zero channels
    """
    print(f"\nComparing datasets at user index {user_idx}...")
    
    # Get channels
    ch_v4 = dataset_v4['ch'][user_idx]
    ch_v3 = dataset_v3[0]['user']['channel'][user_idx]
    
    # Print last 10 values
    print("\nLast 10 values from V4:")
    pprint(ch_v4.flatten()[-10:])
    print("\nLast 10 values from V3:")
    pprint(ch_v3.flatten()[-10:])
    
    # Calculate maximum absolute difference
    max_diff = np.max(np.abs(ch_v4 - ch_v3))
    print(f"\nMaximum absolute difference: {max_diff}")
    
    # Calculate number of non-zero channels in V4
    user_norms = np.linalg.norm(dataset_v4['channel'], axis=(2,3)).squeeze()
    non_zero_count = np.count_nonzero(user_norms)
    print(f"\nNumber of users with non-zero channels in V4: {non_zero_count}")
    
    return max_diff, non_zero_count

def generate_param_combinations() -> List[Dict[str, Any]]:
    """Generate all combinations of parameters to test.
    
    Returns:
        List[Dict[str, Any]]: List of parameter dictionaries
    """
    # Generate base combinations
    base_params = {
        'num_paths': PARAM_COMBINATIONS['num_paths'][0],
        'subcarriers': PARAM_COMBINATIONS['subcarriers'][0],
        'selected_subcarriers': PARAM_COMBINATIONS['selected_subcarriers'][0],
        'antenna_shape': PARAM_COMBINATIONS['antenna_shape'][0],
        'freq_domain': None,
        'bs_rotation': None,
        'bs_fov': None,
        'ue_fov': None
    }
    
    param_combinations = []
    
    # Test each parameter individually
    for param_name, param_values in PARAM_COMBINATIONS.items():
        for value in param_values:
            new_params = base_params.copy()
            new_params[param_name] = value
            param_combinations.append(new_params)
    
    # Add some combined parameter tests
    combined_params = base_params.copy()
    combined_params.update({
        'freq_domain': True,
        'bs_rotation': PARAM_COMBINATIONS['bs_rotation'][1],
        'bs_fov': PARAM_COMBINATIONS['bs_fov'][1],
        'ue_fov': PARAM_COMBINATIONS['ue_fov'][1]
    })
    param_combinations.append(combined_params)
    
    return param_combinations

def run_test_suite(scen_name: str):
    """Run full test suite comparing v3 and v4 implementations.
    
    Args:
        scen_name (str): Name of the scenario to test
    """
    print(f"\nRunning test suite for scenario: {scen_name}")
    print("=" * 80)
    
    results = []
    param_combinations = generate_param_combinations()
    
    for i, params in enumerate(param_combinations, 1):
        print(f"\nTest {i}/{len(param_combinations)}")
        print("-" * 80)
        
        try:
            # Generate datasets
            dataset_v4 = generate_v4_dataset(scen_name, params)
            dataset_v3 = generate_v3_dataset(scen_name, params)
            
            # Compare results
            max_diff, non_zero_count = compare_datasets(dataset_v4, dataset_v3)
            
            results.append({
                'params': params,
                'max_diff': max_diff,
                'non_zero_count': non_zero_count,
                'status': 'SUCCESS' if max_diff < 1e-10 else 'MISMATCH'
            })
            
        except Exception as e:
            print(f"Error during test: {str(e)}")
            results.append({
                'params': params,
                'status': 'ERROR',
                'error': str(e)
            })
    
    # Print summary
    print("\nTest Suite Summary")
    print("=" * 80)
    for i, result in enumerate(results, 1):
        print(f"\nTest {i}:")
        print(f"Status: {result['status']}")
        print("Parameters:", {k: str(v) for k, v in result['params'].items()})
        if result['status'] == 'SUCCESS':
            print(f"Max difference: {result['max_diff']}")
            print(f"Non-zero channels: {result['non_zero_count']}")
        elif result['status'] == 'ERROR':
            print(f"Error: {result['error']}")

def run_single_test(scen_name: str, params: Dict[str, Any]) -> Dict:
    """Run a single test with specified parameters.
    
    Args:
        scen_name (str): Name of the scenario to test
        params (Dict[str, Any]): Parameter combination to test
        
    Returns:
        Dict: Test result
    """
    print(f"\nRunning single test for scenario: {scen_name}")
    print("=" * 80)
    print("Parameters:", {k: str(v) for k, v in params.items()})
    
    try:
        # Generate datasets
        dataset_v4 = generate_v4_dataset(scen_name, params)
        dataset_v3 = generate_v3_dataset(scen_name, params)
        
        # Compare results
        max_diff, non_zero_count = compare_datasets(dataset_v4, dataset_v3)
        
        result = {
            'params': params,
            'max_diff': max_diff,
            'non_zero_count': non_zero_count,
            'status': 'SUCCESS' if max_diff < 1e-10 else 'MISMATCH'
        }
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        result = {
            'params': params,
            'status': 'ERROR',
            'error': str(e)
        }
    
    # Print result
    print("\nTest Result:")
    print("-" * 80)
    print(f"Status: {result['status']}")
    if result['status'] == 'SUCCESS':
        print(f"Max difference: {result['max_diff']}")
        print(f"Non-zero channels: {result['non_zero_count']}")
    elif result['status'] == 'ERROR':
        print(f"Error: {result['error']}")
    
    return result

#%%
if __name__ == "__main__":
    import os
    
    # Example usage
    rt_folder = r'.\P2Ms\asu_campus'  # Change this to your scenario path
    scen_name = 'asu_campus'  # Change this to your scenario name
    # Convert scenario if not yet converted
    # convert_scenario(rt_folder, use_v3=False)
    # convert_scenario(rt_folder, use_v3=True)
    
    # Option 1: Run all tests
    results = run_test_suite(scen_name)
    
    # # Test with different rotation scenarios
    # rotation_tests = [
    #     # Test 1: Single fixed rotation
    #     {
    #         'num_paths': 5,
    #         'subcarriers': 64,
    #         'selected_subcarriers': np.arange(1),
    #         'antenna_shape': np.array([1,1]),
    #         'freq_domain': True,
    #         'bs_rotation': np.array([30, 30, 30]),  # Ensure 3D vector
    #         'bs_fov': np.array([140, 120]),
    #         'ue_fov': None
    #     },
    #     # Test 2: Per-user rotation (Note: v3 will use first rotation only)
    #     {
    #         'num_paths': 5,
    #         'subcarriers': 64,
    #         'selected_subcarriers': np.arange(1),
    #         'antenna_shape': np.array([1,1]),
    #         'freq_domain': True,
    #         'bs_rotation': 'per_user',
    #         'bs_fov': np.array([140, 120]),
    #         'ue_fov': None
    #     }
    # ]
    
    # # Run each test
    # for i, test_params in enumerate(rotation_tests, 1):
    #     print(f"\n{'='*80}\nRunning rotation test {i}\n{'='*80}")
    #     result = run_single_test(scen_name, test_params)

#%%