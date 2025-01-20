"""
End-to-end tests for DeepMIMO functionality.

This test suite verifies the core functionality of DeepMIMO by comparing
the new implementation against the old one using a small subset of users.

To run these tests:
    pytest test/test_end_to_end.py -v -s

The flags:
    -v : verbose output
    -s : allows output capturing (required for scenario creation)
"""

import pytest
import numpy as np
import os
import sys

# Add the parent directory to the path so we can import deepmimo
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import deepmimo as dm

def test_end_to_end_simple():
    """Test end-to-end DeepMIMO functionality with a small subset of users for speed"""
    
    # Test scenario creation with absolute paths
    path_to_p2m = os.path.join(PROJECT_ROOT, 'P2Ms', 'simple_street_canyon_test', 'study_rays=0.25_res=2m_3ghz')
    old_params_dict = {'num_bs': 1, 'user_grid': [1, 91, 61], 'freq': 3.5e9}
    
    # Create both old and new scenarios
    scen_name_new = dm.create_scenario(path_to_p2m,
                                     overwrite=True,
                                     old=False,
                                     old_params=old_params_dict,
                                     scenario_name='test_simple_canyon')
    
    scen_name_old = dm.create_scenario(path_to_p2m,
                                     overwrite=True,
                                     old=True,
                                     old_params=old_params_dict,
                                     scenario_name='test_simple_canyon_old')
    
    assert isinstance(scen_name_new, str)
    assert scen_name_new == 'test_simple_canyon'
    
    # Test loading with specific tx/rx sets for new version
    tx_sets = {1: [0]}  # First TX in set 1
    rx_sets = {2: [0,1,2,3,4]}  # First 5 RXs in set 2
    
    load_params = {
        'tx_sets': tx_sets,
        'rx_sets': rx_sets,
        'max_paths': 5,
        'matrices': None
    }
    
    dataset = dm.load_scenario(scen_name_new, **load_params)
    
    # Verify basic dataset structure
    assert 'tx_pos' in dataset
    assert 'rx_pos' in dataset
    assert 'power' in dataset
    assert 'phase' in dataset
    
    # Test channel generation for new version
    ch_params = dm.ChannelGenParameters()
    
    # Compute necessary parameters for new version
    dataset['num_paths'] = dm.compute_num_paths(dataset)
    dataset['power_linear'] = dm.dbm2watt(dataset['power']) * 1000  # x1000 needed to match old version
    dataset['channel'] = dm.compute_channels(dataset, ch_params)
    dataset['pathloss'] = dm.compute_pathloss(dataset['power'], dataset['phase'])
    dataset['distances'] = dm.compute_distances(dataset['rx_pos'], dataset['tx_pos'])
    
    # Generate old version channels
    params = dm.Parameters_old(scen_name_old)
    params['user_rows'] = np.arange(1)  # First row is sufficient to check users 0-4
    dataset_old = dm.generate_old(params)
    channels_old = dataset_old[0]['user']['channel']
    
    # Verify computed parameters
    assert dataset['channel'].ndim == 4  # [users, antennas, subcarriers, paths]
    assert dataset['pathloss'].ndim == 1  # [users]
    assert dataset['distances'].ndim == 1  # [users]
    
    # Test shape consistency
    num_users = len(rx_sets[2])
    assert dataset['channel'].shape[0] == num_users
    assert dataset['pathloss'].shape[0] == num_users
    assert dataset['distances'].shape[0] == num_users
    
    # Test physical constraints
    assert np.all(dataset['power_linear'] >= 0)  # Power should be non-negative
    assert np.all(dataset['pathloss'] > 0)  # Pathloss should be positive
    assert np.all(dataset['distances'] >= 0)  # Distances should be non-negative
    
    # Compare new and old channel generation
    # Note: channels might not be exactly equal due to numerical precision
    # and implementation differences, so we use a relative error margin
    relative_error_margin = 1e-5  # < 0.001% error
    
    # Compute relative error between old and new channels
    usr_idx = 4
    error = np.abs(dataset['channel'][usr_idx] - channels_old[usr_idx])
    relative_error = error / np.abs(channels_old[usr_idx])
    
    # Check if relative error is within margin
    assert np.all(relative_error <= relative_error_margin), \
        f"Channel mismatch: max relative error = {np.max(relative_error)}" 