"""Sionna Ray Tracing Function."""


# Standard library imports
import os
from typing import Any

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow (excessive) logging

# Third-party imports
import numpy as np
import tensorflow as tf  # type: ignore
from tqdm import tqdm

# Sionna RT imports
from sionna.rt import Transmitter, Receiver  # type: ignore

# Local imports
from .sionna_utils import create_base_scene, set_materials
from ...converter.sionna_rt import sionna_exporter

tf.random.set_seed(1)
gpus = tf.config.list_physical_devices('GPU')
print("TensorFlow sees GPUs:", gpus)

class DataLoader:
    def __init__(self, data, batch_size):
        self.data = np.array(data)
        self.batch_size = batch_size
        self.num_samples = len(data)
        self.indices = np.arange(self.num_samples)

    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.num_samples:
            raise StopIteration
        start_idx = self.current_idx
        end_idx = min(self.current_idx + self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx
        return self.data[batch_indices]

def raytrace_sionna(osm_folder: str, tx_pos: np.ndarray, rx_pos: np.ndarray, **rt_params: Any) -> str:
        """Run ray tracing for the scene."""
        # Create scene
        scene_name = (f"sionna_{rt_params['carrier_freq']/1e9:.1f}GHz_"
                      f"{rt_params['max_reflections']}R_{rt_params['max_diffractions']}D_"
                      f"{1 if rt_params['ds_enable'] else 0}S")

        scene_folder = os.path.join(osm_folder, scene_name)
        xml_path = os.path.join(osm_folder, "scene.xml")  # Created by Blender OSM Export!
        scene = create_base_scene(xml_path, rt_params['carrier_freq'])
        scene = set_materials(scene)
        
        # Map general parameters to Sionna RT parameters
        compute_paths_rt_params = {
            "max_depth": rt_params['max_reflections'],
            "diffraction": bool(rt_params['max_diffractions']),
            "scattering": rt_params['ds_enable']
        }

        # Add BSs
        num_bs = len(tx_pos)
        for b in range(num_bs): 
            tx = Transmitter(position=tx_pos[b], name=f"BS_{b}",
                             power_dbm=tf.Variable(0, dtype=tf.float32))
            scene.add(tx)
            print(f"Added BS_{b} at position {tx_pos[b]}")

        indices = np.arange(rx_pos.shape[0])

        data_loader = DataLoader(indices, rt_params['batch_size'])
        path_list = []

        # Ray-tracing BS-BS paths
        print("Ray-tracing BS-BS paths")
        for b in range(num_bs):
            scene.add(Receiver(name=f"rx_{b}", position=tx_pos[b]))

        paths = scene.compute_paths(**compute_paths_rt_params)
        paths.normalize_delays = False
        path_list.append(paths)

        for b in range(num_bs):
            scene.remove(f"rx_{b}")

        # Ray-tracing BS-UE paths
        for batch in tqdm(data_loader, desc="Ray-tracing BS-UE paths", unit='batch'):
            for i in batch:
                scene.add(Receiver(name=f"rx_{i}", position=rx_pos[i]))

            paths = scene.compute_paths(**compute_paths_rt_params)
            paths.normalize_delays = False
            path_list.append(paths)
            
            for i in batch:
                scene.remove(f"rx_{i}")

        # Save Sionna outputs (ideally export only the FULL when it's working)
        print("Saving Sionna outputs")
        sionna_rt_folder_FULL = os.path.join(scene_folder, "sionna_export_full/")
        sionna_exporter.export_to_deepmimo(scene, path_list, rt_params, sionna_rt_folder_FULL)

        sionna_rt_folder_RX = os.path.join(scene_folder, "sionna_export_RX/")
        sionna_exporter.export_to_deepmimo(scene, path_list[1:], rt_params, sionna_rt_folder_RX)

        return sionna_rt_folder_FULL