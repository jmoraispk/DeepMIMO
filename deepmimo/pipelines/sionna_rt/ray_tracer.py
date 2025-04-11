import os
import numpy as np
from tqdm import tqdm
from constants import BATCH_SIZE
from utils.sionna_utils import create_base_scene, set_materials
from deepmimo.converter.sionna_rt import sionna_exporter

import tensorflow as tf
from sionna.rt import Transmitter, Receiver

tf.random.set_seed(1)
gpus = tf.config.list_physical_devices('GPU')
print("TensorFlow sees GPUs:" if gpus else "No GPUs found.", [gpu.name for gpu in gpus] if gpus else "")

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

class RayTracer:
    def __init__(self, root_folder):
        self.root_folder = root_folder    

    def run(self, scene_name, bs_pos, user_grid, batch_size=5,
            carrier_freq=3.5, n_reflections=1, diffraction=True, scattering=True):
        """Run ray tracing for the scene."""
        # Create scene
        scene_folder = os.path.join(self.root_folder, scene_name)
        scene = create_base_scene(os.path.join(scene_folder, "scene.xml"), carrier_freq)
        scene = set_materials(scene)
        rt_params = {
            "max_depth": n_reflections,
            "diffraction": diffraction,
            "scattering": scattering
        }

        # Add BSs
        num_bs = len(bs_pos)
        for b in range(num_bs): 
            tx = Transmitter(
                position=bs_pos[b], 
                name=f"BS_{b}",
                power_dbm=tf.Variable(0, dtype=tf.float32)
            )
            scene.add(tx)
            print(f"Added BS_{b} at position {bs_pos[b]}")

        indices = np.arange(user_grid.shape[0])
        indices = np.arange(100)

        data_loader = DataLoader(indices, BATCH_SIZE)
        path_list = []
        # Ray-tracing BS-BS paths
        print("Ray-tracing BS-BS paths")
        for b in range(num_bs):
            scene.add(Receiver(name=f"rx_{b}", position=bs_pos[b]))

        paths = scene.compute_paths(**rt_params)
        paths.normalize_delays = False
        path_list.append(paths)

        for b in range(num_bs):
            scene.remove(f"rx_{b}")

        # Ray-tracing BS-UE paths
        for batch in tqdm(data_loader, desc="Ray-tracing BS-UE paths", unit='batch'):
            for i in batch:
                scene.add(Receiver(name=f"rx_{i}", position=user_grid[i]))

            paths = scene.compute_paths(**rt_params)
            paths.normalize_delays = False
            path_list.append(paths)
            
            for i in batch:
                scene.remove(f"rx_{i}")

        # Save Sionna outputs
        print("Saving Sionna outputs")
        sionna_rt_folder_FULL = os.path.join(scene_folder, "sionna_export_full/")
        sionna_exporter.export_to_deepmimo(scene, path_list, rt_params, sionna_rt_folder_FULL)

        sionna_rt_folder_RX = os.path.join(scene_folder, "sionna_export_RX/")
        sionna_exporter.export_to_deepmimo(scene, path_list[1:], rt_params, sionna_rt_folder_RX)

        return 