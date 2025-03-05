# core/ray_tracer.py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import savemat
import tensorflow as tf
from sionna.rt import Transmitter, Receiver
from constants import UE_HEIGHT, BS_HEIGHT, BATCH_SIZE, GRID_SPACING
from utils.geo_utils import *
from utils.sionna_utils import create_base_scene, set_materials

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
        self.df = pd.read_csv('params.csv')

    def generate_user_grid(self, row, origin_lat, origin_lon):
        """Generate user grid in Cartesian coordinates."""
        min_lat, min_lon = row['min_lat'], row['min_lon']
        max_lat, max_lon = row['max_lat'], row['max_lon']
        xmin, ymin, xmax, ymax = convert_GpsBBox2CartesianBBox(
            min_lat, min_lon, 
            max_lat, max_lon, 
            origin_lat, origin_lon)
        grid_x = np.arange(xmin, xmax + GRID_SPACING, GRID_SPACING)
        grid_y = np.arange(ymin, ymax + GRID_SPACING, GRID_SPACING)
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        grid_z = np.zeros_like(grid_x) + UE_HEIGHT  # 1.5m above ground
        return np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=-1)

    def process_row(self, row_idx):
        """Process a single CSV row for ray tracing."""
        scene_folder = os.path.join(self.root_folder, f'scene_{row_idx}')
        row = self.df.iloc[row_idx]
        carrier_freq = row['freq (ghz)'] * 1e9
        n_reflections = row['n_reflections']
        diffraction = bool(row['diffraction'])
        scattering = bool(row['scattering'])
        bs_lats = np.array(row['bs_lat'].split(',')).astype(np.float32)
        bs_lons = np.array(row['bs_lon'].split(',')).astype(np.float32)

        with open(os.path.join(scene_folder, "osm_gps_origin.txt"), "r") as f:
            origin_lat, origin_lon = map(float, f.read().split())
        print(f"origin_lat: {origin_lat}, origin_lon: {origin_lon}")

        num_bs = len(bs_lats)
        bs_pos_xyz = [[convert_Gps2RelativeCartesian(bs_lats[i], bs_lons[i], origin_lat, origin_lon)[0],
                       convert_Gps2RelativeCartesian(bs_lats[i], bs_lons[i], origin_lat, origin_lon)[1], 
                       BS_HEIGHT]
                      for i in range(num_bs)]

        user_grid = self.generate_user_grid(row, origin_lat, origin_lon)
        print(f"User grid shape: {user_grid.shape}")
        scene = create_base_scene(os.path.join(scene_folder, "scene.xml"), carrier_freq)
        scene = set_materials(scene)

        for i in range(num_bs): 
            tx = Transmitter(
                position=bs_pos_xyz[i], 
                name=f"BS_{i}",
                power_dbm=tf.Variable(0, dtype=tf.float32)
            )
            scene.add(tx)
            print(f"Added BS_{i} at position {bs_pos_xyz[i]}")

        indices = np.arange(user_grid.shape[0])
        indices = np.arange(1500)
        data_loader = DataLoader(indices, BATCH_SIZE)
        raytracing_results = {}
        for b in range(num_bs):
            raytracing_results[f"BS{b}_BS"] = []
            raytracing_results[f"BS{b}_UE"] = []

        print("Ray-tracing BS-BS paths")
        for b in range(num_bs):
            scene.add(Receiver(name=f"rx_{b}", position=bs_pos_xyz[b]))

        # Ray tracing
        paths = scene.compute_paths(max_depth=n_reflections, diffraction=diffraction, scattering=scattering)
        paths.normalize_delays = False

        mask, (a, tau) = paths.mask.numpy(), paths.cir()
        a, tau = a.numpy(), tau.numpy()
        doa_phi, doa_theta = paths.phi_r.numpy(), paths.theta_r.numpy()
        dod_phi, dod_theta = paths.phi_t.numpy(), paths.theta_t.numpy()
        types = paths.types.numpy()

        mask, a, tau = np.squeeze(mask), np.squeeze(a), np.squeeze(tau)
        doa_phi, doa_theta = np.squeeze(doa_phi), np.squeeze(doa_theta)
        dod_phi, dod_theta = np.squeeze(dod_phi), np.squeeze(dod_theta)
        types = np.squeeze(types)
        power = np.abs(a)**2
        phase = np.angle(a, deg=True)

        for bs_idx in range(num_bs):
            raytracing_results[f"BS{bs_idx}_BS"].append({
                'a': a[:, bs_idx], 
                'mask': mask[:, bs_idx], 
                'phase': phase[:, bs_idx], 
                'delay': tau[:, bs_idx], 
                'power': power[:, bs_idx],
                'doa_phi': doa_phi[:, bs_idx], 
                'doa_theta': doa_theta[:, bs_idx], 
                'dod_phi': dod_phi[:, bs_idx],
                'dod_theta': dod_theta[:, bs_idx], 
                'Tx_loc': np.array(bs_pos_xyz[bs_idx]), 
                'Rx_locs': np.array(bs_pos_xyz),
                'types': types
            })

        for b in range(num_bs):
            scene.remove(f"rx_{b}")

        for batch in tqdm(data_loader, desc="Ray-tracing BS-UE paths", unit='batch'):
            for i in batch:
                scene.add(Receiver(name=f"rx_{i}", position=user_grid[i]))

            # Ray tracing
            paths = scene.compute_paths(max_depth=n_reflections, diffraction=diffraction, scattering=scattering)
            paths.normalize_delays = False

            mask, (a, tau) = paths.mask.numpy(), paths.cir()
            a, tau = a.numpy(), tau.numpy()
            doa_phi, doa_theta = paths.phi_r.numpy(), paths.theta_r.numpy()
            dod_phi, dod_theta = paths.phi_t.numpy(), paths.theta_t.numpy()
            types = paths.types.numpy()

            mask, a, tau = np.squeeze(mask), np.squeeze(a), np.squeeze(tau)
            doa_phi, doa_theta = np.squeeze(doa_phi), np.squeeze(doa_theta)
            dod_phi, dod_theta = np.squeeze(dod_phi), np.squeeze(dod_theta)
            types = np.squeeze(types)
            power = np.abs(a)**2
            phase = np.angle(a, deg=True)
            
            for bs_idx in range(num_bs):
                raytracing_results[f"BS{bs_idx}_UE"].append({
                    'a': a[:, bs_idx], 
                    'mask': mask[:, bs_idx], 
                    'phase': phase[:, bs_idx], 
                    'delay': tau[:, bs_idx], 
                    'power': power[:, bs_idx],
                    'doa_phi': doa_phi[:, bs_idx], 
                    'doa_theta': doa_theta[:, bs_idx], 
                    'dod_phi': dod_phi[:, bs_idx],
                    'dod_theta': dod_theta[:, bs_idx], 
                    'Tx_loc': bs_pos_xyz[bs_idx], 
                    'Rx_locs': user_grid[batch],
                    'types': types
                })
            
            for i in batch:
                scene.remove(f"rx_{i}")

        return self.process_results(carrier_freq, raytracing_results, scene_folder, num_bs, bs_pos_xyz, indices.shape[0])

    def process_results(self, carrier_freq, results, scene_folder, num_bs, bs_pos_xyz, num_ue):
        """Process ray tracing results into DeepMIMO format."""
        ch_dicts = {}
        loc_dicts = {}
        for b in range(num_bs):
            ch_dicts[f"BS{b}_BS"] = []
            loc_dicts[f"BS{b}_BS"] = []
            ch_dicts[f"BS{b}_UE"] = []
            loc_dicts[f"BS{b}_UE"] = []
        num_features = 8

        for b in range(num_bs):
            for batch in tqdm(results[f"BS{b}_BS"], desc=f"Processing BS{b+1}-BS paths", unit='batch'):
                num_sample = batch['phase'].shape[0]
                for i in range(num_sample):
                    Rx_loc = batch['Rx_locs'][i]
                    mask = np.squeeze(batch['mask'][i])
                    num_paths = np.sum(mask)

                    if num_paths == 0:
                        store_array = np.zeros((num_features, num_paths), dtype=np.single)
                        dist = np.linalg.norm(Rx_loc - batch['Tx_loc'])
                        ch_dicts[f"BS{b}_UE"].append({'p': store_array})
                        loc_dicts[f"BS{b}_UE"].append(list(Rx_loc) + [dist] + [0])
                        continue

                    indices = np.argsort(batch['power'][i, mask])[::-1]
                    phase = batch['phase'][i, mask][indices]
                    delay = batch['delay'][i, mask][indices]
                    power = 10 * np.log10(batch['power'][i, mask][indices] + 1e-25)
                    doa_phi = np.degrees(batch['doa_phi'][i, mask][indices])
                    doa_theta = np.degrees(batch['doa_theta'][i, mask][indices])
                    dod_phi = np.degrees(batch['dod_phi'][i, mask][indices])
                    dod_theta = np.degrees(batch['dod_theta'][i, mask][indices])
                    types = batch['types'][mask][indices]

                    LOS = np.zeros((1, num_paths))
                    if types[0] == 0:
                        LOS[0, 0] = 1
                    dist = np.linalg.norm(Rx_loc - batch['Tx_loc'])

                    store_array = np.zeros((num_features, num_paths), dtype=np.single)
                    store_array[0, :], store_array[1, :], store_array[2, :] = phase, delay, power
                    store_array[3, :], store_array[4, :] = doa_phi, doa_theta
                    store_array[5, :], store_array[6, :] = dod_phi, dod_theta
                    store_array[7, :] = LOS
                    ch_dicts[f"BS{b}_BS"].append({'p': store_array})
                    loc_dicts[f"BS{b}_BS"].append(list(Rx_loc) + [dist] + [0])

            for batch in tqdm(results[f"BS{b}_UE"], desc=f"Processing BS{b+1}-UE paths", unit='batch'):
                num_sample = batch['phase'].shape[0]
                for i in range(num_sample):
                    Rx_loc = batch['Rx_locs'][i, :]
                    mask = np.squeeze(batch['mask'][i])
                    num_paths = np.sum(mask)

                    if num_paths == 0:
                        store_array = np.zeros((num_features, num_paths), dtype=np.single)
                        dist = np.linalg.norm(Rx_loc - batch['Tx_loc'])
                        ch_dicts[f"BS{b}_UE"].append({'p': store_array})
                        loc_dicts[f"BS{b}_UE"].append(list(Rx_loc) + [dist] + [0])
                        continue

                    indices = np.argsort(batch['power'][i, mask])[::-1]
                    phase = batch['phase'][i, mask][indices]
                    delay = batch['delay'][i, mask][indices]
                    power = 10 * np.log10(batch['power'][i, mask][indices] + 1e-25)
                    doa_phi = np.degrees(batch['doa_phi'][i, mask][indices])
                    doa_theta = np.degrees(batch['doa_theta'][i, mask][indices])
                    dod_phi = np.degrees(batch['dod_phi'][i, mask][indices])
                    dod_theta = np.degrees(batch['dod_theta'][i, mask][indices])
                    types = batch['types'][mask][indices]

                    LOS = np.zeros((1, num_paths))
                    if types[0] == 0:
                        LOS[0, 0] = 1
                    dist = np.linalg.norm(Rx_loc - batch['Tx_loc'])

                    store_array = np.zeros((num_features, num_paths), dtype=np.single)
                    store_array[0, :], store_array[1, :], store_array[2, :] = phase, delay, power
                    store_array[3, :], store_array[4, :] = doa_phi, doa_theta
                    store_array[5, :], store_array[6, :] = dod_phi, dod_theta
                    store_array[7, :] = LOS

                    ch_dicts[f"BS{b}_UE"].append({'p': store_array})
                    loc_dicts[f"BS{b}_UE"].append(list(Rx_loc) + [dist] + [0])

        mat_folder = os.path.join(scene_folder, "DeepMIMO_folder")
        os.makedirs(mat_folder, exist_ok=True)

        for b in range(num_bs):
            savemat(os.path.join(mat_folder, f'BS{b+1}_BS.mat'), 
                    {'channels': np.array(ch_dicts[f"BS{b}_BS"]).T,
                     'rx_locs': np.array(loc_dicts[f"BS{b}_BS"], dtype=np.single),
                     'tx_loc': bs_pos_xyz[b]})
            savemat(os.path.join(mat_folder, f'BS{b+1}_UE_0-{num_ue}.mat'),
                    {'channels': np.array(ch_dicts[f"BS{b}_UE"]).T, 
                     'rx_locs': np.array(loc_dicts[f"BS{b}_UE"], dtype=np.single),
                     'tx_loc': bs_pos_xyz[b]})

        params_file = {
            'carrier_freq': carrier_freq, 
            'doppler_available': 0, 
            'dual_polar_available': 0,
            'num_BS': num_bs, 
            'transmit_power': 0.0, 
            'user_grids': [1.0, 1.0, num_ue], 
            'version': 2
        }
        savemat(os.path.join(mat_folder, 'params.mat'), params_file)

    def run(self):
        """Run ray tracing for all scenes."""
        for row_idx in range(len(self.df)):
            self.process_row(row_idx)