"""Script to generate example images for DeepMIMO documentation.

This script generates the images used in the quickstart guide:
- basic_scene.png: Basic scene visualization
- scene_analysis.png: Scene with material analysis
- coverage_map.png: Coverage analysis visualization
- ray_paths.png: Ray tracing paths visualization
"""

import os
import deepmimo as dm
import matplotlib.pyplot as plt
import numpy as np

# Create _static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(__file__), '_static')
os.makedirs(static_dir, exist_ok=True)

def save_fig(name, tight=True, dpi=300):
    """Helper to save figures with consistent settings."""
    if tight:
        plt.tight_layout()
    plt.savefig(os.path.join(static_dir, f'{name}.png'), 
                dpi=dpi, bbox_inches='tight')
    plt.close()

# Load a sample scenario (ASU campus or simple street canyon)
print("Loading scenario...")
dataset = dm.load_scenario('asu_campus_3p5', tx_sets={1: [0]}, rx_sets={2: 'all'})

# 1. Basic Scene Visualization
print("Generating basic scene visualization...")
_, ax = dataset.scene.plot(show=False)
ax.set_title(f"DeepMIMO Scene Example: ASU Campus 3.5 GHz ({ax.get_title()})")
save_fig('basic_scene')

# 2. Scene Analysis with Materials
print("Generating scene analysis visualization...")
scene = dataset.scene
buildings = scene.get_objects('buildings')

# Plot buildings colored by material
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Get unique materials
materials = buildings.get_materials()
colors = plt.cm.tab10(np.linspace(0, 1, len(materials)))

for mat_idx, color in zip(materials, colors):
    buildings_mat = scene.get_objects(label='buildings', material=mat_idx)
    if len(buildings_mat) > 0:
        buildings_mat.plot(ax=ax, color=color, alpha=0.7)

ax.set_title("Buildings Colored by Material")
save_fig('scene_analysis')

# 3. Coverage Map
print("Generating coverage map...")
plt.figure(figsize=(10, 8))
dm.plot_coverage(dataset.rx_pos, dataset.power[:,0], 
                bs_pos=dataset.tx_pos.T,
                title="Power Coverage Map (dB)")
save_fig('coverage_map')

# 4. Ray Paths
print("Generating ray paths visualization...")
# Find a user with multiple paths for better visualization
user_idx = np.argmax(dataset.num_paths)
dm.plot_rays(dataset.rx_pos[user_idx], 
            dataset.tx_pos[0],
            dataset.inter_pos[user_idx], 
            dataset.inter[user_idx],
            proj_3D=True, 
            color_by_type=True)
plt.title("Ray Paths Example")
save_fig('ray_paths')

print("Done! Images saved in docs/_static/") 