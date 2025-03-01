# main_sionna.py
import tensorflow as tf
from core.ray_tracer import RayTracer

tf.random.set_seed(1)

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    print("TensorFlow sees GPUs:" if gpus else "No GPUs found.", [gpu.name for gpu in gpus] if gpus else "")
    
    with open('scenes_folder.txt', 'r') as f:
        root_folder = f.read().strip()
    print(f"Loaded scenes folder: {root_folder}")
    
    tracer = RayTracer(root_folder)
    tracer.run()