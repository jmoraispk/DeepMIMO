# Constants used throughout the automated scene creation script

# Project root directory (use absolute path)
PROJ_ROOT = '.'
BLENDER_PATH = r"/home/joao/blender-3.6.0-linux-x64/blender"

# Material names for scene objects
BUILDING_MATERIAL = 'itu_concrete'
ROAD_MATERIAL = 'itu_brick'  # Note: Manually changed to asphalt in Sionna
FLOOR_MATERIAL = 'itu_wet_ground'

# Add-ons to install, mapping name to zip file
ADDONS = {
    "blosm": "blosm_2.7.11.zip",
    "mitsuba-blender": "mitsuba-blender.zip",
}

# Ray-tracing parameters
UE_HEIGHT = 1.5  # meters
BS_HEIGHT = 20  # meters
BATCH_SIZE = 5
GRID_SPACING = 1.0  # meters