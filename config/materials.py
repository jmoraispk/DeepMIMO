# config/materials.py

BUILDING_MATERIAL = 'itu_concrete'
ROAD_MATERIAL = 'itu_brick'  # Changed to asphalt in Sionna
FLOOR_MATERIAL = 'itu_wet_ground'
OTHERS_MATERIAL = 'itu_wood'

DEFAULT_MATERIALS = [BUILDING_MATERIAL, ROAD_MATERIAL, FLOOR_MATERIAL, OTHERS_MATERIAL]

KNOWN_BUILDING_OSM_MATERIALS = ['wall', 'roof']
KNOWN_ROAD_OSM_MATERIALS = [
    'areas_pedestrian', 'paths_footway', 'paths_steps', 'roads_pedestrian',
    'roads_residential', 'roads_service', 'roads_tertiary', 'roads_secondary',
    'roads_primary', 'roads_unclassified'
]

MATERIAL_COLORS = {
    ROAD_MATERIAL: (0.29, 0.25, 0.21, 1),    # dark grey
    BUILDING_MATERIAL: (0.75, 0.40, 0.16, 1), # beige
    OTHERS_MATERIAL: (0.17, 0.09, 0.02, 1),  # brown
    FLOOR_MATERIAL: (0.8, 0.8, 0.8, 1)       # light grey for wet ground
}

WORLD_EMITTER_COLOR = (0.517334, 0.517334, 0.517334, 1.0)  # RGBA for XML