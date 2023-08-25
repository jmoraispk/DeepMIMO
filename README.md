# AutoRayTracing

Provides 3 scripts for automating Ray Tracing simulations: 

- param_generator: turns the raw_params.csv into params.csv to facilitate large-scale simulations
    1. reads the map bounding boxes (4 coordinates: min/max lon/lat) from raw_params
    2. generates bboxes of smaller cells of size 500 x 500 meters (default)
    3. saves them to params.csv
- scene_creation: creates scenes for ray tracing (to be executed inside Blender)
    1. reads coordinates from params.csv
    2. fetches maps from OSM using those coordinates
    3. imports (only!) the buildings into Blender
    4. adds ground plane in accordance with coordinate range
    5. configures building materials
    6. exports scene in mitsuba format (.xml)
    7. writes scenes folder to scene_folder.txt
- ray_tracing_base: computes the ray tracing on the generated scenes (to be executed in an environment with Sionna)
    1. reads scene_folder.txt and params.csv
    2. loads a scene
    2. distributes users uniformely across the scene
    3. places the transmitter in a building close to the center of the scene
    4. compute ray tracing simulations with the configured parameters
    5. saves the results (paths)

# Requirements

## Blender (+ OSM plug-in & Mitsuba-blender plug-in)

- Download [Blender 3.3](https://download.blender.org/release/) (no install needed, it's ready to use!)
- Download the zip files of [OSM plug-in](https://prochitecture.gumroad.com/l/blender-osm) and [mitsuba-blender plug-in](https://github.com/mitsuba-renderer/mitsuba-blender/releases/tag/latest)
- Add (and enable after adding!) these zipfiles to Blender via Edit -> Preferences -> Addons -> Install
- In the mitsuba plug-in, make sure the dependencies are installed.
- Restart Blender
- Go into the Scripting Tab of Blender
- Run in the console:
    - `import pip`
    - `pip.main(['install', 'pandas', '--user'])`
    - `pip.main(['install', 'geopy', '--user'])`

## Python Environment with Sionna

For the time being, refer to the [install instructions from Sionna](https://nvlabs.github.io/sionna/installation.html)

If installing on Ubuntu, ask João for the latest install steps.

Other packages necessary: tqdm, geopy, scipy

# Execution: Watch the [step-by-step video](https://www.youtube.com/watch?v=usxQ6gtEekY)