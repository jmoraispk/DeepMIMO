# AutoRayTracing

Provides 2 scripts for automating Ray Tracing simulations: 

- scene_creation.py: is supposed to be executed inside Blender
    1. reads coordinates from coords.csv
    2. fetches maps from OSM those coordinates
    3. imports the buildings into Blender
    4. adds ground plane in accordance with coordinate range
    5. configures building materials
    6. exports scene in mitsuba format (.xml)
- compute_ray_tracing.py: needs to be executed in an environment with Sionna
    1. loads the scene
    2. distributes users uniformely across the scene
    3. places the transmitter in a building close to the center of the scene
    4. compute ray tracing simulations with the configured parameters
    5. saves the results (paths)


Additonally, the repository contains compute_ray_tracing.ipynb, which is the .ipynb version of the .py script. It has visualizations and enables easier development, testing and debugging. The .py above is generated from this file.

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
    - `pip.main(['install', 'pandas', '-user'])`

## Python Environment with Sionna

For the time being, refer to the install instructions from Sionna: https://nvlabs.github.io/sionna/installation.html

