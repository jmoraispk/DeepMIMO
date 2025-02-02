# How to add another addon?

- Add-on zips must contain the add-on folder inside (with the name matching the add-on name)
- Inside the add-on folder, there must be a `__init__.py`
- Remember to update this information in `scene_creation.py`

That's it.

# IMPORTANT: Special packages

Packages like mitsuba are "interesting"...

dependencies...

requires manual install

1. Download the zip in https://github.com/mitsuba-renderer/mitsuba-blender/releases/tag/v0.4.0
2. Follow the manual install instructions:
    2.1. In Blender, go to Edit -> Preferences -> Add-ons -> Install.
    2.2. Select the downloaded ZIP archive.
    2.3. Find the add-on using the search bar and enable it.
    2.4. Click on "Install dependencies using pip" to download the latest package