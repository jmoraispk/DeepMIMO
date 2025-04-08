"""
TerrainEditor - A utility for creating and editing terrain (.ter) files for electromagnetic simulation.

This module provides functionality to create and modify terrain files used in electromagnetic
simulation software. The TerrainEditor class focuses on three main operations:

1. Setting vertex positions for a flat rectangular terrain (creating two triangles)
2. Incorporating material properties from a material file into the .ter file
3. Saving the resulting .ter file.

The class is designed to work with a template terrain file, modify specific sections of it,
and output a new terrain file with the desired properties. 

Example usage:
    editor = TerrainEditor()
    editor.set_vertex(-200, -200, 200, 200, 0)  # Create a 400x400 flat terrain at z=0
    editor.set_material("path/to/material.mtl")  # Apply material properties
    editor.save("output.ter")  # Save the terrain file
"""

import numpy as np

class TerrainEditor:
    """Class for creating and editing terrain (.ter) files."""
    
    def __init__(self, template_ter_file="resources/feature/newTerrain.ter"):
        self.template_ter_file = template_ter_file
        with open(self.template_ter_file, "r") as f:
            self.file = f.readlines()

    def set_vertex(self, xmin, ymin, xmax, ymax, z=0):
        """ Write vertices of the flat rectangular terrain to the file: 2 triangles """
        v1 = np.asarray([xmin, ymin, z])
        v2 = np.asarray([xmax, ymin, z])
        v3 = np.asarray([xmax, ymax, z])
        v4 = np.asarray([xmin, ymax, z])

        self.file[40] = "%.10f %.10f %.10f\n" % (v1[0], v1[1], v1[2])
        self.file[41] = "%.10f %.10f %.10f\n" % (v2[0], v2[1], v2[2])
        self.file[42] = "%.10f %.10f %.10f\n" % (v3[0], v3[1], v3[2])

        self.file[47] = "%.10f %.10f %.10f\n" % (v4[0], v4[1], v4[2])
        self.file[48] = "%.10f %.10f %.10f\n" % (v1[0], v1[1], v1[2])
        self.file[49] = "%.10f %.10f %.10f\n" % (v3[0], v3[1], v3[2])

    def set_material(self, material_path):
        """ Write material properties to the file """
        with open(material_path, "r") as f:
            self.material_file = f.readlines()

        for i in range(len(self.file)):
            if self.file[i].startswith("begin_<Material>"):
                start = i
            if self.file[i].startswith("end_<Material>"):
                end = i

        self.file = self.file[:start] + self.material_file + self.file[end + 1 :]

    def save(self, outfile_path):
        """Save the terrain file."""
        # clean the output file before writing
        open(outfile_path, "w+").close()

        with open(outfile_path, "w") as out:
            out.writelines(self.file)


if __name__ == "__main__":
    material_path = "resources/material/ITU Wet earth 2.4 GHz.mtl"
    outfile_path = "test/newTerrain.ter"
    editor = TerrainEditor()
    editor.set_vertex(-200, -200, 200, 200, 0)
    editor.set_material(material_path)
    editor.save(outfile_path)
    print("done")
