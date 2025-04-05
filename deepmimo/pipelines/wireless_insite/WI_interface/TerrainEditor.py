from math import cos, sin, radians
import numpy as np


class Terrain:
    def __init__(
        self,
        fields_diffusively_scattered,
        cross_polarized_power,
        directive_alpha,
        directive_beta,
        directive_lambda,
        conductivity,
        permittivity,
        roughness,
        thickness,
    ):
        self.fields_diffusively_scattered = fields_diffusively_scattered
        self.cross_polarized_power = cross_polarized_power
        self.directive_alpha = directive_alpha
        self.directive_beta = directive_beta
        self.directive_lambda = directive_lambda
        self.conductivity = conductivity
        self.permittivity = permittivity
        self.roughness = roughness
        self.thickness = thickness


class TerrainEditor:
    def __init__(self, infile_path="resource/template/feature/newTerrain.ter"):
        self.infile_path = infile_path
        self.parse()

    def parse(self):
        with open(self.infile_path, "r") as f:
            self.file = f.readlines()

        for i, line in enumerate(self.file):
            if line.startswith("fields_diffusively_scattered"):
                fields_diffusively_scattered = np.float64(line.split(" ")[-1][:-1])
                continue
            if line.startswith("cross_polarized_power"):
                cross_polarized_power = np.float64(line.split(" ")[-1][:-1])
                continue
            if line.startswith("directive_alpha"):
                directive_alpha = np.int64(line.split(" ")[-1][:-1])
                continue
            if line.startswith("directive_beta"):
                directive_beta = np.int64(line.split(" ")[-1][:-1])
                continue
            if line.startswith("directive_lambda"):
                directive_lambda = np.float64(line.split(" ")[-1][:-1])
                continue
            if line.startswith("conductivity"):
                conductivity = np.float64(line.split(" ")[-1][:-1])
                continue
            if line.startswith("permittivity"):
                permittivity = np.float64(line.split(" ")[-1][:-1])
                continue
            if line.startswith("roughness"):
                roughness = np.float64(line.split(" ")[-1][:-1])
                continue
            if line.startswith("thickness"):
                thickness = np.float64(line.split(" ")[-1][:-1])
                continue

        self.terrain = Terrain(
            fields_diffusively_scattered,
            cross_polarized_power,
            directive_alpha,
            directive_beta,
            directive_lambda,
            conductivity,
            permittivity,
            roughness,
            thickness,
        )

    def set_vertex(self, xmin, ymin, xmax, ymax, z=0):
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
        with open(material_path, "r") as f:
            self.material_file = f.readlines()

        for i in range(len(self.file)):
            if self.file[i].startswith("begin_<Material>"):
                start = i
            if self.file[i].startswith("end_<Material>"):
                end = i

        self.file = self.file[:start] + self.material_file + self.file[end + 1 :]

    def save(self, outfile_path):
        # clean the output file before writing
        open(outfile_path, "w+").close()

        with open(outfile_path, "w") as out:
            out.writelines(self.file)


if __name__ == "__main__":
    # infile_path = "resource/template/feature/newTerrain.ter"
    material_path = "resource/material/ITU Wet earth 2.4 GHz.mtl"
    outfile_path = "test/newTerrain.ter"
    editor = TerrainEditor()
    editor.set_vertex(-200, -200, 200, 200, 0)
    editor.set_material(material_path)
    editor.save(outfile_path)
    print("done")
