from math import cos, sin, radians


class ObjectEditor:
    def __init__(self, infile_path):
        self.infile_path = infile_path
        self.read_file()

    def read_file(self):
        with open(self.infile_path, "r") as f:
            self.file = f.readlines()
        print("")

    def transform(self, translate=[0, 0, 0], rotate_angle=0):
        for i in range(len(self.file)):
            if "nVertices" in self.file[i]:
                num_vertex = int(self.file[i].split()[1])
                for j in range(i + 1, i + 1 + num_vertex):
                    vals = self.file[j].split()
                    x = float(vals[0])
                    y = float(vals[1])
                    z = float(vals[2])

                    # Rotation (Flat earth assumption) - should be extended for slopes
                    cos_z_rot = cos(radians(rotate_angle))
                    sin_z_rot = sin(radians(rotate_angle))
                    x_new = cos_z_rot * x - sin_z_rot * y
                    y_new = sin_z_rot * x + cos_z_rot * y
                    z_new = z

                    # Translation
                    x_new += translate[0]
                    y_new += translate[1]
                    z_new += translate[2]

                    self.file[j] = str.format(
                        "{:12.10f} {:12.10f} {:12.10f} \n", x_new, y_new, z_new
                    )

    def save(self, outfile_path):
        # clean the output file before writing
        open(outfile_path, "w+").close()

        with open(outfile_path, "w") as out:
            out.writelines(self.file)


if __name__ == "__main__":
    infile_path = "scenario/car_models/audiA7_lowpoly.object"
    outfile_path = "scenario/car_models/audiA7_lowpoly_test.object"
    editor = ObjectEditor(infile_path)
    editor.transform([30, 40, 50], 90)
    editor.save(outfile_path)
    print("done")
