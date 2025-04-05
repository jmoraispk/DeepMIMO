import numpy as np


class StudyArea:
    def __init__(
        self,
        zmin,
        zmax,
        num_vertex,
        all_vertex,
    ):
        self.zmin = zmin
        self.zmax = zmax
        self.num_vertex = num_vertex
        self.all_vertex = all_vertex  # np.empty((num_vertex, 3))


class RayTracingParam:
    def __init__(
        self,
        max_paths,
        ray_spacing,
        max_reflections,
        max_transmissions,
        max_diffractions,
        ds_enable,
        ds_max_reflections,
        ds_max_transmissions,
        ds_max_diffractions,
        ds_final_interaction_only,
    ):
        self.max_paths = max_paths
        self.ray_spacing = ray_spacing
        self.max_reflections = max_reflections
        self.max_transmissions = max_transmissions
        self.max_diffractions = max_diffractions
        self.ds_enable = ds_enable
        self.ds_max_reflections = ds_max_reflections
        self.ds_max_transmissions = ds_max_transmissions
        self.ds_max_diffractions = ds_max_diffractions
        self.ds_final_interaction_only = ds_final_interaction_only


class Feature:
    def __init__(self, index, type, path):
        self.index = index
        self.type = type
        self.path = path


class SetupEditor:
    def __init__(self, scenario_path, name=None):
        self.scenario_path = scenario_path

        with open("resource/template/setup/feature.txt") as f1:
            self.feature_template = f1.readlines()
        with open("resource/template/setup/txrx.txt") as f1:
            self.txrx_template = f1.readlines()

        self.num_feature = 0
        self.feature_sec = []
        self.txrx_sec = self.txrx_template.copy()

        if not name:
            with open("resource/template/setup/template.setup", "r") as f:
                self.setup_file = f.readlines()
        else:
            with open(self.scenario_path + name, "r") as f:
                self.setup_file = f.readlines()

        self.parse()

    def parse(self):
        self.name = self.setup_file[1].split(" ")[-1][:-1]
        self.features = []
        for i, line in enumerate(self.setup_file):
            if line.startswith("CarrierFrequency "):
                self.carrier_frequency = np.float64(line.split(" ")[-1])
                continue
            if line.startswith("bandwidth "):
                self.bandwidth = np.float64(line.split(" ")[-1])
                continue
            if line.startswith("begin_<boundary>"):  # study area boundary
                zmin = np.float64(self.setup_file[i + 8].split(" ")[-1])
                zmax = np.float64(self.setup_file[i + 9].split(" ")[-1])
                num_vertex = np.int64(self.setup_file[i + 10].split(" ")[-1])
                all_vertex = np.empty((num_vertex, 3))
                for j in range(num_vertex):
                    all_vertex[j, 0] = np.float64(
                        self.setup_file[i + j + 11].split(" ")[0]
                    )
                    all_vertex[j, 1] = np.float64(
                        self.setup_file[i + j + 11].split(" ")[1]
                    )
                    all_vertex[j, 2] = np.float64(
                        self.setup_file[i + j + 11].split(" ")[2]
                    )
                continue

            if line.startswith("MaxRenderedPaths"):
                max_paths = np.float64(line.split(" ")[-1])
                continue
            if line.startswith("ray_spacing"):
                ray_spacing = np.float64(line.split(" ")[-1])
                continue
            if line.startswith("max_reflections"):
                max_reflections = np.float64(line.split(" ")[-1])
                continue
            if line.startswith("max_transmissions"):
                max_transmissions = np.float64(line.split(" ")[-1])
                continue
            if line.startswith("max_wedge_diffractions"):
                max_diffractions = np.float64(line.split(" ")[-1])
                continue
            if line.startswith("begin_<diffuse_scattering>"):
                ds_enable = (
                    True
                    if self.setup_file[i + 1].split(" ")[-1][:-1] == "yes"
                    else False
                )
                ds_max_reflections = np.float64(self.setup_file[i + 2].split(" ")[-1])
                ds_max_diffractions = np.float64(self.setup_file[i + 3].split(" ")[-1])
                ds_max_transmissions = np.float64(self.setup_file[i + 4].split(" ")[-1])
                ds_final_interaction_only = (
                    True
                    if self.setup_file[i + 5].split(" ")[-1][:-1] == "yes"
                    else False
                )
                continue

            if line.startswith("begin_<txrx_sets>"):  # txrx
                self.txrx_file_path = self.setup_file[i + 1].split(" ")[-1][2:-1]
                self.first_available_txrx = np.int64(
                    self.setup_file[i + 2].split(" ")[-1]
                )
                continue

            if line.startswith("begin_<feature>"):  # feature
                feature_idx = np.int64(self.setup_file[i + 1].split(" ")[-1])
                feature_type = self.setup_file[i + 2][:-1]
                feature_path = self.setup_file[i + 4].split(" ")[-1][:-1]
                self.features.append(Feature(feature_idx, feature_type, feature_path))
                if feature_type == "terrain":
                    self.terrain_file_path = feature_path
                continue

        self.study_area = StudyArea(
            zmin,
            zmax,
            num_vertex,
            all_vertex,
        )
        self.ray_tracing_param = RayTracingParam(
            max_paths,
            ray_spacing,
            max_reflections,
            max_transmissions,
            max_diffractions,
            ds_enable,
            ds_max_reflections,
            ds_max_transmissions,
            ds_max_diffractions,
            ds_final_interaction_only,
        )
        return

    def set_carrierFreq_and_bandwidth(self, carrier_frequency, bandwidth):
        self.carrier_frequency = carrier_frequency
        self.bandwidth = bandwidth

    def set_study_area(self, zmin, zmax, all_vertex):
        self.study_area = StudyArea(zmin, zmax, all_vertex.shape[0], all_vertex)

    def set_ray_tracing_param(
        self,
        max_paths,
        ray_spacing,
        max_reflections,
        max_transmissions,
        max_diffractions,
        ds_enable,
        ds_max_reflections,
        ds_max_transmissions,
        ds_max_diffractions,
        ds_final_interaction_only,
    ):
        self.ray_tracing_param = RayTracingParam(
            max_paths,
            ray_spacing,
            max_reflections,
            max_transmissions,
            max_diffractions,
            ds_enable,
            ds_max_reflections,
            ds_max_transmissions,
            ds_max_diffractions,
            ds_final_interaction_only,
        )

    def set_txrx(self, txrx_file_path):
        num_txrx = 0
        with open(self.scenario_path + txrx_file_path) as f1:
            txrx_file = f1.readlines()

        for line in txrx_file:
            if line.startswith("project_id"):
                num_txrx += 1
                first_available_txrx = np.int64(line.split(" ")[-1][:-1]) + 1

        if num_txrx <= 0:
            raise ValueError("Zero TxRx is defined!")

        self.txrx_sec[1] = self.txrx_sec[1].replace("[path]", "./" + txrx_file_path)
        self.txrx_sec[2] = self.txrx_sec[2].replace(
            "[index]", str(first_available_txrx)
        )

        self.first_available_txrx = first_available_txrx
        return

    def add_feature(self, feature_file_path, feature_type="object"):
        tmp = self.feature_template.copy()
        tmp[1] = tmp[1].replace("[index]", str(self.num_feature))
        tmp[2] = tmp[2].replace("[type]", feature_type)
        tmp[4] = tmp[4].replace("[path]", "./" + feature_file_path)

        self.feature_sec += tmp
        self.num_feature += 1

    def update_carrier_frequency_bandwidth(self):
        for i, line in enumerate(self.setup_file):
            if line.startswith("CarrierFrequency "):
                self.setup_file[i] = "CarrierFrequency %.6f\n" % self.carrier_frequency
                continue
            if line.startswith("bandwidth "):
                self.setup_file[i] = "bandwidth %.6f\n" % self.bandwidth
                continue
    
    def update_study_area(self):
        for i, line in enumerate(self.setup_file):
            if line.startswith("begin_<boundary>"):  # study area boundary
                self.setup_file[i + 8] = "zmin %.6f\n" % self.study_area.zmin
                self.setup_file[i + 9] = "zmax %.6f\n" % self.study_area.zmax
                self.setup_file[i + 10] = "nVertices %d\n" % self.study_area.num_vertex
                for j in range(self.study_area.num_vertex):
                    self.setup_file[i + j + 11] = "%.6f %.6f %.6f\n" % (
                        self.study_area.all_vertex[j, 0],
                        self.study_area.all_vertex[j, 1],
                        self.study_area.all_vertex[j, 2],
                    )
                return
    
    def update_ray_tracing_param(self):
        for i, line in enumerate(self.setup_file):
            if line.startswith("MaxRenderedPaths"):
                self.setup_file[i] = "MaxRenderedPaths %d\n" % self.ray_tracing_param.max_paths
                continue
            if line.startswith("ray_spacing"):
                self.setup_file[i] = "ray_spacing %.6f\n" % self.ray_tracing_param.ray_spacing
                continue

            if line.startswith("max_reflections"):
                self.setup_file[i] = "max_reflections %d\n" % self.ray_tracing_param.max_reflections
                continue

            if line.startswith("max_transmissions"):
                self.setup_file[i] = "max_transmissions %d\n" % self.ray_tracing_param.max_transmissions
                continue

            if line.startswith("max_wedge_diffractions"):
                self.setup_file[i] = "max_wedge_diffractions %d\n" % self.ray_tracing_param.max_diffractions
                continue

            if line.startswith("begin_<diffuse_scattering>"):
                self.setup_file[i + 1] = "enabled yes\n" if self.ray_tracing_param.ds_enable else "enabled no\n"
                self.setup_file[i + 2] = "diffuse_reflections %d\n" % self.ray_tracing_param.ds_max_reflections
                self.setup_file[i + 3] = "diffuse_diffractions %d\n" % self.ray_tracing_param.ds_max_diffractions
                self.setup_file[i + 4] = "diffuse_transmissions %d\n" % self.ray_tracing_param.ds_max_transmissions
                self.setup_file[i + 5] = "final_interaction_only yes\n" if self.ray_tracing_param.ds_final_interaction_only else "final_interaction_only no\n"
                continue

    def update_features(self):
        for i, line in enumerate(self.setup_file):
            if line.startswith("end_<studyarea>"):
                self.feature_start = i + 1
                break
        self.setup_file = (
            self.setup_file[: self.feature_start]
            + self.feature_sec
            + self.txrx_sec
            + self.setup_file[self.feature_start :]
        )

    def update_all(self):
        self.update_carrier_frequency_bandwidth()
        self.update_study_area()
        self.update_ray_tracing_param()
        self.update_features()

    def save(self, name, save_path=None):
        if not save_path:
            save_path = self.scenario_path + name + ".setup"
        self.update_all()
        self.setup_file[1] = self.setup_file[1].replace("template", name)

        # clean the output file before writing
        open(save_path, "w+").close()
        with open(save_path, "w") as f:
            for line in self.setup_file:
                f.write(line)


if __name__ == "__main__":
    scenario = SetupEditor(scenario_path="scenario_test/")
    scenario.set_txrx("gwc.txrx")
    scenario.set_study_area(
        0,
        17.5,
        np.asarray([[-200, -165, 0], [200, -165, 0], [200, 165, 0], [-200, 165, 0]]),
    )
    scenario.add_feature("newTerrain.ter", "terrain")
    scenario.add_feature("gwc_building.city", "city")
    scenario.add_feature("gwc_road.city", "city")

    scenario.save("gwc")
    print("done")
