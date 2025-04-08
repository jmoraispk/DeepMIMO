import numpy as np

from lxml import etree
import xml.etree.ElementTree as ET
from WI_interface.SetupEditor import SetupEditor
from WI_interface.TxRxEditor import TxRxEditor
from WI_interface.Material import Material

# XML template folder path
XML_TEMPLATE_PATH = "resources/xml/"

# XML parser
XML_PARSER = etree.XMLParser(recover=True)

FLOAT_STR = "%.17g"

class XmlGenerator:
    def __init__(self, scenario_path, setup_path, version=3):
        self.version = version
        self.scenario_path = scenario_path
        self.scenario = SetupEditor(scenario_path, setup_path)
        self.name = self.scenario.name
        self.txrx = TxRxEditor(scenario_path + self.scenario.txrx_file_path)
        
        # Extract material properties from terrain / city / road files
        self.terrain = Material.from_file(scenario_path + self.scenario.terrain_file_path)
        self.city = Material.from_file(scenario_path + self.scenario.city_file_path)
        self.road = Material.from_file(scenario_path + self.scenario.road_file_path)

        study_area_template = XML_TEMPLATE_PATH + "template.study_area.xml"
        if self.version >= 4:
            study_area_template = study_area_template.replace('.xml', '.v4.xml')
        self.xml = etree.parse(study_area_template, XML_PARSER)

        self.root = self.xml.getroot()
        self.update_name()

        self.scene_root = self.root.find(".//Scene")[0]
        self.load_templates()

    def load_templates(self):
        self.antenna_template_xml = etree.parse(XML_TEMPLATE_PATH + "Antenna.xml", XML_PARSER)

        self.geometry_city_template_xml = etree.parse(XML_TEMPLATE_PATH + "GeometryCity.xml", XML_PARSER)

        self.geometry_terrain_template_xml = etree.parse(XML_TEMPLATE_PATH + "GeometryTerrain.xml", XML_PARSER)
        
        tx_point_path = XML_TEMPLATE_PATH + "TxRxPoint.xml"
        tx_grid_path = XML_TEMPLATE_PATH + "TxRxGrid.xml"

        if self.version >= 4:
            tx_point_path = tx_point_path.replace('.xml', '.v4.xml')
            tx_grid_path = tx_grid_path.replace('.xml', '.v4.xml')

        self.txrx_point_template_xml = etree.parse(tx_point_path, XML_PARSER)
        self.txrx_grid_template_xml = etree.parse(tx_grid_path, XML_PARSER)

    def update_name(self):
        tmp = self.root.find(".//OutputPrefix")
        tmp[0].attrib["Value"] = self.name

        tmp = self.root.find(".//PathResultsDatabase")
        tmp[0][0][0][0][0].attrib["Value"] = tmp[0][0][0][0][0].attrib["Value"].replace('template', self.name)

    def set_carrier_freq(self):
        tmp = self.root.findall(".//CarrierFrequency")
        for t in tmp:
            t[0].attrib["Value"] = FLOAT_STR % (self.scenario.carrier_frequency)

    def set_bandwidth(self):
        tmp = self.root.findall(".//Bandwidth")
        for t in tmp:
            t[0].attrib["Value"] = FLOAT_STR % (
                self.scenario.bandwidth / 1e6
            )  # bandwidth is in MHz unit in the xml file

    def set_study_area(self):
        tmp = self.root.findall(".//StudyArea")[0]

        MaxZ = tmp.findall(".//MaxZ")[0]
        MaxZ[0].attrib["Value"] = FLOAT_STR % (self.scenario.study_area.zmax)

        MinZ = tmp.findall(".//MinZ")[0]
        MinZ[0].attrib["Value"] = FLOAT_STR % (self.scenario.study_area.zmin)

        X = tmp.findall(".//X")[0]
        X[0].attrib["Value"] = " ".join(
            ["%.6g" % i for i in self.scenario.study_area.all_vertex[:, 0]]
        )

        Y = tmp.findall(".//Y")[0]
        Y[0].attrib["Value"] = " ".join(
            ["%.6g" % i for i in self.scenario.study_area.all_vertex[:, 1]]
        )
    
    def set_ray_tracing_param(self):
        tmp = self.root.findall(".//Model")[0]

        x = tmp.findall(".//MaximumPathsPerReceiverPoint")[0]
        x[0].attrib["Value"] = "%d" % (self.scenario.ray_tracing_param.max_paths)

        x = tmp.findall(".//RaySpacing")[0]
        x[0].attrib["Value"] = FLOAT_STR % (self.scenario.ray_tracing_param.ray_spacing)

        x = tmp.findall(".//Reflections")[0]
        x[0].attrib["Value"] = "%d" % (self.scenario.ray_tracing_param.max_reflections)

        x = tmp.findall(".//Transmissions")[0]
        x[0].attrib["Value"] = "%d" % (self.scenario.ray_tracing_param.max_transmissions)

        x = tmp.findall(".//Diffractions")[0]
        x[0].attrib["Value"] = "%d" % (self.scenario.ray_tracing_param.max_diffractions)

        x = tmp.findall(".//DiffuseScatteringEnabled")[0]
        x[0].attrib["Value"] = str(self.scenario.ray_tracing_param.ds_enable).lower()

        x = tmp.findall(".//DiffuseScatteringReflections")[0]
        x[0].attrib["Value"] = "%d" % (self.scenario.ray_tracing_param.ds_max_reflections)

        x = tmp.findall(".//DiffuseScatteringTransmissions")[0]
        x[0].attrib["Value"] = "%d" % (self.scenario.ray_tracing_param.ds_max_transmissions)

        x = tmp.findall(".//DiffuseScatteringDiffractions")[0]
        x[0].attrib["Value"] = "%d" % (self.scenario.ray_tracing_param.ds_max_diffractions)

        x = tmp.findall(".//DiffuseScatteringFinalInteractionOnly")[0]
        x[0].attrib["Value"] = str(self.scenario.ray_tracing_param.ds_final_interaction_only).lower()

    def set_antenna(self):
        antenna_parent = self.scene_root.findall('AntennaList')[0][0]
        new_antenna = etree.fromstring(etree.tostring(self.antenna_template_xml), XML_PARSER)
        antenna_parent.append(new_antenna) # insert b before a

    def set_txrx(self):
        txrx_parent = self.scene_root.findall('TxRxSetList')[0][0]
        for txrx in self.txrx.txrx[::-1]:
            if txrx.txrx_type == 'points':
                new_txrx = etree.fromstring(etree.tostring(self.txrx_point_template_xml), XML_PARSER)
                x = new_txrx.findall(".//X")[0]
                x[0].attrib["Value"] = FLOAT_STR %txrx.txrx_pos[0]
                y = new_txrx.findall(".//Y")[0]
                y[0].attrib["Value"] = FLOAT_STR %txrx.txrx_pos[1]
                z = new_txrx.findall(".//Z")[0]
                z[0].attrib["Value"] = FLOAT_STR %txrx.txrx_pos[2]

            elif txrx.txrx_type == 'grid':
                new_txrx = etree.fromstring(etree.tostring(self.txrx_grid_template_xml), XML_PARSER)
                x = new_txrx.findall(".//X")[0]
                x[0].attrib["Value"] = FLOAT_STR %txrx.txrx_pos[0]
                y = new_txrx.findall(".//Y")[0]
                y[0].attrib["Value"] = FLOAT_STR %txrx.txrx_pos[1]
                z = new_txrx.findall(".//Z")[0]
                z[0].attrib["Value"] = FLOAT_STR %txrx.txrx_pos[2]

                len_x = new_txrx.findall(".//LengthX")[0]
                len_x[0].attrib["Value"] = FLOAT_STR %np.float32(txrx.grid_side[0])
                len_y = new_txrx.findall(".//LengthY")[0]
                len_y[0].attrib["Value"] = FLOAT_STR %np.float32(txrx.grid_side[1])
                grid_spacing = new_txrx.findall(".//Spacing")[0]
                grid_spacing[0].attrib["Value"] = FLOAT_STR %txrx.grid_spacing

            else:
                raise ValueError("Unsupported TxRx type: "+txrx.txrx_type)
            
            OutputID = new_txrx.findall(".//OutputID")[0]
            OutputID[0].attrib["Value"] = "%d"%txrx.txrx_id
            ShortDescription = new_txrx.findall(".//ShortDescription")[0]
            ShortDescription[0].attrib["Value"] = txrx.txrx_name
            
            receiver_parent = new_txrx.findall('.//Receiver')[0]
            antenna = receiver_parent.findall('.//Antenna')[0]
            new_antenna = etree.fromstring(etree.tostring(self.antenna_template_xml), XML_PARSER)
            receiver_parent[0].insert(receiver_parent[0].index(antenna), new_antenna) # insert b before a
            receiver_parent[0].remove(antenna)

            transmitter_parent = new_txrx.findall('.//Transmitter')[0]
            antenna = transmitter_parent.findall('.//Antenna')[0]
            new_antenna = etree.fromstring(etree.tostring(self.antenna_template_xml), XML_PARSER)
            transmitter_parent[0].insert(transmitter_parent[0].index(antenna), new_antenna) # insert b before a
            transmitter_parent[0].remove(antenna)

            if not txrx.is_transmitter:
                new_txrx[0].remove(transmitter_parent)

            if not txrx.is_receiver:
                new_txrx[0].remove(receiver_parent)

            txrx_parent.append(new_txrx)

    def set_geometry(self):
        geometry_parent = self.scene_root.findall('GeometryList')[0][0]
        
        # Map feature types to their corresponding materials and templates
        feature_config = {
            'terrain': {
                'material': self.terrain,
                'template': self.geometry_terrain_template_xml,
                'path_replace': './pathto.ter'
            },
            'city': {
                'material': self.city,
                'template': self.geometry_city_template_xml,
                'path_replace': './pathto.city'
            },
            'road': {
                'material': self.road,
                'template': self.geometry_city_template_xml,
                'path_replace': './pathto.city'
            }
        }
        
        for feature in self.scenario.features:
            if feature.type not in feature_config:
                raise ValueError("Unsupported Geometry type: "+feature.type)
                
            config = feature_config[feature.type]
            new_geometry = etree.fromstring(etree.tostring(config['template']), XML_PARSER)
            
            # Set material properties
            self._set_material_properties(new_geometry, config['material'])
            
            # Replace path placeholder
            new_geometry = etree.tostring(new_geometry, encoding="unicode")
            new_geometry = new_geometry.replace(config['path_replace'], feature.path)
            
            # Convert back to XML and append
            new_geometry = bytes(new_geometry, 'utf-8')
            new_geometry = etree.fromstring(new_geometry, XML_PARSER)
            geometry_parent.append(new_geometry)
    
    def _set_material_properties(self, geometry, material):
        """Helper method to set material properties in the geometry XML."""
        properties = [
            ("Conductivity", material.conductivity, FLOAT_STR),
            ("Permittivity", material.permittivity, FLOAT_STR),
            ("Roughness", material.roughness, FLOAT_STR),
            ("Thickness", material.thickness, FLOAT_STR),
            ("Alpha", material.directive_alpha, "%d"),
            ("Beta", material.directive_beta, "%d"),
            ("CrossPolFraction", material.cross_polarized_power, FLOAT_STR),
            ("Lambda", material.directive_lambda, FLOAT_STR),
            ("ScatteringFactor", material.fields_diffusively_scattered, FLOAT_STR)
        ]
        
        for prop_name, value, format_str in properties:
            x = geometry.findall(f".//{prop_name}")[0]
            x[0].attrib["Value"] = format_str % value

    def update(self):
        self.set_antenna()
        self.set_txrx()
        self.set_geometry()
        self.set_study_area()

        self.set_carrier_freq()
        self.set_bandwidth()

        self.set_ray_tracing_param()

    def save(self, save_path):
        ET.indent(self.root, space="  ", level=0)
        t = str(etree.tostring(self.root, pretty_print=True, encoding="unicode"))
        t = "<!DOCTYPE InSite>\n" + t

        # clean the output file before writing
        open(save_path, "w+").close()
        with open(save_path, "w") as f:
            f.write(t)


if __name__ == "__main__":
    xml = XmlGenerator("scenario_test/", "gwc.setup")
    xml.update()
    xml.save("scenario_test/gwc.study_area.xml")
