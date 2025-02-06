from lxml import etree
import xml.etree.ElementTree as ET

xml_path = 'resource/template/xml/TxRxPoint.xml'
tree = etree.parse(xml_path, etree.XMLParser(recover=True))
root = tree.getroot()
# Scene = root[0][5][0]
# for child in Scene:
#     print(child.tag)

# print('done')

for child in root:
    print(child.tag)
    print(child.attrib)
    for node in child:
        print(node.tag)
        for n in node:
            print(n.tag)


Bandwidths = root.findall('.//Bandwidth')

for a in Bandwidths:
    print(a.tag)
    for b in a:
        print(b.tag)
        print(b.attrib["Value"])



ET.indent(root, space="  ", level=0)
t = str(etree.tostring(root, pretty_print=True, encoding="unicode"))
with open(xml_path,'w') as f:
    f.write(t)




'''
Model
    study area: x, y, and z

OutputLocation
OutputPrefix
PathResultsDatabase
ProjectID
Scene
    AntennaList
        CarrierFrequency
        Bandwidth
    GeometryList
    Origin
    TxRxSetList
    WaveformList
'''