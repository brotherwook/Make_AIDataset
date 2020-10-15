import xml.etree.ElementTree as et
import os

dir_path = "D:\F20001_3_output"
dates = os.listdir(dir_path)

for date in dates:
    files = os.listdir(os.path.join(dir_path, date))
    for file in files:
        xml_path = os.path.join(dir_path, date, file, file + ".xml")
        xml = et.parse(xml_path)
        root = xml.getroot()
        temp = root.findall("image")
        for i, image in enumerate(temp):
            image.attrib["id"] = str(i)
        xml.write(xml_path)
