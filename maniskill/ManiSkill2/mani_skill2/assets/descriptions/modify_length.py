import os, sys 
import xml.etree.ElementTree as ET
import numpy as np


for factor in np.arange(0.4, 2.5, 0.1):
    tree = ET.parse(sys.argv[1])
    root = tree.getroot()

    visual_path = f'franka_description/meshes/visual/link5_{factor:.1f}.dae'
    collision_path = f'franka_description/meshes/collision/link5_{factor:.1f}.stl'

    visual = root.find(".//mesh[@filename='franka_description/meshes/visual/link5.dae']")
    collision = root.find(".//mesh[@filename='franka_description/meshes/collision/link5.stl']") 

    visual.attrib['filename'] = visual_path
    collision.attrib['filename'] = collision_path

    # Find the 'panda_joint6' element
    joint6 = root.find(".//joint[@name='panda_joint6']")
    origin = joint6.find('origin')
    origin.attrib['xyz'] = f"0 0 {0.259 * (factor - 1):.4f}"

    tree.write(sys.argv[1].replace('.urdf', f'_{factor:.1f}.urdf'), xml_declaration=True)
    
    os.system(f'cp {sys.argv[1].replace(".urdf", f".srdf")} {sys.argv[1].replace(".urdf", f"_{factor:.1f}.srdf")}') 
