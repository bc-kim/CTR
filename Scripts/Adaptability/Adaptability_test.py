# <!-- /
# *  Copyright(c) 2024 Byungchul Kim
# *
# *  Permission is hereby granted, free of charge, to any person obtaining a copy
# *  of this software and associated documentation files(the "Software"), to deal
# * in the Software without restriction, including without limitation the rights
# *  to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# *  copies of the Software, and to permit persons to whom the Software is
# *  furnished to do so, subject to the following conditions:
#  *
#  *  The above copyright notice and this permission notice shall be included in all
#  *  copies or substantial portions of the Software.
#  *
#  *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#  *  WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
#  *  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#  *
#  *  Created 2024 by Byungchul Kim[https://bc-kim.github.io]. - ->

import os
import mujoco as mj
import mujoco_viewer
import xml.etree.ElementTree as ET
import numpy as np
from scipy import stats

def combine_xml_files(file1, file2, output_file):
    tree1 = ET.parse(file1)
    root1 = tree1.getroot()
    tree2 = ET.parse(file2)
    root2 = tree2.getroot()
    body_element = root2.find('body')
    if body_element is not None:
        root1.find('worldbody').append(body_element)
    tree1.write(output_file, encoding='utf-8', xml_declaration=True)

def remove_equality_element_and_save(input_file, output_file):
    tree = ET.parse(input_file)
    root = tree.getroot()
    equality_element = root.find('equality')
    if equality_element is not None:
        root.remove(equality_element)
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def do_simulation(opt, routing_type, xml_dir):
    model = mj.MjModel.from_xml_path(xml_dir)
    data = mj.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data) if opt == 1 else None

    time_list, ncon_list, contact_force_list = [], [], []
    total_steps = 10000
    rate = -1 / total_steps
    low_limit_force = 2

    for step in range(total_steps):
        if opt == 1 and not viewer.is_alive:
            break
        tension = rate * step
        if abs(tension) > low_limit_force:
            return tension, time_list, ncon_list, contact_force_list, step

        mj.mj_step(model, data)
        data.ctrl[0] = tension
        if routing_type == 'fcr':
            data.ctrl[1] = tension

        hand_object_contacts, total_contact_force = record_contact_forces(data, model)

        time_list.append(data.time)
        ncon_list.append(hand_object_contacts)
        contact_force_list.append(total_contact_force)

        if viewer:
            viewer.render()

    if viewer:
        viewer.close()

    return tension, time_list, ncon_list, contact_force_list, step

def record_contact_forces(data, model):
    hand_object_contacts, total_contact_force = 0, 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1, geom2 = model.geom(contact.geom1), model.geom(contact.geom2)
        if (geom1.contype == 1 and geom2.contype == 2) or (geom1.contype == 2 and geom2.contype == 1):
            hand_object_contacts += 1
            contact_force_vec = np.zeros(6)
            mj.mj_contactForce(model, data, i, contact_force_vec)
            total_contact_force += np.linalg.norm(contact_force_vec[:3])
    return hand_object_contacts, total_contact_force

def filter_contact_forces(contact_force_list, threshold=3):
    z_scores = stats.zscore(np.array(contact_force_list))
    return [force for force, z in zip(contact_force_list, z_scores) if abs(z) < threshold]

def main(opt, routing_type, xml_dir):
    tension, time_list, ncon_list, contact_force_list, step = do_simulation(opt, routing_type, xml_dir)
    filtered_contact_force_list = filter_contact_forces(contact_force_list)
    max_contact_force = max(contact_force_list) if contact_force_list else 0

    print(f"Tension: {tension}")
    print(f"Max Contact Force: {max_contact_force}")

if __name__ == "__main__":
    Visualize = 1
    Routing_type = 'fcr'
    object_number = 3

    print(f"Routing type: {Routing_type}\nObject number: {object_number}")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Normalize and resolve paths to absolute paths
    base_path = os.path.normpath(os.path.join(script_dir, '..', '..', 'xml', 'Adapt', 'object'))
    file1 = os.path.normpath(os.path.join(script_dir, '..', '..', 'xml', 'Adapt', 'hand_without_object_pcr.xml'))
    file2 = os.path.normpath(os.path.join(base_path, f'obj{object_number}.xml'))
    file3 = os.path.normpath(os.path.join(script_dir, '..', '..', 'xml', 'Adapt', 'hand_without_object_fcr.xml'))
    combined_file = os.path.normpath(os.path.join(script_dir, '..', '..', 'xml', 'Adapt', 'final.xml'))

    if Routing_type == 'fcr':
        remove_equality_element_and_save(file1, file3)
        combine_xml_files(file3, file2, combined_file)
    elif Routing_type == 'pcr':
        combine_xml_files(file1, file2, combined_file)

    main(Visualize, Routing_type, combined_file)
