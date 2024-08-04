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


import mujoco as mj
import mujoco_viewer
import argparse
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import shutil
from scipy import stats


def combine_xml_files(file1, file2, output_file):
    tree1 = ET.parse(file1)
    root1 = tree1.getroot()

    tree2 = ET.parse(file2)
    root2 = tree2.getroot()

    body_element = root2.find('body')
    root1.find('worldbody').append(body_element)
    tree1.write(output_file, encoding='utf-8', xml_declaration=True)


def remove_element_and_save(input_file, output_file, element_name, attribute=None, value=None):
    tree = ET.parse(input_file)
    root = tree.getroot()

    for element in root.findall(element_name):
        if attribute is None or element.get(attribute) == value:
            root.remove(element)

    tree.write(output_file, encoding='utf-8', xml_declaration=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_dir', type=str, default=r'D:\Git\CTR\xml\Force_cap\final.xml')
    return parser.parse_args()


def filter_contact_forces(contact_force_list, threshold=3):
    contact_force_array = np.array(contact_force_list)
    z_scores = stats.zscore(contact_force_array)
    return contact_force_array[np.abs(z_scores) < threshold].tolist()


def do_simulation(opt, routing_type, xml_path):
    t = 0
    tt = 1
    low_limit_force = 100

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data) if opt == 1 else None

    time_list = []
    ncon_list = []
    contact_force_list = []
    time_step = 10000
    MaxTime = 1000 * time_step
    rate = -0.1 / time_step
    prev_tension = 0
    time_contact = 0
    contact_occurred = 0
    prev_contact_force = 0
    try:
        for step in range(MaxTime):
            if step % 1000000 == 10 and contact_force_list:
                print(f"tension: {tension}")
                print(f"contact_force: {contact_force_list[-1]}")
                print(f"rate:{rate}")
                current_contact_force = contact_force_list[-1]
                max_contact_force = max(current_contact_force,prev_contact_force)
                if max_contact_force - current_contact_force > 50:
                    return finalize_simulation(tension, time_list, ncon_list, contact_force_list, step, viewer)
                prev_contact_force = current_contact_force

            tension = rate * (t-time_contact) + prev_tension
            if abs(tension) > low_limit_force:
                return finalize_simulation(tension, time_list, ncon_list, contact_force_list, step, viewer)

            t += 1
            mj.mj_step(model, data)
            update_controls(data, tension, routing_type)

            hand_object_contacts, total_contact_force = record_contact_forces(data, model)

            time_list.append(data.time)
            ncon_list.append(hand_object_contacts)
            contact_force_list.append(total_contact_force)

            if opt == 1 and viewer.is_alive:
                viewer.render()
            elif opt == 1:
                break

            if step > 3 and total_contact_force > 350:
                if tt == 1:
                    prev_tension = tension
                    time_contact = t
                    tt = 2
                    rate = rate / 2

            elif step > 3 and total_contact_force > 375:
                if tt == 3:
                    prev_tension = tension
                    time_contact = t
                    tt = 4
                    rate = rate / 5000

            if total_contact_force > 60:
                contact_occurred = 1
                print(contact_occurred)

            if contact_occurred == 1 and total_contact_force < 0.1:
                print("Release of contact")
                return finalize_simulation(tension, time_list, ncon_list, contact_force_list, step, viewer)

    except KeyboardInterrupt:
        print("Simulation interrupted. Plotting results...")

    return finalize_simulation(tension, time_list, ncon_list, contact_force_list, step, viewer)


def update_controls(data, tension, routing_type):
    data.ctrl[2] = tension
    data.ctrl[1] = tension
    ratio_btw_tension = 0.85
    if routing_type == "ffcr":
        data.ctrl[0] = ratio_btw_tension * tension
        data.ctrl[3] = ratio_btw_tension * tension
    elif routing_type == "fcr":
        data.ctrl[0] = 0


def record_contact_forces(data, model):
    hand_object_contacts = 0
    total_contact_force = 0.0

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1 = model.geom(contact.geom1)
        geom2 = model.geom(contact.geom2)
        if ((geom1.contype == 1 and geom2.contype == 2) or (geom1.contype == 2 and geom2.contype == 1)):
            hand_object_contacts += 1
            contact_force_vec = np.zeros(6)
            mj.mj_contactForce(model, data, i, contact_force_vec)
            total_contact_force += np.linalg.norm(contact_force_vec[:3])

    return hand_object_contacts, total_contact_force


def finalize_simulation(tension, time_list, ncon_list, contact_force_list, step, viewer=None):
    filtered_contact_force_list = filter_contact_forces(contact_force_list)
    if viewer:
        viewer.close()
    return tension, time_list, ncon_list, contact_force_list, filtered_contact_force_list, step


def main(opt, routing_type):
    args = parse_args()
    results = do_simulation(opt, routing_type, args.xml_dir)

if __name__ == "__main__":
    visualize = 1
    routing_type = 'ffcr'  # 'ffcr' or 'fcr'
    object_number = 8

    print(f"Routing type: {routing_type}\nObject number: {object_number}")
    base_path = r'D:\Git\CTR\xml\Force_cap\object'
    file1 = f"{base_path}\\obj{object_number}.xml"
    file2 = r'D:\Git\CTR\xml\Force_cap\hand_without_object.xml'
    combined_file = r'D:\Git\CTR\xml\Force_cap\final.xml'
    combine_xml_files(file2, file1, combined_file)

    main(visualize, routing_type)
