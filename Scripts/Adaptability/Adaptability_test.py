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
import os
import json
from scipy.interpolate import interp1d

def combine_xml_files(file1, file2, output_file):
    # Parse the first XML file
    tree1 = ET.parse(file1)
    root1 = tree1.getroot()
    # Parse the second XML file
    tree2 = ET.parse(file2)
    root2 = tree2.getroot()
    # Find the body element in the second file
    body_element = root2.find('body')
    worldbody = root1.find('worldbody')
    worldbody.append(body_element)
    tree1.write(output_file, encoding='utf-8', xml_declaration=True)

def remove_equality_element_and_save(input_file, output_file):
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Find and remove the equality element
    equality_element = root.find('equality')
    if equality_element is not None:
        root.remove(equality_element)

    # Write the updated XML to the output file
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def parse_args():
    parser = make_parser()
    args = parser.parse_args()
    return args

def make_parser():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--xml_dir', type=str, default=r'D:\Git\CTR\xml\Adapt\final.xml')
    return parser


def do_simulation(opt, Routing_type):
    args = parse_args()
    t = 0
    # Load model, data and viewer
    model = mj.MjModel.from_xml_path(args.xml_dir)
    data = mj.MjData(model)
    if opt == 1:
        viewer = mujoco_viewer.MujocoViewer(model, data)

    # Lists to store time, number of contacts, and contact forces
    time_list = []
    ncon_list = []
    contact_force_list = []
    time = 10000
    rate = -1/time
    low_limit_force = 2

    # simulate and render
    for kk in range(time):
        if opt == 1:
            if viewer.is_alive:
                simstart = data.time
                while (data.time - simstart < 1.0 / 60.0):
                    tension = rate * t
                    if abs(tension) >low_limit_force:
                        return [tension, time_list, ncon_list, contact_force_list, kk]
                        viewer.close()
                    t += 1
                    mj.mj_step(model, data)
                    data.ctrl[0] = tension
                    if Routing_type == 'fcr':
                        data.ctrl[1] = tension
                    hand_object_contacts = 0
                    total_contact_force = 0.0

                    for i in range(data.ncon):
                        contact = data.contact[i]
                        geom1 = model.geom(contact.geom1)
                        geom2 = model.geom(contact.geom2)
                        if ((geom1.contype == 1 and geom2.contype == 2) or
                            (geom1.contype == 2 and geom2.contype == 1)):
                            hand_object_contacts += 1
                            # Get the contact force
                            contact_force_vec = np.zeros(6)
                            mj.mj_contactForce(model, data, i, contact_force_vec)
                            # Extract the 3D force part from the contact force
                            contact_force_magnitude = np.linalg.norm(contact_force_vec[:3])
                            total_contact_force += contact_force_magnitude

                    time_list.append(data.time)
                    ncon_list.append(hand_object_contacts)
                    contact_force_list.append(total_contact_force)
                    viewer.render()
                    print(kk)
                    print(f"tension: {tension}")
            else:
                break

        else:
            simstart = data.time
            while (data.time - simstart < 1.0 / 60.0):
                t += 1
                tension = rate * t
                if abs(tension) > low_limit_force:
                    return [tension, time_list, ncon_list, contact_force_list, kk]
                mj.mj_step(model, data)
                data.ctrl[0] = tension
                if Routing_type == 'fcr':
                    data.ctrl[1] = tension

                # Record time, number of contacts, and contact forces (between hand and object only)
                hand_object_contacts = 0
                total_contact_force = 0.0

                for i in range(data.ncon):
                    contact = data.contact[i]
                    geom1 = model.geom(contact.geom1)
                    geom2 = model.geom(contact.geom2)
                    if ((geom1.contype == 1 and geom2.contype == 2) or
                            (geom1.contype == 2 and geom2.contype == 1)):
                        hand_object_contacts += 1
                        # Get the contact force
                        contact_force_vec = np.zeros(6)
                        mj.mj_contactForce(model, data, i, contact_force_vec)
                        # Extract the 3D force part from the contact force
                        contact_force_magnitude = np.linalg.norm(contact_force_vec[:3])
                        total_contact_force += contact_force_magnitude

                time_list.append(data.time)
                ncon_list.append(hand_object_contacts)
                contact_force_list.append(total_contact_force)

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

def update_controls(data, tension, routing_type):
    data.ctrl[1] = tension
    data.ctrl[2] = tension
    ratio_btw_tension = 0.85
    if routing_type == "with extensor control":
        data.ctrl[0] = ratio_btw_tension * tension
        data.ctrl[3] = ratio_btw_tension * tension
    elif routing_type == "without extensor control":
        data.ctrl[0] = 0
        data.ctrl[3] = 0


def finalize_simulation(tension, time_list, ncon_list, contact_force_list, step, viewer=None):
    filtered_contact_force_list = filter_contact_forces(contact_force_list)
    max_contact_force = max(contact_force_list) if contact_force_list else 0
    if viewer:
        viewer.close()
    return tension, time_list, ncon_list, contact_force_list, filtered_contact_force_list, step, max_contact_force
def filter_contact_forces(contact_force_list, threshold=3):
    contact_force_array = np.array(contact_force_list)
    z_scores = stats.zscore(contact_force_array)
    return contact_force_array[np.abs(z_scores) < threshold].tolist()

def main(opt, Routing_type):
    args = parse_args()
    [tension, time_list, ncon_list, contact_force_list, kk] = do_simulation(opt, Routing_type)
    print(f"tension is {tension}")


if __name__ == "__main__":
    Visualize = 1
    Routing_type = 'pcr'
    object_number = 3

    print(f"Routing type: {Routing_type}\nObject number: {object_number}")
    base_path = r'D:\Git\CTR\xml\Adapt\object'
    file1 = r'D:\Git\CTR\xml\Adapt\hand_without_object_pcr.xml'
    file2 = f"{base_path}\\obj{object_number}.xml"
    file3 = r'D:\Git\CTR\xml\Adapt\hand_without_object_fcr.xml'
    combined_file = r'D:\Git\CTR\xml\Adapt\final.xml'

    if Routing_type == 'fcr':
        remove_equality_element_and_save(file1,file3)
        combine_xml_files(file3, file2, combined_file)
    elif Routing_type == 'pcr':
        combine_xml_files(file1, file2, combined_file)
    main(Visualize, Routing_type)
