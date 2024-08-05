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
import numpy as np
import xml.etree.ElementTree as ET
from scipy import stats

def combine_xml_files(file1, file2, output_file):
    """Combine two XML files by appending the body element from the second to the first."""
    tree1 = ET.parse(file1)
    root1 = tree1.getroot()

    tree2 = ET.parse(file2)
    root2 = tree2.getroot()

    body_element = root2.find('body')
    if body_element is not None:
        root1.find('worldbody').append(body_element)
        tree1.write(output_file, encoding='utf-8', xml_declaration=True)

def remove_element_and_save(input_file, output_file, element_name, attribute=None, value=None):
    """Remove an XML element with optional filtering by attribute and value."""
    tree = ET.parse(input_file)
    root = tree.getroot()

    for element in root.findall(element_name):
        if attribute is None or element.get(attribute) == value:
            root.remove(element)

    tree.write(output_file, encoding='utf-8', xml_declaration=True)

def filter_contact_forces(contact_force_list, threshold=3):
    """Filter out contact forces that are statistical outliers based on z-scores."""
    contact_force_array = np.array(contact_force_list)
    z_scores = stats.zscore(contact_force_array)
    return contact_force_array[np.abs(z_scores) < threshold].tolist()

def do_simulation(opt, control_method, xml_path):
    """Run a simulation using MuJoCo and capture contact forces and other metrics."""
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data) if opt == 1 else None

    time_list, ncon_list, contact_force_list = [], [], []
    time_step, MaxTime = 100000, 1000 * 100000
    rate, prev_tension, time_contact = -0.1 / time_step, 0, 0
    contact_occurred, prev_contact_force, tt = 0, 0, 1

    try:
        for step in range(MaxTime):
            if step % 1000000 == 10 and contact_force_list:
                current_contact_force = contact_force_list[-1]
                max_contact_force = max(current_contact_force, prev_contact_force)

                print(f"Tension: {tension}, Contact Force: {current_contact_force}, Rate: {rate}, Max Contact Force: {max_contact_force}")

                if max_contact_force - current_contact_force > 50 or (max_contact_force == 0 and tension < -10):
                    return finalize_simulation(tension, time_list, ncon_list, contact_force_list, step, viewer)

                prev_contact_force = current_contact_force

            tension = rate * (step - time_contact) + prev_tension
            if abs(tension) > 100:
                return finalize_simulation(tension, time_list, ncon_list, contact_force_list, step, viewer)

            mj.mj_step(model, data)
            update_controls(data, tension, control_method)

            hand_object_contacts, total_contact_force = record_contact_forces(data, model)

            time_list.append(data.time)
            ncon_list.append(hand_object_contacts)
            contact_force_list.append(total_contact_force)

            if opt == 1:
                if viewer.is_alive:
                    viewer.render()
                else:
                    break

            if step > 3:
                if total_contact_force > 350 and tt == 2:
                    prev_tension, time_contact, tt, rate = tension, step, 3, rate / 10
                elif total_contact_force > 150 and tt == 1:
                    prev_tension, time_contact, tt, rate = tension, step, 2, rate / 10

            if total_contact_force > 20:
                contact_occurred = 1

            if contact_occurred == 1 and total_contact_force < 0.1:
                print("Release of contact")
                return finalize_simulation(tension, time_list, ncon_list, contact_force_list, step, viewer)

    except KeyboardInterrupt:
        print("Simulation interrupted. Plotting results...")

    return finalize_simulation(tension, time_list, ncon_list, contact_force_list, step, viewer)

def update_controls(data, tension, control_method):
    """Update control values based on the tension and the control method."""
    ratio_btw_tension = 0.85
    data.ctrl[2] = data.ctrl[1] = tension

    if control_method == "with_extensor_control":
        data.ctrl[0] = data.ctrl[3] = ratio_btw_tension * tension
    elif control_method == "without_extensor_control":
        data.ctrl[0] = data.ctrl[3] = 0

def record_contact_forces(data, model):
    """Record the number of contacts and the total contact force."""
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

def finalize_simulation(tension, time_list, ncon_list, contact_force_list, step, viewer=None):
    """Finalize the simulation by filtering contact forces and closing the viewer."""
    filtered_contact_force_list = filter_contact_forces(contact_force_list)
    if viewer:
        viewer.close()
    return tension, time_list, ncon_list, contact_force_list, filtered_contact_force_list, step

def main(opt, control_method, xml_path):
    """Main function to run the simulation."""
    results = do_simulation(opt, control_method, xml_path)


if __name__ == "__main__":
    visualize = 1
    control_method = 'with_extensor_control' # it can be also 'without_extensor_control'
    object_number = 8

    print(f"Control method: {control_method}\nObject number: {object_number}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_path = os.path.normpath(os.path.join(script_dir, '..', '..', 'xml', 'Force_cap', 'object'))
    file1 = os.path.normpath(os.path.join(script_dir, '..', '..', 'xml', 'Force_cap', 'hand_without_object.xml'))
    file2 = os.path.normpath(os.path.join(base_path, f'obj{object_number}.xml'))
    combined_file = os.path.normpath(os.path.join(script_dir, '..', '..', 'xml', 'Force_cap', 'final.xml'))

    combine_xml_files(file1, file2, combined_file)

    main(visualize, control_method, combined_file)
