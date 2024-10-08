# Code for paper "Analysis and Classification of the Constrained Tendon Routings for the Soft Hand-wearable Robots" submitted at RAL
#### Overview
- Run ``Adaptability_test.py`` to see the result for adaptability simulation.
- You can test force-constrained tendon routing and position-constrained tendon routing by changing the ``Routing_type`` of the code as ``fcr`` or ``pcr``. 
- Run ``Force_capability.py`` to see the result for force capability simulation.
- You can test ``with_extensor_control`` case and ``without_extensor_control``case in Force_capability simulation by changing ``control_method`` of the code as ``with_extensor_control`` or ``without_extensor_control``.
- You can change the object_number to test different object.
- If ``Visualize = 1``, the code visualize the simulation. 

#### Environment
`conda create -n mujoco python=3.10`

#### Dependencies
- `pip install mujoco`
- `pip install mujoco-viewer` (Try `pip install mujoco-python-viewer`, if it doesn't work).
- `pip install mediapy`

- Version (6/29/2024): mujoco (3.1.6), mujoco-python-viewer (0.1.4), mediapy(1.2.2)

#### Other useful commands
- `python -m mujoco.viewer` to see the mujoco viewer.

#### Bibtex
``` bibtex
@inproceedings{
  Kim2024CTR,
  title={Analysis and Classification of the Constrained Tendon Routings for the Soft Hand-wearable Robots},
  author={Byungchul Kim and Useok Jeong and Kyu-Jin Cho},
  journal={IEEE robotics and automation letters},
  year={submitted},
  url={https://sites.google.com/view/constrained-tendon-routings/overview}
}