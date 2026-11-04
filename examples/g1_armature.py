"""Reflected motor inertia from Unitree's G1 29-DOF MIMIC configuration.

Source: https://github.com/unitreerobotics/unitree_rl_lab/blob/main/
source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py
Units: kg m^2. Parallel ankle and waist actuators use twice the 5020 inertia.
"""
import numpy as np


def unitree_armatures(joint_names):
    values = []
    for name in joint_names:
        if any(part in name for part in ('hip_pitch', 'hip_yaw', 'waist_yaw')):
            value = 0.010177520
        elif any(part in name for part in ('hip_roll', 'knee')):
            value = 0.025101925
        elif any(part in name for part in ('ankle_', 'waist_roll', 'waist_pitch')):
            value = 2 * 0.003609725
        elif any(part in name for part in ('wrist_pitch', 'wrist_yaw')):
            value = 0.00425
        elif any(part in name for part in ('shoulder_', 'elbow', 'wrist_roll')):
            value = 0.003609725
        else:
            raise ValueError(f'No G1 motor armature mapping for {name}')
        values.append(value)
    return np.asarray([values], dtype=np.float32)
