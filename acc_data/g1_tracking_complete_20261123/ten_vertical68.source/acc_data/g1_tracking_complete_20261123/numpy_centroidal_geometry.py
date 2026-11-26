"""URDF COM/FK/Jacobians for offline reference preparation only."""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


class CentroidalGeometry:
    def __init__(self, urdf, joint_names):
        root = ET.parse(urdf).getroot()
        self.names = list(joint_names)
        self.links = {}
        for link in root.findall('link'):
            inertia = link.find('inertial')
            if inertia is None:
                self.links[link.attrib['name']] = (0., np.zeros(3))
            else:
                origin = inertia.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                self.links[link.attrib['name']] = (float(inertia.find('mass').get('value')), xyz)
        pending = []
        for joint in root.findall('joint'):
            origin = joint.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            axis = joint.find('axis')
            axis = np.fromstring(axis.get('xyz', '1 0 0') if axis is not None else '1 0 0', sep=' ')
            index = self.names.index(joint.get('name')) if joint.get('type') != 'fixed' else None
            pending.append((joint.find('parent').get('link'), joint.find('child').get('link'),
                            xyz, Rotation.from_euler('xyz', rpy).as_matrix(), axis, index))
        roots = set(self.links)-{j[1] for j in pending}
        if len(roots) != 1:
            raise ValueError('Expected one floating root link')
        self.root = roots.pop(); known = {self.root}; self.joints = []
        while pending:
            ready = [j for j in pending if j[0] in known]
            if not ready:
                raise ValueError('Disconnected URDF tree')
            for joint in ready:
                self.joints.append(joint); known.add(joint[1]); pending.remove(joint)
        self.mass = sum(mass for mass, _ in self.links.values())

    def evaluate(self, q):
        poses = {self.root: (np.asarray(q[:3], dtype=float), Rotation.from_quat(q[3:7]).as_matrix())}
        ancestors = {self.root: ()}; axes = {}; origins = {}
        for parent, child, xyz, local_rotation, axis, index in self.joints:
            p, R = poses[parent]
            p = p+R@xyz; R = R@local_rotation
            chain = ancestors[parent]
            if index is not None:
                origins[index] = p; axes[index] = R@axis
                R = R@Rotation.from_rotvec(axis*q[7+index]).as_matrix()
                chain = (*chain, index)
            poses[child] = p, R; ancestors[child] = chain
        com = np.zeros(3); J = np.zeros((3, 29))
        for name, (mass, local) in self.links.items():
            p, R = poses[name]; point = p+R@local
            com += mass*point
            for j in ancestors[name]:
                J[:, j] += mass*np.cross(axes[j], point-origins[j])
        return com/self.mass, J/self.mass, poses
