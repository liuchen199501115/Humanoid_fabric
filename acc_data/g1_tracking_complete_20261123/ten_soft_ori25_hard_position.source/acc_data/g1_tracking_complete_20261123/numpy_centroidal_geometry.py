"""URDF COM/FK/Jacobians for offline reference preparation only."""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


class CentroidalGeometry:
    def __init__(self, urdf, joint_names):
        root = ET.parse(urdf).getroot()
        self.names = list(joint_names)
        self.links = {}
        self.inertias = {}
        for link in root.findall('link'):
            inertia = link.find('inertial')
            if inertia is None:
                self.links[link.attrib['name']] = (0., np.zeros(3))
                self.inertias[link.attrib['name']] = np.zeros((3,3))
            else:
                origin = inertia.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
                self.links[link.attrib['name']] = (float(inertia.find('mass').get('value')), xyz)
                rpy = np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ')
                R = Rotation.from_euler('xyz',rpy).as_matrix(); I = inertia.find('inertia')
                matrix = np.array([[float(I.get('i'+a+b)) for b in 'xyz'] for a in 'xyz']) if all(I.get('i'+a+b) is not None for a in 'xyz' for b in 'xyz') else np.array([
                    [float(I.get('ixx')),float(I.get('ixy')),float(I.get('ixz'))],
                    [float(I.get('ixy')),float(I.get('iyy')),float(I.get('iyz'))],
                    [float(I.get('ixz')),float(I.get('iyz')),float(I.get('izz'))]])
                self.inertias[link.attrib['name']] = R@matrix@R.T
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

    def angular_momentum(self, q, v):
        """Centroidal angular momentum for world-frame floating-base velocity."""
        poses={self.root:(np.asarray(q[:3],dtype=float),Rotation.from_quat(q[3:7]).as_matrix())}
        linear={self.root:np.asarray(v[:3],dtype=float)}
        angular={self.root:np.asarray(v[3:6],dtype=float)}
        for parent,child,xyz,local_rotation,axis,index in self.joints:
            p,R=poses[parent];offset=R@xyz
            point=p+offset;child_R=R@local_rotation
            linear[child]=linear[parent]+np.cross(angular[parent],offset)
            angular[child]=angular[parent].copy()
            if index is not None:
                angular[child]+=child_R@axis*v[6+index]
                child_R=child_R@Rotation.from_rotvec(axis*q[7+index]).as_matrix()
            poses[child]=point,child_R
        points=[];velocities=[];masses=[];spin=np.zeros(3)
        for name,(mass,local) in self.links.items():
            p,R=poses[name];offset=R@local
            points.append(p+offset);velocities.append(linear[name]+np.cross(angular[name],offset));masses.append(mass)
            spin+=(R@self.inertias[name]@R.T)@angular[name]
        points,velocities,masses=map(np.asarray,(points,velocities,masses))
        com=(masses[:,None]*points).sum(0)/self.mass
        velocity=(masses[:,None]*velocities).sum(0)/self.mass
        return spin+(masses[:,None]*np.cross(points-com,velocities-velocity)).sum(0)

    def centroidal_matrices(self, q):
        """World-frame COM Jacobian and angular momentum matrix, offline only."""
        def skew(v):
            x,y,z=v; return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        com,_,poses=self.evaluate(q)
        ancestors={self.root:()}; axes={}; origins={}
        for parent,child,xyz,local_rotation,axis,index in self.joints:
            p,R=poses[parent]; chain=ancestors[parent]
            if index is not None:
                origins[index]=p+R@xyz; axes[index]=R@local_rotation@axis
                chain=(*chain,index)
            ancestors[child]=chain
        Jc=np.zeros((3,35)); A=np.zeros((3,35))
        for name,(mass,local) in self.links.items():
            p,R=poses[name]; point=p+R@local
            Jp=np.zeros((3,35)); Jw=np.zeros((3,35))
            Jp[:,:3]=np.eye(3);Jp[:,3:6]=-skew(point-q[:3]);Jw[:,3:6]=np.eye(3)
            for j in ancestors[name]:
                Jp[:,6+j]=np.cross(axes[j],point-origins[j]);Jw[:,6+j]=axes[j]
            Jc+=mass*Jp
            A+=(R@self.inertias[name]@R.T)@Jw+mass*skew(point-com)@Jp
        return Jc/self.mass,A

    def inverse_dynamics(self, q, v, a):
        """World-frame recursive Newton–Euler, gravity included, no contacts.

        Upper-body joint torques are independent of the unknown foot forces.
        Motor armature is deliberately supplied by the caller when needed.
        """
        poses={self.root:(np.asarray(q[:3],dtype=float),Rotation.from_quat(q[3:7]).as_matrix())}
        omega={self.root:np.asarray(v[3:6],dtype=float)}
        alpha={self.root:np.asarray(a[3:6],dtype=float)}
        acceleration={self.root:np.asarray(a[:3],dtype=float)}
        axes={}
        for parent,child,xyz,local_rotation,axis,index in self.joints:
            p,R=poses[parent];r=R@xyz;Rc=R@local_rotation
            w=omega[parent].copy();al=alpha[parent].copy()
            ac=acceleration[parent]+np.cross(al,r)+np.cross(w,np.cross(w,r))
            if index is not None:
                axis_world=Rc@axis;axes[index]=axis_world
                al+=axis_world*a[6+index]+np.cross(w,axis_world*v[6+index])
                w+=axis_world*v[6+index]
                Rc=Rc@Rotation.from_rotvec(axis*q[7+index]).as_matrix()
            poses[child]=(p+r,Rc);omega[child]=w;alpha[child]=al;acceleration[child]=ac
        forces={};moments={}
        for name,(mass,local) in self.links.items():
            p,R=poses[name];r=R@local;w=omega[name];al=alpha[name];inertia=R@self.inertias[name]@R.T
            f=mass*(acceleration[name]+np.cross(al,r)+np.cross(w,np.cross(w,r))+[0,0,9.81])
            forces[name]=f;moments[name]=inertia@al+np.cross(w,inertia@w)+np.cross(r,f)
        result=np.zeros(35)
        for parent,child,xyz,local_rotation,axis,index in reversed(self.joints):
            if index is not None:result[6+index]=axes[index]@moments[child]
            forces[parent]+=forces[child]
            moments[parent]+=moments[child]+np.cross(poses[child][0]-poses[parent][0],forces[child])
        result[:3]=forces[self.root];result[3:6]=moments[self.root]
        return result
