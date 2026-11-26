"""Pure NumPy URDF COM evaluation, avoiding a second simulation context."""
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

class CenterOfMassModel:
    def __init__(self,urdf,joint_names):
        root=ET.parse(urdf).getroot();self.indices={str(name):i for i,name in enumerate(joint_names)}
        self.links={};self.inertias={}
        for link in root.findall('link'):
            inertia=link.find('inertial')
            if inertia is None:continue
            origin=inertia.find('origin');center=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            self.links[link.attrib['name']]=(float(inertia.find('mass').attrib['value']),center)
            entries=inertia.find('inertia').attrib
            tensor=np.array([[float(entries['ixx']),float(entries['ixy']),float(entries['ixz'])],[float(entries['ixy']),float(entries['iyy']),float(entries['iyz'])],[float(entries['ixz']),float(entries['iyz']),float(entries['izz'])]])
            rpy=np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            axes=Rotation.from_euler('xyz',rpy).as_matrix();self.inertias[link.attrib['name']]=axes@tensor@axes.T
        self.joints=[];children=set()
        for joint in root.findall('joint'):
            parent=joint.find('parent').attrib['link'];child=joint.find('child').attrib['link'];children.add(child)
            origin=joint.find('origin');xyz=np.zeros(3);rpy=np.zeros(3)
            if origin is not None:
                xyz=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ')
            axis=joint.find('axis');axis=np.fromstring(axis.attrib.get('xyz','1 0 0'),sep=' ') if axis is not None else np.array([1.,0.,0.])
            kind=joint.attrib['type'];name=joint.attrib['name']
            if kind!='fixed' and name not in self.indices:raise ValueError('Unknown articulated joint '+name)
            self.joints.append((parent,child,xyz,Rotation.from_euler('xyz',rpy).as_matrix(),axis,None if kind=='fixed' else self.indices[name]))
        roots={x.attrib['name'] for x in root.findall('link')}-children
        if len(roots)!=1:raise ValueError('Expected one URDF root')
        self.root=next(iter(roots));self.mass=sum(m for m,c in self.links.values())
    def evaluate(self,q):
        transforms={self.root:(np.asarray(q[:3],float),Rotation.from_quat(q[3:7]).as_matrix())}
        remaining=list(self.joints)
        while remaining:
            following=[]
            for parent,child,xyz,R,axis,index in remaining:
                if parent not in transforms:following.append((parent,child,xyz,R,axis,index));continue
                pp,pr=transforms[parent];rotation=R if index is None else R@Rotation.from_rotvec(axis*float(q[7+index])).as_matrix()
                transforms[child]=(pp+pr@xyz,pr@rotation)
            if len(following)==len(remaining):raise ValueError('Disconnected URDF')
            remaining=following
        return sum(m*(transforms[name][0]+transforms[name][1]@c) for name,(m,c) in self.links.items())/self.mass

    def momentum(self,q,v,return_state=False):
        transforms={self.root:(np.asarray(q[:3],float),Rotation.from_quat(q[3:7]).as_matrix(),np.asarray(v[:3],float),np.asarray(v[3:6],float))}
        remaining=list(self.joints)
        while remaining:
            following=[]
            for parent,child,xyz,R,axis,index in remaining:
                if parent not in transforms:following.append((parent,child,xyz,R,axis,index));continue
                pp,pr,pv,pw=transforms[parent];offset=pr@xyz
                rotation=R if index is None else R@Rotation.from_rotvec(axis*float(q[7+index])).as_matrix()
                omega=pw if index is None else pw+pr@R@axis*float(v[6+index])
                transforms[child]=(pp+offset,pr@rotation,pv+np.cross(pw,offset),omega)
            if len(following)==len(remaining):raise ValueError('Disconnected URDF')
            remaining=following
        centers=[];velocities=[];spin=[];masses=[]
        for name,(mass,center) in self.links.items():
            pos,R,linear,angular=transforms[name];offset=R@center
            centers.append(pos+offset);velocities.append(linear+np.cross(angular,offset));spin.append(R@self.inertias[name]@R.T@angular);masses.append(mass)
        masses=np.array(masses);centers=np.array(centers);velocities=np.array(velocities)
        com=(masses[:,None]*centers).sum(0)/self.mass
        momentum=np.sum(spin,axis=0)+np.sum(masses[:,None]*np.cross(centers-com,velocities),axis=0)
        if return_state:return com,(masses[:,None]*velocities).sum(0)/self.mass,momentum
        return momentum
