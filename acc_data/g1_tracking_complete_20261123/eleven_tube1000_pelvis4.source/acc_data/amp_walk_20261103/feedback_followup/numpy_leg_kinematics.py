"""Small URDF leg FK/Jacobian for reference-only IK and offline audits."""
import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

class LegKinematics:
 def __init__(self,urdf,joint_names):
  self.names=list(joint_names);nodes=ET.parse(urdf).getroot().findall('joint');self.by_child={n.find('child').attrib['link']:n for n in nodes};self.chains=[];self.lower=[];self.upper=[]
  for name in self.names[:12]:
   node=next(n for n in nodes if n.attrib['name']==name);limit=node.find('limit');self.lower.append(float(limit.attrib['lower']));self.upper.append(float(limit.attrib['upper']))
  self.lower=np.array(self.lower);self.upper=np.array(self.upper)
  for side in ['left','right']:
   chain=[];link=side+'_ankle_roll_link'
   while link in self.by_child:
    n=self.by_child[link];origin=n.find('origin');xyz=np.fromstring(origin.get('xyz','0 0 0') if origin is not None else '0 0 0',sep=' ');rpy=np.fromstring(origin.get('rpy','0 0 0') if origin is not None else '0 0 0',sep=' ');axis=n.find('axis');axis=np.fromstring(axis.get('xyz','1 0 0') if axis is not None else '1 0 0',sep=' ');index=self.names.index(n.attrib['name']) if n.attrib['type']!='fixed' else None
    chain.append((xyz,Rotation.from_euler('xyz',rpy).as_matrix(),axis,index));link=n.find('parent').attrib['link']
   self.chains.append(chain[::-1])
 def evaluate(self,q,side):
  position=np.array(q[:3],dtype=float);rotation=Rotation.from_quat(q[3:7]).as_matrix();axes=[];origins=[]
  for xyz,R,axis,index in self.chains[side]:
   position=position+rotation@xyz;rotation=rotation@R
   if index is not None:
    axes.append(rotation@axis);origins.append(position.copy());rotation=rotation@Rotation.from_rotvec(axis*q[7+index]).as_matrix()
  axes=np.asarray(axes);origins=np.asarray(origins);J=np.concatenate((np.cross(axes,position-origins).T,axes.T),axis=0)
  return position,rotation,J
