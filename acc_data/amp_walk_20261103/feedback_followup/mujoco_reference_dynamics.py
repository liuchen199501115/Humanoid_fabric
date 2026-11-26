"""URDF rigid-body dynamics for offline reference fitting, without physics stepping."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from g1_armature import unitree_armatures

def numbers(x):return ' '.join(str(float(v)) for v in x)
class ReferenceDynamics:
 def __init__(self,urdf,names):
  self.names=list(names);root=ET.parse(urdf).getroot();links={x.get('name'):x for x in root.findall('link')};children={};childnames=set()
  for j in root.findall('joint'):
   parent=j.find('parent').get('link');child=j.find('child').get('link');children.setdefault(parent,[]).append(j);childnames.add(child)
  base=next(iter(set(links)-childnames));xml=ET.Element('mujoco',model='g1_reference_dynamics');ET.SubElement(xml,'compiler',angle='radian',fusestatic='false');ET.SubElement(xml,'option',gravity='0 0 -9.81');world=ET.SubElement(xml,'worldbody');armature=dict(zip(self.names,np.asarray(unitree_armatures(self.names)).ravel()))
  def body(parent,name,joint=None):
   attrs={'name':name}
   if joint is not None:
    origin=joint.find('origin');attrs['pos']=origin.get('xyz','0 0 0') if origin is not None else '0 0 0';rpy=origin.get('rpy','0 0 0') if origin is not None else '0 0 0';attrs['quat']=numbers(np.roll(Rotation.from_euler('xyz',np.fromstring(rpy,sep=' ')).as_quat(),1))
   node=ET.SubElement(parent,'body',**attrs);inertial=links[name].find('inertial')
   if inertial is not None:
    origin=inertial.find('origin');pos=origin.get('xyz','0 0 0') if origin is not None else '0 0 0';rpy=origin.get('rpy','0 0 0') if origin is not None else '0 0 0';i=inertial.find('inertia');I=np.array([[float(i.get('ixx')),float(i.get('ixy')),float(i.get('ixz'))],[float(i.get('ixy')),float(i.get('iyy')),float(i.get('iyz'))],[float(i.get('ixz')),float(i.get('iyz')),float(i.get('izz'))]]);R=Rotation.from_euler('xyz',np.fromstring(rpy,sep=' ')).as_matrix();I=R@I@R.T
    ET.SubElement(node,'inertial',pos=pos,mass=inertial.find('mass').get('value'),fullinertia=numbers([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
   if joint is None:ET.SubElement(node,'freejoint',name='root')
   elif joint.get('type')!='fixed':
    axis=joint.find('axis');ET.SubElement(node,'joint',name=joint.get('name'),type='hinge',axis=axis.get('xyz') if axis is not None else '1 0 0',armature=str(armature[joint.get('name')]))
   for j in children.get(name,[]):body(node,j.find('child').get('link'),j)
  body(world,base);self.model=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'));self.data=mujoco.MjData(self.model);self.qindex=[];self.vindex=[]
  for name in self.names:
   j=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,name);self.qindex.append(self.model.jnt_qposadr[j]);self.vindex.append(self.model.jnt_dofadr[j])
  self.feet=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,f'{side}_ankle_roll_link') for side in ['left','right']]
  assert self.model.nv==35 and self.model.nq==36
 def evaluate(self,q,v,a=None,kinematic_bias=False):
  m,d=self.model,self.data;R=Rotation.from_quat(q[3:7]).as_matrix();T=np.zeros((35,35));T[:3,:3]=np.eye(3);T[3:6,3:6]=R.T;T[self.vindex,np.arange(6,35)]=1
  d.qpos[:3]=q[:3];d.qpos[3:7]=np.roll(q[3:7],1);d.qpos[self.qindex]=q[7:];d.qvel[:]=T@v;mujoco.mj_forward(m,d);M=np.empty((35,35));mujoco.mj_fullM(m,d,M);mass=T.T@M@T;bias=T.T@d.qfrc_bias;feet=[];jac=[];foot_bias=[]
  for body in self.feet:
   jp=np.zeros((3,35));jr=jp.copy();mujoco.mj_jacBody(m,d,jp,jr,body);feet.append(np.r_[d.xpos[body],Rotation.from_matrix(d.xmat[body].reshape(3,3)).as_quat()]);jac.append(np.vstack((jp,jr))@T)
   if kinematic_bias:
    jpd=np.zeros((3,35));jrd=np.zeros((3,35));mujoco.mj_jacDot(m,d,jpd,jrd,d.xpos[body],body);foot_bias.append(np.r_[jpd@d.qvel,jrd@d.qvel])
  result=dict(mass=mass,bias=bias,feet=np.array(feet),jacobian=np.array(jac))
  if kinematic_bias:result['foot_bias']=np.asarray(foot_bias)
  if a is not None:result['generalized_force']=mass@a+bias
  return result
