from pathlib import Path
import numpy as np,json,xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation
p=Path(__file__).parent;d=np.load(p/'flight_ankles_100_after_two.failure.npz');names=list(d['joint_names']);t=d['history_time'].ravel();q=d['history_q'].reshape(-1,36);ref=d['history_control_reference_q'].reshape(-1,36);feet=d['history_feet'].reshape(-1,2,7)
root=ET.parse('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf').getroot();parents={j.find('child').attrib['link']:j for j in root.findall('joint')};chains=[]
for side in ['left','right']:
 chain=[];link=side+'_ankle_roll_link'
 while link in parents:
  j=parents[link];chain.insert(0,j);link=j.find('parent').attrib['link']
 chains.append(chain)
def fk(q,side):
 pos=q[:3].astype(float).copy();rot=Rotation.from_quat(q[3:7]).as_matrix()
 for j in chains[side]:
  origin=j.find('origin');xyz=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ');rpy=np.fromstring(origin.attrib.get('rpy','0 0 0'),sep=' ')
  pos+=rot@xyz;rot=rot@Rotation.from_euler('xyz',rpy).as_matrix()
  if j.attrib['type']=='revolute':
   axis=np.fromstring(j.find('axis').attrib['xyz'],sep=' ');rot=rot@Rotation.from_rotvec(axis*q[7+names.index(j.attrib['name'])]).as_matrix()
 return pos,rot
rows=[]
for tm in [1.3,1.4,1.5,1.6,1.7,1.75,1.8,1.9]:
 i=abs(t-tm).argmin();_,ar=fk(q[i],0);rp,rr=fk(ref[i],0)
 actual=Rotation.from_quat(feet[i,0,3:]);assert np.linalg.norm((Rotation.from_matrix(ar)*actual.inv()).as_rotvec())<1e-4
 row=dict(time=tm,actual_pitch=float(actual.as_euler('xyz')[1]),target_pitch=float(Rotation.from_matrix(rr).as_euler('xyz')[1]),target_position=rp.tolist(),actual_position=feet[i,0,:3].tolist());rows.append(row);print(row)
(p/'foot_pose_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
