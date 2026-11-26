"""Small joint-space IK adaptation of one landing; original audit poses stay intact."""
import numpy as np,torch
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
from scipy.optimize import lsq_linear

@torch.no_grad()
def adapt_landing(fabric,trajectory,joint_names,event,next_event,xy,smoothing=2.):
 side=1-event['support'];lo=event['takeoff_frame'];land=event['end_frame'];release=next_event['takeoff_frame'];end=next_event['end_frame']
 if 1-next_event['support']!=side:raise ValueError('Expected the next swing of the same foot')
 result=trajectory.clone();model=fabric.dynamics_tasks.model;link=model.kinematics.get_link_index(('left','right')[side]+'_ankle_roll_link')
 joints=[j for j,n in enumerate(joint_names) if j<12 and n.startswith(('left_' if side==0 else 'right_'))];columns=np.array(joints)+6
 limits={e.attrib['name']:e.find('limit') for e in ET.parse(fabric.urdf_path).getroot().findall('joint')}
 low=np.array([float(limits[joint_names[j]].attrib['lower']) for j in joints]);high=np.array([float(limits[joint_names[j]].attrib['upper']) for j in joints])
 def blend(u):u=min(1.,max(0.,u));return u**3*(10-15*u+6*u*u)
 worst_position=worst_orientation=0.;worst_frame=None;goal_records=[]
 for k in range(lo+1,end):
  weight=blend((k-lo)/(land-lo))*(1-blend((k-release)/(end-release)))
  if weight<1e-12:continue
  original=result[k:k+1].clone();pose=original.clone();state=model.snapshot(pose)
  links=[model.kinematics.get_link_index(n+'_ankle_roll_link') for n in ('left','right')]
  goals=state['transforms'][0,links,:3].numpy().astype(float);goals[side,:2]+=weight*np.asarray(xy)
  target_R=Rotation.from_matrix(state['rotations'][0,links].numpy());root_R=Rotation.from_quat(original[0,3:7].numpy())
  all_low=np.array([float(limits[joint_names[j]].attrib['lower']) for j in range(12)]);all_high=np.array([float(limits[joint_names[j]].attrib['upper']) for j in range(12)])
  for iteration in range(45):
   state=model.snapshot(pose);pe=goals-state['transforms'][0,links,:3].numpy();re=(target_R*Rotation.from_matrix(state['rotations'][0,links].numpy()).inv()).as_rotvec();error=np.c_[pe,re].ravel()
   if np.linalg.norm(pe,axis=1).max()<3e-5 and np.linalg.norm(re,axis=1).max()<3e-4:break
   J=np.concatenate((state['origin_jac'][0,links].numpy(),state['angular_jac'][0,links].numpy()),axis=1).reshape(12,35)[:,:18]
   root_translation=(pose[0,:3]-original[0,:3]).numpy();root_rotation=(Rotation.from_quat(pose[0,3:7].numpy())*root_R.inv()).as_rotvec()
   lower=np.r_[np.maximum(-.004,np.array([-.03,-.03,-.05])-root_translation),np.maximum(-.01,-.05-root_rotation),np.maximum(-.06,all_low-pose[0,7:19].numpy())]
   upper=np.r_[np.minimum(.004,np.array([.03,.03,.015])-root_translation),np.minimum(.01,.05-root_rotation),np.minimum(.06,all_high-pose[0,7:19].numpy())]
   lower=np.minimum(lower,upper-1e-10)
   regularizer=np.r_[np.full(6,.02),np.full(12,.001)]
   update=lsq_linear(np.r_[J,np.diag(regularizer)],np.r_[error,np.zeros(18)],bounds=(lower,upper),tol=1e-8,lsmr_tol=1e-10,max_iter=80).x
   pose[0,:3]+=pose.new_tensor(update[:3]);pose[0,3:7]=pose.new_tensor((Rotation.from_rotvec(update[3:6])*Rotation.from_quat(pose[0,3:7].numpy())).as_quat());pose[0,7:19]+=pose.new_tensor(update[6:])
  goal_records.append((k,goals.copy(),target_R))
  pe=np.array([np.linalg.norm(pe,axis=1).max()]);re=np.array([np.linalg.norm(re,axis=1).max()])
  if np.linalg.norm(pe)>worst_position:worst_position=float(np.linalg.norm(pe));worst_frame=k
  worst_orientation=max(worst_orientation,float(np.linalg.norm(re)));result[k]=pose[0]
 if smoothing>0:
  from scipy.ndimage import gaussian_filter1d
  old=trajectory.numpy().astype(float);new=result.numpy().astype(float)
  rotation_delta=(Rotation.from_quat(new[:,3:7])*Rotation.from_quat(old[:,3:7]).inv()).as_rotvec()
  delta=np.c_[new[:,:3]-old[:,:3],rotation_delta,new[:,7:]-old[:,7:]]
  delta=gaussian_filter1d(delta,smoothing,axis=0,mode='constant')
  window=np.array([blend((k-lo)/(4*smoothing))*blend((end-k)/(4*smoothing)) for k in range(len(result))]);delta*=window[:,None]
  new[:,:3]=old[:,:3]+delta[:,:3];new[:,3:7]=(Rotation.from_rotvec(delta[:,3:6])*Rotation.from_quat(old[:,3:7])).as_quat();new[:,7:]=old[:,7:]+delta[:,6:]
  result=trajectory.new_tensor(new)
  worst_position=worst_orientation=0.
  for k,goals,target_R in goal_records:
   st=model.snapshot(result[k:k+1]);pe=np.linalg.norm(goals-st['transforms'][0,links,:3].numpy(),axis=1).max();re=(target_R*Rotation.from_matrix(st['rotations'][0,links].numpy()).inv()).magnitude().max()
   if pe>worst_position:worst_position=float(pe);worst_frame=k
   worst_orientation=max(worst_orientation,float(re))
  low_all=np.array([float(limits[name].attrib['lower']) for name in joint_names]);high_all=np.array([float(limits[name].attrib['upper']) for name in joint_names])
  violation=max(float((low_all-new[:,7:]).max()),float((new[:,7:]-high_all).max()),0.)
  if violation>1e-5:raise ValueError('Smoothed landing reference exceeds joint limits: '+str(violation))
 report=dict(event_end_frame=land,next_release_frame=release,smoothing_frames=smoothing,xy_offset_m=list(xy),peak_ik_position_error_m=worst_position,worst_frame=worst_frame,peak_ik_orientation_error_rad=worst_orientation,peak_joint_reference_change_rmse=float(torch.sqrt(torch.mean((result[:,7:]-trajectory[:,7:])**2,dim=-1)).max()),peak_root_reference_change_m=float(torch.linalg.vector_norm(result[:,:3]-trajectory[:,:3],dim=-1).max()),full_tracking_passed=False)
 if worst_position>.003 or worst_orientation>.01:raise ValueError('Landing offset IK failed preflight: '+str(report))
 return result,report
