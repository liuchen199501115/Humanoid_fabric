"""Re-solve a saved physical first-stage QP with a direct root-position task."""
import json
from pathlib import Path
import numpy as np
import daqp,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from smooth_clock import smooth_clock
root=Path(__file__).parent
d=np.load(root/'rolling_contact_damping8.qp_2150.npz')
p={k:d['priority_level_0_'+k][0].astype(float) for k in ['H','h','G','b','lb','ub','A_eq','b_eq']}
n=len(p['h']); q=d['q'][0]; v=d['qd'][0]
reference=np.load(root/'rolling_contact_damping8.control_reference.npz')
trajectory=torch.tensor(reference['control_q'],dtype=torch.float64)
curve=BlendedSplineReference(MotionInterpolator(trajectory),trajectory,stride=int(reference['smooth_stride']),start_frame=float(reference['smooth_start'])*200,ramp_frames=float(reference['smooth_ramp'])*200)
phase,rate,acc=smooth_clock(float(d['time']),.9,.35,1.2)
rq,rv,ra=[value.numpy()[0] for value in curve.kinematics(float(d['motion_frame_position']),rate*200,acc*200)]
desired=ra[:3]+np.clip(100*(rq[:3]-q[:3])+20*(rv[:3]-v[:3]),-30,30)
A=np.zeros((3,n));A[:,:3]=np.eye(3)
E,e=p['A_eq'],p['b_eq'];matrix=np.ascontiguousarray(np.vstack((E,p['G'])));lo=np.r_[p['lb'],e,np.full(len(p['b']),-np.inf)];hi=np.r_[p['ub'],e,p['b']]
sense=np.zeros(len(hi),dtype=np.int32);sense[:n][p['lb']==p['ub']]=5;sense[n:n+len(e)]=5
rows=[]
for weight in [0,200,1000,4000]:
 for pelvis in [0,.3,1.]:
  H=p['H'].copy();h=p['h'].copy();H+=weight*A.T@A;h-=weight*A.T@desired
  B=d['task_source_pelvis_orientation_A'][0].astype(float);W=d['task_source_pelvis_orientation_metric'][0].astype(float)*pelvis
  target=d['task_source_pelvis_orientation_desired'][0]-d['task_source_pelvis_orientation_bias'][0]
  H[:35,:35]+=B.T@W@B;h[:35]-=B.T@W@target
  scale=max(abs(H).max(),abs(h).max(),1.)
  x,obj,flag,info=daqp.solve(np.ascontiguousarray((H+H.T)/2/scale),np.ascontiguousarray(h/scale),matrix,hi,lo,sense,primal_tol=1e-8,dual_tol=1e-10,eps_prox=1e-8,eta_prox=1e-10,iter_limit=10000)
  violation=max(np.max(p['G']@x-p['b']),np.max(abs(E@x-e)),np.max(p['lb']-x),np.max(x-p['ub']),0.)
  row=dict(weight=weight,pelvis=pelvis,status=int(flag),violation=float(violation),root_acc=x[:6].tolist(),root_target=desired.tolist())
  for name in ['center_of_mass','feet_pose','joint_posture']:
   error=d['task_'+name+'_A'][0]@x[:35]+d['task_'+name+'_bias'][0]-d['task_'+name+'_desired'][0]
   row[name+'_error_norm']=float(np.linalg.norm(error*d['task_'+name+'_active'][0]))
  rows.append(row)
(root/'root_task_audit.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r)
