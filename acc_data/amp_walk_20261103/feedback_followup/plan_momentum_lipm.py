"""Offline bounded COM planning with source footsteps and unchanged audit poses."""
from pathlib import Path
import json,pickle
import numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
import osqp
from numpy_g1_com import CenterOfMassModel
from numpy_leg_kinematics import LegKinematics
p=Path(__file__).parent;baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];base=np.load(p/'early_flat_lower15.npz')['control_q'];history=np.load(p/'fixed_five_recovery.npz');names=history['final_joint_order'];urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
com_model=CenterOfMassModel(urdf,names);leg=LegKinematics(urdf,names)
with open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb') as f:contacts=pickle.load(f)['contact_sequence']
com=np.array([com_model.evaluate(q) for q in source]);com_curve=CubicSpline(np.arange(len(source))/200,com)
tensor=torch.tensor(base);sampler=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20)
prepared=[];momenta=[]
for f in range(len(base)):
 q,v,_=sampler.kinematics(float(f),200.);prepared.append(com_model.evaluate(q.numpy()[0]));momenta.append(com_model.momentum(q.numpy()[0],v.numpy()[0]))
height_curve=CubicSpline(np.arange(len(base))/200,np.array(prepared)[:,2]);momentum_curve=CubicSpline(np.arange(len(base))/200,np.array(momenta))
frames=np.unique(np.r_[np.arange(180,2000,4),1999]);N=len(frames)-1;dt=np.diff(frames)/200
ref=com_curve(frames/200);velocity=com_curve(frames/200,1)
i=np.argmin(abs(history['history_frame']-frames[0]));initial=np.r_[history['history_com'][i,0,:2],history['history_com_velocity'][i,0,:2]]
for radius in [.10,.14,.18]:
 nx=4*(N+1);n=nx+2*N;rows=[];lo=[];hi=[]
 def add(indices,values,lower,upper):
  rows.append((indices,values));lo.append(lower);hi.append(upper)
 for j in range(4):add([j],[1.],initial[j],initial[j])
 for k in range(N):
  height=float(height_curve(frames[k]/200));vertical_acc=float(np.clip(height_curve(frames[k]/200,2),-6.,10.));hd=momentum_curve(frames[k]/200,1);forcing=np.array([-hd[1],hd[0]])/(com_model.mass*height)
  omega=np.sqrt((9.81+vertical_acc)/height);ch=np.cosh(omega*dt[k]);sh=np.sinh(omega*dt[k]);A=np.array([[ch,sh/omega],[omega*sh,ch]]);B=np.array([1-ch,-omega*sh])
  for axis in range(2):
   for j in range(2):
    add([4*(k+1)+2*j+axis,4*k+axis,4*k+2+axis,nx+2*k+axis],[1.,-A[j,0],-A[j,1],-B[j]],-B[j]/omega**2*forcing[axis],-B[j]/omega**2*forcing[axis])
  f=int(frames[k]);active=np.asarray(contacts[f]).astype(bool);vertices=[]
  for side in range(2):
   if not active[side]:continue
   pos,R,_=leg.evaluate(source[f],side);yaw=np.arctan2(R[1,0],R[0,0]);rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
   for x in [-.025,.085]:
    for y in [-.02,.02]:vertices.append(pos[:2]+rotation@np.array([x,y]))
  if len(vertices)<3:raise ValueError('Flight phase is outside this walking planner')
  hull=ConvexHull(np.array(vertices))
  for a,b,c in hull.equations:add([nx+2*k,nx+2*k+1],[a,b],-np.inf,-c)
 for k in range(N+1):
  # An inscribed octagonal bound on horizontal COM displacement.
  for theta in np.arange(8)*np.pi/4:
   normal=np.array([np.cos(theta),np.sin(theta)]);add([4*k,4*k+1],normal,-np.inf,float(normal@ref[k,:2]+radius*np.cos(np.pi/8)))
 target=np.zeros(n);weights=np.ones(n)
 for k in range(N+1):target[4*k:4*k+4]=np.r_[ref[k,:2],velocity[k,:2]];weights[4*k:4*k+4]=[4000,4000,80,80]
 weights[4*N:4*(N+1)]*=4
 ii=[];jj=[];vv=[]
 for r,(indices,values) in enumerate(rows):
  ii.extend([r]*len(indices));jj.extend(indices);vv.extend(values)
 matrix=sparse.csc_matrix((vv,(ii,jj)),shape=(len(rows),n));shift=matrix@target
 solver=osqp.OSQP();solver.setup(P=sparse.diags(2*weights,format='csc'),q=np.zeros(n),A=matrix,l=np.array(lo)-shift,u=np.array(hi)-shift,eps_abs=1e-6,eps_rel=1e-6,max_iter=50000,polish=True,verbose=False)
 answer=solver.solve();report=dict(radius=radius,status=answer.info.status,initial_state=initial.tolist(),source_frames=[int(frames[0]),int(frames[-1])],physical_tracking_certified=False)
 if answer.x is not None and answer.info.status_val in (1,2):
  result=answer.x+target;state=result[:nx].reshape(-1,4);u=result[nx:].reshape(-1,2);residual=matrix@result;violation=float(max(np.max(np.array(lo)-residual),np.max(residual-np.array(hi)),0.));report.update(constraint_violation=violation,peak_com_offset=float(np.linalg.norm(state[:,:2]-ref[:,:2],axis=1).max()),preflight_passed=violation<1e-4)
  np.savez_compressed(p/f'momentum_lipm_plan{int(radius*100)}.npz',frames=frames,state=state,zmp=u,source_com=ref,source_com_velocity=velocity)
 (p/f'momentum_lipm_plan{int(radius*100)}.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
