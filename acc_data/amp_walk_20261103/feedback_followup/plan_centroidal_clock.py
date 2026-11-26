"""Plan source clock energy against centroidal wrench feasibility along a fixed q path."""
from pathlib import Path
import json,pickle
import numpy as np,torch
from scipy.interpolate import CubicSpline
from scipy.spatial import ConvexHull
from scipy import sparse
import osqp
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from numpy_g1_com import CenterOfMassModel
from numpy_leg_kinematics import LegKinematics
p=Path(__file__).parent;raw=np.load(p/'seventh_clearance10.npz')['control_q'];names=np.load(p/'wrench_fixed_ten.npz')['final_joint_order'];urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';model=CenterOfMassModel(urdf,names);leg=LegKinematics(urdf,names)
with open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb') as f:contacts=pickle.load(f)['contact_sequence']
tensor=torch.tensor(raw);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);frames=np.unique(np.r_[np.arange(450,2000,2),1999]);phase=frames/200.;data=[];poses=[]
for f in frames:
 q,v,_=curve.kinematics(float(f),200.);poses.append(q.numpy()[0]);data.append(model.momentum(q.numpy()[0],v.numpy()[0],return_state=True))
data=np.array(data);com,cv,H=data[:,0],data[:,1],data[:,2];ca=CubicSpline(phase,cv)(phase,1);Hd=CubicSpline(phase,H)(phase,1);N=len(frames);ds=np.diff(phase)
for slack_weight in [100.,10000.]:
 n=N+N-1;rows=[];lo=[];hi=[]
 def add(indices,values,l,u):rows.append((indices,values));lo.append(l);hi.append(u)
 for k in range(N):add([k],[1.],1.44 if k==0 else .8**2,1.44 if k==0 else 1.5**2)
 for k in range(N-1):
  add([k,k+1],[-1/(2*ds[k]),1/(2*ds[k])],-2.,2.);add([N+k],[1.],0.,np.inf)
  # F = Fc*x + Fv*u + Fg; dH = Hc*x + Hv*u.
  fc=model.mass*ca[k];fv=model.mass*cv[k];fg=np.array([0,0,model.mass*9.81]);hc=Hd[k];hv=H[k]
  def affine(coeff_x,coeff_u,constant,upper,soft=False):
   indices=[k,k+1];values=[coeff_x-coeff_u/(2*ds[k]),coeff_u/(2*ds[k])]
   if soft:indices.append(N+k);values.append(-1.)
   add(indices,values,-np.inf,upper-constant)
  active=np.asarray(contacts[frames[k]],bool);vertices=[]
  for side in range(2):
   if not active[side]:continue
   pos,R,_=leg.evaluate(poses[k],side);yaw=np.arctan2(R[1,0],R[0,0]);rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
   for x in [-.025,.085]:
    for y in [-.02,.02]:vertices.append(pos[:2]+rotation@np.array([x,y]))
  hull=ConvexHull(np.array(vertices))
  for nx,ny,b in hull.equations:
   w=np.array([-com[k,2]*nx,-com[k,2]*ny,nx*com[k,0]+ny*com[k,1]+b]);hh=np.array([ny,-nx,0.]);affine(w@fc+hh@hc,w@fv+hh@hv,w@fg,0.,True)
  affine(-fc[2],-fv[2],-fg[2],-80.);affine(fc[2],fv[2],fg[2],800.*sum(active))
  for x in [-1,1]:
   for y in [-1,1]:
    w=np.array([x,y,-.6]);affine(w@fc,w@fv,w@fg,0.)
 ii=[];jj=[];vv=[]
 for i,(ix,va) in enumerate(rows):ii.extend([i]*len(ix));jj.extend(ix);vv.extend(va)
 A=sparse.csc_matrix((vv,(ii,jj)),shape=(len(rows),n));P=sparse.diags(np.r_[np.full(N,2.),np.full(N-1,2*slack_weight)],format='csc');q=np.r_[np.full(N,-2*1.44),np.zeros(N-1)];D=sparse.diags([-np.ones(N-1),np.ones(N-1)],[0,1],shape=(N-1,n),format='csc');P=P+100.*D.T@D
 solver=osqp.OSQP();solver.setup(P=sparse.triu(P).tocsc(),q=q,A=A,l=np.array(lo),u=np.array(hi),eps_abs=1e-6,eps_rel=1e-6,max_iter=100000,polish=True,verbose=False);result=solver.solve();report=dict(status=result.info.status,physical_tracking_certified=False,source_contact_polygons_are_planning_approximation=True)
 if result.x is not None and result.info.status_val in (1,2):
  z=result.x;x=z[:N];rate=np.sqrt(x);wall=np.r_[0.,np.cumsum(2*ds/(rate[:-1]+rate[1:]))];u=np.diff(rate)/np.diff(wall);report.update(minimum_speed=float(rate.min()),maximum_speed=float(rate.max()),maximum_moment_slack=float(z[N:].max()),mean_moment_slack=float(z[N:].mean()),duration=float(wall[-1]))
  np.savez_compressed(p/f'centroidal_clock{int(slack_weight)}.npz',frames=frames,source_phase=phase,wall_offset=wall,rate=rate,acceleration=np.r_[0.,.5*(u[:-1]+u[1:]),u[-1]],moment_slack=z[N:])
 (p/f'centroidal_clock{int(slack_weight)}.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
