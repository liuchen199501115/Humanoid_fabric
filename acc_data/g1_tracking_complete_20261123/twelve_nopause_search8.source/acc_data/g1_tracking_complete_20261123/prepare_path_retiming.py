"""Offline whole-turn path retiming, never a runtime predictive controller.

For phase speed squared x and phase acceleration u, exact RNEA along a fixed
pose path is G + X*x + U*u. Explicit slacks report physical path conflicts;
a solved optimization does not certify closed-loop tracking or impact behavior.
"""
import hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import osqp
import torch
from scipy import sparse
from scipy.spatial import ConvexHull
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_centroidal_geometry import CentroidalGeometry
from batched_reference_dynamics import reference_outputs
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from smooth_clock import smooth_clock
from g1_armature import unitree_armatures
label,parent_name=sys.argv[1:3];parent=OUT/(parent_name+'.npz');target=OUT/('prepared_path_clock_'+label+'.npz')
assert not target.exists()
base=np.load(parent);reference=torch.tensor(base['control_q']);sampler=BlendedSplineReference(MotionInterpolator(reference),reference,stride=6,start_frame=130,ramp_frames=20)
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();urdf=ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
model=CentroidalGeometry(str(urdf),names);armature=np.asarray(unitree_armatures(names)).ravel();cache=np.load(OUT/'reference_com_cache.npz')
first_frame=int(sys.argv[3]) if len(sys.argv)>3 else 660
if first_frame<660 or first_frame>850 or first_frame%2:
 raise ValueError('Retiming must start on an even frame between 660 and 850')
frames=np.arange(first_frame,1033,2);n=len(frames);ds=.01;contacts=cache['contacts'][frames]
q,v,a=[np.concatenate([sampler.kinematics(float(k),200.,0.)[j].numpy() for k in frames]).astype(float) for j in range(3)]
zero=np.zeros_like(v);G=reference_outputs(model,q,zero,zero,contacts,armature)
X=reference_outputs(model,q,v,a,contacts,armature)-G
U=reference_outputs(model,q,zero,v,contacts,armature)-G
# An independent nonlinear evaluation checks the phase-chain-rule model.
x_probe=np.linspace(.4,1.8,n);u_probe=np.sin(np.arange(n))
exact=reference_outputs(model,q,v*np.sqrt(x_probe[:,None]),a*x_probe[:,None]+v*u_probe[:,None],contacts,armature)
error=float(np.max(abs(exact-(G+X*x_probe[:,None]+U*u_probe[:,None]))));assert error<1e-8,error
print('phase RNEA scaling error',error,flush=True)
root=ET.parse(urdf).getroot();limits={j.get('name'):[float(j.find('limit').get(key)) for key in ('effort','velocity')] for j in root.findall('joint') if j.find('limit') is not None}
effort,velocity_limit=np.array([limits[name] for name in names]).T
base_clock=lambda t:smooth_clock(t,.9,.35,1.2,2.3,.25,.8,3.,.2,1.2)
old_time=np.array([brentq(lambda t:base_clock(t)[0]-k/200.,0.,10.,xtol=1e-14) for k in frames])
old=np.array([base_clock(t)[1:] for t in old_time])
D=sparse.diags([-np.ones(n-1)/(2*ds),np.ones(n-1)/(2*ds)],[0,1],shape=(n-1,n),format='csr')
Du=sparse.vstack([D,sparse.csr_matrix((1,n))],format='csr')
rows=[];rhs=[];kinds=[];sample=[];baseline_violation=[];mg=model.mass*9.81
corners=np.array([[.12,.025,-.035],[.12,-.025,-.035],[-.05,.025,-.035],[-.05,-.025,-.035]])
def constraint(i,coeff,bound,kind,normalizer):
 row=sparse.csr_matrix(([float(coeff@X[i])/normalizer],([0],[i])),shape=(1,n))+(float(coeff@U[i])/normalizer)*Du[i]
 rows.append(row);rhs.append((bound-float(coeff@G[i]))/normalizer);kinds.append(kind);sample.append(int(frames[i]))
 baseline_violation.append((float(coeff@(G[i]+X[i]*old[i,0]**2+U[i]*old[i,1]))-bound)/normalizer)
for i,k in enumerate(frames):
 c,_,poses=model.evaluate(q[i]);active=contacts[i]
 if active.any():
  points=np.concatenate([(poses[side+'_ankle_roll_link'][1]@corners.T).T+poses[side+'_ankle_roll_link'][0] for side,on in zip(('left','right'),active) if on])[:,:2]
  for nx,ny,b in ConvexHull(points).equations:
   coeff=np.zeros(35);coeff[:6]=[-nx*c[2],-ny*c[2],nx*c[0]+ny*c[1]+b,ny,-nx,0.]
   constraint(i,coeff,0.,'cop',mg*.05)
  for angle in np.arange(8)*np.pi/4:
   coeff=np.zeros(35);coeff[:3]=[np.cos(angle),np.sin(angle),-.55*np.cos(np.pi/8)]
   constraint(i,coeff,0.,'friction',mg)
  coeff=np.zeros(35);coeff[2]=-1.;constraint(i,coeff,-50.,'normal',mg)
 else:
  for axis in range(6):
   for sign in [-1.,1.]:
    coeff=np.zeros(35);coeff[axis]=sign;constraint(i,coeff,0.,'flight_wrench',mg if axis<3 else mg*.05)
 for j in range(29):
  if j<12 and active.sum()!=1:continue
  for sign in [-1.,1.]:
   coeff=np.zeros(35);coeff[6+j]=sign
   budget=effort[j]*(.8 if k<810 else 1.)
   constraint(i,coeff,budget,'motor',budget)
A=sparse.vstack(rows,format='csc');b=np.asarray(rhs);ns=len(b)
# Smooth acceleration and keep speed near the existing clock unless physical
# slack reduction warrants a change. All physical slacks remain explicit.
accel_smooth=sparse.diags([-np.ones(n-2),np.ones(n-2)],[0,1],shape=(n-2,n-1))@D
P0=.1*sparse.eye(n)+.001*(D.T@D)+.01*(accel_smooth.T@accel_smooth)
P=sparse.block_diag([P0,1e4*sparse.eye(ns)],format='csc');linear=np.r_[-.1*old[:,0]**2,np.zeros(ns)]
mat=sparse.vstack([sparse.eye(n+ns),sparse.hstack([A,-sparse.eye(ns)]),sparse.hstack([D,sparse.csc_matrix((n-1,ns))])],format='csc')
lower=np.full(n,.4**2);upper=np.minimum(1.6**2,np.min((velocity_limit[None]/np.maximum(abs(v[:,6:]),1e-8))**2,axis=1))
lower[0]=upper[0]=old[0,0]**2;lower[1]=upper[1]=lower[0]+2*ds*old[0,1];lower[-2:]=upper[-2:]=1.2**2
assert np.all(lower<=upper)
max_iterations=int(sys.argv[4]) if len(sys.argv)>4 else 100000
solver=osqp.OSQP();solver.setup(P=P,q=linear,A=mat,l=np.r_[lower,np.zeros(ns),np.full(ns,-np.inf),np.full(n-1,-6.)],u=np.r_[upper,np.full(ns,np.inf),b,np.full(n-1,6.)],eps_abs=1e-5,eps_rel=1e-5,rho=10.,max_iter=max_iterations,polish=True,verbose=True)
if len(sys.argv)>5:
 warm=np.load(sys.argv[5]);solver.warm_start(x=warm['x'],y=warm['y'])
fit=solver.solve();print('solver',fit.info.status,flush=True)
np.savez_compressed(target.with_suffix('.solver_iterate.npz'),x=fit.x,y=fit.y)
report=dict(method=__doc__,parent_file=parent.name,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),solver_status=fit.info.status,linearization_error=error,full_tracking_passed=False,preflight_passed=False,physical_scope='Optimistic full nominal sole support; leg effort only in single support; upper-body effort in all modes; no impact model')
if fit.info.status_val in (1,2):
 x=fit.x[:n];speed=np.sqrt(np.maximum(x,1e-12));phase_acceleration=Du@x
 t=np.r_[old_time[0],old_time[0]+np.cumsum(2*ds/(speed[:-1]+speed[1:]))]
 violation=A@x-b;old_violation=np.array(baseline_violation)
 report.update(speed_range=[float(speed.min()),float(speed.max())],maximum_phase_acceleration=float(abs(phase_acceleration).max()),time_change_s=float(t[-1]-old_time[-1]),
  maximum_physical_violation_by_kind={k:float(np.max(violation[np.array(kinds)==k])) for k in set(kinds)},baseline_violation_by_kind={k:float(np.max(old_violation[np.array(kinds)==k])) for k in set(kinds)},
  max_slack=float(fit.x[n:].max()),preflight_passed=bool(np.all(x>0) and np.all(np.diff(t)>0) and abs(phase_acceleration).max()<6.001))
 if report['preflight_passed']:
  np.savez_compressed(target,frame=frames,time=t,speed=speed,phase_acceleration=phase_acceleration,parent_sha256=report['parent_sha256'],source_sha256=report['source_sha256'],physical_slack=fit.x[n:])
  report['samples']=[dict(frame=int(k),time_s=float(tt),speed=float(vv),phase_acceleration=float(aa)) for k,tt,vv,aa in zip(frames,t,speed,phase_acceleration)]
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='samples'},flush=True)
