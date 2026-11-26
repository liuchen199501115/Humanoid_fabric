"""Independent directional check of cached world-left root RNEA variations."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_centroidal_geometry import CentroidalGeometry
from numpy_leg_kinematics import LegKinematics
from root_rotation_variation import skew
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from smooth_clock import smooth_clock
from g1_armature import unitree_armatures
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf');model=CentroidalGeometry(urdf,names);legs=LegKinematics(urdf,names)
armature=np.asarray(unitree_armatures(names)).ravel()
custom=len(sys.argv)>1
parent=OUT/((sys.argv[1] if custom else 'prepared_combined_root_y50')+'.npz');q0=torch.tensor(np.load(parent)['control_q']);sampler=BlendedSplineReference(MotionInterpolator(q0),q0,stride=6,start_frame=130,ramp_frames=20)
rot=np.load(OUT/(parent.stem+'.fullroot_linearization.npz' if custom else 'fullroot_rotation_linearization.npz'));assert rot['parent_sha256'].item()==hashlib.sha256(parent.read_bytes()).hexdigest()
columns=slice(3,6) if custom else slice(None)
cache=np.load(OUT/'reference_com_cache.npz')
def output(q,v,a,contact):
 c=model.evaluate(q)[0];tau=model.inverse_dynamics(q,v,a);F=tau[:3].copy();M=tau[3:6].copy();H=M-np.cross(c-q[:3],F)
 for side in range(2):
  if contact[side]:
   foot,_,J=legs.evaluate(q,side);tau[6+6*side:12+6*side]-=J.T@np.r_[F,M-np.cross(foot-q[:3],F)]/max(1,int(np.sum(contact)))
 return np.r_[F,H,tau[6:]+armature*a[6:]]
d=np.array([.03,-.04,.02]);dv=np.array([.2,.1,-.3]);da=np.array([1.,-.5,.2]);eps=1e-3;dt=1e-4;rows=[]
for frame in [730,790,830,900]:
 index=(frame-648)//2;t=cache['time'][frame];rate,rate_a=smooth_clock(t,.9,.35,1.2,2.3,.25,.8,3.,.2,1.2)[1:]
 q,v,a=[x.numpy()[0].astype(float) for x in sampler.kinematics(float(frame),200*rate,200*rate_a)]
 def omega(t,e):
  r=e*(d+t*dv+.5*t*t*da);rd=e*(dv+t*da);angle=np.linalg.norm(r);S=skew(r)
  J=np.eye(3)+(1-np.cos(angle))/angle**2*S+(angle-np.sin(angle))/angle**3*S@S
  return J@rd+Rotation.from_rotvec(r).apply(v[3:6]+t*a[3:6])
 def value(e):
  qq=q.copy();qq[3:7]=(Rotation.from_rotvec(e*d)*Rotation.from_quat(q[3:7])).as_quat()
  vv=v.copy();vv[3:6]=omega(0,e)
  aa=a.copy();aa[3:6]=(omega(dt,e)-omega(-dt,e))/(2*dt)
  return output(qq,vv,aa,cache['contacts'][frame])
 predicted=rot['Lq'][index,:,columns]@d+rot['Lv'][index,:,columns]@(dv-np.cross(v[3:6],d))+rot['La'][index,:,columns]@(da-np.cross(v[3:6],dv)-np.cross(a[3:6],d))
 observed=(value(eps)-value(-eps))/(2*eps)
 error=float(np.max(abs(predicted-observed)));assert error<.002,(frame,error)
 rows.append(dict(source_frame=frame,max_directional_generalized_force_error=error))
report=dict(scope=__doc__,passed=True,cases=rows)
(OUT/(parent.stem+'.fullroot_linearization_verification.json' if custom else 'fullroot_linearization_verification.json')).write_text(json.dumps(report,indent=2)+'\n');print(report)
