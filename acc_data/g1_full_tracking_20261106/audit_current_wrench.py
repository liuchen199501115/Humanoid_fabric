"""Necessary reference ground-wrench audit with a deliberately generous support hull."""
from pathlib import Path
import json,pickle
import numpy as np,torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from mujoco_reference_dynamics import ReferenceDynamics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;d=np.load(p/'rolling_contact.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order']);contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence'])
reports=[]
for name,path in [('parent','acc_data/amp_walk_20261103/feedback_followup/wbc_convex_catch15_25_clear35.npz'),('force_fit150',p/'current_force_fit150.npz')]:
 q=torch.tensor(np.load(path)['control_q'],dtype=torch.float64);curve=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20);rows=[]
 for f in np.arange(740,1110.01,.5):
  pose,v,a=[x.numpy()[0] for x in curve.kinematics(float(f),240.)];state=model.evaluate(pose,v,a);F=state['generalized_force'][:3];M=state['generalized_force'][3:6]+np.cross(pose[:3],F);vertices=[]
  for foot in np.flatnonzero(contacts[int(f)]):
   fp=state['feet'][foot];R=Rotation.from_quat(fp[3:]).as_matrix();vertices.extend((fp[:3]+np.array([[x,y,-.035] for x in [-.025,.085] for y in [-.02,.02]])@R.T)[:,:2])
  cop=np.array([-M[1],M[0]])/F[2] if F[2]>0 else np.full(2,np.nan)
  violation=None
  if len(vertices)>=3 and F[2]>0:
   h=ConvexHull(vertices).equations;violation=float(np.maximum(0,h[:,:2]@cop+h[:,2]).max())
  rows.append(dict(frame=float(f),force=F.tolist(),moment_origin=M.tolist(),cop=cop.tolist(),cop_violation=violation))
 valid=[r for r in rows if r['cop_violation'] is not None]
 report=dict(candidate=name,scope='Exact reference q/v/a; root generalized wrench shifted to ground origin. Hull uses complete projected soles from source contact labels, so satisfying it does not certify observed point/line contacts, friction, joint torque or stance velocity.',negative_normal_samples=sum(r['force'][2]<0 for r in rows),maximum_cop_violation_m=max(r['cop_violation'] for r in valid),worst_rows=sorted(valid,key=lambda r:r['cop_violation'],reverse=True)[:8],ranges=[])
 for lo,hi in [(740,880),(880,955),(955,1073)]:
  subset=[r for r in valid if lo<=r['frame']<=hi];report['ranges'].append(dict(frames=[lo,hi],maximum_cop_violation_m=max(r['cop_violation'] for r in subset),samples_outside_2mm=sum(r['cop_violation']>.002 for r in subset)))
 reports.append(report);print(json.dumps(report),flush=True)
(p/'current_wrench_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
