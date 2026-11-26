from pathlib import Path
import numpy as np,json,pickle,torch
from mujoco_reference_dynamics import ReferenceDynamics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);limits=saved['effort_limits'].ravel();reports=[]
for name in ['turn_recovery_y40','late_centroidal_upper_fit_floor','turn_error_feedforward_lower_842_floor']:
 q=np.load(p/(name+'.npz'))['control_q'];t=torch.tensor(q);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);rows=[]
 for frame in np.arange(842,1014,.5):
  active=np.flatnonzero(contacts[int(frame)]); 
  if len(active)!=1:continue
  foot=active[0];q,v,a=[x.numpy().ravel() for x in curve.kinematics(float(frame),240.)];d=model.evaluate(q,v,a);J=d['jacobian'][foot];force=np.linalg.solve(J[:,:6].T,d['generalized_force'][:6]);tau=d['generalized_force'][6:]-J[:,6:].T@force;speed=J@v;ratio=abs(tau)/limits;rows.append(dict(frame=float(frame),maximum_effort_ratio=float(ratio.max()),joint=str(saved['final_joint_order'][ratio.argmax()]),torque=float(tau[ratio.argmax()]),force=force.tolist(),foot_velocity=speed.tolist()))
 report=dict(candidate=name,maximum_required_effort_ratio=max(r['maximum_effort_ratio'] for r in rows),largest_cases=sorted(rows,key=lambda r:r['maximum_effort_ratio'],reverse=True)[:8]);reports.append(report);print(report,flush=True)
(p/'full_reference_effort_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
