from pathlib import Path
import numpy as np,json,torch,sys
from full_body_preview import integrate
from mujoco_reference_dynamics import ReferenceDynamics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from scipy.spatial.transform import Rotation
p=Path(__file__).parent;name=sys.argv[1];d=np.load(p/(name+'.full_body_preview.first_plan.npz'));saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);q=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);curve=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20);plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]];rows=[]
for k in range(1,len(d['controls'])):
 q,v,a=[x.numpy().ravel() for x in curve.kinematics(float(d['frames'][k]),float(d['rate']))];q=integrate(q,d['states'][k,:35]);v+=d['states'][k,35:];control=d['controls'][k];state=model.evaluate(q,v,control[:35]);res=state['generalized_force'].copy();res[6:]-=control[47:]
 for side in [0,1]:res-=state['jacobian'][side].T@control[35+6*side:41+6*side]
 clear=[]
 for side in [0,1]:
  points,radii=geometry[side];foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();clear.append(float(foot[2]+np.min((points@R.T)[:,2]-radii)))
 rows.append(dict(frame=float(d['frames'][k]),base_residual=res[:6].tolist(),peak_joint_residual=float(abs(res[6:]).max()),native_clearance=clear,effort_ratio=float((abs(control[47:])/saved['effort_limits'].ravel()).max())))
report=dict(run=name,first_plan_time=float(d['time']),stages=rows,max_base_force_residual=float(max(np.linalg.norm(r['base_residual'][:3]) for r in rows)),max_base_moment_residual=float(max(np.linalg.norm(r['base_residual'][3:]) for r in rows)),max_joint_residual=float(max(r['peak_joint_residual'] for r in rows)),minimum_native_clearance=min(min(r['native_clearance']) for r in rows));(p/(name+'.preview_nonlinear_audit.json')).write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='stages'})
