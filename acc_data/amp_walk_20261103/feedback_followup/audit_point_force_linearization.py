from pathlib import Path
import json,numpy as np
from scipy.spatial.transform import Rotation
from mujoco_reference_dynamics import ReferenceDynamics
from generalized_dynamics_linearization import perturb
from centroidal_pose_linearization import skew
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);geometry=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));rng=np.random.default_rng(321);reports=[]
for t in [3.5,3.7,3.8]:
 d=np.load(p/f'fixed_clock120_body10_preview_snapshots.snapshot_{t:.3f}.npz');q=d['q'][0].astype(float);v=d['v'][0].astype(float);state=model.evaluate(q,v)
 for side in [0,1]:
  points=np.asarray(geometry['collision_points'][side]);radii=np.asarray(geometry['collision_radii'][side]);R=Rotation.from_quat(state['feet'][side,3:]).as_matrix();heights=(points@R.T)[:,2]-radii;ids=np.flatnonzero(heights<=heights.min()+.0005);points=points[ids];radii=radii[ids];forces=np.tile([8.,-5.,100.],(len(ids),1));dq=rng.normal(size=35)*1e-5;df=rng.normal(size=forces.shape)*1e-4;Jw=state['jacobian'][side,3:]
  def wrench(rotation,f):
   offsets=points@rotation.T-radii[:,None]*np.array([0.,0.,1.]);return np.r_[f.sum(0),np.cross(offsets,f).sum(0)]
  derivative=np.zeros((6,35));B=np.zeros((6,3*len(ids)))
  for i,point in enumerate(points):
   derivative[3:]+=skew(forces[i])@skew(R@point)@Jw;offset=R@point-np.array([0.,0.,radii[i]]);B[:3,3*i:3*i+3]=np.eye(3);B[3:,3*i:3*i+3]=skew(offset)
  moved=model.evaluate(perturb(q,dq),v);moved_R=Rotation.from_quat(moved['feet'][side,3:]).as_matrix();actual=wrench(moved_R,forces+df)-wrench(R,forces);predicted=derivative@dq+B@df.ravel();relative=float(np.linalg.norm(actual-predicted)/max(np.linalg.norm(actual),1e-12))
  yaw=Rotation.from_matrix(R).as_euler('xyz')[2];Y=Rotation.from_euler('z',yaw).as_matrix();moved_yaw=Rotation.from_matrix(moved_R).as_euler('xyz')[2];moved_Y=Rotation.from_euler('z',moved_yaw).as_matrix();column=R[:,0];yaw_gradient=np.array([-column[0]*column[2],-column[1]*column[2],column[0]**2+column[1]**2])/max(column[0]**2+column[1]**2,1e-8);cone=np.array([[sx,sy,-.5] for sx in [-1.,1.] for sy in [-1.,1.]]);friction_errors=[]
  for force,delta_force in zip(forces,df):
   actual_friction=cone@(moved_Y.T@(force+delta_force)-Y.T@force);predicted_friction=cone@Y.T@delta_force+(cone@(-skew(np.array([0.,0.,1.])))@Y.T@force)*(yaw_gradient@Jw@dq);friction_errors.append(float(np.linalg.norm(actual_friction-predicted_friction)/max(np.linalg.norm(actual_friction),1e-12)))
  row=dict(time=t,foot=side,force_points=len(ids),relative_error=relative,friction_relative_error=max(friction_errors),maximum_absolute_error=float(abs(actual-predicted).max()));reports.append(row);print(row)
assert max(max(r['relative_error'],r['friction_relative_error']) for r in reports)<1e-3
(p/'point_force_linearization_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
