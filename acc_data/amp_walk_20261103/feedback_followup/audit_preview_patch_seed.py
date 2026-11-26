from pathlib import Path
import numpy as np,torch,json,pickle,time
from full_body_preview import FullBodyPreview
from mujoco_reference_dynamics import ReferenceDynamics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);base=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);reference=BlendedSplineReference(MotionInterpolator(base),base,stride=6,start_frame=130,ramp_frames=20);source=MotionInterpolator(torch.tensor(np.load(p/'fixed_smooth_early.control_reference.npz')['source_q']));contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);modes=contacts[:,0]+2*contacts[:,1];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]];history=np.load(p/'fixed_clock120_body10_overdue.npz');velocity=history['history_joint_velocity_limits'][0].ravel() if 'history_joint_velocity_limits' in history else np.full(29,30.);reports=[]

from full_body_preview import difference,integrate
from scipy.spatial.transform import Rotation
d=np.load(p/'fixed_clock120_body10_preview_snapshots.snapshot_3.700.npz');q=d['q'][0].astype(float);v=d['v'][0].astype(float);rq=reference.kinematics(845.,240.)[0].numpy().ravel();err=difference(q,rq)
for k in range(5):
 frame=845.+6*k;r=reference.kinematics(frame,240.)[0].numpy().ravel();nq=q if k==0 else integrate(r,err*np.exp(-k*.025/.3));state=model.evaluate(nq,v)
 for side in [0,1]:
  foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=geometry[side];z=foot[2]+(points@R.T)[:,2]-radii;print(k,side,np.flatnonzero(z<=z.min()+.003),np.round(np.sort(z)[:4],5))
