from pathlib import Path
import numpy as np,torch,json,pickle,time
from full_body_preview import FullBodyPreview
from mujoco_reference_dynamics import ReferenceDynamics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;saved=np.load(p/'wrench_fixed_ten.npz');model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',saved['final_joint_order']);base=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);reference=BlendedSplineReference(MotionInterpolator(base),base,stride=6,start_frame=130,ramp_frames=20);source=MotionInterpolator(torch.tensor(np.load(p/'fixed_smooth_early.control_reference.npz')['source_q']));contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence']);modes=contacts[:,0]+2*contacts[:,1];plan=json.load(open(p/'wrench_fixed_ten.steps_plan.json'));geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in [0,1]];history=np.load(p/'fixed_clock120_body10_overdue.npz');velocity=history['history_joint_velocity_limits'][0].ravel() if 'history_joint_velocity_limits' in history else np.full(29,30.);reports=[]


from scipy.spatial.transform import Rotation
from full_body_preview import integrate,skew
for file in [p/'fixed_clock120_body10_full_body_preview42_guarded24.full_body_preview.first_plan.npz',p/'full_body_preview_native_patch_3.700.npz']:
 if not file.exists():continue
 d=np.load(file,allow_pickle=True);rate=float(d['rate']) if 'rate' in d else 240.;worst=dict(acceleration=0.)
 for k in range(1,len(d['controls'])):
  rq,rv,_=[x.numpy().ravel() for x in reference.kinematics(float(d['frames'][k]),rate)];q=integrate(rq,d['states'][k,:35]);v=rv+d['states'][k,35:];state=model.evaluate(q,v,kinematic_bias=True);control=d['controls'][k]
  for side in [0,1]:
   if control[37+6*side]<20:continue
   foot=state['feet'][side];R=Rotation.from_quat(foot[3:]).as_matrix();points,radii=geometry[side];heights=(points@R.T)[:,2]-radii;near=np.flatnonzero(heights<=heights.min()+.0015);J=state['jacobian'][side];omega=J[3:]@v
   for point in near:
    offset=R@points[point]-np.array([0.,0.,radii[point]]);bias=state['foot_bias'][side,:3]-skew(offset)@state['foot_bias'][side,3:]+np.cross(omega,np.cross(omega,offset));value=float(np.linalg.norm((J[:3]-skew(offset)@J[3:])@control[:35]+bias))
    if value>worst['acceleration']:worst=dict(acceleration=value,stage=k,frame=float(d['frames'][k]),foot=side,point=int(point))
 reports.append(dict(file=file.name,worst=worst));print(reports[-1])
(p/'full_body_preview_contact_acceleration_audit.json').write_text(json.dumps(reports,indent=2)+'\n')
