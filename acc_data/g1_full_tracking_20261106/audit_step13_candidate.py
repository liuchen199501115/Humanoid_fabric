"""Independent dense kinematic and HiGHS wrench/effort feasibility audit.
No simulator stepping, no physical success claim, and no candidate promotion.
"""
from pathlib import Path
import argparse,json,pickle,hashlib
import numpy as np,torch
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation
from blended_spline import BlendedSplineReference
from mujoco_reference_dynamics import ReferenceDynamics
from g1_model_effort import model_effort_limits
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
parser=argparse.ArgumentParser();parser.add_argument('candidate');parser.add_argument('--name',required=True);parser.add_argument('--start-frame',type=float,default=960.);parser.add_argument('--end-frame',type=float,default=1035.);parser.add_argument('--frame-rate',type=float,default=240.);parser.add_argument('--spacing',type=float,default=.125);args=parser.parse_args()
if not (0 <= args.start_frame < args.end_frame < 2000 and .125 <= args.spacing <= 1 and 0 < args.frame_rate <= 400):raise ValueError('Invalid audit frame range, spacing or frame rate')
p=Path(__file__).parent
if (p/(args.name+'.candidate_audit.json')).exists():raise RuntimeError('Existing audit report')
candidate=np.load(args.candidate)
q=candidate['control_q'] if 'control_q' in candidate else candidate['rejected_or_unexecuted_control_q']
q=q.astype(np.float32).astype(np.float64)  # Match the controller's prepared-reference input.
record=np.load(p/'rolling_contact.npz');names=record['final_joint_order']
raw=pickle.load(open('retarget/amp_reference/amp_walk_1ms.pkl','rb'));source_order=json.loads(Path('acc_data/amp_walk_20261102/input_report.json').read_text())['source_joint_order']
raw_joints=raw['dof_pos'][:,[source_order.index(n) for n in names]]
model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',names)
effort=model_effort_limits('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',names,record['effort_limits'].reshape(-1))
contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence'])
tensor=torch.tensor(q,dtype=torch.float64);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20)
plan=json.loads((p/'rolling_contact.steps_plan.json').read_text());geometry=[(np.array(plan['collision_points'][j]),np.array(plan['collision_radii'][j])) for j in (0,1)]
rows=[];clearance=[];feet_samples=[];frames=np.arange(args.start_frame,args.end_frame+1e-5,args.spacing)
for f in frames:
 pose,v,a=[t.numpy()[0] for t in curve.kinematics(float(f),args.frame_rate)];state=model.evaluate(pose,v,a);R=Rotation.from_quat(state['feet'][:,3:]).as_matrix();active=contacts[int(f)];mode=3 if active.all() else 1 if active[0] else 2
 G,b=build_single_support_wrench_constraints(contact_mode=torch.tensor([mode]),foot_rotation_world=torch.tensor(R[None]),friction_coefficient=.6,coupled_yaw=True,toe_length=.085,heel_length=.025,half_width=.02,sole_depth=.035,minimum_normal_force=0.,maximum_normal_force=800.)
 G,b=G.numpy()[0],b.numpy()[0];selected=np.flatnonzero(active);cols=np.concatenate([np.arange(6*foot,6*foot+6) for foot in selected]);G=G[:,cols];used=np.any(abs(G)>0,axis=1);G,b=G[used],b[used];scale=np.linalg.norm(G*np.tile([350.,350.,350.,35.,35.,35.],len(selected)),axis=1);G=G/scale[:,None];b=b/scale
 J=np.concatenate(state['jacobian'][selected],axis=0);force=state['generalized_force'];nw=J.shape[0]
 # Directly optimize every active foot wrench, rather than reuse fit allocation.
 # Rigid-body base dynamics is an exact equality; joint torques are eliminated.
 C=np.vstack((G,J[:,6:].T/effort[:,None],-J[:,6:].T/effort[:,None]));rhs=np.r_[b,1+force[6:]/effort,1-force[6:]/effort]
 floors=[]
 for k,foot in enumerate(selected):
  row=np.zeros(nw);row[6*k:6*k+3]=-R[foot,:,2]/350.;floors.append(row)
 C=np.vstack((C,floors));rhs=np.r_[rhs,[-(20. if len(selected)==2 and foot==0 else 100.)/350. for foot in selected]]
 result=linprog(np.r_[np.zeros(nw),1.],A_ub=np.c_[C,-np.ones(len(C))],b_ub=rhs,A_eq=np.c_[J[:,:6].T,np.zeros(6)],b_eq=force[:6],bounds=[(None,None)]*nw+[(0,None)],method='highs')
 if not result.success:raise RuntimeError(result.message)
 w=result.x[:nw];tau=force[6:]-J[:,6:].T@w;eq=float(abs(J[:,:6].T@w-force[:6]).max());j=int(np.argmax(abs(tau)/effort))
 rows.append(dict(frame=float(f),minimum_normalized_slack=float(result.x[-1]),base_equality_error=eq,effort_fraction=float(abs(tau[j])/effort[j]),joint=str(names[j]),torque=float(tau[j])))
 feet_samples.append(state['feet']);heights=[]
 for foot in (0,1):
  pts,radius=geometry[foot];heights.append(float(state['feet'][foot,2]+min((pts@R[foot].T)[:,2]-radius)))
 clearance.append(heights)
clearance=np.array(clearance);raw_rmse=np.sqrt(np.mean((q[:,7:]-raw_joints)**2,axis=1))
report=dict(candidate=str(args.candidate),candidate_sha256=hashlib.sha256(Path(args.candidate).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path('retarget/amp_reference/amp_walk_1ms.pkl').read_bytes()).hexdigest(),reference_quantization='float32 as loaded by the controller',normal_floor_frame='local foot normal',audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),frame_rate=args.frame_rate,frame_spacing=args.spacing,scope='Independent HiGHS nominal rigid-body full-sole audit, original motor caps; excludes actual contact/impact and physical tracking.',frames=[float(frames[0]),float(frames[-1])],sample_count=len(frames),worst_sample=max(rows,key=lambda r:r['minimum_normalized_slack']),peak_base_equality_error=max(r['base_equality_error'] for r in rows),peak_raw_joint_rmse=float(raw_rmse.max()),minimum_sole_clearance=clearance.min(axis=0).tolist(),peak_left_swing_clearance=float(clearance[frames<1013,0].max()),physical_tracking_certified=False)
report['nominal_dynamics_passed']=bool(report['worst_sample']['minimum_normalized_slack']<1e-4 and report['peak_base_equality_error']<1e-7)
(p/(args.name+'.candidate_audit.json')).write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(p/(args.name+'.candidate_audit.npz'),frames=frames,clearance=clearance,feet=feet_samples,minimum_slack=[r['minimum_normalized_slack'] for r in rows])
print(json.dumps(report,indent=2),flush=True)
