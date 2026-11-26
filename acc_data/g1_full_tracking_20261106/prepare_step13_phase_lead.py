"""Reference-only monotone phase lead; source clock and raw acceptance unchanged."""
from pathlib import Path
import json,pickle,hashlib
import numpy as np,torch
from scipy.spatial.transform import Rotation
from blended_spline import BlendedSplineReference
from reference_joint_extrema import joint_extrema
from mujoco_reference_dynamics import ReferenceDynamics
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path(__file__).parent;name='step13_phase_lead12';parent_path=p/'step13_whole_body_dense.npz'
if (p/(name+'.json')).exists():raise RuntimeError('Existing report')
parent=np.load(parent_path);base=parent['control_q'].astype(float);out=base.copy();record=np.load(p/'rolling_contact.npz');names=record['final_joint_order'];raw=pickle.load(open('retarget/amp_reference/amp_walk_1ms.pkl','rb'));order=json.load(open('acc_data/amp_walk_20261102/input_report.json'))['source_joint_order'];raw_joints=raw['dof_pos'][:,[order.index(n) for n in names]]
model=ReferenceDynamics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',names);plan=json.load(open(p/'rolling_contact.steps_plan.json'));points=[np.array(x) for x in plan['collision_points']];radii=[np.array(x) for x in plan['collision_radii']]
start,end,lead=952.,1056.,12.
original=MotionInterpolator(torch.tensor(base,dtype=torch.float64),method='pchip')
def phase(f):
 u=np.clip((f-start)/(end-start),0.,1.);return f+lead*64*u**3*(1-u)**3
for f in range(int(start)+1,int(end)):out[f]=original(float(phase(f))).numpy()[0]
t=torch.tensor(out,dtype=torch.float64);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);fs=np.arange(start-6,end+6.01,.25);clear=[];dense_joint=[]
for f in fs:
 q,v,a=[x.numpy()[0] for x in curve.kinematics(float(f),240.)];s=model.evaluate(q,v,a)
 for foot in (0,1):
  R=Rotation.from_quat(s['feet'][foot,3:]).as_matrix();clear.append(float(s['feet'][foot,2]+np.min((points[foot]@R.T)[:,2]-radii[foot])))
 ref=np.array([np.interp(f,np.arange(len(raw_joints)),raw_joints[:,j]) for j in range(29)]);dense_joint.append(float(np.sqrt(np.mean((q[7:]-ref)**2))))
u=np.linspace(0,1,10001);rate=1+lead*192*u**2*(1-u)**2*(1-2*u)/(end-start)
report=dict(parent=str(parent_path),parent_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),scope='Monotone whole-body reference phase lead only. Audit source clock, contact labels, physical state, and original-PKL acceptance unchanged. No physical tracking certification.',frames=[start,end],maximum_phase_lead_frames=lead,minimum_reference_phase_rate=float(rate.min()),peak_source_joint_reference_rmse=float(np.sqrt(np.mean((out[:,7:]-raw_joints)**2,axis=1)).max()),peak_dense_source_joint_rmse=max(dense_joint),peak_root_translation_change_m=float(np.linalg.norm(out[:,:3]-base[:,:3],axis=1).max()),peak_root_orientation_change_rad=float((Rotation.from_quat(out[:,3:7])*Rotation.from_quat(base[:,3:7]).inv()).magnitude().max()),minimum_native_sole_height=min(clear),physical_tracking_certified=False)
report.update(joint_extrema(curve.smooth,start-6,end+6,240.,record['joint_position_limits'][0],record['history_joint_velocity_limits'][-1].reshape(-1)))
report['kinematic_preflight_passed']=bool(report['peak_source_joint_reference_rmse']<.19 and max(dense_joint)<.19 and min(clear)>-.0026 and rate.min()>0 and report['minimum_continuous_joint_margin']>0 and report['maximum_continuous_joint_velocity_fraction']<=1)
(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(p/(name+'.diagnostic.npz'),rejected_or_unexecuted_control_q=out)
if report['kinematic_preflight_passed']:np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
print(json.dumps({k:v for k,v in report.items() if not k.startswith('joint_')},indent=2),flush=True)
