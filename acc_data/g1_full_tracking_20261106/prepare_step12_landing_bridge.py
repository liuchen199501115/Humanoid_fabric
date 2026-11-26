"""Reference-only right landing bridge and stationary support through step 13.
The original source, contact labels, physical state, and acceptance rules remain unchanged.
"""
from pathlib import Path
import json,hashlib,pickle
import numpy as np,torch
from scipy.spatial.transform import Rotation
from numpy_leg_kinematics import LegKinematics
from bounded_leg_ik import solve_leg_pose
from blended_spline import BlendedSplineReference
from reference_joint_extrema import joint_extrema
from mujoco_reference_dynamics import ReferenceDynamics
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.reference.bridge import ReferenceBridge,RotationReferenceBridge
p=Path(__file__).parent;name='step12_landing_bridge';parent_path=p/'step13_whole_body_dense.npz'
if (p/(name+'.json')).exists():raise RuntimeError('Existing report')
parent=np.load(parent_path);base=parent['control_q'].astype(float);out=base.copy();record=np.load(p/'rolling_contact.npz');source=np.load(p/'rolling_contact.control_reference.npz')['source_q'];urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';leg=LegKinematics(urdf,record['final_joint_order']);model=ReferenceDynamics(urdf,record['final_joint_order']);plan=json.load(open(p/'rolling_contact.steps_plan.json'));points=[np.array(x) for x in plan['collision_points']];radii=[np.array(x) for x in plan['collision_radii']]
raw_pickle=pickle.load(open('retarget/amp_reference/amp_walk_1ms.pkl','rb'));source_order=json.load(open('acc_data/amp_walk_20261102/input_report.json'))['source_joint_order'];source=source.copy();source[:,7:]=raw_pickle['dof_pos'][:,[source_order.index(n) for n in record['final_joint_order']]]
raw=MotionInterpolator(torch.tensor(base,dtype=torch.float64),method='pchip')
def foot_kinematics(f):
 q,v,a=[x.numpy()[0] for x in raw.kinematics(float(f),240.)];s=model.evaluate(q,v,a,kinematic_bias=True);J=s['jacobian'][1]
 return s['feet'][1],J@v,J@a+s['foot_bias'][1]
def height(pos,R,foot):return float(pos[2]+np.min((points[foot]@R.T)[:,2]-radii[foot]))
start,landing,hold,end=928,952,970,1000
anchor,_,_=foot_kinematics(970);Ranchor=Rotation.from_quat(anchor[3:]);anchor[2]-=height(anchor[:3],Ranchor.as_matrix(),1)
initial,iv,ia=foot_kinematics(start);duration=(landing-start)/240.
pbridge=ReferenceBridge(initial[:3]-anchor[:3],iv[:3],ia[:3],duration)
rbridge=RotationReferenceBridge(initial[3:],iv[3:],anchor[3:],np.zeros(3),np.zeros(3),duration,measured_acceleration=ia[3:])
source_start,sv,sa=foot_kinematics(hold);duration_out=(end-hold)/240.
return_p=ReferenceBridge(anchor[:3]-source_start[:3],-sv[:3],-sa[:3],duration_out)
return_r=RotationReferenceBridge(anchor[3:],np.zeros(3),source_start[3:],sv[3:],sa[3:],duration_out,measured_acceleration=np.zeros(3))
errors=[]
for f in range(start+1,end):
 if f<landing:
  elapsed=(f-start)/240.;target=anchor[:3]+pbridge.sample(elapsed)[0];quat=rbridge.sample(elapsed,anchor[3:],np.zeros(3),np.zeros(3))[0]
 elif f<=hold:target=anchor[:3].copy();quat=anchor[3:]
 else:
  elapsed=(f-hold)/240.;pose,v,a=foot_kinematics(f);target=pose[:3]+return_p.sample(elapsed)[0];quat=return_r.sample(elapsed,pose[3:],v[3:],a[3:])[0]
  u=(f-hold)/(end-hold);target[2]+=.005*64*u**3*(1-u)**3
 out[f],pe,re=solve_leg_pose(leg,base[f],1,target,Rotation.from_quat(quat).as_matrix(),position_tolerance=1e-9,orientation_tolerance=1e-8);errors.append([pe,re])
t=torch.tensor(out,dtype=torch.float64);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);fs=np.arange(start-6,end+6.01,.25);clear=[];stance_speed=[];raw_joint=[]
for f in fs:
 q,v,a=[x.numpy()[0] for x in curve.kinematics(float(f),240.)];s=model.evaluate(q,v,a)
 for foot in (0,1):clear.append(height(s['feet'][foot,:3],Rotation.from_quat(s['feet'][foot,3:]).as_matrix(),foot))
 if landing+6<=f<=hold-6:stance_speed.append(float(np.linalg.norm((s['jacobian'][1]@v)[:3])))
 ref=np.array([np.interp(f,np.arange(len(source)),source[:,j]) for j in range(7,36)]);raw_joint.append(float(np.sqrt(np.mean((q[7:]-ref)**2))))
report=dict(parent=str(parent_path),parent_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),scope='Kinematic reference experiment only; no nominal dynamics or physical tracking certification.',bridge_frames=[start,landing,hold,end],anchor_pose=anchor.tolist(),maximum_ik_errors=np.max(errors,axis=0).tolist(),peak_source_joint_reference_rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max()),peak_dense_source_joint_rmse=max(raw_joint),minimum_native_sole_height=min(clear),maximum_stance_foot_origin_speed=max(stance_speed),source_contact_labels_unchanged=True,physical_tracking_certified=False)
report.update(joint_extrema(curve.smooth,start-6,end+6,240.,record['joint_position_limits'][0],record['history_joint_velocity_limits'][-1].reshape(-1)))
report['kinematic_preflight_passed']=bool(report['peak_source_joint_reference_rmse']<.19 and max(raw_joint)<.19 and min(clear)>-.0026 and max(stance_speed)<.02 and report['minimum_continuous_joint_margin']>0 and report['maximum_continuous_joint_velocity_fraction']<=1 and np.max(errors,axis=0)[0]<.001 and np.max(errors,axis=0)[1]<.01)
(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
np.savez_compressed(p/(name+'.diagnostic.npz'),rejected_or_unexecuted_control_q=out)
if report['kinematic_preflight_passed']:np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=parent['baseline_sha256'],source_sha256=parent['source_sha256'])
print(json.dumps({k:v for k,v in report.items() if not k.startswith('joint_')},indent=2),flush=True)
