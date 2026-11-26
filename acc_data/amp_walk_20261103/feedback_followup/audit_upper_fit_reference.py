"""Audit the reconstructed q/v/a, rather than the linearized offline QP."""
from pathlib import Path
import json,pickle,sys
import numpy as np,torch
from scipy.interpolate import CubicSpline,BSpline
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from numpy_g1_com import CenterOfMassModel
from numpy_leg_kinematics import LegKinematics
p=Path(__file__).parent;name=sys.argv[1] if len(sys.argv)>1 else 'wbc_upper_fit_projected'
metadata=json.loads((p/(name+'.json')).read_text());parent_name=metadata.get('parent','wbc_tenth_held_out33')
q=np.load(p/(name+'.npz'))['control_q'];parent=np.load(p/(parent_name+'.npz'))['control_q'];source=np.load(p/'fixed_smooth_early.control_reference.npz')['source_q'];d=np.load(p/'wrench_fixed_ten.npz');urdf='src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'
com=CenterOfMassModel(urdf,d['final_joint_order']);leg=LegKinematics(urdf,d['final_joint_order']);limits=d['joint_position_limits'][0];speed=d['history_joint_velocity_limits'][-1].ravel();plan=json.loads((p/'wrench_fixed_ten.steps_plan.json').read_text());contacts=np.asarray(pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))['contact_sequence'])
def sampler(trajectory):
 t=torch.tensor(trajectory,dtype=torch.float64);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
start,end=780.,1100.
with np.load(p/(name+'.npz')) as candidate:
 if 'knots' in candidate:start,end=float(candidate['knots'][3]),float(candidate['knots'][-4])
if (p/(name+'.centroidal.npz')).exists():
 with np.load(p/(name+'.centroidal.npz')) as planned:
  start,end=float(planned['frames'][0]),float(planned['frames'][-1])
  if 'planning_contacts' in planned:contacts=planned['planning_contacts'].copy()
switches=np.flatnonzero(np.any(contacts[1:]!=contacts[:-1],axis=1))+1;switches=switches[(switches>start)&(switches<end)]
curve=sampler(q);original=sampler(parent);frames=np.unique(np.r_[np.arange(start-4,end+4.01,.5),switches-1e-4]);values=[curve.kinematics(float(f),240.) for f in frames];poses=np.array([v[0].numpy()[0] for v in values]);velocities=np.array([v[1].numpy()[0] for v in values]);states=np.array([com.momentum(a,b,return_state=True) for a,b in zip(poses,velocities)]);rate=CubicSpline(frames/240.,states[:,1:])(frames/240.,1);force=com.mass*(rate[:,0]+[0,0,9.81]);hdot=rate[:,1];clear=[];foot_errors=[];orientation_errors=[];cop_violations=[];friction=[];normal=[];dynamic_frames=[];cop_values=[]
for k,(f,pose) in enumerate(zip(frames,poses)):
 base=original.kinematics(float(f),240.)[0].numpy()[0];vertices=[]
 for side in [0,1]:
  pos,R,_=leg.evaluate(pose,side);pp,RR,_=leg.evaluate(base,side);points=np.array(plan['collision_points'][side]);radii=np.array(plan['collision_radii'][side]);clear.append(pos[2]+np.min((points@R.T)[:,2]-radii));foot_errors.append(np.linalg.norm(pos-pp));orientation_errors.append(np.linalg.norm(Rotation.from_matrix(R@RR.T).as_rotvec()))
  if contacts[int(f),side]:vertices.extend((pos+np.array([[x,y,-.035] for x in [-.025,.085] for y in [-.02,.02]])@R.T)[:,:2])
 if start+20<=f<=end-40:
  F=force[k];dynamic_frames.append(float(f));normal.append(F[2]);friction.append(np.sum(abs(F[:2]))/max(F[2],1e-9));cop=states[k,0,:2]-states[k,0,2]*F[:2]/max(F[2],1e-9)+np.array([-hdot[k,1],hdot[k,0]])/max(F[2],1e-9)
  cop_values.append(cop.tolist());equations=ConvexHull(np.array(vertices)).equations;cop_violations.append(max(0.,float(np.max(equations[:,:2]@cop+equations[:,2]))))
report=json.loads((p/(name+'.json')).read_text());report.update(sample_frame_stride=.5,minimum_joint_margin=float(np.minimum(poses[:,7:]-limits[:,0],limits[:,1]-poses[:,7:]).min()),maximum_joint_velocity_fraction=float(np.max(abs(velocities[:,6:])/speed)),minimum_native_sole_clearance=float(min(clear)),peak_source_joint_reference_rmse=float(np.sqrt(np.mean((q[:,7:]-source[:,7:])**2,axis=1)).max()),peak_foot_position_change=float(max(foot_errors)),peak_foot_orientation_change=float(max(orientation_errors)),nonlinear_normal_force_range=[float(min(normal)),float(max(normal))],nonlinear_friction_l1_required=float(max(friction)),nonlinear_cop_violation=float(max(cop_violations)),physical_tracking_certified=False)
report['worst_cop_samples']=[dict(frame=dynamic_frames[j],violation=float(cop_violations[j]),cop=cop_values[j],normal_force=float(normal[j])) for j in np.argsort(cop_violations)[-8:][::-1]]
from reference_joint_extrema import joint_extrema
report.update(joint_extrema(curve.smooth,start-4,end+4,240.,limits,speed));report['requires_exact_curve_joint_audit']=False
report['preflight_passed']=bool(report['minimum_continuous_joint_margin']>=0 and report['maximum_continuous_joint_velocity_fraction']<=1 and min(clear)>=-.002 and report['peak_source_joint_reference_rmse']<.19 and max(foot_errors)<.002 and max(orientation_errors)<.01 and min(normal)>=100 and max(normal)<=800 and max(friction)<=.6 and max(cop_violations)<=.002)
report['requires_nonlinear_dynamic_audit']=False;report['requires_native_body_collision_audit']=True
(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='repaired_frames'},flush=True)
if not report['preflight_passed']:raise RuntimeError('Reconstructed reference failed preflight; do not execute')
