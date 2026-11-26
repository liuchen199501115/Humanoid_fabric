from pathlib import Path
import json,pickle,numpy as np,torch
from measured_stance_reference import MeasuredStanceReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from fabrics_sim.reference.whole_body import WholeBodyReference
from blended_spline import BlendedSplineReference
p=Path(__file__).parent
d=np.load(p/'fixed_clock120_body10_turn_recovery40.npz');data=pickle.load(open('acc_data/amp_walk_20261102/amp_walk_native_contacts.pkl','rb'))
contact=np.asarray(data['contact_sequence']);modes=contact[:,0].astype(int)+2*contact[:,1].astype(int)
ref=torch.tensor(np.load(p/'turn_recovery_y40.npz')['control_q']);curve=BlendedSplineReference(MotionInterpolator(ref),ref,stride=6,start_frame=130,ramp_frames=20)
projector=MeasuredStanceReference('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order'],modes,200,limit=.01)
fields=['normal_force','point','normal','distance','count','start']
vel_errors=[];acc_errors=[];records=[]
for i,f in enumerate(d['history_frame']):
 if f<740:continue
 projector.observe(f,d['history_q'][i,0],d['history_ground_foot_forces'][i,:,2],tuple(d['history_contact_'+k][i] for k in fields))
 if i%5:continue
 def sample(t):
  frame=f+240*t;values=curve.kinematics(frame,240,0.)
  return projector.project(frame,240,0.,curve,WholeBodyReference(*values))
 o=sample(0.);minus=sample(-.001);plus=sample(.001)
 vel_errors.append(float(abs((plus.q[:,7:]-minus.q[:,7:])/.002-o.v[:,6:]).max()))
 acc_errors.append(float(abs((plus.v[:,6:]-minus.v[:,6:])/.002-o.a[:,6:]).max()))
 records.append([float(f),float((o.q-ref[int(f)][None]).norm()),float(o.a.abs().max())])
report=dict(events=projector.events,peak_ik_position_residual=projector.peak_position_residual,peak_ik_orientation_residual=projector.peak_orientation_residual,peak_joint_correction=projector.peak_joint_correction,velocity_derivative_max_error=max(vel_errors),acceleration_derivative_max_error=max(acc_errors),samples=records)
report['passed']=bool(report['peak_ik_position_residual']<.002 and report['peak_ik_orientation_residual']<.01 and max(vel_errors)<.03 and max(acc_errors)<5.)
(p/'measured_stance_reference_audit.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items() if k!='samples'})
