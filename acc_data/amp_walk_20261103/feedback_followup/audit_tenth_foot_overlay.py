from pathlib import Path
import json
import numpy as np,torch
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from joint_correction_overlay import JointCorrectionOverlay
from numpy_leg_kinematics import LegKinematics
p=Path(__file__).parent;source=np.load(p/'eight_smooth6_tube2.control_reference.npz')['source_q'];names=np.load(p/'eight_smooth6_tube2.failure.npz')['joint_names'];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',names)
for key in ['004','006']:
 trajectory=np.load(p/('balance_y'+key+'.npz'))['control_q'];tensor=torch.tensor(trajectory,dtype=torch.float64);base=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=564,ramp_frames=16)
 # Hash uses original float32 bytes, calculation uses float64 for derivative checks.
 overlay=JointCorrectionOverlay(base,torch.tensor(trajectory),torch.tensor(source),p/('tenth_flat50_y'+key+'.npz'))
 qvals=[];feet=[];v_error=0.;a_error=0.;h=1e-4
 for f in np.arange(760.,850.01,.2):
  q,v,a=[x.numpy()[0] for x in overlay.kinematics(f,200.,13.)];qvals.append(q);feet.append(model.evaluate(q,0)[0])
  if 760.01<f<849.99:
   prev=overlay.kinematics(f-h,200.,13.);nxt=overlay.kinematics(f+h,200.,13.)
   fdv=(nxt[0].numpy()[0,7:]-prev[0].numpy()[0,7:])/(2*h)*200
   # d/dt v = dv/dframe * frame_rate plus q' * frame_acceleration.
   fda=(nxt[1].numpy()[0,6:]-prev[1].numpy()[0,6:])/(2*h)*200+v[6:]/200*13
   v_error=max(v_error,float(abs(fdv-v[6:]).max()));a_error=max(a_error,float(abs(fda-a[6:]).max()))
 qvals=np.array(qvals);feet=np.array(feet);joints=qvals[:,7:19];report=dict(velocity_fd_error=v_error,acceleration_fd_error=a_error,min_joint_margin=float(min((joints-model.lower).min(),(model.upper-joints).min())),held_stance_position_span_m=float(np.ptp(feet[75:216],axis=0).max()),outside_window_exact=all(torch.equal(x,y) for f in [0.,759.,760.,850.,851.,1999.] for x,y in zip(base.kinematics(f,200),overlay.kinematics(f,200))),physical_tracking_certified=False)
 print('AUDIT',key,report,flush=True)
 assert report['outside_window_exact'] and v_error<1e-4 and a_error<.02 and report['min_joint_margin']>-.00001
 (p/('tenth_flat50_y'+key+'.curve_audit.json')).write_text(json.dumps(report,indent=2)+'\n');print(key,report,flush=True)
