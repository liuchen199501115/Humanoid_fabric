from pathlib import Path
import sys,json,numpy as np,torch
from scipy.interpolate import CubicSpline
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from numpy_g1_com import CenterOfMassModel
p=Path(__file__).parent;model=CenterOfMassModel('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',np.load(p/'wrench_fixed_ten.npz')['final_joint_order']);reports=[]
for name in ['tenth_landing_forward15_right20','turn_capture_sole_y60_drop15_floor']:
 t=torch.tensor(np.load(p/(name+'.npz'))['control_q']);curve=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);frames=np.arange(660,1067,.5);states=[]
 for f in frames:
  q,v,_=curve.kinematics(float(f),240.);states.append(model.momentum(q.numpy()[0],v.numpy()[0],return_state=True))
 states=np.array(states);time=frames/240.;force=model.mass*(CubicSpline(time,states[:,1])(time,1)+[0,0,9.81]);rate=CubicSpline(time,states[:,2])(time,1);turn=(frames>=740)&(frames<=920);positive=turn&(force[:,2]>0);report=dict(candidate=name,turn_min_fz=float(force[turn,2].min()),negative_turn_fz_samples=int(np.sum(force[turn,2]<0)),peak_turn_mu_norm=float(np.max(np.linalg.norm(force[positive,:2],axis=1)/force[positive,2])),peak_turn_momentum_rate=float(abs(rate[turn]).max()),samples=[dict(frame=f,com=states[abs(frames-f).argmin(),0].tolist(),velocity=states[abs(frames-f).argmin(),1].tolist()) for f in [740,760,780,800,820,840]],physical_tracking_certified=False);reports.append(report)
(p/'early_capture_dynamics_audit.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports,indent=2))
