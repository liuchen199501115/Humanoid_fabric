from pathlib import Path
import json
import numpy as np,torch
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from numpy_g1_com import CenterOfMassModel
from smooth_clock import smooth_clock
p=Path(__file__).parent;names=np.load(p/'wrench_fixed_ten.npz')['final_joint_order'];model=CenterOfMassModel('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',names);name='root_turn10_smooth61_3';q=torch.tensor(np.load(p/(name+'.npz'))['control_q']);curve=BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
for rate in [.8]:
 frames=np.arange(740,2000);times=[];data=[]
 for f in frames:
  t=brentq(lambda t:200*smooth_clock(t,.9,.35,1.2,3.35,.2,rate)[0]-f,0,18);s,r,a=smooth_clock(t,.9,.35,1.2,3.35,.2,rate);q,v,_=curve.kinematics(float(f),200*r,200*a);times.append(t);data.append(model.momentum(q.numpy()[0],v.numpy()[0],return_state=True))
 data=np.array(data);times=np.array(times);acc=CubicSpline(times,data[:,1])(times,1);force=model.mass*(acc+[0,0,9.81]);ratio=np.linalg.norm(force[:,:2],axis=1)/force[:,2];valid=force[:,2]>0;peak=np.argmax(np.where(valid,ratio,np.inf));report=dict(clock_final_speed=rate,source_frames=[740,1999],minimum_normal_force=float(force[:,2].min()),maximum_normal_force=float(force[:,2].max()),negative_force_samples=int((~valid).sum()),peak_required_friction_norm=float(ratio[valid].max()),peak_friction_frame=int(frames[peak]),physical_tracking_certified=False)
 np.savez_compressed(p/f'turn10_smoothed_retimed{int(rate*100)}.centroidal_path.npz',frames=frames,time=times,com=data[:,0],velocity=data[:,1],acceleration=acc,force=force);(p/f'turn10_smoothed_retimed{int(rate*100)}.centroidal_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
