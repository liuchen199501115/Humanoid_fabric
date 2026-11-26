from pathlib import Path
import json
import numpy as np,torch
from scipy.interpolate import CubicSpline
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
from numpy_g1_com import CenterOfMassModel
p=Path(__file__).parent;names=np.load(p/'wrench_fixed_ten.npz')['final_joint_order'];model=CenterOfMassModel('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',names)
for name in ['early_flat_lower15','lipm_reference50','momentum_lipm_reference50']:
 raw=np.load(p/(name+'.npz'))['control_q'];tensor=torch.tensor(raw);curve=BlendedSplineReference(MotionInterpolator(tensor),tensor,stride=6,start_frame=130,ramp_frames=20);frames=np.arange(180,1999.01,.5);data=[]
 for f in frames:
  q,v,_=curve.kinematics(float(f),200);data.append(model.momentum(q.numpy()[0],v.numpy()[0],return_state=True))
 data=np.array(data);c,velocity,h=data[:,0],data[:,1],data[:,2];a=CubicSpline(frames/200,velocity)(frames/200,1);hd=CubicSpline(frames/200,h)(frames/200,1);force=model.mass*(a+[0,0,9.81]);zmp=c[:,:2]-c[:,2,None]*force[:,:2]/force[:,2,None];zmp[:,0]-=hd[:,1]/force[:,2];zmp[:,1]+=hd[:,0]/force[:,2]
 report=dict(name=name,minimum_reference_normal_force=float(force[:,2].min()),maximum_reference_normal_force=float(force[:,2].max()),negative_force_samples=int(np.sum(force[:,2]<0)),peak_momentum_rate=float(np.linalg.norm(hd,axis=1).max()),physical_tracking_certified=False)
 np.savez_compressed(p/(name+'.centroidal_path.npz'),frames=frames,com=c,velocity=velocity,acceleration=a,momentum=h,momentum_rate=hd,force=force,zmp=zmp);(p/(name+'.centroidal_audit.json')).write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
