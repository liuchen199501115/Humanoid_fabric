import sys,json,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.signal import savgol_filter
from fabrics_sim.reference.interpolation import MotionInterpolator
p=Path('acc_data/amp_walk_20261103/feedback_followup');sys.path.insert(0,str(p))
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
basefile=np.load(p/'turn_recovery_y40.npz');base=basefile['control_q'];baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];d=np.load(p/'wrench_fixed_ten.npz');limits=d['joint_position_limits'][0];model=LegKinematics('src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf',d['final_joint_order'])
f=np.arange(len(base));x=np.clip((f-830)/40,0,1);w=(x**3*(10-15*x+6*x*x))[:,None];x=np.clip((f-940)/40,0,1);w*=1-(x**3*(10-15*x+6*x*x))[:,None]
for window in [31,51]:
 out=base.copy();out[831:980]=(base+w*(savgol_filter(base,window,3,axis=0)-base))[831:980];out[831:980,3:7]/=np.linalg.norm(out[831:980,3:7],axis=1,keepdims=True)
 errors=[]
 for i in range(830,981):
  errors.append([np.linalg.norm(model.evaluate(out[i],s)[0]-model.evaluate(source[i],s)[0]) for s in [0,1]])
 errors=np.array(errors);joint=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));margin=float(np.minimum(out[830:981,7:]-limits[:,0],limits[:,1]-out[830:981,7:]).min());report=dict(window=window,peak_source_foot_error=float(errors.max()),peak_source_joint_rmse=float(joint.max()),joint_margin=margin,preflight_passed=bool(joint.max()<.19 and margin>=0 and np.max(np.linalg.norm(out[830:981,:3]-source[830:981,:3],axis=1))<.19),physical_tracking_certified=False)
 name='recovery_wholebody_smooth'+str(window);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=basefile['baseline_sha256'],source_sha256=basefile['source_sha256']);(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
