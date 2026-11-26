"""Smooth bounded upper-body reference precompensation; source audits unchanged."""
from pathlib import Path
import numpy as np,json,hashlib,torch
from scipy.signal import savgol_filter
from fabrics_sim.reference.interpolation import MotionInterpolator
from blended_spline import BlendedSplineReference
p=Path(__file__).parent;base=np.load(p/'turn_recovery_y40.npz')['control_q'];baseline=np.load(p/'fixed_smooth_early.control_reference.npz');source=baseline['source_q'];d=np.load(p/'fixed_clock120_body10_overdue.npz');limits=d['joint_position_limits'][0];frames=d['history_frame'].ravel();actual=d['history_q'][:,0,7:];reference=d['history_whole_body_reference_q'].reshape(-1,36)[:,7:];error=reference-actual;f=np.arange(len(base));future=f+16.;curve=np.stack([np.interp(future,frames,error[:,j]) for j in range(29)],axis=-1);curve=savgol_filter(curve,81,3,axis=0)
def smooth(x):x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
w=smooth((f-740)/60)*(1-smooth((f-940)/60))
for first,label in [(12,'upper'),(15,'arms')]:
 out=base.copy();correction=.15*np.tanh(curve[:,first:]/.15)*w[:,None];out[:,7+first:]+=correction;out[:,7+first:]=np.clip(out[:,7+first:],limits[first:,0]+.02,limits[first:,1]-.02)
 # Preserve every frame outside the active interval exactly.
 out[w==0]=base[w==0]
 t=torch.tensor(out);sampler=BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20);dense=np.array([sampler.kinematics(x,240)[0].numpy()[0] for x in np.arange(734,1007,.2)]);margin=float(np.minimum(dense[:,7:]-limits[:,0],limits[:,1]-dense[:,7:]).min());rmse=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1));maxchange=float(abs(out-base).max());name='turn_'+label+'_precomp150';report=dict(precompensation_maximum_rad=.15,anticipation_frames=16,peak_source_joint_reference_rmse=float(rmse.max()),minimum_dense_joint_limit_margin=margin,maximum_coordinate_change=maxchange,root_and_leg_reference_unchanged=bool(np.array_equal(out[:,:19],base[:,:19])),physical_tracking_certified=False);report['preflight_passed']=bool(rmse.max()<.19 and margin>=0 and maxchange<.151 and report['root_and_leg_reference_unchanged']);np.savez_compressed(p/(name+'.npz'),control_q=out,fps=200.,baseline_sha256=hashlib.sha256(baseline['control_q'].tobytes()).hexdigest(),source_sha256=hashlib.sha256(source.tobytes()).hexdigest());(p/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(name,report,flush=True)
