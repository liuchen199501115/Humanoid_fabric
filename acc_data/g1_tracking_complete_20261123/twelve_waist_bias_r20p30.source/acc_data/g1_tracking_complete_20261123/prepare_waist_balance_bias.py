"""Smooth offline waist bias during transfer12, retaining source acceptance and all leg/root coordinates."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
label=sys.argv[1];roll,pitch=map(float,sys.argv[2:4]);assert np.isfinite([roll,pitch]).all() and max(abs(roll),abs(pitch))<=.35
parent=OUT/'prepared_return_source_transfer60.npz';target=OUT/f'prepared_waist_balance_{label}.npz';assert not target.exists();base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q'];assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
frames=np.arange(len(source));weight=smooth((frames-906)/48)*(1-smooth((frames-1000)/60));out=base['control_q'].copy();out[:,20]+=weight*roll;out[:,21]+=weight*pitch
np.testing.assert_array_equal(out[:907],base['control_q'][:907]);np.testing.assert_array_equal(out[1060:],base['control_q'][1060:]);np.testing.assert_array_equal(out[:,:20],base['control_q'][:,:20]);np.testing.assert_array_equal(out[:,22:],base['control_q'][:,22:])
def sampler(q):
 q=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
a,b=sampler(base['control_q']),sampler(out)
for f in np.arange(0,900.01,.5):
 for x,y in zip(a.kinematics(f,280.,0.),b.kinematics(f,280.,0.)):np.testing.assert_array_equal(x.numpy(),y.numpy())
limits=np.load(OUT/'ten_verify.npz')['joint_position_limits'][0];violation=float(np.maximum(limits[:,0]-out[:,7:],out[:,7:]-limits[:,1]).clip(0).max());rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=-1)).max())
r=dict(parent_file=parent.name,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),waist_roll_bias_rad=roll,waist_pitch_bias_rad=pitch,start_frame=906,end_frame=954,fade_out_frames=[1000,1060],maximum_joint_limit_violation_rad=violation,maximum_source_joint_rmse_rad=rmse,prefix_q_v_a_exact_through_frame=900,root_legs_arms_unchanged=True,preflight_passed=violation<1e-6 and rmse<.19,full_tracking_passed=False);target.with_suffix('.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
if r['preflight_passed']:
 with target.open('xb') as file:np.savez_compressed(file,**dict(base,control_q=out))
