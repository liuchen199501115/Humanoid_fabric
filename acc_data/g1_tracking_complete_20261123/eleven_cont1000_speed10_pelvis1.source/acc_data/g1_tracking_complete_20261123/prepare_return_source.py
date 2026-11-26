"""Return a unified prepared pose to the original source after the verified prefix.

This is an offline reference candidate, not a dynamics-feasibility certificate.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
label,start,end=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]);assert 906<=start<end<=1050
parent=OUT/'prepared_rooty_clearance10_lift10.npz';target=OUT/('prepared_return_source_'+label+'.npz');assert not target.exists()
base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
u=np.clip((np.arange(len(source))-start)/(end-start),0,1);blend=u**3*(10-15*u+6*u*u)
out=base['control_q'].copy();rows=blend>0
out[rows,:3]+=blend[rows,None]*(source[rows,:3]-out[rows,:3]);out[rows,7:]+=blend[rows,None]*(source[rows,7:]-out[rows,7:])
old_R=Rotation.from_quat(out[rows,3:7]);target_R=Rotation.from_quat(source[rows,3:7]);delta=(target_R*old_R.inv()).as_rotvec();new_q=(Rotation.from_rotvec(blend[rows,None]*delta)*old_R).as_quat();new_q*=np.where(np.sum(new_q*out[rows,3:7],axis=-1)<0,-1,1)[:,None];out[rows,3:7]=new_q
np.testing.assert_array_equal(out[:start+1],base['control_q'][:start+1])
old_error=np.sqrt(np.mean((base['control_q'][:,7:]-source[:,7:])**2,axis=-1));new_error=np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=-1));assert np.all(new_error<=old_error+2e-7)
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist();legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
g=json.loads((OUT/'ten_verify.steps_plan.json').read_text());clearance={}
for label_q,poses in [('parent',base['control_q']),('source',source),('candidate',out)]:
 minima=[]
 for side in (0,1):
  points=np.asarray(g['collision_points'][side]);radii=np.asarray(g['collision_radii'][side]);vals=[]
  for q in poses[start:end+1]:
   p,R,_=legs.evaluate(q,side);vals.append(float(p[2]+np.min((points@R.T)[:,2]-radii)))
  minima.append(min(vals))
 clearance[label_q]=minima
def sampler(q):
 t=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
a,b=sampler(base['control_q']),sampler(out);prefix=6*(start//6)-6
for k in np.arange(0,prefix+.01,.5):
 for x,y in zip(a.kinematics(k,280.,0.),b.kinematics(k,280.,0.)):np.testing.assert_array_equal(x.numpy(),y.numpy())
geometry_ok=all(c>=min(p,s)-.002 for c,p,s in zip(clearance['candidate'],clearance['parent'],clearance['source']))
limits=np.load(OUT/'ten_verify.npz')['joint_position_limits'][0]
joint_violation=float(np.maximum(limits[:,0]-out[:,7:],out[:,7:]-limits[:,1]).clip(0).max())
report=dict(method=__doc__,start_frame=start,end_frame=end,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),prefix_q_v_a_equal_through_frame=prefix,pointwise_source_joint_error_not_increased=True,peak_source_joint_rmse_rad=float(new_error.max()),minimum_native_clearance_m=clearance,geometry_not_more_than_2mm_worse_than_inputs=geometry_ok,preflight_passed=geometry_ok,full_tracking_passed=False)
report['maximum_joint_limit_violation_rad']=joint_violation
report['preflight_passed']=bool(geometry_ok and joint_violation<1e-6)
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
 with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
print(target.name,report,flush=True)
