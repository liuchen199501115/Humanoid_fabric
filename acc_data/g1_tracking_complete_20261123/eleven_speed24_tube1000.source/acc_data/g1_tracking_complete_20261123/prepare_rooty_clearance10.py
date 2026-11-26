"""Small bounded right swing lift for the earlier source-Y tracking branch."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
label,height_mm,start,end=sys.argv[1],float(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
assert 0<height_mm<=10 and 762<=start<end<=826
parent=OUT/'prepared_root_smooth_floor3_late11_sigma16.npz'
target=OUT/('prepared_rooty_clearance10_'+label+'.npz');assert not target.exists()
base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
legs=LegKinematics(str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf'),names)
out=base['control_q'].copy();peakp=peakr=0.
for k in range(start+1,end):
    original=base['control_q'][k];p,R,_=legs.evaluate(original,1)
    phase=(k-start)/(end-start);p=p.copy();p[2]+=height_mm/1000.*64*phase**3*(1-phase)**3
    q=original.astype(float).copy()
    for _ in range(160):
        pos,rot,J=legs.evaluate(q,1);error=np.r_[p-pos,Rotation.from_matrix(R@rot.T).as_rotvec()]
        if np.max(abs(error))<5e-7:break
        lo=np.maximum(-.05,legs.lower[6:]-q[13:19]);hi=np.minimum(.05,legs.upper[6:]-q[13:19])
        step=lsq_linear(np.vstack((J,.0001*np.eye(6))),np.r_[error,np.zeros(6)],bounds=(lo,hi),tol=1e-10).x
        q[13:19]+=step
    out[k]=q;pos,rot,_=legs.evaluate(out[k],1)
    peakp=max(peakp,float(np.linalg.norm(p-pos)));peakr=max(peakr,float(Rotation.from_matrix(R@rot.T).magnitude()))
np.testing.assert_array_equal(out[:start+1],base['control_q'][:start+1])
np.testing.assert_array_equal(out[end:],base['control_q'][end:])
np.testing.assert_array_equal(out[:,:13],base['control_q'][:,:13])
np.testing.assert_array_equal(out[:,19:],base['control_q'][:,19:])
def sampler(q):
    t=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
old,new=sampler(base['control_q']),sampler(out)
prefix_frame=6*(start//6)-6
for k in np.arange(0,prefix_frame+.01,.5):
    for a,b in zip(old.kinematics(k,240.,0.),new.kinematics(k,240.,0.)):
        np.testing.assert_array_equal(a.numpy(),b.numpy())
rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max())
report=dict(method=__doc__,parent_file=parent.name,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
    source_sha256=base['source_sha256'].item(),start_frame=start,end_frame=end,added_height_m=height_mm/1000.,
    maximum_foot_position_residual_m=peakp,maximum_foot_rotation_residual_rad=peakr,
    peak_source_joint_rmse_rad=rmse,prefix_q_v_a_equal_through_frame=prefix_frame,
    root_other_leg_and_upper_body_exactly_unchanged=True,
    preflight_passed=peakp<2e-5 and peakr<2e-4 and rmse<.19,full_tracking_passed=False)
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
    with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
print(target.name,report,flush=True)
