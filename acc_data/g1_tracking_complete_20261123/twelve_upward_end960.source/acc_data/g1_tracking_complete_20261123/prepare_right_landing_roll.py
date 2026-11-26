"""Offline right-foot roll preparation; fixed root and foot-link position."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
parent=OUT/'prepared_margin30_turn_com_y15.npz'
with np.load(parent) as a:base={k:a[k] for k in a.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs=LegKinematics(urdf,names);model=CentroidalGeometry(urdf,names)
def smooth(u):
    u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
for amplitude in map(float,sys.argv[1:] or ['.08','.12']):
    target=OUT/f'prepared_com15_right_roll_minus{round(amplitude*100):02}.npz'
    assert not target.exists()
    out=base['control_q'].copy();maxp=maxr=maxcom=0.
    for k in range(771,826):
        original=base['control_q'][k];q=original.astype(float).copy()
        p0,R0,_=legs.evaluate(original,1)
        weight=smooth((k-770)/16)*(1-smooth((k-802)/24))
        goal=Rotation.from_rotvec([-amplitude*weight,0,0]).as_matrix()@R0
        for iteration in range(160):
            p,R,J=legs.evaluate(q,1);error=np.r_[p0-p,Rotation.from_matrix(goal@R.T).as_rotvec()]
            if np.max(abs(error))<5e-7:break
            lower=np.maximum(-.05,legs.lower[6:]-q[13:19]);upper=np.minimum(.05,legs.upper[6:]-q[13:19])
            if np.any(lower>=upper):
                q[13:19]=np.clip(q[13:19],legs.lower[6:]+1e-8,legs.upper[6:]-1e-8);continue
            step=lsq_linear(np.vstack((J,.0001*np.eye(6))),np.r_[error,np.zeros(6)],bounds=(lower,upper),tol=1e-10).x
            q[13:19]+=step
        out[k]=q;p,R,_=legs.evaluate(out[k],1)
        maxp=max(maxp,float(np.linalg.norm(p-p0)))
        maxr=max(maxr,float(Rotation.from_matrix(goal@R.T).magnitude()))
        maxcom=max(maxcom,float(np.linalg.norm(model.evaluate(out[k])[0]-model.evaluate(original)[0])))
    for columns in (slice(0,13),slice(19,None)):
        np.testing.assert_array_equal(out[:,columns],base['control_q'][:,columns])
    np.testing.assert_array_equal(out[:771],base['control_q'][:771])
    np.testing.assert_array_equal(out[826:],base['control_q'][826:])
    rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max())
    report=dict(method='Offline right-foot negative world-roll target, root and other joints fixed; right foot origin preserved',
        amplitude_rad=amplitude,start_frame=770,rise_end_frame=786,hold_end_frame=802,end_frame=826,
        maximum_foot_position_residual_m=maxp,maximum_foot_rotation_residual_rad=maxr,
        maximum_com_change_m=maxcom,peak_source_joint_rmse_rad=rmse,
        source_sha256=base['source_sha256'].item(),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        preflight_passed=maxp<2e-5 and maxr<2e-4 and rmse<.19,full_tracking_passed=False)
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['preflight_passed']:
        with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
    print(target.name,report,flush=True)
