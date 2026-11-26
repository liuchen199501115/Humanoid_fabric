"""Prepare small lateral COM changes, preserving both control foot poses.

Offline bounded IK only. Original source and closed-loop acceptance are unchanged.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
from reference_com_ik import fit_com_reference
parent=OUT/'prepared_freeleg80_waist30_swing10lower25.npz'
if '--parent' in sys.argv:
 index=sys.argv.index('--parent');parent=OUT/(sys.argv[index+1]+'.npz')
 del sys.argv[index:index+2]
with np.load(parent) as a:base={k:a[k] for k in a.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs=LegKinematics(urdf,names);model=CentroidalGeometry(urdf,names)
def smooth(u):
    u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
for offset in map(float,sys.argv[1:] or ['.015','.030']):
    target=OUT/f'prepared_margin30_turn_com_y{round(offset*1000)}.npz'
    if parent.stem!='prepared_freeleg80_waist30_swing10lower25':
        target=OUT/f'{parent.stem}_com_y{round(offset*1000)}.npz'
    assert not target.exists()
    frames=np.arange(len(source));delta=np.zeros((len(source),3))
    delta[:,1]=offset*smooth((frames-680)/80)*(1-smooth((frames-900)/100))
    out,report=fit_com_reference(base,source,delta,legs,model,max_iterations=600)
    np.testing.assert_array_equal(out[:681],base['control_q'][:681])
    np.testing.assert_array_equal(out[1000:],base['control_q'][1000:])
    report.update(method='Offline bounded lateral COM offset with both foot poses preserved',
        offset_y_m=offset,start_frame=680,rise_end_frame=760,hold_end_frame=900,end_frame=1000,
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        source_sha256=base['source_sha256'].item(),full_tracking_passed=False)
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['preflight_passed']:
        with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
    print(target.name,report,flush=True)
