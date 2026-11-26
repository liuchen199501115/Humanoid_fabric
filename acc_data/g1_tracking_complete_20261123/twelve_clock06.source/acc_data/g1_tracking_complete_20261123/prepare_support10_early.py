"""Move tenth landing laterally earlier in its preparation; audit raw source."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
from reference_com_ik import fit_com_reference

label,parent_name=sys.argv[1:3];parent=OUT/(parent_name+'.npz');target=OUT/('prepared_support10_early_'+label+'.npz')
offset=np.array(list(map(float,sys.argv[3:6])) if len(sys.argv)>3 else [0.,-.025,0.])
assert offset.shape==(3,) and np.isfinite(offset).all() and np.linalg.norm(offset)<=.035
if target.exists():raise FileExistsError(target)
with np.load(parent) as a:base={k:a[k] for k in a.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf');legs=LegKinematics(urdf,names);model=CentroidalGeometry(urdf,names)
def smooth(u):
    u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
k=np.arange(len(source));weight=smooth((k-762)/40)*(1-smooth((k-908)/44));offsets=np.zeros((len(k),2,3));offsets[:,1]=weight[:,None]*offset
out,report=fit_com_reference(base,source,np.zeros((len(k),3)),legs,model,foot_offsets=offsets,max_iterations=200)
np.testing.assert_array_equal(out[:763],base['control_q'][:763]);np.testing.assert_array_equal(out[952:],base['control_q'][952:])
report.update(method='Offline tenth landing offset, earlier smooth onset, full-body foot/COM IK',
    start_frame=762,landing_frame=802,offset_m=offset.tolist(),source_sha256=base['source_sha256'].item(),
    parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),full_tracking_passed=False)
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
    with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
print(report,flush=True)
