"""Combine early left touchdown and smoother right swing using world-pose IK."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
from reference_com_ik import fit_com_reference
parent=OUT/'prepared_early_landing9_full_f8_z15.npz'
with np.load(parent) as data: base={k:data[k] for k in data.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs=LegKinematics(urdf,names); model=CentroidalGeometry(urdf,names)
for end in (802,812):
    target=OUT/f'prepared_early9_full_f8_z15_swing10_{end}.npz'
    if target.exists(): raise FileExistsError(target)
    right_parent=OUT/f'prepared_swing10_smooth_{end}.npz'
    right=np.load(right_parent)['control_q']
    offsets=np.zeros((len(source),2,3)); orientations=np.tile(np.eye(3),(len(source),2,1,1))
    for k in range(731,end):
        goals=[legs.evaluate(base['control_q'][k],s)[:2] for s in (0,1)]
        p,R,_=legs.evaluate(right[k],1)
        offsets[k,1]=p-goals[1][0]
        orientations[k,0]=goals[0][1];orientations[k,1]=R
    out,report=fit_com_reference(base,source,np.zeros((len(source),3)),legs,model,
        foot_offsets=offsets,foot_rotations=orientations,max_iterations=600)
    report.update(method='Earlier left touchdown plus smooth right swing, full-body offline IK',
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),right_parent_sha256=hashlib.sha256(right_parent.read_bytes()).hexdigest(),
        source_sha256=base['source_sha256'].item(),full_tracking_passed=False)
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['preflight_passed']:
        with target.open('xb') as stream: np.savez_compressed(stream,**dict(base,control_q=out))
    print(target.name,report,flush=True)
