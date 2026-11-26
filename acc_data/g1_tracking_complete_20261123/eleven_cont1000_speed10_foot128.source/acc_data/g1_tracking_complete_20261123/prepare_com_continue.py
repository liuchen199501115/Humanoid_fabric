"""Continue a bounded lateral COM correction with lower height and fixed feet."""
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
parent=OUT/'prepared_combined_root_y50.npz'
with np.load(parent) as a:base={k:a[k] for k in a.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf');legs=LegKinematics(urdf,names);model=CentroidalGeometry(urdf,names)
def smooth(u):
    u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
for y,z,label in [(.035,-.015,'y35_z15'),(.05,-.025,'y50_z25')]:
    target=OUT/('prepared_com_continue_'+label+'.npz')
    if target.exists():raise FileExistsError(target)
    t=np.arange(len(source))/200;weight=smooth((t-3.2)/.5)*(1-smooth((t-4.5)/.5))
    delta=weight[:,None]*np.array([0,y,z]);out,report=fit_com_reference(base,source,delta,legs,model,max_iterations=200)
    report.update(method='Offline continued lateral COM correction with lowered height and fixed foot poses',
        offset_m=[0,y,z],hold_until_source_s=4.5,source_sha256=base['source_sha256'].item(),
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),full_tracking_passed=False)
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['preflight_passed']:
        with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
    print(report,flush=True)
