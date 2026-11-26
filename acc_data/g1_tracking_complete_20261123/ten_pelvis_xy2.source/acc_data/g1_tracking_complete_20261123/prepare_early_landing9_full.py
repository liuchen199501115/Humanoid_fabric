"""Prepare genuinely earlier touchdown with both feet and COM constrained."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
from reference_com_ik import fit_com_reference
parent=OUT/'prepared_combined_root_y50.npz'
with np.load(parent) as data: base={k:data[k] for k in data.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs=LegKinematics(urdf,names); model=CentroidalGeometry(urdf,names)
def smooth(u):
    u=np.clip(u,0,1); return u**3*(10-15*u+6*u*u)
start,land,end=730,762,782
poses=[legs.evaluate(base['control_q'][k],0)[:2] for k in range(start,end+1)]
positions=np.array([p for p,R in poses])
rotations=Slerp(np.arange(start,end+1),Rotation.from_matrix([R for p,R in poses]))
for lead in map(int,sys.argv[1:] or [4,8,12]):
    target=OUT/f'prepared_early_landing9_full_f{lead}.npz'
    if target.exists(): raise FileExistsError(target)
    offsets=np.zeros((len(source),2,3)); orientations=np.tile(np.eye(3),(len(source),2,1,1))
    stop=land-lead; join=stop-12
    for k in range(start+1,end):
        if k<join: f=k+lead*smooth((k-start)/8)
        elif k<stop:
            u=(k-join)/12; f=land-12+12*(u+4*u**3-7*u**4+3*u**5)
        else: f=float(land)
        j=min(len(positions)-2,int(f)-start); alpha=f-(start+j)
        p=(1-alpha)*positions[j]+alpha*positions[j+1]; R=rotations(f).as_matrix()
        oldp,oldR,_=legs.evaluate(base['control_q'][k],0)
        if k>land:
            blend=smooth((k-land)/(end-land));p+=blend*(oldp-p)
            R=Rotation.from_rotvec(blend*Rotation.from_matrix(oldR@R.T).as_rotvec()).as_matrix()@R
        offsets[k,0]=p-oldp; orientations[k,0]=R
        orientations[k,1]=legs.evaluate(base['control_q'][k],1)[1]
    out,report=fit_com_reference(base,source,np.zeros((len(source),3)),legs,model,
        foot_offsets=offsets,foot_rotations=orientations,max_iterations=600)
    np.testing.assert_array_equal(out[:start+1],base['control_q'][:start+1])
    np.testing.assert_array_equal(out[end:],base['control_q'][end:])
    report.update(method='Earlier touchdown, both feet and COM constrained offline IK',advance_frames=lead,
        target_touchdown_frame=stop,original_touchdown_frame=land,rejoin_frame=end,
        parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),full_tracking_passed=False)
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['preflight_passed']:
        with target.open('xb') as stream: np.savez_compressed(stream,**dict(base,control_q=out))
    print(target.name,report,flush=True)
