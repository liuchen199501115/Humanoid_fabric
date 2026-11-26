"""Bounded pose-reference offsets to leave margin at observed mechanical stops.

These are control-reference candidates only. Original PKL joint/root fidelity,
native clearance and original landing checks remain authoritative.
"""
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
parent=OUT/'prepared_combined_root_y50.npz'
with np.load(parent) as data:base={k:data[k] for k in data.files}
source=np.load(ROOT/'acc_data/g1_tracking_contacts_20261123/eight_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf';legs=LegKinematics(str(urdf),names)
limits={j.get('name'):[float(j.find('limit').get(key)) for key in ('lower','upper')] for j in ET.parse(urdf).getroot().findall('joint') if j.find('limit') is not None}
lower,upper=np.array([limits[name] for name in names]).T
def smooth(u):
    u=np.clip(u,0,1);return u**3*(10-15*u+6*u*u)
k=np.arange(len(source));waist=smooth((k-730)/60)*(1-smooth((k-850)/110));ankle=smooth((k-780)/30)*(1-smooth((k-842)/42))
for label,waist_offset,pitch_offset,roll_offset in [('waist10',-.1,0.,0.),('ankle10_8',0.,.1,-.08),('combined10',-.1,.1,-.08)]:
    target=OUT/('prepared_stop_margin_'+label+'.npz')
    if target.exists():raise FileExistsError(target)
    out=base['control_q'].copy()
    for name,offset,weight in [('waist_roll_joint',waist_offset,waist),('left_ankle_pitch_joint',pitch_offset,ankle),('left_ankle_roll_joint',roll_offset,ankle)]:out[:,7+names.index(name)]+=offset*weight
    max_position=max_rotation=0.
    for frame in range(730,961):
        for side in (0,1):
            p,R,_=legs.evaluate(out[frame],side);p0,R0,_=legs.evaluate(base['control_q'][frame],side)
            max_position=max(max_position,float(np.linalg.norm(p-p0)));max_rotation=max(max_rotation,float(Rotation.from_matrix(R@R0.T).magnitude()))
    limit_error=float(np.maximum(lower-out[:,7:],out[:,7:]-upper).max())
    rmse=float(np.sqrt(np.mean((out[:,7:]-source[:,7:])**2,axis=1)).max())
    report=dict(method='C2 waist/ankle posture margin around measured step-ten stop events',
        waist_roll_offset_rad=waist_offset,left_ankle_pitch_offset_rad=pitch_offset,left_ankle_roll_offset_rad=roll_offset,
        maximum_control_foot_position_change_m=max_position,maximum_control_foot_rotation_change_rad=max_rotation,
        maximum_joint_limit_violation_rad=limit_error,peak_source_joint_rmse_rad=rmse,
        source_sha256=base['source_sha256'].item(),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
        full_tracking_passed=False,preflight_passed=limit_error<1e-6 and rmse<.19 and max_position<.003 and max_rotation<.15)
    np.testing.assert_array_equal(out[:731],base['control_q'][:731]);np.testing.assert_array_equal(out[960:],base['control_q'][960:]);np.testing.assert_array_equal(out[:,:7],base['control_q'][:,:7])
    target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['preflight_passed']:
        with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
    print(target.name,report,flush=True)
