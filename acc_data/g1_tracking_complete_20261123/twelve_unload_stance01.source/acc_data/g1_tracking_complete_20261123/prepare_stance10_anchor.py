"""Offline bounded stance-foot anchoring after verified landing ten.

Uses the recorded loaded pose at source816. No simulator reset or runtime
state overwrite; original-source fidelity and landing checks stay unchanged.
"""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
from reference_com_ik import fit_com_reference
label=sys.argv[1];rotation_fraction=float(sys.argv[2])
start_frame=int(sys.argv[3]) if len(sys.argv)>3 else 822
assert start_frame in (802,822)
assert 0<=rotation_fraction<=1
parent=OUT/'prepared_support10_early_com15_y20.npz'
if '--parent' in sys.argv:parent=OUT/(sys.argv[sys.argv.index('--parent')+1]+'.npz')
hold_frame=(int(sys.argv[sys.argv.index('--hold-frame')+1]) if '--hold-frame' in sys.argv else 860)
end_frame=(int(sys.argv[sys.argv.index('--end-frame')+1]) if '--end-frame' in sys.argv else 884)
com_drop=(float(sys.argv[sys.argv.index('--com-drop')+1]) if '--com-drop' in sys.argv else 0.)
fixed_root='--fixed-root' in sys.argv
assert start_frame+28<=hold_frame<end_frame<=964 and 0<=com_drop<=.04
target=OUT/('prepared_stance10_anchor_'+label+'.npz');assert not target.exists()
base=dict(np.load(parent));history=np.load(OUT/'ten_support10_com15_y20.npz')
source=np.load(OUT/'ten_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=history['final_joint_order'].tolist();urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs=LegKinematics(urdf,names);model=CentroidalGeometry(urdf,names)
index=int(np.argmin(abs(history['history_frame'].ravel()-816)))
anchor,anchor_rotation,_=legs.evaluate(history['history_q'][index,0],1)
def smooth(x):
 x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
frames=np.arange(len(source));blend=smooth((frames-start_frame)/28)*(1-smooth((frames-hold_frame)/(end_frame-hold_frame)))
offsets=np.zeros((len(source),2,3));rotations=np.zeros((len(source),2,3,3))
for k,q in enumerate(base['control_q']):
 for side in [0,1]:
  pos,rot,_=legs.evaluate(q,side);rotations[k,side]=rot
  if side==1:
   offsets[k,side]=blend[k]*(anchor-pos)
   rotations[k,side]=Rotation.from_rotvec(blend[k]*rotation_fraction*Rotation.from_matrix(anchor_rotation@rot.T).as_rotvec()).as_matrix()@rot
com_delta=np.zeros((len(source),3));com_delta[:,2]=-com_drop*blend
out,report=fit_com_reference(base,source,com_delta,legs,model,
                             foot_offsets=offsets,foot_rotations=rotations,max_iterations=600,fixed_root=fixed_root)
np.testing.assert_array_equal(out[:start_frame+1],base['control_q'][:start_frame+1])
report.update(method=__doc__,rotation_fraction=rotation_fraction,anchor_position=anchor.tolist(),
 anchor_rotation=anchor_rotation.tolist(),observed_anchor_source_frame=float(history['history_frame'][index]),
 maximum_foot_position_correction_m=float(np.linalg.norm(offsets,axis=-1).max()),
 start_frame=start_frame,rise_end_frame=start_frame+28,fade_start_frame=hold_frame,fade_end_frame=end_frame,
 additional_com_drop_m=com_drop,fixed_root_rotation=fixed_root,parent_file=parent.name,
 parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),full_tracking_passed=False)
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
 with target.open('xb') as f:np.savez_compressed(f,**dict(base,control_q=out))
print(target.name,report,flush=True)
