"""Offline stance-foot flattening with native sole height and preserved COM.

Preserves the left foot and the right-foot XY path. No physical geometry,
friction coefficient, source audit target, or runtime state is changed.
"""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
from reference_com_ik import fit_com_reference
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator

label,parent_name=sys.argv[1:3]
parent=OUT/(parent_name+'.npz');target=OUT/('prepared_stance10_flat_'+label+'.npz')
assert not target.exists()
base=dict(np.load(parent));source=np.load(OUT/'ten_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest()==base['source_sha256'].item()
names=np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf=str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs,model=LegKinematics(urdf,names),CentroidalGeometry(urdf,names)
geometry=json.loads((OUT/'ten_verify.steps_plan.json').read_text())
points=np.asarray(geometry['collision_points'][1]);radii=np.asarray(geometry['collision_radii'][1])
def smooth(x):
    x=np.clip(x,0,1);return x**3*(10-15*x+6*x*x)
frames=np.arange(len(source));blend=smooth((frames-822)/28)*(1-smooth((frames-884)/24))
offsets=np.zeros((len(source),2,3));rotations=np.zeros((len(source),2,3,3))
for k,q in enumerate(base['control_q']):
    for side in (0,1):
        position,rotation,_=legs.evaluate(q,side);rotations[k,side]=rotation
        if side==1 and blend[k]:
            yaw=np.arctan2(rotation[1,0],rotation[0,0])
            flat=Rotation.from_euler('z',yaw).as_matrix()
            rotations[k,side]=Rotation.from_rotvec(blend[k]*Rotation.from_matrix(flat@rotation.T).as_rotvec()).as_matrix()@rotation
            flat_height=.001-float(np.min((points@flat.T)[:,2]-radii))
            offsets[k,side,2]=blend[k]*(flat_height-position[2])
out,report=fit_com_reference(base,source,np.zeros((len(source),3)),legs,model,
    foot_offsets=offsets,foot_rotations=rotations,fixed_root=False,max_iterations=600)
np.testing.assert_array_equal(out[:823],base['control_q'][:823])
np.testing.assert_array_equal(out[908:],base['control_q'][908:])
def sampler(q):
    t=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(t),t,stride=6,start_frame=130,ramp_frames=20)
old,new=sampler(base['control_q']),sampler(out)
for k in np.arange(0,816.01,.5):
    for a,b in zip(old.kinematics(k,240.,0.),new.kinematics(k,240.,0.)):
        np.testing.assert_array_equal(a.numpy(),b.numpy())
report.update(method=__doc__,parent_file=parent.name,
    parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256=base['source_sha256'].item(),
    maximum_foot_height_change_m=float(abs(offsets[:,:,2]).max()),prefix_q_v_a_equal_through_frame=816,
    full_tracking_passed=False)
target.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
if report['preflight_passed']:
    with target.open('xb') as stream:np.savez_compressed(stream,**dict(base,control_q=out))
print(target.name,report,flush=True)
