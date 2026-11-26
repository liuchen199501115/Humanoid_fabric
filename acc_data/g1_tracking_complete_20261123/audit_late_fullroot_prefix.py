"""Check unchanged runtime q/v/a through the tenth landing, before source816."""
import json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator
from scipy.spatial.transform import Rotation

def sampler(q):
 q=torch.tensor(q);return BlendedSplineReference(MotionInterpolator(q),q,stride=6,start_frame=130,ramp_frames=20)
base=np.load(OUT/'prepared_margin30_turn_com_y15.npz')['control_q'];original=sampler(base)
source=np.load(OUT/'ten_verify.control_reference.npz')['source_q'];records=[]
for scale in [60,90]:
 name=f'prepared_fullroot_dynamics_com15_after816_cop1_scale{scale}';q=np.load(OUT/(name+'.npz'))['control_q'];new=sampler(q)
 np.testing.assert_array_equal(q[:823],base[:823]);np.testing.assert_array_equal(q[1014:],base[1014:])
 frames=np.unique(np.r_[np.arange(0,816.01,.5),814.52,815.9999])
 for frame in frames:
  for old,modified in zip(original.kinematics(float(frame),240.,0.),new.kinematics(float(frame),240.,0.)):
   np.testing.assert_array_equal(old.numpy(),modified.numpy())
 angle=(Rotation.from_quat(q[:,3:7])*Rotation.from_quat(source[:,3:7]).inv()).magnitude()
 records.append(dict(candidate=name,exact_runtime_prefix_last_frame=816.,samples=len(frames),peak_source_root_orientation_rad=float(angle.max()),full_tracking_passed=False))
(OUT/'late_fullroot_prefix_verification.json').write_text(json.dumps(records,indent=2)+'\n');print(records)
