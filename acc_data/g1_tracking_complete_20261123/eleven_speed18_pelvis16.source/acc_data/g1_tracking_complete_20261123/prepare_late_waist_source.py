"""Offline source-waist restoration preserving COM, both feet and root attitude.

This is a candidate reference only. Original PKL and runtime acceptance are
unchanged. Geometric preflight does not establish dynamic tracking feasibility.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics
from numpy_centroidal_geometry import CentroidalGeometry
from reference_com_ik import fit_com_reference
from blended_spline import BlendedSplineReference
from fabrics_sim.reference.interpolation import MotionInterpolator

label, parent_name, fraction = sys.argv[1], sys.argv[2], float(sys.argv[3])
fixed_root = '--free-root-attitude' not in sys.argv
assert 0 < fraction <= 1
parent = OUT/(parent_name+'.npz')
target = OUT/('prepared_late_waist_source_'+label+'.npz')
assert not target.exists()
base = dict(np.load(parent))
source = np.load(OUT/'ten_verify.control_reference.npz')['source_q']
assert hashlib.sha256(source.tobytes()).hexdigest() == base['source_sha256'].item()
names = np.load(OUT/'ten_verify.npz')['final_joint_order'].tolist()
urdf = str(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
legs, model = LegKinematics(urdf, names), CentroidalGeometry(urdf, names)

def smooth(x):
    x = np.clip(x, 0, 1)
    return x**3*(10-15*x+6*x*x)

frames = np.arange(len(source))
envelope = fraction*smooth((frames-822)/60)*(1-smooth((frames-940)/60))
modified = base['control_q'].copy()
modified[:, 19:22] += envelope[:, None]*(source[:, 19:22]-modified[:, 19:22])
delta = np.zeros((len(source), 3))
for k in np.flatnonzero(envelope):
    delta[k] = model.evaluate(base['control_q'][k])[0]-model.evaluate(modified[k])[0]
out, report = fit_com_reference(dict(base, control_q=modified), source, delta,
    legs, model, fixed_root=fixed_root, max_iterations=600)
np.testing.assert_array_equal(out[:823], base['control_q'][:823])
np.testing.assert_array_equal(out[1000:], base['control_q'][1000:])
if fixed_root:
    np.testing.assert_array_equal(out[:, 3:7], base['control_q'][:, 3:7])

def sampler(q):
    tensor = torch.tensor(q)
    return BlendedSplineReference(MotionInterpolator(tensor), tensor,
        stride=6, start_frame=130, ramp_frames=20)

old, new = sampler(base['control_q']), sampler(out)
for k in np.arange(0, 816.01, .5):
    for a, b in zip(old.kinematics(k, 240., 0.), new.kinematics(k, 240., 0.)):
        np.testing.assert_array_equal(a.numpy(), b.numpy())
report.update(method='Offline source-waist restoration preserving COM and both feet; root attitude '+('fixed' if fixed_root else 'bounded by existing IK'), parent_file=parent.name,
    parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
    source_sha256=base['source_sha256'].item(), waist_restoration_fraction=fraction,
    prefix_q_v_a_equal_through_frame=816, full_tracking_passed=False)
target.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
if report['preflight_passed']:
    with target.open('xb') as stream:
        np.savez_compressed(stream, **dict(base, control_q=out))
print(target.name, report, flush=True)
