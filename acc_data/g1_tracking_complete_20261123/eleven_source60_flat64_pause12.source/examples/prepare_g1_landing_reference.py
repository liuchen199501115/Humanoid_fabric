"""Prepare a smooth landing correction in full-body poses, keeping the raw audit.

Only the selected leg changes. The pelvis, other foot, foot orientation and
unaffected source frames stay fixed. This is offline IK, not a runtime WBC.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'acc_data/amp_walk_20261103/feedback_followup'))
from numpy_leg_kinematics import LegKinematics


def correct_landing(model, poses, event, next_event, offset):
    side = 1 - event['support']
    start, land = event['takeoff_frame'], event['end_frame']
    release, end = next_event['takeoff_frame'], next_event['end_frame']
    offset = np.asarray(offset, dtype=float)
    if (1 - next_event['support'] != side or
            not 0 <= start < land <= release < end < len(poses) or
            offset.shape != (3,) or not np.isfinite(offset).all() or
            not np.isfinite(poses).all()):
        raise ValueError('Invalid landing correction inputs')
    result = poses.copy()
    joint_slice = slice(7 + 6 * side, 13 + 6 * side)
    lower, upper = model.lower[6*side:6*side+6], model.upper[6*side:6*side+6]
    worst_pos = worst_rot = 0.

    def smooth(x):
        x = np.clip(x, 0., 1.)
        return x**3 * (10 - 15*x + 6*x*x)

    for frame in range(start + 1, end):
        weight = smooth((frame-start)/(land-start)) * (1-smooth((frame-release)/(end-release)))
        q = poses[frame].astype(float).copy()
        target, target_rotation, _ = model.evaluate(q, side)
        target += weight * offset
        for _ in range(100):
            position, rotation, jacobian = model.evaluate(q, side)
            error = np.r_[target-position, Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()]
            if np.linalg.norm(error[:3]) < 2e-7 and np.linalg.norm(error[3:]) < 2e-6:
                break
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-8*np.eye(6), error)
            delta *= min(1., .05/max(np.abs(delta).max(), 1e-12))
            q[joint_slice] = np.clip(q[joint_slice]+delta, lower, upper)
        result[frame] = q
        # Check the saved precision, not just the double precision iterate.
        actual, rotation, _ = model.evaluate(result[frame], side)
        pe = float(np.linalg.norm(target-actual))
        re = float(Rotation.from_matrix(target_rotation @ rotation.T).magnitude())
        if pe > 2e-5 or re > 2e-4:
            raise ValueError(f'Landing IK failed at source frame {frame}: position={pe}, rotation={re}')
        worst_pos, worst_rot = max(worst_pos, pe), max(worst_rot, re)
    return result, dict(start_frame=start, landing_frame=land, release_frame=release,
                       end_frame=end, side=side, offset_m=offset.tolist(),
                       peak_position_residual_m=worst_pos,
                       peak_orientation_residual_rad=worst_rot)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--verified-reference', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--offset', type=float, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Never overwrite a prepared reference or its audit')
    with np.load(args.candidate) as data:
        candidate = {k: data[k] for k in data.files}
    with np.load(args.verified_reference) as data:
        source = data['source_q']
        if not np.array_equal(candidate['control_q'], data['control_q']):
            raise ValueError('Candidate differs from the verified control reference')
        if hashlib.sha256(source.tobytes()).hexdigest() != candidate['source_sha256'].item():
            raise ValueError('Original PKL hash mismatch')
    with np.load(args.history) as history:
        names = history['joint_names'].tolist()
    model = LegKinematics(ROOT/'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf', names)
    events = json.loads(args.plan.read_text())['events']
    event = events[args.step-1]
    next_event = next(e for e in events[args.step:] if e['support'] == event['support'])
    corrected, report = correct_landing(model, candidate['control_q'], event, next_event, args.offset)
    report.update(parent=str(args.candidate), parent_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  original_source_sha256=candidate['source_sha256'].item(),
                  peak_source_joint_rmse_rad=float(np.sqrt(np.mean((corrected[:,7:]-source[:,7:])**2, axis=1)).max()),
                  root_unchanged=bool(np.array_equal(corrected[:,:7], candidate['control_q'][:,:7])),
                  original_pkl_retained_for_audit=True, physical_tracking_certified=False)
    if report['peak_source_joint_rmse_rad'] >= .2:
        raise ValueError('Corrected reference exceeds original source joint tolerance')
    candidate['control_q'] = corrected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as output:
        np.savez_compressed(output, **candidate)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
