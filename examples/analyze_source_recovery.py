"""Audit physical recovery toward an unchanged PKL frame, without claiming source progress."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_source_sequence import summarize_source_sequence


def summarize_source_recovery(data, source_steps=3, recovery_steps=1, fixed_frame=176):
    sequence = summarize_source_sequence(data, source_steps + recovery_steps, fixed_frame)
    sequence['scope'] = f'{source_steps + recovery_steps}_physical_steps'
    report = dict(scope='fixed_frame_recovery', recovery_passed=False,
                  source_steps=source_steps, recovery_steps=recovery_steps,
                  fixed_source_frame=fixed_frame, full_pkl_tracking=False,
                  continuous_walking=False, physical_sequence=sequence)
    required = ('history_source_recovery', 'history_source_recovery_target_feet',
                'history_source_recovery_root', 'history_q')
    if any(key not in data for key in required):
        report['reason'] = 'Missing fixed-reference recovery observations'
        return report
    active = np.asarray(data['history_source_recovery']).ravel().astype(bool)
    indices = np.flatnonzero(active)
    if not len(indices):
        report['reason'] = 'No recovery observed'
        return report
    first = indices[0]
    t = np.asarray(data['history_time']).ravel()
    baseline = (t >= t[first] - .2) & (t < t[first])
    final = t >= t[-1] - 1. + 1e-7
    if baseline.sum() < 50 or final.sum() < 499:
        report['reason'] = 'Insufficient baseline or final observations'
        return report
    targets = np.asarray(data['history_source_recovery_target_feet']).reshape(-1,2,7)
    roots = np.asarray(data['history_source_recovery_root']).reshape(-1,3)
    fixed = bool(np.isfinite(targets[active]).all() and np.isfinite(roots[active]).all()
                 and np.allclose(targets[active], targets[first], atol=1e-7, rtol=0)
                 and np.allclose(roots[active], roots[first], atol=1e-7, rtol=0))
    frame = np.asarray(data['history_frame']).ravel()
    clock_held = bool(np.all(np.abs(frame[first:] - fixed_frame) <= 1e-6))
    numbers = np.asarray(data['history_source_step_number']).ravel()
    counted = bool(np.array_equal(np.unique(numbers[active]),
                                 np.arange(source_steps+1, source_steps+recovery_steps+1))
                   and active[first:].all())
    feet = np.asarray(data['history_feet']).reshape(-1,2,7)
    q = np.asarray(data['history_q']).reshape(len(t), -1)
    errors = np.linalg.norm(feet[:,:,:2] - targets[first,:,:2], axis=-1)
    root_errors = np.linalg.norm(q[:,:2] - roots[first,:2], axis=-1)
    before, after = errors[baseline].mean(axis=0), errors[final].mean(axis=0)
    rms_before, rms_after = float(np.sqrt(np.mean(before**2))), float(np.sqrt(np.mean(after**2)))
    root_before, root_after = float(root_errors[baseline].mean()), float(root_errors[final].mean())
    improved = bool(rms_before-rms_after >= .05 and root_before-root_after >= .05
                    and np.all(after <= before + .02))
    report.update(recovery_passed=bool(sequence['sequence_passed'] and fixed and clock_held and counted and improved),
                  reference_unchanged=fixed, source_clock_held=clock_held, recovery_count_verified=counted,
                  initial_foot_xy_error_m=before.tolist(), final_foot_xy_error_m=after.tolist(),
                  initial_foot_rms_error_m=rms_before, final_foot_rms_error_m=rms_after,
                  initial_root_xy_error_m=root_before, final_root_xy_error_m=root_after,
                  error_reduction_verified=improved,
                  fixed_target_reached=bool(fixed and max(after.max(), root_after) <= .05))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--source-steps', type=int, default=3)
    parser.add_argument('--recovery-steps', type=int, default=1)
    parser.add_argument('--fixed-frame', type=int, default=176)
    args = parser.parse_args()
    with np.load(args.recording) as data:
        report = summarize_source_recovery(data, args.source_steps, args.recovery_steps, args.fixed_frame)
    output = json.dumps(report, indent=2) + '\n'
    args.recording.with_suffix('.recovery.json').write_text(output)
    print(output, end='')
    raise SystemExit(0 if report['recovery_passed'] else 1)
