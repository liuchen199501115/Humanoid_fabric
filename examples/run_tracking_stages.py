"""Run a fixed standing -> transfer baseline; never certify full PKL tracking."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from analyze_contact_transfer import summarize_transfer


def summarize_standing(data):
    result = dict(scope='initial_standing', standing_passed=False, full_pkl_tracking=False)
    required = ('history_time', 'history_frame', 'history_commanded_mode',
                'history_com', 'history_com_target', 'history_foot_forces')
    if any(key not in data for key in required):
        return dict(result, reason='Missing standing observations')
    t = np.asarray(data['history_time']).ravel()
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        return dict(result, reason='Invalid control clock')
    frame = np.asarray(data['history_frame']).ravel()
    com = np.asarray(data['history_com']).reshape(-1, 3)
    target = np.asarray(data['history_com_target']).reshape(-1, 3)
    force = np.asarray(data['history_foot_forces']).reshape(-1, 2, 3)[..., 2]
    mode = np.asarray(data['history_commanded_mode']).ravel()
    if any(len(x) != len(t) or not np.isfinite(x).all() for x in (frame, com, target, force, mode)):
        return dict(result, reason='Invalid standing observations')
    recent = t >= t[-1]-1.+1e-7
    speed = np.linalg.norm(np.diff(com, axis=0)/np.diff(t)[:, None], axis=1)
    error = np.linalg.norm(com-target, axis=1)
    controller = bool(np.all(data.get('motion_tracking_valid', False)))
    result.update(duration_s=float(t[-1]-t[0]), final_source_frame=float(frame[-1]),
                  peak_com_error_m=float(error.max()),
                  final_second_min_foot_force_n=force[recent].min(axis=0).tolist(),
                  final_second_max_com_speed_m_s=float(speed[recent[1:]].max()),
                  controller_valid=controller)
    result['standing_passed'] = bool(t[-1]-t[0] >= 3.99 and recent.sum() >= 499
        and np.max(np.diff(t)) <= .00201 and np.ptp(frame) < 1e-6
        and np.all(mode == 3) and error.max() <= .03
        and force[recent].min() >= 20. and speed[recent[1:]].max() <= .02 and controller)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--first-step-profile', type=Path,
                        help='After transfer passes, audit this first-step profile without execution')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    common = [sys.executable, str(root/'examples/test_g1.py'),
              '--device', 'cpu', '--motion-pkl', 'retarget/lafan/walk1_g1.pkl',
              '--reference-mode', 'unified', '--reference-manager', '--reference-footsteps', 'source',
              '--motion-interpolation', 'pchip-analytic', '--motion-ramp-duration', '.25',
              '--direct-wbc', '--control-frequency', '500', '--source-pelvis-weight', '1',
              '--observed-contact-patches', '--early-touchdown', '--align-takeoff-frame',
              '--joint-limit-avoidance-weight', '20', '--joint-limit-barrier-bandwidth', '8',
              '--tracking-servo', '--contact-aware-servo', '--nominal-implicit-servo',
              '--qp-solver', 'daqp', '--wrist-hip-avoidance', '--wrist-hip-barrier', '--wrist-body-avoidance',
              '--report-self-contacts', '--diagnose-com', '--motor-armature',
              '--flat-foot-spawn', '--motion-warmup-duration', '5', '--source-arm-roll-offset', '0',
              '--diagnose-qp-time', '0', '--diagnose-qp-time', '.012']
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env['PYTHONPATH'] = os.pathsep.join((str(root/'src'), str(root/'examples'),
        str(root/'.runtime/qp_solvers'), env.get('PYTHONPATH', '')))
    manifest = dict(full_pkl_tracking=False, stages=[],
                    policy='Stop on first failed stage; unchanged control configuration across stages')
    for name, options, audit in (
            ('standing', ['--duration', '4'], summarize_standing),
            ('transfer', ['--duration', '16', '--unified-transfer-test'], summarize_transfer)):
        record = out/(name+'.npz')
        command = common+options+['--output-path', str(record)]
        row = dict(stage=name, command=command, passed=False)
        manifest['stages'].append(row)
        (out/'status.json').write_text(json.dumps(manifest, indent=2)+'\n')
        with (out/(name+'.log')).open('w') as log:
            completed = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        row['returncode'] = completed.returncode
        if completed.returncode == 0 and record.exists():
            try:
                with np.load(record, allow_pickle=False) as data:
                    row['audit'] = audit(data)
                row['passed'] = row['audit'][name+'_passed']
            except (ValueError, KeyError, IndexError) as exc:
                row['audit_error'] = str(exc)
        row['artifacts'] = [str(p) for p in sorted(out.glob(name+'*'))]
        (out/'status.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(row, indent=2), flush=True)
        if not row['passed']:
            return 1
    if args.first_step_profile is not None:
        report_path = out/'first_step.json'
        command = [sys.executable, str(root/'examples/plan_tracking_first_step.py'),
                   str(out/'transfer.npz'), '--profile', str(args.first_step_profile.resolve()),
                   '--output', str(report_path)]
        row = dict(stage='first_step_preflight', command=command, passed=False)
        manifest['stages'].append(row)
        (out/'status.json').write_text(json.dumps(manifest, indent=2)+'\n')
        with (out/'first_step.log').open('w') as log:
            completed = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        row['returncode'] = completed.returncode
        if report_path.exists():
            row['audit'] = json.loads(report_path.read_text())
            row['passed'] = bool(completed.returncode == 0 and row['audit']['first_step_preflight_passed'])
        (out/'status.json').write_text(json.dumps(manifest, indent=2)+'\n')
        if not row['passed']:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
