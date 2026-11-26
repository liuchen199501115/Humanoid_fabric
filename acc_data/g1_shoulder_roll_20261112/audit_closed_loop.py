"""Reload original PKL, verify measured landings, and compare shoulder errors."""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent / 'closed_loop'
sys.path.insert(0, str(ROOT / 'examples'))
from inspect_mocap_reference import urdf_fk
from analyze_bounded_steps import summarize as summarize_steps
spec = importlib.util.spec_from_file_location('prefix_audit',
    ROOT / 'acc_data/amp_walk_20261102/contact_clock_followup/summarize_trials.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT = OUT
source = ROOT / 'retarget/amp_reference/amp_walk_1ms.pkl'
raw = pickle.loads(source.read_bytes())
input_report = json.loads((ROOT / 'acc_data/amp_walk_20261102/input_report.json').read_text())
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash == input_report['source_sha256']
tree = ast.parse((ROOT / 'examples/test_g1_two_tasks.py').read_text())
names = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'G1_JOINT_NAMES' for t in n.targets))
order = [input_report['source_joint_order'].index(n) for n in names]
shoulders = [names.index(s + '_shoulder_roll_joint') for s in ('left', 'right')]
raw_q = np.concatenate((raw['root_pos'], raw['root_rot'][:, [1, 2, 3, 0]], raw['dof_pos'][:, order]), 1).astype(np.float32)
raw_q[:, 3:7] /= np.linalg.norm(raw_q[:, 3:7], axis=1, keepdims=True)
node = next(n for n in ast.parse((ROOT / 'examples/test_g1.py').read_text()).body
            if getattr(n, 'name', None) == 'interpolate_motion_trajectory')
namespace = {'torch': torch, 'math': math}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'source_sampler', 'exec'), namespace)
rows = []
baseline_path = OUT / 'baseline_live.npz'
baseline_q = baseline_times = None
if baseline_path.exists():
    with np.load(baseline_path) as d:
        baseline_q, baseline_times = d['history_q'].reshape(-1,36), d['history_time'].ravel()
for manifest_path in sorted(OUT.glob('*.command.json')):
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'finished':
        continue
    name = manifest_path.name.removesuffix('.command.json')
    tracking_path = OUT / (name + '.tracking.json')
    if not tracking_path.exists():
        rows.append(dict(run=name, returncode=manifest['returncode'], full_motion_tracking_passed=False,
                         audit_unavailable='No tracking report; inspect launch log'))
        continue
    tracking = json.loads(tracking_path.read_text())
    report = module.summarize(name)
    saved_entry = OUT / 'sources' / (manifest['sha256']['examples/test_g1.py'] + '_test_g1.py')
    tree = ast.parse(saved_entry.read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                      and any(isinstance(target,ast.Name) and target.id=='shoulder_weight' for target in n.targets))
    report['weight_clock'] = assignment.value.args[0].id
    report.update(returncode=manifest['returncode'], command=manifest['command'],
                  stop_reason=tracking.get('stop_reason'), failed_time_s=tracking.get('failed_time_s'))
    with np.load(report['recording'], allow_pickle=False) as data:
        options = json.loads(data['controller_options'].item())
        report['execution'] = {key: options.get(key) for key in (
            'tracking_profile', 'direct_wbc', 'tracking_servo', 'track_full_motion', 'tracking_steps',
            'stop_at_fidelity_loss', 'shoulder_roll_weight_scale', 'shoulder_roll_weight_start',
            'shoulder_roll_weight_ramp', 'shoulder_roll_weight_side')}
        report['execution'].update({key:options.get(key) for key in (
            'shoulder_roll_guard','shoulder_roll_guard_relative','shoulder_roll_guard_slack')})
        report['execution'].update({key:options.get(key) for key in (
            'com_velocity_soft_weight','com_velocity_soft_horizon','com_velocity_soft_contraction',
            'com_velocity_soft_onset','com_velocity_soft_width')})
        assert options['track_full_motion'] and options['tracking_steps'] == 0 and options['stop_at_fidelity_loss']
        times, frames = data['history_time'].ravel(), data['history_frame'].ravel()
        trajectory = torch.from_numpy(raw_q.copy())
        trajectory[:, 2] += float(data['motion_ground_shift'])
        reference = torch.cat([namespace['interpolate_motion_trajectory'](trajectory, float(f)) for f in frames]).numpy()
        actual = data['history_q'].reshape(-1, 36)
        errors = actual[:, 7:] - reference[:, 7:]
        joint = np.sqrt(np.mean(errors**2, axis=1))
        position = np.linalg.norm(actual[:, :3] - reference[:, :3], axis=1)
        angle = (Rotation.from_quat(actual[:, 3:7]) * Rotation.from_quat(reference[:, 3:7]).inv()).magnitude()
        good = (joint <= .2) & (position <= .2) & (angle <= .35)
        plan = json.loads((OUT / (name + '.steps_plan.json')).read_text())
        steps = summarize_steps(data, plan)
        report['independent_measured_steps'] = steps['measured_steps']
        report['independent_next_step_diagnostics'] = steps.get('next_step_diagnostics')
        poses, _, _ = urdf_fk(reference, names, ROOT / 'src/fabrics_sim/models/robots/urdf/g1/g1_29dof_rev_1_0.urdf')
        measured_feet = data['history_feet'].reshape(-1, 2, 7)
        events = []
        for event in steps['landings']:
            foot = 0 if event['swing'] == 'left' else 1
            index = int(abs(times - event['time']).argmin())
            target = poses[event['swing'] + '_ankle_roll_link'][0][index]
            error = float(np.linalg.norm(measured_feet[index, foot, :3] - target))
            events.append(dict(step=event['step'], source_foot_error_m=error,
                prefix_pose_passed=bool(good[times <= event['time'] + 1e-9].all()), foot_position_passed=error <= .04))
        verified_prefix = 0
        for event in events:
            if event['step'] != verified_prefix + 1 or not (event['prefix_pose_passed'] and event['foot_position_passed']):
                break
            verified_prefix += 1
        report['raw_source_audit'] = dict(source_sha256=source_hash, frame_count=len(raw_q),
            peak_joint_rmse_rad=float(joint.max()), peak_root_position_m=float(position.max()),
            peak_root_orientation_rad=float(angle.max()), prefix_landings=verified_prefix,
            reference_max_difference=float(abs(reference - data['history_reference_q'].reshape(-1, 36)).max()),
            landings=events)
        report['samples'] = []
        for sec in (3.41, 3.48, 3.58, 3.65, 3.75, 3.96, 4.216):
            if sec > times[-1] + 1e-9:
                continue
            i = int(np.argmin(abs(times-sec)))
            report['samples'].append(dict(time_s=float(times[i]), shoulder_error_rad=errors[i,shoulders].tolist(),
                joint_rmse_rad=float(joint[i]), root_position_m=float(position[i]), root_orientation_rad=float(angle[i])))
        report['shoulder_intervals'] = []
        for lo, hi in ((3.3,3.58),(3.58,3.8),(3.8,4.2)):
            mask = (times >= lo) & (times <= hi)
            if times[-1] < hi or not mask.any():
                continue
            report['shoulder_intervals'].append(dict(start_s=lo,end_s=hi,
                per_shoulder_rmse_rad=np.sqrt(np.mean(errors[mask][:, shoulders]**2,axis=0)).tolist(),
                combined_rmse_rad=float(np.sqrt(np.mean(errors[mask][:,shoulders]**2))),
                full_joint_rmse_rad=float(np.sqrt(np.mean(errors[mask]**2)))))
        if 'history_shoulder_roll_weight_multiplier' in data:
            mult = data['history_shoulder_roll_weight_multiplier'].ravel()
            report['weight_range'] = [float(mult.min()),float(mult.max())]
            indices = np.flatnonzero(mult > 1)
            report['first_weight_activation_s'] = float(times[indices[0]]) if len(indices) else None
        if 'history_shoulder_guard_accepted' in data:
            trials=data['history_shoulder_guard_attempts'].ravel()>0
            accepted=data['history_shoulder_guard_accepted'].ravel().astype(bool)
            reasons,counts=np.unique(data['history_shoulder_guard_reason'][trials],return_counts=True)
            report['guard']=dict(attempted_ticks=int(trials.sum()),accepted_ticks=int(accepted.sum()),
                accepted_fraction=float(accepted[trials].mean()) if trials.any() else 0.,
                reasons=dict(zip(reasons.tolist(),counts.tolist())),
                rejected_candidates=int(data['history_shoulder_guard_rejected_candidates'].sum()))
            if accepted.any():
                before=data['history_shoulder_guard_before_residual'].ravel()[accepted]
                after=data['history_shoulder_guard_after_residual'].ravel()[accepted]
                report['guard']['accepted_residual_ratio_max']=float((after/np.maximum(before,1e-12)).max())
                assert np.all(after <= .99*before+1e-5)
        if 'history_qp_wall_seconds' in data:
            mask=times>3.58
            if mask.any():
                report['qp_latency_after_3_58_ms']=dict(zip(('median','p95'),
                    np.percentile(data['history_qp_wall_seconds'].ravel()[mask]*1000,[50,95]).tolist()))
        if 'history_com_soft_activation' in data:
            active=data['history_com_soft_activation'].ravel()>0
            ids=np.flatnonzero(active)
            error=data['history_com_soft_velocity_error'].reshape(-1,3)
            forecast=data['history_com_soft_forecast_error'].reshape(-1,3)
            slack=data['history_com_soft_required_slack'].reshape(-1,3)
            report['com_velocity_soft']=dict(active_ticks=int(active.sum()),
                first_activation_s=float(times[ids[0]]) if len(ids) else None,
                peak_required_slack_mps=float(slack[active].max()) if active.any() else 0.,
                active_forecast_improves_fraction=float((np.linalg.norm(forecast[active],axis=1)<
                    np.linalg.norm(error[active],axis=1)).mean()) if active.any() else None)
        if baseline_q is not None:
            n = min(len(actual),len(baseline_q))
            np.testing.assert_array_equal(times[:n],baseline_times[:n])
            diff = np.max(abs(actual[:n]-baseline_q[:n]),axis=1)
            indices = np.flatnonzero(diff>0)
            report['first_state_difference_s'] = float(times[indices[0]]) if len(indices) else None
            if 'com_velocity_soft' in report:
                activation=report['com_velocity_soft']['first_activation_s']
                before=times[:n]<=activation if activation is not None else np.ones(n,dtype=bool)
                report['com_velocity_soft']['before_activation_max_pose_difference']=float(diff[before].max())
                assert report['com_velocity_soft']['before_activation_max_pose_difference']==0
            before_requested = times[:n] <= options['shoulder_roll_weight_start']
            report['before_requested_start_max_pose_difference'] = float(diff[before_requested].max())
            activation = report.get('first_weight_activation_s')
            before = times[:n] < activation if activation is not None else np.ones(n,dtype=bool)
            report['before_actual_activation_max_pose_difference'] = float(diff[before].max())
            if report['weight_clock'] == 'control_time' and activation is not None:
                expected = times[np.flatnonzero(times > options['shoulder_roll_weight_start'] + 1e-9)[0]]
                if options.get('shoulder_roll_guard'):
                    assert activation >= expected-1e-9
                else:
                    np.testing.assert_allclose(activation, expected, atol=1e-9, rtol=0)
                assert report['before_requested_start_max_pose_difference'] == 0
                report['control_clock_and_prefix_check_passed'] = True
        for key in ('servo_valid','servo_saturation_fraction','qp_validated'):
            if key in data:
                report[key] = data[key].tolist()
        if 'self_contact_events' in data:
            report['self_contact_events'] = data['self_contact_events'].tolist()
    (OUT / (name + '.audit.json')).write_text(json.dumps(report, indent=2) + '\n')
    rows.append(report)
    print(json.dumps({key: report[key] for key in ('run','independent_measured_steps',
        'last_time_s','stop_reason','full_motion_tracking_passed','shoulder_intervals')}),flush=True)
(OUT / 'results.json').write_text(json.dumps(dict(
    scope='Fresh physics from frame zero; original PKL reloaded and measured landings independently reconstructed.',
    source_sha256=source_hash,
    full_motion_tracking_passed=any(row['full_motion_tracking_passed'] for row in rows), trials=rows), indent=2)+'\n')
