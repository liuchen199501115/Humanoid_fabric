"""Audit saved physics against raw PKL; a failed trial remains a failed trial."""
import ast
import hashlib
import importlib.util
import json
import math
import pickle
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'examples'))
from inspect_mocap_reference import urdf_fk
spec = importlib.util.spec_from_file_location('prefix_audit', ROOT / 'acc_data/amp_walk_20261102/contact_clock_followup/summarize_trials.py')
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
raw_q = np.concatenate((raw['root_pos'], raw['root_rot'][:, [1, 2, 3, 0]], raw['dof_pos'][:, order]), 1).astype(np.float32)
raw_q[:, 3:7] /= np.linalg.norm(raw_q[:, 3:7], axis=1, keepdims=True)
node = next(n for n in ast.parse((ROOT / 'examples/test_g1_2.py').read_text()).body
            if getattr(n, 'name', None) == 'interpolate_motion_trajectory')
namespace = {'torch': torch, 'math': math}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'source_sampler', 'exec'), namespace)
rows = []
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
    report.update(returncode=manifest['returncode'], command=manifest['command'],
                  stop_reason=tracking.get('stop_reason'), failed_time_s=tracking.get('failed_time_s'),
                  failed_source_frame=tracking.get('failed_source_frame'),
                  tested_source_sha256=manifest['sha256']['examples/test_g1_2.py'])
    with np.load(report['recording']) as data:
        options = json.loads(data['controller_options'].item())
        report['execution'] = {key: options.get(key) for key in ('tracking_profile', 'formula_scope',
            'direct_wbc', 'tracking_servo', 'formula_source_reference', 'physical_braking_start', 'clock_final_speed','response_gain','response_error_onset','state_limit_priority','state_braking_budget','state_limit_weight','state_root_metric','state_root_onset','state_root_angular_scale','state_braking_hard','state_soft_orientation')}
        for key in ('servo_valid', 'servo_saturation_fraction', 'qp_validated'):
            if key in data:
                report[key] = data[key].tolist()
        for key in ('formula_weights', 'formula_risk', 'formula_activation','response_estimate','response_activation','state_braking_active','soft_orientation_recovery_used','state_root_activation'):
            if 'history_' + key in data:
                values = data['history_' + key]
                report[key + '_range'] = [float(values.min()), float(values.max())]
        times = data['history_time'].ravel()
        frames = data['history_frame'].ravel()
        trajectory = torch.from_numpy(raw_q.copy())
        trajectory[:, 2] += float(data['motion_ground_shift'])
        reference = torch.cat([namespace['interpolate_motion_trajectory'](trajectory, float(f)) for f in frames]).numpy()
        actual = data['history_q'].reshape(-1, 36)
        joint = np.sqrt(((actual[:, 7:] - reference[:, 7:])**2).mean(-1))
        position = np.linalg.norm(actual[:, :3] - reference[:, :3], axis=-1)
        angle = (Rotation.from_quat(actual[:, 3:7]) * Rotation.from_quat(reference[:, 3:7]).inv()).magnitude()
        good = (joint <= .2) & (position <= .2) & (angle <= .35)
        steps = json.loads((OUT / (name + '.steps.json')).read_text())
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
        report['raw_source_audit'] = dict(source_sha256=source_hash, frame_count=len(raw_q),
            peak_joint_rmse_rad=float(joint.max()), peak_root_position_m=float(position.max()),
            peak_root_orientation_rad=float(angle.max()),
            prefix_landings=sum(e['prefix_pose_passed'] and e['foot_position_passed'] for e in events),
            reference_max_difference=float(abs(reference - data['history_reference_q'].reshape(-1, 36)).max()),
            landings=events)
    (OUT / (name + '.audit.json')).write_text(json.dumps(report, indent=2) + '\n')
    rows.append(report)
    print(json.dumps({key: report[key] for key in ('run', 'measured_landing_events',
        'prefix_fidelity_passing_landings', 'last_time_s', 'stop_reason', 'full_motion_tracking_passed')}))
result = dict(scope='Fresh sequential physics trials from source frame zero; unchanged pose and measured landing checks. Raw PKL is independently reloaded and FK rechecks event positions.',
    full_motion_tracking_passed=any(row['full_motion_tracking_passed'] for row in rows), trials=rows)
(OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
