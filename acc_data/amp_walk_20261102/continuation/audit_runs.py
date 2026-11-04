"""Summarize physical runs; simulation duration alone is never tracking success."""
import hashlib
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent
repo = root.parents[2]
source = repo / 'retarget/amp_reference/amp_walk_1ms.pkl'
expected_hash = 'a870f6642380a5d7e23ae34fbc8a9614bf4164d6a9ce0a26432ae35f3401b8c4'
trials = []
for path in sorted(root.glob('*.steps.json')):
    steps = json.loads(path.read_text())
    trials.append(dict(name=path.name.removesuffix('.steps.json'),
                       measured_landings=steps['measured_steps'],
                       expected_landings=steps['expected_steps'],
                       failed_time_s=steps.get('failed_time_s'),
                       steps_tracking_passed=steps['steps_tracking_passed'],
                       peak_joint_rmse_rad=steps.get('pkl_fidelity', {}).get('peak_joint_rmse')))

def contact_audit(filename):
    with np.load(root / filename) as d:
        t = d['history_time'].reshape(-1)
        mode = d['history_commanded_mode'].reshape(-1)
        forces = d['history_ground_foot_forces'].reshape(-1, 2, 3)[..., 2]
        loaded = (forces >= 20.) & (forces > .25*forces.sum(axis=1, keepdims=True))
        ignored = ((mode == 1) & loaded[:, 1]) | ((mode == 2) & loaded[:, 0])
        mask = ignored & (t <= 1.7)
        return dict(window_s=[0., 1.7], ignored_loaded_contact_samples=int(mask.sum()),
                    times_s=t[mask].tolist())

steps = json.loads((root / 'measured_load.steps.json').read_text())
with np.load(root / 'measured_load.failure.npz') as d:
    first_four = d['history_time'].reshape(-1) <= steps['landings'][3]['time'] + 1e-8
    peak = float(d['history_pkl_joint_rmse'].reshape(-1)[first_four].max())
report = dict(source=str(source.relative_to(repo)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              retained_change='Pass measured ground normal forces to dynamic-start contact switching, independently of legacy source-clock gating.',
              default_record=str((root / 'measured_load.failure.npz').relative_to(repo)),
              default_result=json.loads((root / 'measured_load.tracking.json').read_text()),
              contact_audit_before=contact_audit('baseline.npz'),
              contact_audit_after=contact_audit('measured_load.failure.npz'),
              first_four_peak_joint_rmse_rad=peak,
              original_joint_rmse_tolerance_rad=.2,
              experimental_profiles_enabled_by_default=False,
              trials=trials)
report['source_unchanged'] = report['source_sha256'] == expected_hash
report['full_motion_tracking_passed'] = report['default_result']['full_motion_tracking_passed']
log = root / 'regression.log'
if log.exists():
    text = log.read_text()
    import re
    match = re.search(r'Ran (\d+) tests in ([\d.]+)s', text)
    report['validation'] = dict(tests=int(match[1]) if match else None,
                               passed=bool(match and '\nOK\n' in text),
                               log=str(log.relative_to(repo)))
(root / 'progress.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
