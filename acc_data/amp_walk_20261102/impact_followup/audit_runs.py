"""Separate landing evidence, controller checks, and original-source fidelity."""
import hashlib,json,re
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent
repo=p.parents[2]
trials=[]
for path in sorted(p.glob('*.steps.json')):
    d=json.loads(path.read_text());name=path.name.removesuffix('.steps.json')
    row=dict(name=name,verified_landings=d['measured_steps'],expected_landings=d['expected_steps'],
             qp_failure_time_s=d.get('failed_time_s'),last_source_frame=d.get('last_source_frame'),
             controller_valid=d.get('controller_valid'),pkl_fidelity_passed=d.get('pkl_fidelity_passed'),
             strict_steps_tracking_passed=d['steps_tracking_passed'],landings=d['landings'])
    trial_path=p/(name+'.failure.npz')
    if not trial_path.exists():trial_path=p/(name+'.npz')
    if trial_path.exists() and d['landings']:
        with np.load(trial_path) as data:
            t=data['history_time'].reshape(-1);q=data['history_q'].reshape(-1,36);r=data['history_reference_q'].reshape(-1,36)
            i=int(abs(t-d['landings'][-1]['time']).argmin())
            quat=q[i,3:7]/np.linalg.norm(q[i,3:7]);ref=r[i,3:7]/np.linalg.norm(r[i,3:7])
            row['last_verified_landing_fidelity']=dict(time_s=float(t[i]),
                joint_rmse_rad=float(data['history_pkl_joint_rmse'].reshape(-1)[i]),
                root_position_error_m=float(np.linalg.norm(q[i,:3]-r[i,:3])),
                root_orientation_error_rad=float(2*np.arccos(np.clip(abs(quat@ref),0,1))))
    trials.append(row)
report=dict(source='retarget/amp_reference/amp_walk_1ms.pkl',
            source_sha256=hashlib.sha256((repo/'retarget/amp_reference/amp_walk_1ms.pkl').read_bytes()).hexdigest(),
            source_unchanged=False,default_controller_changed=False,
            retained_fix='COM diagnostics now record the actual WBC task request, including preserved feedforward and overrides.',
            full_motion_tracking_passed=False,trials=trials,
            limitations=['Verified landings alone do not certify stable tracking.',
                         'The five-landing candidate already has excessive body error at the fifth landing.',
                         'Experimental controllers and reference modifications are not enabled by default.'])
report['source_unchanged']=report['source_sha256']=='a870f6642380a5d7e23ae34fbc8a9614bf4164d6a9ce0a26432ae35f3401b8c4'
for filename,key in [('five_landings_reproduction.json','five_landings_reproduction'),('diagnostic_verification.json','diagnostic_verification')]:
    if (p/filename).exists():report[key]=json.loads((p/filename).read_text())
log=p/'regression.log'
if log.exists():
    text=log.read_text();match=re.search(r'Ran (\d+) tests in ([\d.]+)s',text)
    report['regression']=dict(tests=int(match[1]) if match else None,passed=bool(match and '\nOK\n' in text),log=str(log.relative_to(repo)))
(p/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('trials','five_landings_reproduction','diagnostic_verification')},indent=2))
for row in trials:print(row['name'],row['verified_landings'],row['qp_failure_time_s'],row.get('last_verified_landing_fidelity'))
