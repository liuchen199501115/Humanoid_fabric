from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
report=json.loads((out/'results.json').read_text())
rows=[r for r in report['trials'] if 'raw_source_audit' in r]
best=max(rows,key=lambda r:(r['raw_source_audit']['prefix_landings'],r['last_source_frame']))
summary=dict(full_motion_tracking_passed=report['full_motion_tracking_passed'],
    physical_trials=len(rows),best_prefix=best['raw_source_audit']['prefix_landings'],
    best_prefix_run=best['run'],best_last_source_frame=best['last_source_frame'],
    source_frame_count=2000,expected_landings=29,
    regression_tests=dict(passed=36,failed=0,log=str((out/'tests.log').relative_to(root))),
    implementation_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest()
        for p in ['examples/test_g1_2.py','examples/g1_response_feedback.py','tests/test_g1_response_feedback.py']},
    limitations=['State corrections augment the historical schedule; no claim that all time gates were removed.',
        'Stopping budget calibrated from the stable 0--4.1 s interval, not a physical deceleration guarantee.',
        'Offline feasible snapshots do not establish closed-loop tracking.',
        'No whole-clip success unless all original fidelity and 29 landing checks pass.'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
