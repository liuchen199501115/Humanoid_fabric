"""Report landing events separately from prefix fidelity, retaining failed trials."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent

def summarize(stem):
    path = ROOT / (stem+'.npz')
    if not path.exists(): path = ROOT / (stem+'.failure.npz')
    steps = json.loads((ROOT/(stem+'.steps.json')).read_text())
    tracking = json.loads((ROOT/(stem+'.tracking.json')).read_text())
    with np.load(path) as d:
        time = d['history_time'].ravel()
        q = d['history_q'].reshape(-1,36)
        ref = d['history_reference_q'].reshape(-1,36)
        joint = np.sqrt(np.mean((q[:,7:]-ref[:,7:])**2, axis=1))
        pos = np.linalg.norm(q[:,:3]-ref[:,:3],axis=1)
        actual = q[:,3:7]/np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
        target = ref[:,3:7]/np.linalg.norm(ref[:,3:7],axis=1,keepdims=True)
        angle = 2*np.arccos(np.clip(np.abs(np.sum(actual.astype(float)*target,axis=1)),0.,1.))
        outside = (joint>.2) | (pos>.2) | (angle>.35)
        landings = []
        for event in steps['landings']:
            index = abs(time-event['time']).argmin()
            mask = time <= event['time']+1e-9
            landings.append(dict(step=event['step'], time_s=event['time'], source_frame=event['source_frame'],
                landing_error_m=event['landing_error_m'], joint_rmse_rad=float(joint[index]),
                root_position_error_m=float(pos[index]), root_orientation_error_rad=float(angle[index]),
                prefix_peak_joint_rmse_rad=float(joint[mask].max()),
                prefix_peak_root_position_error_m=float(pos[mask].max()),
                prefix_peak_root_orientation_error_rad=float(angle[mask].max()),
                prefix_fidelity_passed=bool(not outside[mask].any())))
        return dict(run=stem, recording=str(path), measured_landing_events=steps['measured_steps'],
            full_motion_tracking_passed=tracking.get('full_motion_tracking_passed',False),
            prefix_fidelity_passing_landings=sum(e['prefix_fidelity_passed'] for e in landings),
            first_fidelity_exceeded_time_s=float(time[np.flatnonzero(outside)[0]]) if outside.any() else None,
            last_source_frame=float(d['history_frame'].ravel()[-1]),
            last_time_s=float(time[-1]), landings=landings,
            next_step_diagnostics=steps.get('next_step_diagnostics'))

if __name__ == '__main__':
    rows=[summarize(p.name.removesuffix('.steps.json')) for p in sorted(ROOT.glob('*.steps.json'))]
    report_path=ROOT/'progress.json'
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update(full_motion_tracking_passed=any(r['full_motion_tracking_passed'] for r in rows),
                  thresholds=dict(joint_rmse_rad=.2,root_position_m=.2,root_angle_rad=.35),trials=rows)
    candidate=report.get('experimental_candidate',{}).get('run')
    if candidate:
        report['experimental_candidate']=next(r for r in rows if r['run']==candidate)
    (ROOT/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
    for r in rows:
        print(r['run'], 'events',r['measured_landing_events'],'prefix fidelity passing events',r['prefix_fidelity_passing_landings'],
              'last landing',r['landings'][-1] if r['landings'] else None)
