"""Audit strict source landing targets, including interrupted simulation records."""
import argparse
import json
from pathlib import Path
import numpy as np
from fabrics_sim.reference.bounded_steps import MeasuredSteps
from analyze_contact_transfer import sole_clearances


def summarize(data, plan):
    monitor = MeasuredSteps(plan['events'], plan['landing_positions_world'])
    t = np.asarray(data['history_time']).ravel()
    frame = np.asarray(data['history_frame']).ravel()
    feet = np.asarray(data['history_feet']).reshape(-1,2,7)
    forces = np.asarray(data['history_foot_forces']).reshape(-1,2,3)[...,2]
    mode = np.asarray(data['history_commanded_mode']).ravel()
    if len(t)<2 or np.any(np.diff(t)<=0):
        raise ValueError('Need a valid control history')
    clearance = sole_clearances(feet)
    for i in range(1,len(t)):
        monitor.update(frame[i], t[i], mode[i], forces[i], clearance[i], feet[i,:,:3], t[i]-t[i-1])
    report = monitor.report()
    controller_available = 'motion_tracking_valid' in data
    controller = bool(np.all(data.get('motion_tracking_valid',False)))
    settled = bool(np.all(data.get('stopped_at_motion_end',False)))
    report.update(last_time_s=float(t[-1]), last_source_frame=float(frame[-1]),
        controller_valid=controller if controller_available else None,
        controller_validation_available=controller_available, stable_final_support=settled,
        steps_tracking_passed=bool(monitor.complete and controller and settled))
    if 'reason' in data:
        report['stop_reason']=str(np.asarray(data['reason']).item())
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with np.load(args.recording,allow_pickle=False) as data:
        report=summarize(data,json.loads(args.plan.read_text()))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['steps_tracking_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
