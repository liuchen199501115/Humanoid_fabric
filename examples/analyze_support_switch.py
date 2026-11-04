"""Check an actual support-side change and a sustained lifted-foot hold."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_contact_transfer import sole_clearances
from analyze_control_rate import summarize_control_rate


def summarize_support_switch(data):
    t=np.asarray(data['history_time']).ravel()
    target=np.asarray(data['history_support_switch_target']).ravel()
    release=np.asarray(data['history_support_switch_release_time']).ravel()
    mode=np.asarray(data['history_commanded_mode']).ravel()
    report=dict(scope='one_measured_support_switch',support_switch_passed=False,
                full_pkl_tracking=False,continuous_walking=False)
    active=np.flatnonzero(target>=0)
    released=np.flatnonzero(release>=0)
    if not len(active) or not len(released):
        report['reason']='New support was not released by the measured gate'
        return report
    support=int(target[active[-1]]);swing=1-support;start=float(release[released[0]])
    recent=t>=t[-1]-1.+1e-7
    feet=np.asarray(data['history_feet']).reshape(-1,2,7)
    forces=np.asarray(data['history_foot_forces']).reshape(-1,2,3)[:,:,2]
    clearance=sole_clearances(feet)
    error=np.linalg.norm((data['history_com']-data['history_com_target']).reshape(-1,3),axis=-1)
    rate=summarize_control_rate(data)
    old_support_seen=bool(np.any(mode[:active[0]]==swing+1))
    controller_ok=bool(np.all(data.get('qp_validated',False)) and np.all(data.get('servo_valid',False)))
    stats=dict(min_swing_clearance_m=float(clearance[recent,swing].min()),
               max_swing_force_n=float(forces[recent,swing].max()),
               min_stance_force_n=float(forces[recent,support].min()),max_com_error_m=float(error[recent].max()))
    report.update(new_support='left' if support==0 else 'right',old_support_observed=old_support_seen,
                  release_time=start,post_release_duration=float(t[-1]-start),final_second=stats,
                  controller_validated=controller_ok,control_rate=rate)
    report['support_switch_passed']=bool(old_support_seen and t[-1]-start>=3. and recent.sum()>=499
        and np.all(mode[recent]==support+1) and stats['min_swing_clearance_m']>=.02
        and stats['max_swing_force_n']<10. and stats['min_stance_force_n']>=100.
        and stats['max_com_error_m']<=.03 and controller_ok and rate['simulation_rate_passed'])
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('recording',type=Path)
    args=parser.parse_args()
    with np.load(args.recording) as data:report=summarize_support_switch(data)
    args.recording.with_suffix('.switch.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
