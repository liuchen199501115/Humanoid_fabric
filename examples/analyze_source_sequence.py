"""Audit alternating measured landings and stopped support, not full PKL fidelity."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_contact_transfer import sole_clearances
from analyze_control_rate import summarize_control_rate


def summarize_source_sequence(data, expected_steps=2, end_frame=None):
    if type(expected_steps) is not int or expected_steps < 1:
        raise ValueError('Expected at least one source step')
    if end_frame is None:
        if expected_steps == 2:
            end_frame = 155.
        elif 'history_source_step_end_frame' in data:
            end_frame = float(np.asarray(data['history_source_step_end_frame']).ravel()[-1])
        else:
            raise ValueError('Expected source endpoint must be supplied')
    if not np.isfinite(end_frame) or end_frame <= 0:
        raise ValueError('Expected a finite positive source endpoint')
    report=dict(scope=f'{expected_steps}_adapted_source_steps',sequence_passed=False,
                expected_steps=expected_steps,expected_end_frame=float(end_frame),
                full_pkl_tracking=False,continuous_walking=False)
    if expected_steps == 2:
        report['two_steps_passed'] = False
    number=np.asarray(data['history_source_step_number']).ravel()
    t=np.asarray(data['history_time']).ravel()
    frame=np.asarray(data['history_frame']).ravel()
    landed=np.asarray(data['history_source_step_touchdown']).ravel().astype(bool)
    finished=np.asarray(data['history_source_steps_finished']).ravel().astype(bool)
    feet=np.asarray(data['history_feet']).reshape(-1,2,7)
    forces=np.asarray(data['history_foot_forces']).reshape(-1,2,3)[:,:,2]
    clearance=sole_clearances(feet)
    goals=np.asarray(data['history_source_step_goal']).reshape(-1,7)
    source_goals=np.asarray(data['history_source_step_source_goal']).reshape(-1,7)
    mode=np.asarray(data['history_commanded_mode']).ravel()
    landings=[]
    for step in range(1,expected_steps+1):
        swing=step%2
        active=number==step
        indices=np.flatnonzero(active & landed)
        if not len(indices):
            report.update(reason=f'Step {step} has no confirmed touchdown',landings=landings,final_source_frame=float(frame[-1]))
            return report
        index=int(indices[0])
        landings.append(dict(step=step,swing='left' if swing==0 else 'right',time=float(t[index]),
            source_frame=float(frame[index]),
            adapted_landing_error_m=float(np.linalg.norm(feet[index,swing,:3]-goals[index,:3])),
            original_landing_error_m=float(np.linalg.norm(feet[index,swing,:3]-source_goals[index,:3])),
            maximum_swing_clearance_m=float(clearance[active,swing].max()),
            measured_load_n=float(forces[index,swing]),
            single_support_observed=bool(np.any(active & (mode==2-swing) & (clearance[:,swing]>=.02)
                & (forces[:,swing]<10.) & (forces[:,1-swing]>=100.)))))
    end=np.flatnonzero(finished)
    recent=t>=t[-1]-1.+1e-7
    com_error=np.linalg.norm((data['history_com']-data['history_com_target']).reshape(-1,3),axis=-1)
    com=np.asarray(data['history_com']).reshape(-1,3)
    speeds=np.linalg.norm(np.diff(com,axis=0)/np.diff(t)[:,None],axis=-1)
    stats=dict(minimum_foot_loads_n=forces[recent].min(axis=0).tolist(),
        maximum_com_error_m=float(com_error[recent].max()),
        maximum_com_speed_m_s=float(speeds[recent[1:]].max()))
    hold_duration=0. if not len(end) else float(t[-1]-t[end[0]])
    rate=summarize_control_rate(data)
    monotonic=bool(np.all(np.diff(frame)>=-1e-6))
    controller=bool(np.all(data.get('qp_validated',False)) and np.all(data.get('servo_valid',False)))
    passed=bool(len(end) and hold_duration>=3. and recent.sum()>=499
        and np.all(np.asarray(data['history_commanded_mode']).ravel()[recent]==3)
        and min(stats['minimum_foot_loads_n'])>=20. and stats['maximum_com_error_m']<=.03
        and stats['maximum_com_speed_m_s']<=.04 and abs(frame[-1]-end_frame)<=1e-6
        and number[-1]==expected_steps and np.max(number)==expected_steps
        and monotonic and all(r['maximum_swing_clearance_m']>=.02
        and r['single_support_observed'] and r['adapted_landing_error_m']<=.04 and r['measured_load_n']>=5. for r in landings)
        and controller and rate['simulation_rate_passed'])
    report.update(sequence_passed=passed,landings=landings,final_source_frame=float(frame[-1]),
        source_clock_monotonic=monotonic,minimum_source_frame_increment=float(np.diff(frame).min()),
        stopped_hold_duration_s=hold_duration,final_second=stats,qp_and_servo_validated=controller,control_rate=rate)
    if expected_steps == 2:
        report['two_steps_passed'] = passed
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('recording',type=Path)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--end-frame',type=float);a=p.parse_args()
    with np.load(a.recording) as data:r=summarize_source_sequence(data,a.steps,a.end_frame)
    text=json.dumps(r,indent=2)+'\n';a.recording.with_suffix('.sequence.json').write_text(text);print(text,end='')
