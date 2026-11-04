"""Audit strict source landing targets, including interrupted simulation records."""
import argparse
import json
from pathlib import Path
import numpy as np
from fabrics_sim.reference.bounded_steps import MeasuredSteps
from analyze_contact_transfer import sole_clearances



def next_step_diagnostics(monitor, time, frame, mode, forces, clearance, feet, reference_feet=None):
    """Explain missing evidence for the next step without changing its gates."""
    if monitor.complete:
        return None
    index = len(monitor.landings)
    event = monitor.events[index]
    support, swing = event['support'], 1-event['support']
    # The sequential monitor starts collecting this step on the tick AFTER
    # confirming its predecessor; earlier observations cannot certify it.
    window = frame >= event['takeoff_frame']-1
    if monitor.landings:
        window &= time > monitor.landings[-1]['time']
    deadline = event.get('landing_window_end_frame') if monitor.continuous else None
    if deadline is not None:
        window &= frame < deadline
    flight = window & (mode == support+1) & (forces[:, support] >= 100.) & (forces[:, swing] < 10.)
    airborne = flight & (clearance[:, swing] >= .02)
    goal = reference_feet[:, swing] if monitor.track_source_pose else monitor.goals[index]
    error = np.linalg.norm(feet[:, swing, :3]-goal, axis=1)
    bearing = ((np.isin(mode, [3, swing+1]) & (forces[:, swing] >= 20.)) if monitor.continuous
               else ((mode == 3) & (forces >= 20.).all(axis=1)))
    loaded = window & (frame >= event['end_frame']-1) & bearing
    ready = loaded & np.maximum.accumulate(airborne) & (error <= monitor.tolerance)
    longest = held = 0.
    for i, valid in enumerate(ready):
        dt = time[i]-time[i-1] if i else time[1]-time[0]
        held = held+dt if valid else 0.
        longest = max(longest, held)
    expired = bool(deadline is not None and frame[-1] >= deadline)
    missing = ('verified_flight' if not airborne.any() else
               'loaded_landing' if not loaded.any() else
               'landing_position' if not (loaded & (error <= monitor.tolerance)).any() else
               'continuous_landing_hold')
    return dict(step=index+1, swing='left' if swing == 0 else 'right',
        missing_evidence=missing, landing_window_expired=expired,
        airborne_verified=bool(airborne.any()),
        peak_unloaded_clearance_m=float(clearance[flight, swing].max()) if flight.any() else None,
        best_loaded_landing_error_m=float(error[loaded].min()) if loaded.any() else None,
        longest_verified_landing_hold_s=float(longest),
        required_clearance_m=.02, required_landing_hold_s=.05,
        landing_position_tolerance_m=float(monitor.tolerance))


def summarize(data, plan):
    continuous = bool(plan.get('continuous', False))
    source_pose = plan.get('landing_goal_policy') == 'source_pose'
    monitor = MeasuredSteps(plan['events'], plan['landing_positions_world'], continuous=continuous, track_source_pose=source_pose)
    t = np.asarray(data['history_time']).ravel()
    frame = np.asarray(data['history_frame']).ravel()
    feet = np.asarray(data['history_feet']).reshape(-1,2,7)
    ground_available = True
    if 'history_ground_foot_forces' in data:
        forces = np.asarray(data['history_ground_foot_forces']).reshape(-1,2,3)[...,2]
    elif 'history_ground_contact_forces' in data:
        from fabrics_sim.solver.contact_observation import ground_support_forces
        forces = np.stack([ground_support_forces(row)[:,2]
                           for row in data['history_ground_contact_forces']])
    else:
        # Old total-force logs can be inspected, but cannot certify support.
        forces = np.asarray(data['history_foot_forces']).reshape(-1,2,3)[...,2]
        ground_available = False
    measured_available = 'history_measured_mode' in data
    mode = np.asarray(data['history_measured_mode'] if measured_available
                      else data['history_commanded_mode']).ravel()
    if (len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0)
            or any(len(x)!=len(t) for x in (frame,feet,forces,mode))):
        raise ValueError('Need a valid control history')
    native_geometry = 'collision_points' in plan and 'collision_radii' in plan
    clearance = sole_clearances(feet, plan.get('collision_points'), plan.get('collision_radii'))
    reference_feet = None
    if source_pose:
        if 'history_source_feet_reference' not in data:
            raise ValueError('Source-pose landing audit requires original source feet in the recording')
        reference_feet = np.asarray(data['history_source_feet_reference']).reshape(-1, 2, 3)
        if reference_feet.shape != (len(t), 2, 3) or not np.isfinite(reference_feet).all():
            raise ValueError('Invalid source foot reference history')
    for i in range(len(t)):
        dt = t[i]-t[i-1] if i else t[1]-t[0]
        monitor.update(frame[i], t[i], mode[i], forces[i].clip(min=0),
                       clearance[i], feet[i,:,:3], dt, reference_feet=None if reference_feet is None else reference_feet[i])
    report = monitor.report()
    report['next_step_diagnostics'] = next_step_diagnostics(monitor, t, frame, mode, forces.clip(min=0), clearance, feet, reference_feet)
    if source_pose:
        legacy_plan = dict(plan, landing_goal_policy='fixed_contact_onset')
        legacy = summarize(data, legacy_plan)
        report['fixed_contact_onset_audit'] = dict(measured_steps=legacy['measured_steps'],
            landings=legacy['landings'], next_step_diagnostics=legacy['next_step_diagnostics'])
    controller_available = 'motion_tracking_valid' in data
    controller = bool(np.all(data.get('motion_tracking_valid',False)))
    endpoint = bool(np.all(data.get('stopped_at_motion_end',False)))
    settled = endpoint and not bool(plan.get('full_motion',False))
    bounded_stop = bool(np.all(data.get('stopped_at_tracking_steps',False)))
    fidelity_available = all(k in data for k in ('history_q','history_reference_q'))
    fidelity = False
    if fidelity_available:
        from analyze_motion_tracking import summarize_tracking
        audit = summarize_tracking(data)
        fidelity = audit['pkl_fidelity_passed']
        report['pkl_fidelity'] = audit
    coverage = bool(np.isfinite(frame).all() and frame[0]<=1e-6
                    and (np.diff(frame)>=-1e-6).all()
                    and frame[-1]>=plan['events'][-1]['end_frame'])
    finished = endpoint if plan.get('full_motion',False) else (bounded_stop if continuous else settled)
    report.update(last_time_s=float(t[-1]), last_source_frame=float(frame[-1]),
        controller_valid=controller if controller_available else None,
        controller_validation_available=controller_available, stable_final_support=settled,
        stopped_at_tracking_steps=bounded_stop, ground_support_evidence_available=ground_available,
        native_clearance_geometry_available=native_geometry,
        measured_mode_available=measured_available, bounded_source_covered=coverage,
        pkl_fidelity_available=fidelity_available, pkl_fidelity_passed=fidelity,
        steps_tracking_passed=bool(monitor.complete and controller and finished and coverage
                                  and fidelity and ground_available
                                  and (measured_available or not continuous)))
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
