"""Audit the bounded first source step; never certify continuous PKL tracking."""
import argparse
import json
from pathlib import Path

import numpy as np
from analyze_contact_transfer import sole_clearances
from analyze_control_rate import summarize_control_rate


def summarize_source_step(data):
    time = np.asarray(data['history_time']).ravel()
    elapsed = np.asarray(data['history_source_step_elapsed']).ravel()
    active = elapsed >= 0.
    result = dict(scope='first_source_swing_and_touchdown', full_pkl_tracking=False,
                  continuous_walking=False, source_step_passed=False,
                  final_source_frame=float(np.asarray(data['history_frame']).ravel()[-1]))
    if not active.any():
        result['reason'] = 'No source swing executed'
        return result
    first = np.flatnonzero(active)[0]
    modes = np.asarray(data['history_commanded_mode']).ravel()
    if modes[first] not in (1, 2):
        result['reason'] = 'Source swing did not start in single support'
        return result
    support = int(modes[first])-1
    swing = 1-support
    feet = np.asarray(data['history_feet']).reshape(-1, 2, 7)
    forces = np.asarray(data['history_foot_forces']).reshape(-1, 2, 3)[..., 2]
    clearance = sole_clearances(feet)
    goal = np.asarray(data['history_source_step_goal']).reshape(-1, 7)[-1]
    source_goals = data.get('history_source_step_source_goal')
    result.update(adapted_landing=None, landing_adaptation_m=None,
                  original_landing_position_error_m=None)
    if source_goals is not None:
        source_goal = np.asarray(source_goals).reshape(-1,7)[-1]
        deviation = float(np.linalg.norm(goal[:2]-source_goal[:2]))
        result.update(adapted_landing=deviation > 1e-6, landing_adaptation_m=deviation,
                      original_landing_position_error_m=float(np.linalg.norm(feet[-1,swing,:3]-source_goal[:3])))
        if result['adapted_landing']:
            result['scope'] = 'adapted_bridge_swing_and_touchdown'
    error = np.linalg.norm((data['history_com']-data['history_com_target']).reshape(-1, 3), axis=-1)
    landing_error = float(np.linalg.norm(feet[-1, swing, :3]-goal[:3]))
    touchdown = np.asarray(data['history_source_step_touchdown']).ravel()
    landed = np.flatnonzero(active & touchdown & (modes == 3))
    rate = summarize_control_rate(data)
    result.update(control_rate=rate, swing_foot='right' if swing else 'left',
        start_time=float(time[first]), planned_elapsed=float(elapsed[-1]),
        measured_touchdown_confirmed=bool(len(landed)),
        landing_time=None if not len(landed) else float(time[landed[0]]),
        landing_position_error_m=landing_error, final_forces_n=forces[-1].tolist(),
        final_sole_clearance_m=clearance[-1].tolist(),
        max_swing_clearance_m=float(clearance[active, swing].max()),
        max_swing_com_error_m=float(error[active].max()), final_com_error_m=float(error[-1]),
        peak_original_pkl_joint_rmse_rad=float(np.max(data['history_pkl_joint_rmse'])),
        peak_original_pkl_root_error_m=float(np.max(data['history_pkl_root_position_error'])))
    phases=np.asarray(data.get('history_reference_phase',[])).ravel()
    stopping=phases == 'LANDING_STOP'
    result['landing_stop_executed']=bool(stopping.any())
    result['landing_stop_endpoint_passed']=False
    result['post_stop_hold_validated']=False
    if stopping.any():
        com_speed=float(np.linalg.norm(np.asarray(data['history_com_velocity']).reshape(-1,3)[-1]))
        ref_speed=float(np.max(np.abs(data['history_whole_body_reference_v'][-1])))
        ref_acceleration=float(np.max(np.abs(data['history_whole_body_reference_a'][-1])))
        result.update(final_com_speed_m_s=com_speed,final_reference_velocity_max=ref_speed,
                      final_reference_acceleration_max=ref_acceleration,
                      landing_stop_endpoint_passed=bool(len(landed) and modes[-1]==3 and com_speed<=.04
                          and error[-1]<=.015 and ref_speed<=1e-4 and ref_acceleration<=1e-3
                          and forces[-1,swing]>=5. and rate['simulation_rate_passed']))
    result['source_step_passed'] = bool(np.all(data.get('motion_tracking_valid', False))
        and np.all(data.get('stopped_at_source_step_end', False)) and len(landed) >= 1
        and rate['simulation_rate_passed'] and landing_error <= .04 and error[-1] <= .05
        and forces[-1, swing] >= 5. and clearance[-1, swing] <= .002)
    return result


def plot_source_step(data, target):
    """Compare measured motion with online targets, including failed probes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    active = np.asarray(data['history_source_step_elapsed']).ravel() >= 0.
    if not active.any():
        raise ValueError('No source swing to plot')
    t = np.asarray(data['history_source_step_elapsed']).ravel()[active]
    modes = np.asarray(data['history_commanded_mode']).ravel()[active]
    if modes[0] not in (1, 2):
        raise ValueError('Expected initial single support')
    swing = 2-int(modes[0])
    feet = np.asarray(data['history_feet']).reshape(-1,2,7)[active]
    targets = np.asarray(data['history_managed_feet_position']).reshape(-1,2,3)[active]
    forces = np.asarray(data['history_foot_forces']).reshape(-1,2,3)[active,:,2]
    com_error = np.linalg.norm((data['history_com']-data['history_com_target']).reshape(-1,3),axis=-1)[active]
    fig, axes = plt.subplots(3,2,figsize=(12,9),sharex=True)
    for i, axis in enumerate('XYZ'):
        axes[i,0].plot(t,feet[:,swing,i],label='Measured swing foot')
        axes[i,0].plot(t,targets[:,swing,i],'--',label='Online target')
        axes[i,0].set_ylabel(axis+' [m]')
    for i, side in enumerate(('Left','Right')):
        axes[0,1].plot(t,forces[:,i],label=side)
    axes[0,1].set_ylabel('Measured vertical force [N]')
    axes[1,1].plot(t,com_error,label='COM error')
    axes[1,1].plot(t,np.linalg.norm(feet[:,swing,:3]-targets[:,swing],axis=-1),label='Swing position error')
    axes[1,1].set_ylabel('Tracking error [m]')
    if 'history_unload_effort_ratio' in data:
        ratio=np.asarray(data['history_unload_effort_ratio']).ravel()[active]
        axes[2,1].plot(t,ratio,label='WBC torque / motor limit')
        axes[2,1].axhline(.8,color='r',linestyle=':',label='WBC reserve budget')
    axes[2,1].set_ylabel('Peak torque ratio')
    for ax in axes.flat:
        ax.grid(True)
        ax.legend()
    for ax in axes[-1]:ax.set_xlabel('Time since first-step release [s]')
    fig.suptitle('Measured first-step probe (not a success certificate)')
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording', type=Path)
    parser.add_argument('--plot', action='store_true', help='Plot measured swing and online targets')
    args = parser.parse_args()
    with np.load(args.recording) as stored:
        data = dict(stored)
    if 'history_source_step_source_goal' not in data:
        stem = args.recording.stem.removesuffix('.reference_failure').removesuffix('.failure')
        sidecar = args.recording.with_name(stem+'.step_plan.json')
        if sidecar.exists():
            original_goal = json.loads(sidecar.read_text()).get('source_landing_pose')
            if original_goal is not None:
                data['history_source_step_source_goal'] = np.asarray(original_goal)[None]
    report = summarize_source_step(data)
    args.recording.with_suffix('.step.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.plot:
        plot_source_step(data,args.recording.with_suffix('.step.png'))
    print(json.dumps(report, indent=2))
