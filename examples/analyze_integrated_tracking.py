"""Audit actual/desired joint drift and support loads in a mocap recording."""
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(data):
    time=np.asarray(data['history_time']).ravel()
    if len(time)<2 or np.any(np.diff(time)<=0):
        raise ValueError('Need a strictly increasing recorded control history')
    measured=np.asarray(data['history_q']).reshape(-1,36)[:,7:]
    desired=np.asarray(data['history_joint_position_desired']).reshape(-1,29)
    error=np.abs(desired-measured)
    com=np.asarray(data['history_com']).reshape(-1,3)
    target=np.asarray(data['history_com_target']).reshape(-1,3)
    force=np.asarray(data['history_foot_forces']).reshape(-1,2,3)[:,:,2]
    if not all(np.isfinite(x).all() for x in (time,measured,desired,com,target,force)):
        raise ValueError('Nonfinite tracking record')
    return dict(last_time_s=float(time[-1]),last_frame=float(np.asarray(data['history_frame']).ravel()[-1]),
        maximum_integrated_joint_error_rad=float(error.max()),
        final_integrated_joint_error_rad=float(error[-1].max()),
        maximum_com_error_m=float(np.linalg.norm(com-target,axis=1).max()),
        final_com_error_m=float(np.linalg.norm(com[-1]-target[-1])),
        minimum_com_height_m=float(com[:,2].min()),
        final_left_right_load_n=force[-1].tolist()),(time,error,com,target,force)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    args=parser.parse_args()
    with np.load(args.recording,allow_pickle=False) as data:
        report,(time,error,com,target,force)=summarize(data)
    report['recording']=str(args.recording)
    report['scope']='Diagnostics only; four-step success requires analyze_bounded_steps.py'
    args.recording.with_suffix('.integrator.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    axes[0].plot(time,error.max(1));axes[0].set_ylabel('Max command error [rad]')
    axes[1].plot(time,np.linalg.norm(com-target,axis=1),label='COM error')
    axes[1].plot(time,com[:,2],label='COM height');axes[1].set_ylabel('Distance [m]');axes[1].legend()
    axes[2].plot(time,force[:,0],label='Left');axes[2].plot(time,force[:,1],label='Right')
    axes[2].set(xlabel='Simulation time [s]',ylabel='Normal load [N]');axes[2].legend()
    for ax in axes:ax.grid(True)
    fig.tight_layout();fig.savefig(args.recording.with_suffix('.integrator.png'));plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
