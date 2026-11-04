"""Audit actual alternating touchdowns and travel, separately from QP validity."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def summarize_walk(data):
    time=np.asarray(data['history_time']).reshape(-1)
    modes=np.asarray(data['history_commanded_mode']).reshape(-1)
    feet=np.asarray(data['history_feet']).reshape(-1,2,7)
    forces=np.asarray(data['history_foot_forces']).reshape(-1,2,3)
    q=np.asarray(data['history_q']).reshape(-1,36)
    phases=np.asarray(data['history_reference_phase'])
    touchdowns=[]
    pending=None
    for i in range(1,len(time)):
        if modes[i-1] in (1,2) and modes[i]==3:
            swing=1 if modes[i-1]==1 else 0
            if forces[i,swing,2]>=5:
                pending=(i,swing)
        if pending is not None and phases[i-1]=='SETTLE' and phases[i]=='TRACK' and modes[i]==3:
            contact,swing=pending
            if forces[i,swing,2]>=20:
                touchdowns.append({'time':float(time[contact]),'settled_time':float(time[i]),
                                   'foot':'right' if swing else 'left',
                                   'position':feet[i,swing,:3].tolist(),
                                   'normal_force':float(forces[i,swing,2])})
            pending=None
        if modes[i]!=3:pending=None
    slip=0.
    for foot in range(2):
        anchor=None
        for i,mode in enumerate(modes):
            active=bool(int(mode)&(1<<foot)) and phases[i]!='INITIALIZE'
            if active:
                if anchor is None:anchor=feet[i,foot,:2]
                slip=max(slip,float(np.linalg.norm(feet[i,foot,:2]-anchor)))
            else:anchor=None
    forward=Rotation.from_quat(q[0,3:7]).as_matrix()[:2,0]
    forward/=np.linalg.norm(forward)
    travel=float((q[-1,:2]-q[0,:2])@forward)
    alternating=all(a['foot']!=b['foot'] for a,b in zip(touchdowns,touchdowns[1:]))
    controller_valid=bool(np.asarray(data.get('motion_tracking_valid',False)).all())
    completed=bool(np.asarray(data.get('simulation_completed',False)).all())
    result=dict(last_time=float(time[-1]),controller_valid=controller_valid,
                simulation_completed=completed,touchdown_count=len(touchdowns),
                alternating=alternating,forward_distance=travel,maximum_stance_slip=slip,
                touchdowns=touchdowns,final_source_frame=float(data['history_frame'][-1]),
                final_phase=str(data['history_reference_phase'][-1]),
                final_wait_seconds=float(data['history_reference_wait'][-1]),
                peak_pkl_joint_rmse=float(np.max(data['history_pkl_joint_rmse'])),
                peak_pkl_root_error=float(np.max(data['history_pkl_root_position_error'])))
    result['walking_segment_passed']=bool(controller_valid and completed and len(touchdowns)>=4
                                         and alternating and travel>=.2 and slip<=.02)
    result['criteria']='Complete simulation and controller checks; >=4 alternating touchdowns that complete load settling; >=0.2 m forward travel; <=0.02 m stance XY slip after initialization. This is a finite flat-ground segment, not full PKL fidelity.'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plot',type=Path,help='Save an actual-motion diagnostic figure (PNG/PDF/SVG).')
    args=parser.parse_args()
    with np.load(args.recording) as data:
        result=summarize_walk(data)
        if args.plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            time=np.asarray(data['history_time']).reshape(-1)
            feet=np.asarray(data['history_feet']).reshape(-1,2,7)
            force=np.asarray(data['history_foot_forces']).reshape(-1,2,3)
            root=np.asarray(data['history_q']).reshape(-1,36)[:,:3]
            fig,axes=plt.subplots(3,1,figsize=(10,9),layout='constrained')
            axes[0].plot(root[:,0],root[:,1],label='Root')
            for index,name in enumerate(('Left','Right')):
                axes[0].plot(feet[:,index,0],feet[:,index,1],label=f'{name} ankle')
                axes[1].plot(time,force[:,index,2],label=name)
                axes[2].plot(time,feet[:,index,2],label=name)
            axes[0].set(xlabel='World X (m)',ylabel='World Y (m)',title='Measured motion')
            axes[0].set_aspect('equal',adjustable='datalim')
            axes[1].set(xlabel='Time (s)',ylabel='Normal load (N)')
            axes[2].axhline(.035,color='gray',linestyle='--',label='Level sole: ankle 0.035 m')
            axes[2].set(xlabel='Time (s)',ylabel='Ankle height (m)')
            for ax in axes:
                ax.grid(alpha=.25);ax.legend()
            fig.suptitle(f"Settled touchdowns: {result['touchdown_count']} | "
                         f"Forward: {result['forward_distance']:.3f} m | "
                         f"Stance slip: {result['maximum_stance_slip']:.4f} m | "
                         f"Audit passed: {result['walking_segment_passed']}")
            args.plot.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(args.plot,dpi=160)
            plt.close(fig)
    content=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(content)
    print(content,end='')


if __name__=='__main__':main()
