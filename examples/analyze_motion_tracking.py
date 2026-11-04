"""Audit full source coverage and original PKL fidelity, separately from walking."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def summarize_tracking(data, source_frames=None, joint_tolerance=.2,
                       root_tolerance=.2, orientation_tolerance=.35):
    limits=np.asarray([joint_tolerance,root_tolerance,orientation_tolerance],dtype=float)
    if not np.isfinite(limits).all() or (limits<=0).any():
        raise ValueError('Tracking tolerances must be finite and positive')
    frames=np.asarray(data['history_frame'],dtype=float).reshape(-1)
    q=np.asarray(data['history_q']).reshape(-1,36)
    reference=np.asarray(data['history_reference_q']).reshape(-1,36)
    if not len(frames) or len(q)!=len(frames) or q.shape!=reference.shape:
        raise ValueError('Expected matching, nonempty frame and pose histories')
    finite=bool(np.isfinite(frames).all() and np.isfinite(q).all() and np.isfinite(reference).all())
    source_frames=int(data.get('source_frame_count',0) if source_frames is None else source_frames)
    source_start=int(data.get('source_start_frame',round(float(frames[0]))))
    joint=np.sqrt(np.mean((q[:,7:]-reference[:,7:])**2,axis=1))
    root=np.linalg.norm(q[:,:3]-reference[:,:3],axis=1)
    valid_quaternions=finite and (np.linalg.norm(q[:,3:7],axis=1)>1e-8).all() and (np.linalg.norm(reference[:,3:7],axis=1)>1e-8).all()
    orientation=(Rotation.from_quat(q[:,3:7])*Rotation.from_quat(reference[:,3:7]).inv()).magnitude() if valid_quaternions else np.full(len(q),np.inf)
    covered=bool(finite and source_frames>0 and source_start==0 and frames[0]<=1e-6
                 and frames[-1]>=source_frames-1-1e-6 and (np.diff(frames)>=-1e-6).all())
    completed=bool(np.asarray(data.get('simulation_completed',False)).all())
    settled=bool(np.asarray(data.get('source_completed',False)).all())
    controller=bool(np.asarray(data.get('motion_tracking_valid',False)).all())
    fidelity=bool(finite and joint.max()<=joint_tolerance and root.max()<=root_tolerance
                  and orientation.max()<=orientation_tolerance)
    return dict(source_frame_count=source_frames,first_source_frame=float(frames[0]),
                last_source_frame=float(frames[-1]),full_source_covered=covered,
                final_support_settled=settled,simulation_completed=completed,
                controller_valid=controller,peak_joint_rmse=float(joint.max()),
                peak_root_position_error=float(root.max()),peak_root_orientation_error=float(orientation.max()),
                tolerances=dict(joint_rmse_rad=joint_tolerance,root_position_m=root_tolerance,
                                root_orientation_rad=orientation_tolerance),
                pkl_fidelity_passed=fidelity,
                full_clip_tracking_passed=bool(covered and completed and controller and fidelity),
                full_tracking_passed=bool(covered and settled and completed and controller and fidelity),
                scope='Original, ground-aligned PKL at the retimed source clock; no per-frame root alignment or stride rescaling. Peak errors include initialization.')


def summarize_full_motion(data, plan):
    """Combine whole-clip fidelity with measured landing evidence."""
    from analyze_bounded_steps import summarize as summarize_steps
    report = summarize_tracking(data)
    steps = summarize_steps(data, plan)
    report.update(expected_landings=steps['expected_steps'],
        measured_landings=steps['measured_steps'],
        all_source_landings_verified=steps['measured_steps_complete'],
        full_motion_tracking_passed=bool(report['full_clip_tracking_passed']
                                        and steps['steps_tracking_passed']))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',type=Path)
    parser.add_argument('--source-frames',type=int,help='For recordings predating source metadata.')
    parser.add_argument('--joint-tolerance',type=float,default=.2)
    parser.add_argument('--root-tolerance',type=float,default=.2)
    parser.add_argument('--orientation-tolerance',type=float,default=.35)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with np.load(args.recording) as data:
        result=summarize_tracking(data,args.source_frames,args.joint_tolerance,
                                  args.root_tolerance,args.orientation_tolerance)
    content=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(content)
    print(content,end='')


if __name__=='__main__':main()
