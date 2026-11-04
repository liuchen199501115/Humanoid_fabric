"""Plan the next PKL landing from a measured, audited support-switch recording."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import warp as wp
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.source_step_rollout import SourceStepRollout
from fabrics_sim.reference.whole_body import WholeBodyReference
from test_g1_two_tasks import G1_JOINT_NAMES, load_motion_trajectory, load_motion_contact_sequence
from analyze_support_switch import summarize_support_switch
from g1_armature import unitree_armatures
from fabrics_sim.reference.source_sequence import next_source_landing


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--motion-pkl',type=Path,default=Path('retarget/lafan/walk1_g1_with_contacts.pkl'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--landing-fraction',type=float,default=.4)
    p.add_argument('--landing-height',type=float,default=.66)
    p.add_argument('--body-join',choices=('first_sample','distributed'),default='distributed')
    p.add_argument('--clearance',type=float,default=.01)
    p.add_argument('--swing-phase-blend',type=float,default=.85)
    p.add_argument('--duration',type=float,default=2.)
    p.add_argument('--pelvis-pitch-offset',type=float,default=-.2)
    args=p.parse_args()
    if args.output.suffix != ".npz":
        p.error("--output must end in .npz")
    with np.load(args.recording) as data:
        audit=summarize_support_switch(data)
        if not audit['support_switch_passed']:
            raise ValueError('Recording must pass measured support-switch audit')
        initial=WholeBodyReference(*[torch.from_numpy(data['history_whole_body_reference_'+k][-1]) for k in ('q','v','a')])
        support=int(data['history_commanded_mode'][-1].item())-1
        shift=float(data['motion_ground_shift'])
        start=int(round(float(data['history_frame'][-1].item())))
    source,_=load_motion_trajectory(args.motion_pkl)
    labels=load_motion_contact_sequence(args.motion_pkl)
    if labels is None:
        raise ValueError('Supply the contact-labelled copy of the motion')
    modes=labels[:,0].numpy().astype(int)+2*labels[:,1].numpy().astype(int)
    takeoff,end=next_source_landing(modes,start,support)
    source[:,2]+=shift
    wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init();torch.set_grad_enabled(False)
    fabric=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
    print(json.dumps(dict(start_frame=start,takeoff_frame=takeoff,end_frame=end,support=support)),flush=True)
    plan=SourceStepRollout(fabric,initial,G1_JOINT_NAMES,source,start,end,support,.002,
        source.new_tensor(unitree_armatures(G1_JOINT_NAMES)),args.output.with_suffix('.json'),
        candidate_options=dict(body_join=args.body_join,clearance=args.clearance,duration=args.duration,landing_height=args.landing_height,
            swing_phase_blend=args.swing_phase_blend,landing_com_fraction=.45,landing_fraction=args.landing_fraction,
            landing_search_depth=.004,terminal_capture='double_support',pelvis_pitch_offset=args.pelvis_pitch_offset),
        stop_duration=.5)
    np.savez_compressed(args.output,**{k:np.asarray([r[k] for r in plan.rows]) for k in ('q','v','a','frame','time')},
        full_pkl_tracking=False,simulation_validated=False)


if __name__=='__main__':
    main()
