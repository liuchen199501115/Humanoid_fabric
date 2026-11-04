"""Audit a stopped sequence boundary, next load transfer, and following source swing."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch
import warp as wp
from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.contact_transfer import ContactTransferReference
from fabrics_sim.reference.source_step_rollout import SourceStepRollout
from fabrics_sim.reference.source_sequence import next_source_landing
from fabrics_sim.reference.whole_body import WholeBodyReference
from test_g1_two_tasks import G1_JOINT_NAMES,load_motion_trajectory,load_motion_contact_sequence
from analyze_source_sequence import summarize_source_sequence
from g1_armature import unitree_armatures


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--motion-pkl',type=Path,default=Path('retarget/lafan/walk1_g1_with_contacts.pkl'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fixed-source-frame',type=int)
    p.add_argument('--maximum-step-distance',type=float,default=.42)
    p.add_argument('--completed-steps',type=int,default=2)
    p.add_argument('--source-end-frame',type=float)
    p.add_argument('--transfer-height-offset',type=float,default=-.02)
    p.add_argument('--transfer-pelvis-roll-offset',type=float,default=0.)
    p.add_argument('--lift-height',type=float,default=.03)
    p.add_argument('--landing-fraction',type=float,default=.3)
    p.add_argument('--stop-duration',type=float,default=.4)
    p.add_argument('--pelvis-pitch-offset',type=float,default=-.2)
    p.add_argument('--landing-com-fraction',type=float,default=.45)
    p.add_argument('--landing-height',type=float,default=.66)
    a=p.parse_args()
    if a.output.suffix!='.npz':p.error('--output must end in .npz')
    with np.load(a.recording) as d:
        if not summarize_source_sequence(d,a.completed_steps,a.source_end_frame)['sequence_passed']:
            raise ValueError('Require an audited stopped source-sequence recording')
        initial=WholeBodyReference(*[torch.from_numpy(d['history_whole_body_reference_'+k][-1]) for k in ('q','v','a')])
        feet=torch.from_numpy(d['history_feet'][-1]).reshape(1,2,7)
        modes=d['history_commanded_mode'].ravel();last_single=np.flatnonzero(np.isin(modes,[1,2]))[-1]
        support=2-int(modes[last_single]);start=int(d['history_frame'][-1].item());shift=float(d['motion_ground_shift'])
    source,_=load_motion_trajectory(a.motion_pkl);labels=load_motion_contact_sequence(a.motion_pkl)
    if labels is None:raise ValueError('A contact-labelled source is required')
    if a.fixed_source_frame is None:
        takeoff,end=next_source_landing(labels[:,0].numpy().astype(int)+2*labels[:,1].numpy().astype(int),start,support)
    else:
        if a.fixed_source_frame != start:
            raise ValueError('Recovery target must equal the stopped source frame')
        takeoff=end=start
    source[:,2]+=shift
    wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init();torch.set_grad_enabled(False)
    f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
    limits={n.attrib['name']:n.find('limit') for n in ET.parse(f.urdf_path).getroot().findall('joint')}
    bounds=dict(lower=source.new_tensor([float(limits[n].attrib['lower']) for n in G1_JOINT_NAMES]),
        upper=source.new_tensor([float(limits[n].attrib['upper']) for n in G1_JOINT_NAMES]),dt=.002,bandwidth=8.)
    transfer=ContactTransferReference(f,initial,G1_JOINT_NAMES,feet,support,0.,preflight_dt=.002,
        joint_bounds=bounds,com_height_offset=a.transfer_height_offset,lift_height=a.lift_height,pelvis_roll_offset=a.transfer_pelvis_roll_offset)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix('.transfer.json').write_text(json.dumps(transfer.preflight,indent=2)+'\n')
    transfer.sample(2.,3);transfer.sample(2.,support+1);boundary=transfer.sample(5.,support+1)
    print(json.dumps(dict(start_frame=start,takeoff_frame=takeoff,end_frame=end,support=support)),flush=True)
    plan=SourceStepRollout(f,boundary,G1_JOINT_NAMES,source,start,end,support,.002,
        source.new_tensor(unitree_armatures(G1_JOINT_NAMES)),a.output.with_suffix('.json'),joint_bounds=bounds,
        candidate_options=dict(duration=2.,landing_height=a.landing_height,body_join='distributed',clearance=.01,
            swing_phase_blend=.85,landing_com_fraction=a.landing_com_fraction,landing_fraction=1. if a.fixed_source_frame is not None else a.landing_fraction,
            landing_search_depth=.004,terminal_capture='double_support',pelvis_pitch_offset=a.pelvis_pitch_offset),stop_duration=a.stop_duration,
        fixed_source_frame=a.fixed_source_frame,maximum_step_distance=a.maximum_step_distance if a.fixed_source_frame is not None else None)
    np.savez_compressed(a.output,**{k:np.asarray([r[k] for r in plan.rows]) for k in ('q','v','a','frame','time')},
        simulation_validated=False,full_pkl_tracking=False)


if __name__=='__main__':main()
