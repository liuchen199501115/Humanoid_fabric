"""Offline fixed-mode contact projection using measured anchors from a recording."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from fabrics_sim.reference.whole_body import WholeBodyReference
from fabrics_sim.reference.anchored_reference import anchor_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('--anchor-recording',type=Path,required=True)
    p.add_argument('--start-frame',type=int,required=True);p.add_argument('--end-frame',type=int,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--warm-start',action='store_true',help='Seed only root translation and legs from previous solved frame')
    p.add_argument('--com-offset-local',nargs=2,type=float,default=[0.,0.],metavar=('X','Y'),help='Constant COM offset in fixed stance axes, metres; maximum norm 0.15')
    p.add_argument('--com-offset-height',type=float,default=0.,help='Constant offset along fixed stance Z, metres')
    p.add_argument('--hold-measured-upper',action='store_true',help='Hold measured waist/arms; retain original task targets before IK')
    p.add_argument('--hold-measured-pelvis',action='store_true',help='Offline experiment: hold measured pelvis orientation, preserving precomputed source task targets')
    args=p.parse_args()
    if not np.isfinite([*args.com_offset_local,args.com_offset_height]).all() or np.linalg.norm([*args.com_offset_local,args.com_offset_height])>.15:
        raise ValueError('COM offset must be finite and at most 0.15 m')
    from test_g1_two_tasks import G1_JOINT_NAMES
    from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
    import warp as wp
    with np.load(args.candidate,allow_pickle=False) as d:
        data={k:d[k].copy() for k in ('q','v','a','time','source_frame','contact_sequence','joint_names','contact_names')}
    if list(data['joint_names'])!=list(G1_JOINT_NAMES) or list(data['contact_names'])!=['left_foot','right_foot']:
        raise ValueError('Candidate order mismatch')
    selected=np.flatnonzero((data['source_frame']>=args.start_frame)&(data['source_frame']<=args.end_frame))
    if not len(selected) or args.end_frame<args.start_frame:raise ValueError('Empty or invalid interval')
    mask=data['contact_sequence'][selected]
    if not np.isin(mask,[0,1]).all() or not np.all(mask==mask[0]) or not mask[0].any():
        raise ValueError('Select a single nonempty fixed contact mode; transitions require a separate plan')
    with np.load(args.anchor_recording,allow_pickle=False) as d:
        anchors=torch.tensor(d['history_feet'][-1].reshape(1,2,7),dtype=torch.float32)
        forces=d['history_foot_forces'][-1].reshape(2,3)[:,2]
        if np.any(forces[mask[0].astype(bool)]<20):raise ValueError('Requested measured anchor is not loaded')
        held_pelvis=torch.tensor(d['history_q'][-1].reshape(1,36)[:,3:7],dtype=torch.float32) if args.hold_measured_pelvis else None
        held_upper=torch.tensor(d['history_q'][-1].reshape(1,36)[:,19:],dtype=torch.float32) if args.hold_measured_upper else None
        shift=float(d['motion_ground_shift']);anchor_time=float(np.asarray(d['history_time']).ravel()[-1])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init();torch.set_grad_enabled(False)
    f=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware');m=f.dynamics_tasks.model
    feet=[m.kinematics.get_link_index(s+'_ankle_roll_link') for s in ('left','right')]
    rows=[];diagnostics=[];failures=[];accepted=[]
    contacts=torch.tensor(mask[:1],dtype=torch.bool)
    from fabrics_sim.taskmaps.g1_end_effector_pose_taskmap import _quat_xyzw_to_matrix
    if (np.any(args.com_offset_local) or args.com_offset_height) and int(mask[0].sum())!=1:
        raise ValueError('Local COM offset requires exactly one fixed stance foot')
    side=int(np.flatnonzero(mask[0])[0])
    local=anchors.new_tensor([*args.com_offset_local,args.com_offset_height])
    com_offset=(_quat_xyzw_to_matrix(anchors[:,side,3:])@local).reshape(1,3)
    for i in selected:
        q=torch.tensor(data['q'][i:i+1],dtype=torch.float32);q[:,2]+=shift
        source=WholeBodyReference(q,*[torch.tensor(data[k][i:i+1],dtype=torch.float32) for k in ('v','a')])
        try:
            ref=anchor_reference(f,source,G1_JOINT_NAMES,anchors,contacts,seed=rows[-1].q if args.warm_start and rows else None,com_offset=com_offset,held_upper=held_upper,held_pelvis=held_pelvis)
            old=m.snapshot(source.q);new,_,_=m.evaluate(ref.q,ref.v,f.integrate_root_position,f.dynamics_tasks.config['difference_step'])
            J=torch.cat((new['origin_jac'][:,feet],new['angular_jac'][:,feet]),2)
            bias=torch.cat((new['origin_jac_bias'][:,feet],new['angular_jac_bias'][:,feet]),2)
            vel=(J@ref.v[:,None,:,None]).squeeze(-1);acc=(J@ref.a[:,None,:,None]).squeeze(-1)+bias
            diagnostics.append(dict(frame=int(data['source_frame'][i]),
                stance_position_error_m=float((new['transforms'][:,feet,:3]-anchors[:,:,:3])[contacts].abs().max()),
                stance_twist_max=float(vel[contacts].abs().max()),stance_acceleration_max=float(acc[contacts].abs().max()),
                swing_position_change_m=float((new['transforms'][:,feet,:3]-old['transforms'][:,feet,:3])[~contacts].abs().max()) if (~contacts).any() else 0.,
                com_change_m=float((new['com']-old['com']).abs().max())))
            rows.append(ref);accepted.append(i)
        except (ValueError,RuntimeError) as exc:failures.append(dict(frame=int(data['source_frame'][i]),reason=str(exc)))
    report=dict(scope='Offline fixed-mode projection; no transition, reset, or physics execution',
        full_tracking_passed=False,hold_measured_upper=args.hold_measured_upper,hold_measured_pelvis=args.hold_measured_pelvis,com_offset_local_m=[*args.com_offset_local,args.com_offset_height],com_offset_world_m=com_offset[0].tolist(),warm_start=args.warm_start,candidate=str(args.candidate),anchor_recording=str(args.anchor_recording),
        anchor_time_s=anchor_time,ground_shift_m=shift,requested_frames=len(selected),projected_frames=len(rows),
        failures=failures,diagnostics=diagnostics)
    if rows:
        report['diagnostic_reference_file']='partial.npz' if failures else 'projected.npz'
        report['complete_interval']=not failures
        np.savez_compressed(args.output_dir/report['diagnostic_reference_file'],**{k:np.concatenate([getattr(r,k).numpy() for r in rows]) for k in ('q','v','a')},
            **{k:data[k][accepted] for k in ('time','source_frame','contact_sequence')},joint_names=data['joint_names'],contact_names=data['contact_names'])
    (args.output_dir/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 1 if failures else 0

if __name__=='__main__':raise SystemExit(main())
