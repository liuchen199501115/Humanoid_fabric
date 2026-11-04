"""Audit exact reference torque and the runtime rectangular contact-wrench cone."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import warp as wp

from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
from fabrics_sim.reference.inverse_dynamics import required_single_support
from fabrics_sim.reference.whole_body import WholeBodyReference
from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
from fabrics_sim.solver.joint_control import measured_joint_acceleration_bounds
from g1_armature import unitree_armatures
from test_g1_two_tasks import G1_JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--support', choices=('left', 'right'), default='left')
    parser.add_argument('--qp-acceleration-limit-scale', type=float, default=5.)
    parser.add_argument('--joint-limit-barrier-bandwidth', type=float, default=8.)
    args = parser.parse_args()
    if not np.isfinite(args.qp_acceleration_limit_scale) or args.qp_acceleration_limit_scale <= 0:
        parser.error('Acceleration limit scale must be finite and positive')
    wp.config.kernel_cache_dir='/tmp/g1_warp_cache'; wp.init(); torch.set_grad_enabled(False)
    fabric=G1PoseFabric(1, 'cpu', robot_name='g1_29dof_rev_1_0', task_profile='dynamics-aware')
    model=fabric.dynamics_tasks.model
    feet=[model.kinematics.get_link_index(side+'_ankle_roll_link') for side in ('left', 'right')]
    support=0 if args.support=='left' else 1
    armature=torch.from_numpy(unitree_armatures(G1_JOINT_NAMES))
    limits={n.attrib['name']:n.find('limit') for n in ET.parse(fabric.urdf_path).getroot().findall('joint')}
    effort=torch.tensor([float(limits[n].attrib['effort']) for n in G1_JOINT_NAMES])
    lower=torch.tensor([float(limits[n].attrib['lower']) for n in G1_JOINT_NAMES])
    upper=torch.tensor([float(limits[n].attrib['upper']) for n in G1_JOINT_NAMES])
    acceleration_limits=args.qp_acceleration_limit_scale*fabric.accel_limits
    rows=[]
    with np.load(args.candidate) as data:
        for i,time in enumerate(data['time']):
            reference=WholeBodyReference(*[torch.from_numpy(data[k][i]) for k in ('q','v','a')])
            result=required_single_support(model,reference,fabric.integrate_root_position,feet[support],armature)
            state=result['state']
            G,b=build_single_support_wrench_constraints(contact_mode=torch.tensor([support+1]),
                foot_rotation_world=state['rotations'][:,feet],toe_length=.12,heel_length=.05,half_width=.025,
                sole_depth=.035,coupled_yaw=True)
            wrench=torch.zeros(1,12);wrench[:,6*support:6*support+6]=result['wrench']
            residual=((G@wrench[...,None]).squeeze(-1)-b).clamp_min(0)
            violation=float(residual.max())
            local=residual[0,20*support:20*(support+1)]
            corners=reference.q.new_tensor([[-.05,-.025,-.035],[-.05,.025,-.035],[.12,-.025,-.035],[.12,.025,-.035]])
            swing=feet[1-support]
            sole=state['transforms'][:,swing,:3,None]+state['rotations'][:,swing]@corners.T
            clearance=float(sole[:,2].min())
            ratios=result['torque'][0].abs()/effort
            index=int(ratios.argmax())
            lo,hi=measured_joint_acceleration_bounds(reference.q[:,7:],reference.v[:,6:],lower,upper,.002,
                                                    barrier_bandwidth=args.joint_limit_barrier_bandwidth)
            joint_violation=torch.maximum(lo-reference.a[:,6:],reference.a[:,6:]-hi).clamp_min(0)
            worst_bound=int(joint_violation[0].argmax())
            rows.append(dict(acceleration_ratio=float((reference.a.abs()/acceleration_limits).max()),
                joint_bound_violation=float(joint_violation.max()),worst_joint_bound=G1_JOINT_NAMES[worst_bound],
                time=float(time),torque_ratio=float(ratios[index]),worst_joint=G1_JOINT_NAMES[index],
                wrench_violation=violation,friction_violation=float(local[:4].max()),
                normal_force_violation=float(local[4:6].max()),cop_violation=float(local[6:10].max()),
                yaw_violation=float(local[10:].max()),swing_clearance=clearance,base_residual=float(result['base_residual'].abs().max()),
                contact_velocity=float(result['contact_velocity'].abs().max()),
                contact_acceleration=float(result['contact_acceleration'].abs().max())))
    worst=max(rows,key=lambda r:r['torque_ratio'])
    passed=all(r['torque_ratio']<=1. and r['wrench_violation']<=1e-3 and r['base_residual']<=1e-3
               and r['contact_velocity']<=1e-3 and r['contact_acceleration']<=1e-3 for r in rows)
    failed=[r for r in rows if r['wrench_violation']>1e-3]
    failure_interval=(None if not failed else dict(start=failed[0]['time'],end=failed[-1]['time'],samples=len(failed),
        swing_clearance_min=min(r['swing_clearance'] for r in failed),swing_clearance_max=max(r['swing_clearance'] for r in failed)))
    runtime_passed=all(r['acceleration_ratio']<=1. and r['joint_bound_violation']<=1e-3 for r in rows)
    report=dict(runtime_acceleration_and_joint_bounds_passed=runtime_passed,
        reference_constraints_passed=passed and runtime_passed,
        qp_acceleration_limit_scale=args.qp_acceleration_limit_scale,
        joint_limit_barrier_bandwidth=args.joint_limit_barrier_bandwidth,
        max_acceleration_ratio=max(r['acceleration_ratio'] for r in rows),
        worst_joint_bound_sample=max(rows,key=lambda r:r['joint_bound_violation']),
        failure_interval=failure_interval, scope='offline_rigid_body_single_support_with_motor_armature',candidate=str(args.candidate),
        support=args.support,samples=len(rows),inverse_dynamics_passed=passed,simulation_validated=False,
        ready_for_tracking=False,full_pkl_tracking=False,worst_torque_sample=worst,
        max_wrench_violation=max(r['wrench_violation'] for r in rows),
        wrench_pass_fraction=float(np.mean([r['wrench_violation']<=1e-3 for r in rows])),
        max_contact_velocity=max(r['contact_velocity'] for r in rows),
        max_contact_acceleration=max(r['contact_acceleration'] for r in rows),
        unchecked=['collision','impact','landing continuation','closed-loop tracking','continuous inter-sample bounds'])
    args.candidate.with_suffix('.dynamics.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.candidate.with_suffix('.dynamics.npz'),**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
