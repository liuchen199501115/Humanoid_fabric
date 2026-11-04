"""Sampled necessary-condition audit of a prepared G1 reference; no physics."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import linprog


def allocate_wrenches(jac, generalized, cone, bound, effort, contacts):
    """Minimize peak torque/limit with fixed reference acceleration and contacts."""
    effort=np.asarray(effort).reshape(29)
    if np.asarray(contacts).shape != (2,) or not np.any(contacts):
        return dict(feasible=False,peak_torque_ratio=None,reason='No support contact')
    constraints=np.r_[np.c_[cone,np.zeros(len(bound))],
                      np.c_[-jac[:,6:].T,-effort],np.c_[jac[:,6:].T,-effort]]
    bounds=np.r_[bound,-generalized[6:],generalized[6:]]
    variable_bounds=[(None,None) if contacts[i//6] else (0.,0.) for i in range(12)]+[(0.,None)]
    result=linprog(np.r_[np.zeros(12),1.],A_ub=constraints,b_ub=bounds,
        A_eq=np.c_[jac[:,:6].T,np.zeros(6)],b_eq=generalized[:6],bounds=variable_bounds,method='highs')
    return dict(feasible=bool(result.success),peak_torque_ratio=float(result.x[-1]) if result.success else None,
                reason=result.message)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stride',type=int,default=30,help='Source-frame stride; transition neighbors are always included')
    args=parser.parse_args()
    if args.stride<1:raise ValueError('Stride must be positive')
    if args.output.exists():raise FileExistsError(args.output)
    import torch,warp as wp
    from fabrics_sim.fabrics.g1_pose_fabric import G1PoseFabric
    from fabrics_sim.reference.whole_body import WholeBodyReference
    from fabrics_sim.reference.inverse_dynamics import required_single_support
    from fabrics_sim.reference.collision_audit import collision_reference_check
    from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
    from fabrics_sim.solver.joint_control import measured_joint_acceleration_bounds
    from test_g1_two_tasks import G1_JOINT_NAMES
    from g1_armature import unitree_armatures
    with np.load(args.candidate,allow_pickle=False) as data:
        q,v,a=(data[k].copy() for k in ('q','v','a'))
        contacts=data['contact_sequence'].copy();time=data['time'].copy()
        source_frame=data['source_frame'].copy() if 'source_frame' in data else np.arange(len(q))
        if list(data['joint_names'])!=list(G1_JOINT_NAMES):raise ValueError('Candidate joint order mismatch')
        if list(data['contact_names'])!=['left_foot','right_foot']:raise ValueError('Contact ordering mismatch')
    n=len(q)
    if (n<1 or q.shape!=(n,36) or v.shape!=(n,35) or a.shape!=(n,35) or contacts.shape!=(n,2)
        or time.shape!=(n,) or source_frame.shape!=(n,) or not np.isfinite(source_frame).all() or not np.isin(contacts,[0,1]).all()
        or np.any(np.diff(time)<=0) or np.any(np.linalg.norm(q[:,3:7],axis=1)<1e-8)
        or any(not np.isfinite(x).all() for x in (q,v,a,time))):raise ValueError('Invalid candidate arrays')
    contacts=contacts.astype(bool)
    events=np.flatnonzero(np.any(contacts[1:]!=contacts[:-1],axis=1))+1
    indices=np.unique(np.r_[np.arange(0,n,args.stride),events,events-1,n-1])
    wp.config.kernel_cache_dir='/tmp/g1_warp_cache';wp.init();torch.set_grad_enabled(False)
    fabric=G1PoseFabric(1,'cpu',robot_name='g1_29dof_rev_1_0',task_profile='dynamics-aware')
    tasks=fabric.dynamics_tasks;model=tasks.model
    tasks.enable_wrist_hip_avoidance(include_upper_body=True);tasks.wrist_hip_barrier=True
    feet=[model.kinematics.get_link_index(side+'_ankle_roll_link') for side in ('left','right')]
    limits={j.attrib['name']:j.find('limit') for j in ET.parse(fabric.urdf_path).getroot().findall('joint')}
    lower=torch.tensor([float(limits[k].attrib['lower']) for k in G1_JOINT_NAMES])
    upper=torch.tensor([float(limits[k].attrib['upper']) for k in G1_JOINT_NAMES])
    effort=np.array([float(limits[k].attrib['effort']) for k in G1_JOINT_NAMES])
    armature=torch.tensor(unitree_armatures(G1_JOINT_NAMES)).reshape(1,29)
    rows=[]
    for index in indices:
        ref=WholeBodyReference(*[torch.tensor(x[index:index+1],dtype=torch.float32) for x in (q,v,a)])
        result=required_single_support(model,ref,fabric.integrate_root_position,feet[0],armature)
        st=result['state'];J=torch.cat((st['origin_jac'][:,feet],st['angular_jac'][:,feet]),2).flatten(1,2)[0]
        gen=torch.cat((ref.q.new_zeros(6),result['torque'][0]))+J[:6].T@result['wrench'][0]
        mode=int(contacts[index,0])+2*int(contacts[index,1])
        G,b=build_single_support_wrench_constraints(contact_mode=torch.tensor([mode]),
            foot_rotation_world=st['rotations'][:,feet],toe_length=.12,heel_length=.05,
            half_width=.025,sole_depth=.035,coupled_yaw=True)
        allocation=allocate_wrenches(J.numpy(),gen.numpy(),G[0].numpy(),b[0].numpy(),effort,contacts[index])
        sole_wrench_diagnostic=None
        if mode in (1,2):
            side=mode-1;Js=J[6*side:6*(side+1)]
            wrench=torch.linalg.solve(Js[:,:6].T,gen[:6])
            rotation=st['rotations'][0,feet[side]].T
            force=rotation@wrench[:3];moment=rotation@wrench[3:]
            moment-=torch.linalg.cross(force.new_tensor([0.,0.,-.035]),force)
            cop=None if abs(float(force[2]))<1e-8 else [-float(moment[1]/force[2]),float(moment[0]/force[2])]
            sole_wrench_diagnostic=dict(force_local_n=force.tolist(),cop_local_m=cop,
                nominal_rectangle_m=dict(x=[-.05,.12],y=[-.025,.025]))

        bias=torch.cat((st['origin_jac_bias'][:,feet],st['angular_jac_bias'][:,feet]),2).flatten(1,2)[0]
        cv=(J@ref.v[0]).reshape(2,6).numpy();ca=(J@ref.a[0]+bias).reshape(2,6).numpy()
        active=contacts[index]
        cvmax=float(abs(cv[active]).max()) if active.any() else 0.
        camax=float(abs(ca[active]).max()) if active.any() else 0.
        joint_reason=None; joint_state=None
        try:
            lo,hi=measured_joint_acceleration_bounds(ref.q[:,7:],ref.v[:,6:],lower,upper,.002,barrier_bandwidth=8.)
            violation=torch.maximum(lo-ref.a[:,6:],ref.a[:,6:]-hi).clamp_min(0)[0]
            joint_error=float(violation.max());joint_passed=joint_error<=1e-5
            j=int(violation.argmax());worst_joint=G1_JOINT_NAMES[j]
            joint_state=dict(position_rad=float(ref.q[0,7+j]),velocity_rad_s=float(ref.v[0,6+j]),
                acceleration_rad_s2=float(ref.a[0,6+j]),allowed_acceleration_rad_s2=[float(lo[0,j]),float(hi[0,j])],
                position_limits_rad=[float(lower[j]),float(upper[j])],
                nearest_position_limit_distance_rad=float(torch.minimum(ref.q[0,7+j]-lower[j],upper[j]-ref.q[0,7+j])))
        except ValueError as exc:
            joint_error=None;joint_passed=False;worst_joint=None;joint_reason=str(exc)
        collision=collision_reference_check(fabric,ref)
        passed=bool(allocation['feasible'] and allocation['peak_torque_ratio']<=1
            and cvmax<=1e-3 and camax<=1e-3 and joint_passed and collision['passed'])
        rows.append(dict(frame=float(source_frame[index]),time=float(time[index]),mode=mode,passed=passed,
            allocation=allocation,sole_wrench_diagnostic=sole_wrench_diagnostic,contact_twist_max=cvmax,contact_acceleration_max=camax,
            joint_braking_violation=joint_error,joint_braking_passed=joint_passed,
            joint_braking_reason=joint_reason,worst_joint=worst_joint,joint_state=joint_state,
            collision=collision))
        if len(rows)%100==0:print(f'Audited {len(rows)}/{len(indices)} source samples',flush=True)
    report=dict(scope='Sampled exact-reference necessary conditions; not a closed-loop WBC test',
        full_tracking_passed=False,simulation_run=False,candidate=str(args.candidate),
        source_frames=n,samples=len(rows),stride=args.stride,includes_transition_neighbors=True,
        contact_model='Rigid full six-dimensional flat soles; rolling/impact not modeled',
        collision_model='Enabled runtime sphere covers; overlap is not proof of mesh penetration',
        thresholds=dict(contact_twist=1e-3,contact_acceleration=1e-3,joint_braking=1e-5,torque_ratio=1),
        sampled_necessary_conditions_passed=all(r['passed'] for r in rows),
        failures=dict(contact=sum(r['contact_twist_max']>1e-3 or r['contact_acceleration_max']>1e-3 for r in rows),
            torque_or_wrench=sum(not r['allocation']['feasible'] or r['allocation']['peak_torque_ratio']>1 for r in rows),
            joint_braking=sum(not r['joint_braking_passed'] for r in rows),
            collision=sum(not r['collision']['passed'] for r in rows)),rows=rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    return 0 if report['sampled_necessary_conditions_passed'] else 1

if __name__=='__main__':raise SystemExit(main())
