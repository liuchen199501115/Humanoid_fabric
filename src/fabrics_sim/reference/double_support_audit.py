"""Full rigid-body double-support reference audit; no trajectory planner."""
import numpy as np
import torch
from scipy.optimize import linprog
from .inverse_dynamics import required_single_support
from fabrics_sim.solver.AccelerationQP import build_single_support_wrench_constraints
from fabrics_sim.solver.joint_control import measured_joint_acceleration_bounds

@torch.no_grad()
def audit_stop_reference(plan, times, armature, effort, acceleration_limits, joint_bounds):
    """Allocate both sole wrenches while retaining motor and braking bounds."""
    import xml.etree.ElementTree as ET
    limits={node.attrib['name']:node.find('limit') for node in ET.parse(plan.fabric.urdf_path).getroot().findall('joint')}
    velocity_limits=plan.q.new_tensor([float(limits[name].attrib['velocity']) for name in plan.names])
    rows=[]
    for time in times:
        ref=plan.sample(float(time))
        result=required_single_support(plan.model,ref,plan.fabric.integrate_root_position,plan.feet[0],armature)
        st=result['state']
        J=torch.cat((st['origin_jac'][:,plan.feet],st['angular_jac'][:,plan.feet]),2).flatten(1,2)[0]
        generalized=torch.cat((ref.q.new_zeros(6),result['torque'][0]))+J[:6].T@result['wrench'][0]
        G,b=build_single_support_wrench_constraints(contact_mode=torch.tensor([3],device=ref.q.device),
            foot_rotation_world=st['rotations'][:,plan.feet],toe_length=.12,heel_length=.05,half_width=.025,
            sole_depth=.035,coupled_yaw=True)
        jac,gen,eff=J.cpu().numpy(),generalized.cpu().numpy(),effort.cpu().numpy()
        constraints=np.r_[np.c_[G[0].cpu().numpy(),np.zeros(G.shape[1])],
                          np.c_[-jac[:,6:].T,-eff],np.c_[jac[:,6:].T,-eff]]
        bound=np.r_[b[0].cpu().numpy(),-gen[6:],gen[6:]]
        solution=linprog(np.r_[np.zeros(12),1.],A_ub=constraints,b_ub=bound,
            A_eq=np.c_[jac[:,:6].T,np.zeros(6)],b_eq=gen[:6],bounds=[(None,None)]*12+[(0,None)],method='highs')
        lo,hi=measured_joint_acceleration_bounds(ref.q[:,7:],ref.v[:,6:],joint_bounds['lower'],joint_bounds['upper'],
            joint_bounds['dt'],barrier_bandwidth=joint_bounds['bandwidth'],
            velocity_damper_time=joint_bounds.get('velocity_damper_time'),impact_recovery=joint_bounds.get('impact_recovery',False))
        joint=float(torch.maximum(lo-ref.a[:,6:],ref.a[:,6:]-hi).clamp_min(0).max())
        ratio=float((ref.a.abs()/acceleration_limits).max())
        bias=torch.cat((st['origin_jac_bias'][:,plan.feet],st['angular_jac_bias'][:,plan.feet]),2).flatten(1,2)[0]
        cv=float((J@ref.v[0]).abs().max()); ca=float((J@ref.a[0]+bias).abs().max())
        velocity_ratio=float((ref.v[:,6:].abs()/velocity_limits).max())
        rows.append(dict(velocity_ratio=velocity_ratio,time=float(time),q=ref.q.cpu().numpy(),v=ref.v.cpu().numpy(),a=ref.a.cpu().numpy(),
            wrench_feasible=bool(solution.success),torque_ratio=float(solution.x[-1]) if solution.success else None,
            joint_bound_violation=joint,acceleration_ratio=ratio,contact_velocity=cv,contact_acceleration=ca))
    from .collision_audit import audit_collision_rows
    collision = audit_collision_rows(plan.fabric, rows, plan.q)
    passed=all(r['wrench_feasible'] and r['torque_ratio']<=1 and r['joint_bound_violation']==0.
               and r['acceleration_ratio']<=1 and r['velocity_ratio']<=1 and max(r['contact_velocity'],r['contact_acceleration'])<=1e-3 for r in rows)
    return rows,dict(sampled_necessary_constraints_passed=passed and collision['passed'],samples=len(rows),
        collision_audit=collision,
        max_velocity_ratio=max(r['velocity_ratio'] for r in rows),
        max_joint_bound_violation=max(r['joint_bound_violation'] for r in rows),
        wrench_infeasible_samples=sum(not r['wrench_feasible'] for r in rows),
        peak_torque_ratio=max((r['torque_ratio'] for r in rows if r['torque_ratio'] is not None),default=None),
        scope='double_support_stop_nominal_reference',simulation_validated=False,full_pkl_tracking=False)
