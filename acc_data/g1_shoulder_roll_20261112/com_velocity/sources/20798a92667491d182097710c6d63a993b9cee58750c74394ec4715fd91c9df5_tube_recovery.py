"""Retry an infeasible experimental joint tube as a soft tracking objective.

Every pre-existing physical inequality/equality and the two root tube rows stay
in force. This never advances physics with an unvalidated solution. Source
fidelity is still independently checked; softening this auxiliary tube is not
permission to exceed the original tracking thresholds.
"""
import torch
from fabrics_sim.fabrics.task_request import TaskRequest

def solve_with_joint_tube_recovery(solver,*,tube_C,tube_d,recovery_metric=1000.,**kwargs):
    solver.tube_recovery_used=False
    result=solver.solve(**kwargs)
    if bool(result.validated.all()):return result
    C,d=kwargs['acceleration_constraints']
    if not torch.equal(C[:,-3:],tube_C) or not torch.equal(d[:,-3:],tube_d):
        raise ValueError('Auxiliary tube rows must be the final three constraints')
    retry=dict(kwargs)
    retry['acceleration_constraints']=(torch.cat((C[:,:-3],C[:,-2:]),1),torch.cat((d[:,:-3],d[:,-2:]),1))
    row=tube_C[:,:1];target=tube_d[:,:1]
    task=TaskRequest('joint_tube_recovery',row,torch.zeros_like(target),target,
        torch.full_like(target[:,:,None],recovery_metric),torch.ones_like(target,dtype=torch.bool))
    retry['task_requests']=list(kwargs['task_requests'])+[task]
    retry['priority_tasks']=tuple(kwargs['priority_tasks'])+(task.name,)
    if kwargs.get('priority_requests') is not None:retry['priority_requests']=list(kwargs['priority_requests'])+[task]
    if kwargs.get('priority_levels') is not None:
        levels=kwargs['priority_levels'];retry['priority_levels']=levels[:-1]+(levels[-1]+(task.name,),)
    solver.reset()
    solver.tube_recovery_used=True
    return solver.solve(**retry)


def solve_with_orientation_tube_recovery(solver,*,tube_C,tube_d,tube_diagnostics,
                                         orientation_bandwidth=10.,recovery_metric=1000.,**kwargs):
    """Retry only after failure, increasing the auxiliary orientation response rate.

    The contact, friction, effort, velocity, collision and joint-limit rows are
    unchanged. This is an experimental tracking heuristic, not an invariance
    certificate; actual source errors are still monitored independently.
    """
    solver.orientation_tube_recovery_bandwidth=float('nan')
    result=solve_with_joint_tube_recovery(solver,tube_C=tube_C,tube_d=tube_d,
        recovery_metric=recovery_metric,**kwargs)
    if bool(result.validated.all()):return result
    C,d=kwargs['acceleration_constraints']
    if not torch.equal(C[:,-3:],tube_C) or not torch.equal(d[:,-3:],tube_d):
        raise ValueError('Auxiliary tube rows must be the final three constraints')
    h,hd=tube_diagnostics[:,2,0],tube_diagnostics[:,2,1]
    for rate in (30.,60.,100.):
        if rate<=orientation_bandwidth:continue
        updated=tube_d.clone()
        updated[:,-1]+=2*(rate-orientation_bandwidth)*hd+(rate**2-orientation_bandwidth**2)*h
        retry=dict(kwargs)
        retry['acceleration_constraints']=(C,torch.cat((d[:,:-3],updated),1))
        solver.reset()
        result=solve_with_joint_tube_recovery(solver,tube_C=tube_C,tube_d=updated,
            recovery_metric=recovery_metric,**retry)
        solver.orientation_tube_recovery_bandwidth=rate
        if bool(result.validated.all()):return result
    return result
