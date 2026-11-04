"""Diagnostic: include configured joint-stop avoidance before posture output locks."""
import test_g1 as app
from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP
original=G1DynamicsAccelerationQP.solve

def limit_solve(self,*args,**kwargs):
    names=kwargs.get('priority_tasks',())
    if names and any(r.name=='joint_limit_avoidance' for r in kwargs.get('task_requests') or []):
        kwargs['priority_tasks']=names+('joint_limit_avoidance',)
        levels=kwargs.get('priority_levels')
        if levels:kwargs['priority_levels']=(levels[0]+('joint_limit_avoidance',),)+levels[1:]
    return original(self,*args,**kwargs)
G1DynamicsAccelerationQP.solve=limit_solve
app.main()
