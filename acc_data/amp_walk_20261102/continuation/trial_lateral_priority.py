"""Diagnostic: lateral COM acceleration before the remaining tracking tasks."""
import torch
import test_g1 as app
from fabrics_sim.solver.AccelerationQP import G1DynamicsAccelerationQP
original=G1DynamicsAccelerationQP.solve

def lateral_solve(self,*args,**kwargs):
    names=kwargs.get('priority_tasks',())
    if 'center_of_mass' in names:
        remaining=tuple(n for n in names if n!='center_of_mass')
        kwargs['priority_levels']=(('center_of_mass',),remaining) if remaining else (('center_of_mass',),)
        rows=dict(kwargs.get('priority_rows') or {});rows.pop('center_of_mass',None)
        kwargs['priority_rows']=rows
        kwargs['priority_projections']={'center_of_mass':kwargs['acc_reference'].new_tensor([[[0.,1.,0.]]])}
    return original(self,*args,**kwargs)
G1DynamicsAccelerationQP.solve=lateral_solve
app.main()
