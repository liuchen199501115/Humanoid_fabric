"""Bounded leg IK with a positive knee seed and monotone residual descent."""
import numpy as np
from scipy.spatial.transform import Rotation

def solve_leg_pose(model,q,side,target,target_rotation,max_iterations=100,orientation_weight=.2,position_tolerance=1e-7,orientation_tolerance=1e-6):
    if not np.isfinite(orientation_weight) or orientation_weight<=0:raise ValueError('Positive orientation weight required')
    if not np.isfinite([position_tolerance,orientation_tolerance]).all() or min(position_tolerance,orientation_tolerance)<=0:raise ValueError('Positive finite IK tolerances required')
    q=q.copy();rows=slice(7+6*side,13+6*side);bounds=slice(6*side,6*side+6)
    angles=q[rows].copy()
    if angles[3]<.15:
        change=.15-angles[3];angles[[0,4]]-=.5*change;angles[3]=.15
    q[rows]=np.clip(angles,model.lower[bounds],model.upper[bounds])
    scale=np.array([1.,1.,1.,orientation_weight,orientation_weight,orientation_weight])
    def evaluate(state):
        actual,rotation,jac=model.evaluate(state,side)
        error=np.r_[target-actual,Rotation.from_matrix(target_rotation@rotation.T).as_rotvec()]
        return error,jac
    for _ in range(max_iterations):
        error,J=evaluate(q)
        if np.linalg.norm(error[:3])<position_tolerance and np.linalg.norm(error[3:])<orientation_tolerance:break
        weighted=scale*error;jac=scale[:,None]*J
        delta=jac.T@np.linalg.solve(jac@jac.T+1e-7*np.eye(6),weighted)
        delta*=min(1.,.08/max(abs(delta).max(),1e-12))
        improved=False
        for alpha in [1.,.5,.25,.1,.05,.01]:
            trial=q.copy();trial[rows]=np.clip(q[rows]+alpha*delta,model.lower[bounds],model.upper[bounds])
            next_error,_=evaluate(trial)
            if np.linalg.norm(scale*next_error)<np.linalg.norm(weighted)-1e-12:
                q=trial;improved=True;break
        if not improved:break
    error,_=evaluate(q)
    return q,float(np.linalg.norm(error[:3])),float(np.linalg.norm(error[3:]))
