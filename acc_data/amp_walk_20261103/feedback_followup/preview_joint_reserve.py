"""Recover a planner's interior reserve without changing hardware joint limits."""
import numpy as np

def joint_reserve(initial_position,limits,elapsed,reserve=.005,recovery_seconds=.05):
 distance=np.minimum(initial_position-limits[:,0],limits[:,1]-initial_position)
 fraction=np.clip(elapsed/recovery_seconds,0.,1.)
 return np.where(distance<reserve,reserve*fraction,reserve)
